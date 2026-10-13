"""Offline tiny-model tests for real updates, exact resume, and portable reload."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import (BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3Model,
                          XLMRobertaConfig, XLMRobertaModel)

from . import trainer
from .budget import Budget, BudgetExceeded


class _PauseAt:
    def __init__(self, stage, allowed):
        self.stage, self.allowed, self.calls = stage, allowed, 0

    def check(self, stage, training=False):
        if stage == self.stage:
            self.calls += 1
            if self.calls > self.allowed:
                raise BudgetExceeded(stage)
        return 100000.


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".ver29_trainer_", dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        source = self.root / "base"
        source.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "patient", "fever", "pain", "nausea", "breath", "no"]
        (source / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"), do_lower_case=True)
        tokenizer.save_pretrained(source)
        trainer.seed_everything(901)
        model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=16, max_position_embeddings=32,
                                    hidden_dropout_prob=.1, attention_probs_dropout_prob=.1), add_pooling_layer=False)
        model.save_pretrained(source)
        self.spec = {"kind": "control", "model_id": str(source), "snapshot_path": str(source), "revision": "a" * 40}
        self.settings = {"seed": 42, "control_max_length": 24, "control_lr": .002, "head_lr": .003,
                         "effective_batch_size": 4, "batch_candidates": [1, 2, 4], "benchmark_examples": 8,
                         "checkpoint_every_updates": 1, "log_every": 0, "allow_cpu": True}
        self.texts = ["patient " + " ".join(["fever", "pain", "nausea", "breath"][:i % 4 + 1]) for i in range(12)]
        self.targets = np.random.default_rng(71).integers(0, 2, size=(12, 9)).astype(np.float32)
        self.rows, self.dev = np.arange(8), np.arange(8, 12)

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, output, budget=None, resume=False, dev=True, epochs=2):
        return trainer.train_model("control", self.spec, self.settings, self.texts, self.targets,
                                   self.rows, self.dev if dev else None, self.root / output,
                                   budget or Budget(None), epochs, resume=resume)

    def _tuning(self):
        return mock.patch.object(trainer, "_autotune", return_value={"selected_batch_size": 2,
                              "effective_batch_size": 4, "gradient_accumulation_steps": 2,
                              "seconds_per_training_example": .01, "candidates": []})

    def test_resume_is_bitwise_equal_to_uninterrupted_training_with_dropout(self):
        with self._tuning():
            complete = self._run("complete")
            paused = self._run("resumed", _PauseAt("classifier_training", 1))
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["runtime"]["global_updates"], 1)
            result = self._run("resumed", resume=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(complete["history"], result["history"])
        self.assertEqual(complete["best_epoch"], result["best_epoch"])
        first = torch.load(Path(complete["checkpoint_dir"]) / "model.pt", weights_only=True)
        second = torch.load(Path(result["checkpoint_dir"]) / "model.pt", weights_only=True)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]), name)
        self.assertFalse((self.root / "resumed" / "training_state.pt").exists())

    def test_evaluation_pause_does_not_retrain_completed_epoch(self):
        with self._tuning():
            paused = self._run("evaluation_pause", _PauseAt("classifier_evaluation", 0))
            self.assertEqual(paused["runtime"]["global_updates"], 2)
            self.assertEqual(paused["runtime"]["phase"], "evaluate")
            with mock.patch.object(trainer, "_update", wraps=trainer._update) as update:
                result = self._run("evaluation_pause", resume=True)
                self.assertEqual(update.call_count, 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["runtime"]["global_updates"], 4)

    def test_offline_reload_and_hash_integrity(self):
        with self._tuning():
            result = self._run("offline", dev=False)
        self.assertIsNone(result["best_score"])
        self.assertEqual(result["best_epoch"], 2)
        with mock.patch.object(trainer.AutoModel, "from_pretrained", side_effect=AssertionError("Base fetch forbidden")):
            logits = trainer.predict_checkpoint(result["checkpoint_dir"], self.texts, self.dev, batch_size=2)
            self.assertEqual(logits.shape, (4, 9))
            self.assertTrue(np.isfinite(logits).all())
            with np.load(self.root / "offline" / "reload_reference.npz") as reference:
                restored = trainer.predict_checkpoint(result["checkpoint_dir"], self.texts,
                                                       reference["indices"], batch_size=int(reference["batch_size"]))
                np.testing.assert_array_equal(reference["logits"], restored)
        manifest = Path(result["checkpoint_dir"]) / "manifest.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["files"]["model.pt"] = "0" * 64
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            trainer.load_checkpoint(result["checkpoint_dir"])

    def test_changed_data_or_scheduler_horizon_cannot_resume(self):
        with self._tuning():
            self._run("identity", _PauseAt("classifier_training", 1))
        with self.assertRaisesRegex(ValueError, "identity"):
            self._run("identity", resume=True, epochs=3)
        self.targets[0, 0] = 1 - self.targets[0, 0]
        with self.assertRaisesRegex(ValueError, "identity"):
            self._run("identity", resume=True)

    def test_autotune_runs_real_optimizer_steps_and_reset_is_independent(self):
        tuning = trainer._autotune("control", self.spec, self.settings, self.texts, self.targets,
                                   self.rows, torch.device("cpu"), Budget(None))
        self.assertEqual([entry["status"] for entry in tuning["candidates"]], ["ok"] * 3)
        self.assertIn(tuning["selected_batch_size"], (1, 2, 4))
        self.assertTrue(all(entry["examples"] == 8 for entry in tuning["candidates"]))
        self.assertGreater(tuning["seconds_per_training_example"], 0)

    def test_weight_formula_and_disjoint_indices(self):
        values = np.zeros((10, 9), dtype=np.float32)
        values[:2, 0] = 1
        values[:, 1] = 1
        weights = trainer.positive_weights(values)
        self.assertEqual(float(weights[0]), 2.)
        self.assertEqual(float(weights[1]), 1.)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            trainer.train_model("control", self.spec, self.settings, self.texts, self.targets,
                                self.rows, self.rows, self.root / "overlap", Budget(None), 1)

    def test_last_partial_effective_batch_and_completed_run_relocation(self):
        self.rows, self.dev = np.arange(9), np.arange(9, 12)
        with self._tuning(), mock.patch.object(trainer, "_update", wraps=trainer._update) as update:
            result = self._run("tail", epochs=1)
            groups = [call.args[4].tolist() for call in update.call_args_list]
            self.assertEqual(sorted(sum(groups, [])), list(range(9)))
            self.assertEqual([len(group) for group in groups], [4, 4, 1])
        self.assertEqual(result["history"][0]["train_examples"], 9)
        shutil.copytree(self.root / "tail", self.root / "moved")
        moved = self._run("moved", resume=True, epochs=1)
        self.assertEqual(Path(moved["checkpoint_dir"]), self.root / "moved" / "best_model")

    def test_kure_and_optional_qwen_architectures_train_and_reload_offline(self):
        for kind in ("kure", "qwen_embedding"):
            with self.subTest(kind=kind):
                source = self.root / (kind + "_base")
                tokenizer = BertTokenizerFast.from_pretrained(self.root / "base", local_files_only=True)
                tokenizer.save_pretrained(source)
                if kind == "kure":
                    encoder = XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tokenizer), hidden_size=8,
                        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                        max_position_embeddings=34, pad_token_id=0), add_pooling_layer=False)
                else:
                    encoder = Qwen3Model(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, intermediate_size=24, max_position_embeddings=96, pad_token_id=0))
                encoder.save_pretrained(source)
                spec = {"kind": kind, "model_id": str(source), "snapshot_path": str(source), "revision": "b" * 40}
                settings = {**self.settings, f"{kind}_max_length": 64 if kind == "qwen_embedding" else 24,
                            f"{kind}_lr": .002}
                with self._tuning():
                    result = trainer.train_model(kind, spec, settings, self.texts, self.targets,
                                self.rows, self.dev, self.root / kind, Budget(None), 1)
                with np.load(self.root / kind / "reload_reference.npz") as reference:
                    logits = trainer.predict_checkpoint(result["checkpoint_dir"], self.texts,
                                      reference["indices"], batch_size=int(reference["batch_size"]))
                    np.testing.assert_array_equal(logits, reference["logits"])
                if kind == "qwen_embedding":
                    stats = json.loads((self.root / kind / "length_statistics.json").read_text(encoding="utf-8"))
                    self.assertEqual(stats["truncated_training_rows"], 0)
                    with self._tuning():
                        trainer.train_model(kind, spec, settings, self.texts, self.targets, self.rows, self.dev,
                                            self.root / "qwen_resume", _PauseAt("classifier_training", 1), 1)
                        resumed = trainer.train_model(kind, spec, settings, self.texts, self.targets,
                            self.rows, self.dev, self.root / "qwen_resume", Budget(None), 1, resume=True)
                    original_model, _, metadata = trainer.load_checkpoint(result["checkpoint_dir"])
                    resumed_model, _, resumed_metadata = trainer.load_checkpoint(resumed["checkpoint_dir"])
                    self.assertEqual(metadata, resumed_metadata)
                    self.assertEqual(metadata["model_protocol"], trainer.model_protocol(kind))
                    for name, parameter in original_model.state_dict().items():
                        self.assertTrue(torch.equal(parameter, resumed_model.state_dict()[name]), name)

    def test_native_bfloat16_only_and_cpu_never_queries_cuda(self):
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True) as supports:
            self.assertIsNone(trainer._amp_dtype("cpu"))
            supports.assert_not_called()
            with mock.patch.object(torch.cuda, "device", return_value=nullcontext()):
                self.assertEqual(trainer._amp_dtype("cuda:0"), torch.bfloat16)
                supports.assert_called_once_with(including_emulation=False)
        with mock.patch.object(torch.cuda, "device", return_value=nullcontext()), \
                mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation):
            self.assertEqual(trainer._amp_dtype("cuda:0"), torch.float16)

    def test_asymmetric_loss_math_gradient_and_extreme_logits(self):
        logits = torch.zeros((1, 9), requires_grad=True)
        labels = torch.zeros_like(logits)
        weights = torch.arange(1, 10, dtype=torch.float32)
        loss = trainer.classification_loss(logits, labels, weights, "klue_asl")
        self.assertAlmostEqual(float(loss.detach()), .25 * np.log(2), places=7)
        loss.backward()
        # This derivative includes the derivative of p**2; detached focusing would be .125 / 9.
        expected = (.125 + .25 * np.log(2)) / 9
        torch.testing.assert_close(logits.grad, torch.full_like(logits, expected))
        positives = torch.zeros((1, 9), requires_grad=True)
        trainer.classification_loss(positives, torch.ones_like(positives), weights, "klue_asl").backward()
        torch.testing.assert_close(positives.grad, -.5 * weights.unsqueeze(0) / 9)
        extremes = torch.tensor([[-10000., 10000., -1000., 1000., -30., 30., -2., 2., 0.]], requires_grad=True)
        loss = trainer.classification_loss(extremes, torch.tensor([[1., 0., 0., 1., 1., 0., 1., 0., 0.]]),
                                           weights, "klue_asl")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extremes.grad).all())
        control = trainer.classification_loss(extremes.detach(), labels, weights, "control")
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(extremes.detach(), labels, pos_weight=weights)
        torch.testing.assert_close(control, expected_bce, rtol=0., atol=0.)

    def test_asymmetric_loss_accumulation_matches_whole_effective_batch(self):
        initial = torch.linspace(-4, 4, 45).reshape(5, 9)
        full = initial.clone().requires_grad_(True)
        parts = initial.clone().requires_grad_(True)
        labels = torch.as_tensor(self.targets[:5])
        weights = trainer.positive_weights(self.targets[self.rows])
        trainer.classification_loss(full, labels, weights, "klue_asl").backward()
        for start, stop in ((0, 2), (2, 4), (4, 5)):
            (trainer.classification_loss(parts[start:stop], labels[start:stop], weights, "klue_asl")
             * ((stop - start) / 5)).backward()
        torch.testing.assert_close(parts.grad, full.grad)

    def test_fixed_instruction_tokenization_lengths_and_identity_are_consistent(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.root / "base", local_files_only=True)
        prepared = trainer.prepare_texts("qwen_embedding", self.texts)
        self.assertEqual(prepared[0], f"Instruct: {trainer.QWEN_INSTRUCTION}\nQuery: {self.texts[0]}")
        self.assertEqual(trainer.prepare_texts("klue_asl", self.texts), self.texts)
        encoded = trainer._tokenize(tokenizer, self.texts, 64, "qwen_embedding")
        direct = tokenizer(prepared, truncation=True, max_length=64, padding=False)
        self.assertEqual([item["input_ids"] for item in encoded], direct["input_ids"])
        stats = trainer._length_statistics(tokenizer, self.texts, self.rows, 64, "qwen_embedding")
        lengths = [len(ids) for ids in tokenizer(prepared[:8], truncation=False, padding=False)["input_ids"]]
        self.assertEqual(stats["raw_token_percentiles"]["50"], float(np.percentile(lengths, 50)))
        identity = trainer._identity("qwen_embedding", self.spec, self.settings, self.texts,
                                     self.targets, self.rows, self.dev, 2)
        with mock.patch.object(trainer, "QWEN_INSTRUCTION", "A different task."):
            changed = trainer._identity("qwen_embedding", self.spec, self.settings, self.texts,
                                        self.targets, self.rows, self.dev, 2)
        self.assertNotEqual(identity, changed)

    def test_asl_updates_encoder_resumes_and_preserves_portable_protocol(self):
        spec = {**self.spec, "kind": "klue_asl"}
        settings = {**self.settings, "klue_asl_max_length": 24, "klue_asl_lr": .002}
        with self._tuning():
            trainer.train_model("klue_asl", spec, settings, self.texts, self.targets, self.rows, self.dev,
                                self.root / "asl", _PauseAt("classifier_training", 1), 2)
            result = trainer.train_model("klue_asl", spec, settings, self.texts, self.targets,
                                         self.rows, self.dev, self.root / "asl", Budget(None), 2, resume=True)
            whole = trainer.train_model("klue_asl", spec, settings, self.texts, self.targets,
                                        self.rows, self.dev, self.root / "asl_whole", Budget(None), 2)
        self.assertEqual(result["history"], whole["history"])
        model, tokenizer, metadata = trainer.load_checkpoint(result["checkpoint_dir"])
        self.assertEqual(metadata["model_protocol"], trainer.model_protocol("klue_asl"))
        self.assertEqual(metadata["pooling"], "masked_mean")
        source = BertModel.from_pretrained(self.spec["snapshot_path"], local_files_only=True, add_pooling_layer=False)
        self.assertFalse(torch.equal(model.encoder.embeddings.word_embeddings.weight,
                                     source.embeddings.word_embeddings.weight))
        other, _, _ = trainer.load_checkpoint(whole["checkpoint_dir"])
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, other.state_dict()[name]), name)
        manifest_path = Path(result["checkpoint_dir"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["model_protocol"]["loss"]["gamma_negative"] = 4.
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "protocol"):
            trainer.load_checkpoint(result["checkpoint_dir"])

    def test_qwen_last_pooling_is_invariant_to_left_right_padding_and_batch(self):
        encoder = Qwen3Model(Qwen3Config(vocab_size=16, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, intermediate_size=24,
            max_position_embeddings=64, pad_token_id=0))
        model = trainer.EncoderClassifier(encoder, "qwen_embedding", dropout=.1).eval()
        with torch.no_grad():
            alone = model(torch.tensor([[2, 5, 6, 3]]), torch.ones((1, 4), dtype=torch.long))
            right = model(torch.tensor([[2, 5, 6, 3, 0, 0], [2, 5, 6, 7, 8, 3]]),
                          torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]))
            left = model(torch.tensor([[0, 0, 2, 5, 6, 3], [2, 5, 6, 7, 8, 3]]),
                         torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]))
        torch.testing.assert_close(alone[0], right[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(alone[0], left[0], rtol=1e-5, atol=1e-6)

    def test_overflow_replay_restores_dropout_and_applies_one_update(self):
        class SimulatedOverflowScaler:
            def __init__(self):
                self.value, self.attempts, self.accepted = 8., 0, 0

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return None

            def is_enabled(self):
                return True

            def get_scale(self):
                return self.value

            def step(self, optimizer):
                self.attempts += 1
                if self.attempts > 1:
                    optimizer.step()
                    self.accepted += 1

            def update(self):
                self.value = 4.

        trainer.seed_everything(42)
        original, tokenizer = trainer.create_model("control", self.spec, self.settings, torch.device("cpu"))
        trainer.seed_everything(42)
        replayed, _ = trainer.create_model("control", self.spec, self.settings, torch.device("cpu"))
        tokens = trainer._tokenize(tokenizer, self.texts, 24)
        weights = trainer.positive_weights(self.targets[self.rows])
        plain = torch.amp.GradScaler("cuda", enabled=False)
        replay = SimulatedOverflowScaler()
        trainer.seed_everything(876)
        expected_loss, retry = trainer._update(original, tokenizer, tokens, self.targets, self.rows[:4], 2,
            trainer._optimizer(original, "control", self.settings), plain, weights, torch.device("cpu"))
        trainer.seed_everything(876)
        actual_loss, retries = trainer._update(replayed, tokenizer, tokens, self.targets, self.rows[:4], 2,
            trainer._optimizer(replayed, "control", self.settings), replay, weights, torch.device("cpu"))
        self.assertEqual(retry, 0)
        self.assertEqual(retries, 1)
        self.assertEqual(replay.attempts, 2)
        self.assertEqual(replay.accepted, 1)
        self.assertEqual(actual_loss, expected_loss)
        for name, parameter in original.state_dict().items():
            self.assertTrue(torch.equal(parameter, replayed.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
