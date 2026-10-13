"""Offline, real tiny-transformer checks for the ver25 training contract."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3Model

from . import trainer


class OverflowScaler:
    """Deterministically exercise GradScaler's skipped-step contract on CPU."""
    def __init__(self, skipped):
        self.skipped, self.attempt, self.value = set(skipped), 0, 64.

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        if self.attempt in self.skipped:
            next(parameter for group in optimizer.param_groups for parameter in group["params"]
                 if parameter.grad is not None).grad.fill_(float("inf"))

    def step(self, optimizer):
        if self.attempt not in self.skipped:
            optimizer.step()

    def update(self):
        if self.attempt in self.skipped:
            self.value /= 2
        self.attempt += 1


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        root = Path(__file__).resolve().parents[3] / "outputs" / "mission3_ver25_verification"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                      "hello", "world", "short", "long", "test", "medical", "sample", "."]
        vocab = self.directory / "vocab.txt"
        vocab.write_text("\n".join(vocabulary), encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        self.texts = ["hello world", "short", "medical sample", "long test hello", "sample"]
        self.targets = np.asarray([[int((i + j) % 3 == 0) for j in range(9)]
                                   for i in range(len(self.texts))], dtype=np.int8)
        self.settings = {"base_store_dir": str(self.directory / "run" / "shared_base"),
                         "test_only_unquantized": True, "qwen_max_length": 32,
                         "control_max_length": 32, "lora_dropout": 0., "control_dropout": 0.}

    def control(self):
        config = BertConfig(vocab_size=len(self.tokenizer), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24, max_position_embeddings=64,
                            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        return trainer.MeanPoolClassifier(BertModel(config), dropout=0.)

    def qwen(self):
        snapshot = self.directory / "pretrained_qwen"
        if not snapshot.exists():
            config = Qwen3Config(vocab_size=len(self.tokenizer), hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                 intermediate_size=24, max_position_embeddings=128,
                                 pad_token_id=0, eos_token_id=3, attention_dropout=0.)
            Qwen3Model(config).save_pretrained(snapshot)
            self.tokenizer.save_pretrained(snapshot)
        spec = {"kind": "qwen", "model_id": str(snapshot), "revision": "a" * 40}
        model, tokenizer = trainer.create_model(spec, self.settings, "cpu", local_files_only=True)
        return model, tokenizer, spec

    def test_weight_rule_and_independent_zero_boundary(self):
        targets = np.zeros((100, 9))
        targets[:25, 0] = 1
        targets[:50, 1] = 1
        weights = trainer.positive_weights(targets)
        self.assertAlmostEqual(weights[0].item(), np.sqrt(3), places=6)
        self.assertEqual(weights[1].item(), 1.)
        self.assertEqual(weights[2].item(), 5.)
        logits = np.full((2, 9), -1.)
        logits[1, :2] = 0.
        np.testing.assert_array_equal(trainer.fixed_decisions(logits).sum(1), [0, 2])
        with self.assertRaises(ValueError):
            trainer.positive_weights(np.full((2, 9), .5))

    def test_qwen_head_tail_preserves_full_suffix(self):
        suffix = self.tokenizer.encode(trainer.JUDGMENT_SUFFIX, add_special_tokens=False)
        text = "hello world short long test medical sample hello world long"
        raw = self.tokenizer.encode(text, add_special_tokens=False)
        actual = trainer.qwen_token_ids(self.tokenizer, text, len(suffix) + 5)
        self.assertEqual(actual, raw[:3] + raw[-2:] + suffix)
        self.assertEqual(trainer.qwen_token_ids(self.tokenizer, "", 32), suffix)
        with self.assertRaises(ValueError):
            trainer.qwen_token_ids(self.tokenizer, text, len(suffix) + 1)
        stats = trainer.length_statistics(self.tokenizer, [text, "short"], "qwen", len(suffix) + 5)
        self.assertEqual(stats["overlength_count"], 1)
        self.assertEqual(stats["suffix_tokens"], len(suffix))

    def test_real_qwen_lora_and_head_receive_gradients_base_stays_frozen(self):
        model, tokenizer, _ = self.qwen()
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        batch = next(iter(trainer.make_loader(tokenizer, self.texts, self.targets,
                                              kind="qwen", batch_size=2, max_length=32)))
        logits = model(**{name: value for name, value in batch.items() if name != "labels"})
        torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"],
                                                             pos_weight=torch.arange(1., 10.)).backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(all("lora_" in name or name.startswith("classifier.") for name, _ in trainable))
        self.assertTrue(any("lora_B" in name and parameter.grad.abs().sum() > 0 for name, parameter in trainable))
        self.assertGreater(model.classifier.weight.grad.abs().sum(), 0)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if "lora_" not in name and not name.startswith("classifier.")))
        config = model.encoder.peft_config["default"]
        self.assertEqual(config.r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.target_modules, set(trainer.LORA_TARGETS))
        self.assertFalse(model.encoder.config.use_cache)
        optimizer = trainer.build_optimizer(model, "qwen", self.settings)
        self.assertEqual({group["lr"] for group in optimizer.param_groups}, {1e-4, 3e-4})
        head_ids = {id(parameter) for parameter in model.classifier.parameters()}
        self.assertTrue(all(group["lr"] == 3e-4 for group in optimizer.param_groups
                            if any(id(parameter) in head_ids for parameter in group["params"])))

    def test_last_nonpadding_pooling_matches_left_right_and_unpadded(self):
        model, _, _ = self.qwen()
        model.eval()
        plain = model(torch.tensor([[5, 6, 7]]), torch.tensor([[1, 1, 1]]))
        right = model(torch.tensor([[5, 6, 7, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        left = model(torch.tensor([[0, 0, 5, 6, 7]]), torch.tensor([[0, 0, 1, 1, 1]]))
        torch.testing.assert_close(plain, right, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(plain, left, atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            model(torch.tensor([[0]]), torch.tensor([[0]]))

    def test_explicit_weighted_bce_and_partial_accumulation(self):
        model = self.control()
        whole = deepcopy(model)
        micro = trainer.make_loader(self.tokenizer, self.texts, self.targets,
                                     kind="control", batch_size=2, max_length=32)
        batch = trainer.make_loader(self.tokenizer, self.texts, self.targets,
                                     kind="control", batch_size=5, max_length=32)
        weights = torch.arange(1., 10.)
        raw = next(iter(batch))
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            whole(**{name: value for name, value in raw.items() if name != "labels"}),
            raw["labels"], pos_weight=weights).item()
        stats = trainer.train_epoch(model, micro, torch.optim.SGD(model.parameters(), lr=.01),
                                    device="cpu", pos_weight=weights, accumulation_steps=3)
        other_stats = trainer.train_epoch(whole, batch, torch.optim.SGD(whole.parameters(), lr=.01),
                                          device="cpu", pos_weight=weights, accumulation_steps=1)
        self.assertAlmostEqual(other_stats["mean_loss"], expected_loss, places=6)
        self.assertTrue(stats["completed_epoch"])
        self.assertEqual(stats["examples"], 5)
        for first, second in zip(model.parameters(), whole.parameters()):
            torch.testing.assert_close(first, second, atol=1e-7, rtol=1e-5)

    def test_pilot_caps_microsteps_flushes_tail_and_counts_timing(self):
        model = self.control()
        texts = ["hello world"] * 20
        targets = np.tile(self.targets[0], (20, 1))
        loader = trainer.make_loader(self.tokenizer, texts, targets, kind="control", batch_size=2, max_length=32)
        optimizer = trainer.build_optimizer(model, "control", self.settings)
        scheduler = trainer.build_scheduler(optimizer, 3)
        result = trainer.train_epoch(model, loader, optimizer, device="cpu", pos_weight=torch.ones(9),
                                      accumulation_steps=4, max_microsteps=6, scheduler=scheduler)
        self.assertEqual(result["microsteps"], 6)
        self.assertEqual(result["examples"], 12)
        self.assertEqual(result["successful_updates"], 2)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(len(result["microstep_seconds"]), 6)
        self.assertEqual(len(result["measured_microstep_seconds"]), 2)
        self.assertGreater(result["seconds_per_microstep"], 0)
        self.assertFalse(result["completed_epoch"])
        self.assertEqual(result["peak_vram_bytes"], 0)

    def test_overflow_replays_every_group_row_once_with_identical_dropout(self):
        replayed = self.control()
        replayed.dropout = torch.nn.Dropout(.2)
        reference = deepcopy(replayed)
        settings = dict(kind="control", batch_size=2, max_length=32)
        first = trainer.make_loader(self.tokenizer, self.texts, self.targets, **settings)
        second = trainer.make_loader(self.tokenizer, self.texts, self.targets, **settings)
        replay_optimizer = torch.optim.SGD(replayed.parameters(), lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        replay_scheduler = trainer.build_scheduler(replay_optimizer, 2)
        reference_scheduler = trainer.build_scheduler(reference_optimizer, 2)
        kwargs = dict(device="cpu", pos_weight=trainer.positive_weights(self.targets), accumulation_steps=2)
        progress = []
        torch.manual_seed(711)
        stats = trainer.train_epoch(replayed, first, replay_optimizer, scheduler=replay_scheduler,
                                    scaler=OverflowScaler([0]), on_update=progress.append, **kwargs)
        torch.manual_seed(711)
        trainer.train_epoch(reference, second, reference_optimizer, scheduler=reference_scheduler, **kwargs)
        self.assertTrue(stats["completed_epoch"])
        self.assertEqual(stats["examples"], 5)
        self.assertEqual(stats["microsteps"], 3)
        self.assertEqual(len(stats["microstep_seconds"]), 3)
        self.assertEqual(stats["successful_updates"], 2)
        self.assertEqual(stats["attempted_updates"], 3)
        self.assertEqual(stats["overflow_retries"], 1)
        self.assertEqual(stats["final_scaler_scale"], 32.)
        self.assertEqual(replay_scheduler.last_epoch, 2)
        self.assertEqual([step["examples"] for step in progress], [4, 5])
        for actual, expected in zip(replayed.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_overflow_replay_is_bounded_and_leaves_no_partial_update(self):
        model = self.control()
        before = deepcopy(model.state_dict())
        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets,
                                     kind="control", batch_size=2, max_length=32)
        optimizer = trainer.build_optimizer(model, "control", self.settings)
        scheduler = trainer.build_scheduler(optimizer, 2)
        scaler = OverflowScaler(range(10))
        with self.assertRaisesRegex(FloatingPointError, "retry limit exceeded"):
            trainer.train_epoch(model, loader, optimizer, device="cpu", pos_weight=torch.ones(9),
                                accumulation_steps=2, scheduler=scheduler, scaler=scaler,
                                max_overflow_retries=2)
        self.assertEqual(scaler.attempt, 3)
        self.assertEqual(scheduler.last_epoch, 0)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_overflow_replay_checks_budget_before_retraining(self):
        class ReplayBudget:
            def check(self, stage, training=False):
                if stage == "classifier_overflow_replay":
                    raise TimeoutError("replay budget exhausted")

        model = self.control()
        before = deepcopy(model.state_dict())
        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets,
                                     kind="control", batch_size=2, max_length=32)
        with self.assertRaisesRegex(TimeoutError, "replay budget"):
            trainer.train_epoch(model, loader, trainer.build_optimizer(model, "control", self.settings),
                                device="cpu", pos_weight=torch.ones(9), accumulation_steps=2,
                                scaler=OverflowScaler([0]), budget=ReplayBudget())
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_nf4_configuration_and_checkpointing_are_explicit(self):
        model, _, spec = self.qwen()
        encoder = Qwen3Model(model.encoder.config)
        settings = dict(self.settings, test_only_unquantized=False)
        with patch.object(trainer, "BitsAndBytesConfig") as quantization, \
                patch.object(trainer.AutoModel, "from_pretrained", return_value=encoder) as load, \
                patch.object(trainer, "_amp_dtype", return_value=torch.bfloat16), \
                patch.object(trainer, "_save_frozen_base", return_value=self.directory), \
                patch("peft.prepare_model_for_kbit_training", side_effect=lambda model, **_: model.requires_grad_(False)) as prepare, \
                patch.object(torch.nn.Module, "to", lambda self, *_, **__: self):
            result, _ = trainer.create_model(spec, settings, "cuda:0", local_files_only=True)
        quantization.assert_called_once_with(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16)
        self.assertEqual(load.call_args.kwargs["device_map"], {"": "cuda:0"})
        prepare.assert_called_once()
        self.assertTrue(prepare.call_args.kwargs["use_gradient_checkpointing"])
        self.assertEqual(prepare.call_args.kwargs["gradient_checkpointing_kwargs"], {"use_reentrant": False})
        self.assertEqual(result.kind, "qwen")

    def test_qwen_shared_base_adapter_head_roundtrip_after_relocation(self):
        model, tokenizer, spec = self.qwen()
        loader = trainer.make_loader(tokenizer, self.texts, self.targets, kind="qwen", batch_size=2, max_length=32)
        trainer.train_epoch(model, loader, trainer.build_optimizer(model, "qwen", self.settings),
                            device="cpu", pos_weight=trainer.positive_weights(self.targets))
        expected = trainer.predict_texts(model, tokenizer, self.texts, max_length=32)
        destination = self.directory / "run" / "fit_qwen"
        metadata = trainer.save_model(destination, model, tokenizer, spec, self.settings, "fit_only", 1)
        base_files = set(Path(model._base_directory).iterdir())
        fresh, _, _ = self.qwen()
        self.assertEqual(set(Path(fresh._base_directory).iterdir()), base_files)
        relocated = self.directory / "relocated"
        shutil.copytree(self.directory / "run", relocated)
        with patch("huggingface_hub.file_download.http_get", side_effect=AssertionError("network access")):
            loaded, loaded_tokenizer, saved = trainer.load_model(relocated / "fit_qwen" / "manifest.json")
        self.assertEqual(metadata, saved)
        actual = trainer.predict_texts(loaded, loaded_tokenizer, self.texts, max_length=saved["max_length"])
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_array_equal(trainer.fixed_decisions(actual), trainer.fixed_decisions(expected))
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("head.pt", manifest["files"])
        base_state = next(Path(model._base_directory).glob("*.safetensors"))
        from safetensors import safe_open
        with safe_open(base_state, framework="pt") as handle:
            self.assertTrue(all("lora_" not in key and "base_layer" not in key for key in handle.keys()))
        (relocated / "fit_qwen" / "head.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "integrity"):
            trainer.load_model(relocated / "fit_qwen")

    def test_control_roundtrip_without_hub(self):
        model = self.control()
        spec = {"kind": "control", "model_id": "tiny-local-bert", "revision": "b" * 40}
        destination = self.directory / "control"
        expected = trainer.predict_texts(model, self.tokenizer, self.texts, max_length=32)
        trainer.save_model(destination, model, self.tokenizer, spec, self.settings, "all_training", 2)
        with patch("huggingface_hub.file_download.http_get", side_effect=AssertionError("network access")):
            loaded, tokenizer, metadata = trainer.load_model(destination)
        self.assertEqual(metadata["selected_epochs"], 2)
        self.assertEqual(metadata["training_scope"], "all_training")
        actual = trainer.predict_texts(loaded, tokenizer, self.texts, max_length=32)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_production_qwen_does_not_silently_fall_back_to_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            trainer.create_model({"kind": "qwen", "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                                  "revision": "a" * 40}, {}, "cpu")
        with self.assertRaisesRegex(ValueError, "immutable"):
            trainer.create_model({"kind": "qwen", "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                                  "revision": "main"}, {}, "cuda:0")


if __name__ == "__main__":
    unittest.main()
