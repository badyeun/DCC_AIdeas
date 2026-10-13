"""Focused real-encoder CPU checks for the ver24 matched recipe."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, RobertaConfig, RobertaModel

from . import trainer


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        # Temporary files stay inside the repository's writable output tree.
        base = Path(__file__).resolve().parents[3] / "outputs" / "mission3_ver24_verification"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                      "hello", "world", "short", "long", "test", "medical", "sample"]
        (self.directory / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(self.directory / "vocab.txt"), do_lower_case=False)
        self.config = BertConfig(vocab_size=len(vocabulary), hidden_size=16,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=24, max_position_embeddings=64,
                                 hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        torch.manual_seed(42)
        self.model = trainer.MeanPoolClassifier(BertModel(self.config), dropout=0.)
        self.texts = ["hello world", "short", "medical sample", "long test hello", "sample"]
        self.targets = np.asarray([[int((i + j) % 3 == 0) for j in range(9)]
                                   for i in range(len(self.texts))], dtype=np.int8)

    def test_sqrt_weights_use_only_supplied_binary_rows(self):
        values = np.zeros((100, 9), dtype=np.int8)
        values[:25, 0] = 1
        values[:50, 1] = 1
        values[:1, 2] = 1
        weights = trainer.positive_weights(values).numpy()
        self.assertAlmostEqual(weights[0], np.sqrt(3), places=6)
        self.assertEqual(weights[1], 1)
        self.assertEqual(weights[2], 5)
        self.assertEqual(weights[3], 5)
        with self.assertRaises(ValueError):
            trainer.positive_weights(np.full((3, 9), .5))

    def test_decisions_are_fixed_and_empty_rows_remain_empty(self):
        logits = np.ones((2, 9)) * -2
        logits[1, 6] = 0
        result = trainer.fixed_decisions(logits)
        self.assertEqual(result[0].sum(), 0)
        self.assertEqual(result[1].sum(), 1)
        self.assertEqual(result[1, 6], 1)
        with self.assertRaises(ValueError):
            trainer.fixed_decisions(np.full((2, 9), np.nan))

    def test_batch_padding_does_not_change_real_logits(self):
        self.model.eval()
        alone = trainer.predict_texts(self.model, self.tokenizer, ["short"], max_length=32)
        together = trainer.predict_texts(self.model, self.tokenizer,
                                         ["short", "hello world medical long test"], max_length=32)
        np.testing.assert_allclose(alone[0], together[0], atol=2e-7)

    def test_loader_uses_unique_indices_and_keeps_final_row(self):
        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets,
                                     indices=np.array([4, 2, 1]), batch_size=2, max_length=32)
        batches = list(loader)
        self.assertEqual([len(batch["labels"]) for batch in batches], [2, 1])
        np.testing.assert_array_equal(batches[0]["labels"].numpy(), self.targets[[4, 2]])
        with self.assertRaises(ValueError):
            trainer.make_loader(self.tokenizer, self.texts, indices=[0, 0])

    def test_partial_accumulation_matches_one_whole_batch(self):
        other = deepcopy(self.model)
        first = trainer.make_loader(self.tokenizer, self.texts, self.targets, batch_size=2, max_length=32)
        second = trainer.make_loader(self.tokenizer, self.texts, self.targets, batch_size=5, max_length=32)
        original = self.model.encoder.embeddings.word_embeddings.weight.detach().clone()
        kwargs = dict(device="cpu", pos_weight=trainer.positive_weights(self.targets))
        stats = trainer.train_epoch(self.model, first, torch.optim.SGD(self.model.parameters(), lr=.01),
                                    accumulation_steps=3, **kwargs)
        trainer.train_epoch(other, second, torch.optim.SGD(other.parameters(), lr=.01),
                            accumulation_steps=1, **kwargs)
        self.assertTrue(stats["completed_epoch"])
        self.assertEqual(stats["examples"], 5)
        self.assertEqual(stats["successful_updates"], 1)
        self.assertFalse(torch.equal(original, self.model.encoder.embeddings.word_embeddings.weight))
        for left, right in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)

    def test_pilot_counts_successful_updates_and_excludes_warmup(self):
        texts = ["hello world"] * 48
        targets = np.tile(self.targets[0], (48, 1))
        loader = trainer.make_loader(self.tokenizer, texts, targets, batch_size=2, max_length=32)
        optimizer = trainer.build_optimizer(self.model)
        scheduler = trainer.build_scheduler(optimizer, 12)
        seen = []
        result = trainer.train_epoch(self.model, loader, optimizer, device="cpu",
                                     pos_weight=trainer.positive_weights(targets), accumulation_steps=2,
                                     scheduler=scheduler, max_updates=11, on_update=seen.append)
        self.assertEqual(result["successful_updates"], 11)
        self.assertEqual(len(result["measured_update_seconds"]), 10)
        self.assertGreater(result["seconds_per_update"], 0)
        self.assertFalse(result["completed_epoch"])
        self.assertEqual(len(seen), 11)
        self.assertEqual(scheduler.last_epoch, 11)

    def test_budget_stop_clears_gradients_without_training(self):
        class Exhausted:
            def check(self, stage, training=False):
                raise TimeoutError("test deadline")
        before = deepcopy(self.model.state_dict())
        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets, max_length=32)
        with self.assertRaises(TimeoutError):
            trainer.train_epoch(self.model, loader, trainer.build_optimizer(self.model),
                                device="cpu", pos_weight=torch.ones(9), budget=Exhausted())
        for key, value in before.items():
            torch.testing.assert_close(value, self.model.state_dict()[key])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_skipped_overflow_updates_do_not_advance_schedule_or_fake_completion(self):
        class OverflowScaler:
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

        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets, batch_size=2, max_length=32)
        optimizer = trainer.build_optimizer(self.model)
        scheduler = trainer.build_scheduler(optimizer, 3)
        stats = trainer.train_epoch(self.model, loader, optimizer, device="cpu", pos_weight=torch.ones(9),
                                    accumulation_steps=1, scheduler=scheduler, scaler=OverflowScaler([0]))
        self.assertEqual(stats["attempted_updates"], 3)
        self.assertEqual(stats["successful_updates"], 2)
        self.assertEqual(scheduler.last_epoch, 2)
        with self.assertRaisesRegex(FloatingPointError, "no successful"):
            trainer.train_epoch(self.model, loader, optimizer, device="cpu", pos_weight=torch.ones(9),
                                accumulation_steps=1, scaler=OverflowScaler([0, 1, 2]))

    def test_inference_pilot_reports_actual_examples(self):
        loader = trainer.make_loader(self.tokenizer, self.texts, batch_size=2, max_length=32)
        result = trainer.predict_logits(self.model, loader, device="cpu", max_batches=2, return_stats=True)
        self.assertEqual(result["logits"].shape, (4, 9))
        self.assertEqual(result["examples"], 4)
        self.assertGreater(result["seconds_per_example"], 0)

    def test_standard_roberta_encoder_is_supported(self):
        config = RobertaConfig(vocab_size=len(self.tokenizer), hidden_size=16,
                                num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                                max_position_embeddings=64, pad_token_id=0)
        model = trainer.MeanPoolClassifier(RobertaModel(config), dropout=0.)
        values = trainer.predict_texts(model, self.tokenizer, self.texts, max_length=32)
        self.assertEqual(values.shape, (5, 9))
        self.assertTrue(np.isfinite(values).all())

    def test_local_pretrained_creation_and_remote_revision_guard(self):
        snapshot = self.directory / "pretrained"
        self.model.encoder.save_pretrained(snapshot)
        self.tokenizer.save_pretrained(snapshot)
        model, tokenizer = trainer.create_model(snapshot, local_files_only=True)
        self.assertEqual(model.encoder.config.model_type, "bert")
        self.assertEqual(len(tokenizer), len(self.tokenizer))
        with self.assertRaises(ValueError):
            trainer.create_model("example/model", revision="main")

    def test_checkpoint_epoch_and_real_portable_logits_roundtrip(self):
        loader = trainer.make_loader(self.tokenizer, self.texts, self.targets, batch_size=2, max_length=32)
        trainer.train_epoch(self.model, loader, trainer.build_optimizer(self.model, lr=1e-3),
                            device="cpu", pos_weight=trainer.positive_weights(self.targets))
        expected = trainer.predict_texts(self.model, self.tokenizer, self.texts, max_length=32)
        destination = self.directory / "portable"
        metadata = {"epoch": 1, "max_length": 32, "trained_on": 5}
        manifest = trainer.export_model(destination, self.model, self.tokenizer, metadata)
        self.assertEqual(manifest["decision"], trainer.DECISION)
        with patch.object(trainer.AutoModel, "from_pretrained", side_effect=AssertionError("network load")):
            loaded, tokenizer, saved = trainer.load_export(destination)
        self.assertEqual(saved, metadata)
        actual = trainer.predict_texts(loaded, tokenizer, self.texts, max_length=saved["max_length"])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_array_equal(trainer.fixed_decisions(actual), trainer.fixed_decisions(expected))
        (destination / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            trainer.load_export(destination)

    def test_export_rejects_changed_target_order(self):
        destination = self.directory / "portable"
        trainer.export_model(destination, self.model, self.tokenizer, {"max_length": 32})
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["target_symptoms"].reverse()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            trainer.load_export(destination)


if __name__ == "__main__":
    unittest.main()
