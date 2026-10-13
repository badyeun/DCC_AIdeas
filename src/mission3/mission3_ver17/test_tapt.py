"""Actual small Torch MLM updates without downloads or Transformers imports."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver17 import tapt, runtime
else:
    from . import tapt, runtime
from mission3_ver14.test_neural import FakeEncoder, FakeTokenizer, FakeConfig


class MaskTokenizer(FakeTokenizer):
    mask_token_id = 103
    all_special_ids = [0, 101, 102, 103]

    def __len__(self):
        return 300


class TinyMaskedModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.base_model = FakeEncoder()
        self.config = self.base_model.config
        self.decoder = nn.Linear(self.config.hidden_size, 300)
        self.calls = 0

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpoint_options = kwargs

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        self.calls += 1
        hidden = self.base_model(input_ids, attention_mask, token_type_ids).last_hidden_state
        logits = self.decoder(hidden)
        return SimpleNamespace(loss=F.cross_entropy(logits.reshape(-1, 300), labels.reshape(-1), ignore_index=-100))


class MLMBudget:
    def __init__(self, admit=True, stop_microbatch=None):
        self.admit = admit
        self.stop_microbatch = stop_microbatch
        self.microbatches = 0

    def check(self, stage, training=False):
        if stage == "MLM microbatch":
            self.microbatches += 1
            if self.microbatches == self.stop_microbatch:
                raise runtime.BudgetExceeded(stage)
        return 10000.

    def can_start(self, estimate, training=True):
        return self.admit


def config(**overrides):
    return {"model_id": "fixture/MLM", "revision": "a" * 40, "seed": 42,
            "allow_cpu": True, "budget_safety_factor": 1.25,
            "tapt_steps": 3, "tapt_batch_size": 2, "tapt_gradient_accumulation": 2,
            "tapt_learning_rate": .01, "tapt_max_length": 8, "tapt_pilot_steps": 2,
            "tapt_fit_time_limit_seconds": 60., "tapt_full_time_limit_seconds": 60.,
            **overrides}


class TAPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".tapt_test_", dir=HERE)
        self.root = Path(self.temporary.name)
        self.prepared, self.initial = [], []

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, model_id, revision, assets):
        model, tokenizer = TinyMaskedModel(), MaskTokenizer()
        for name in ("encoder", "tokenizer"):
            (assets / name).mkdir(parents=True)
        model.config.save_pretrained(assets / "encoder")
        tokenizer.save_pretrained(assets / "tokenizer")
        self.prepared.append(model)
        self.initial.append(model.base_model.embedding.weight.detach().clone())
        return model, tokenizer, revision

    def test_special_padding_tokens_never_targeted_and_short_windows_have_a_mask(self):
        tokenizer = MaskTokenizer()
        windows = tapt.WindowDataset(["token_1\ntoken_2", "token_3"], tokenizer, 8)
        batch = tapt.MaskedCollator(tokenizer, .000001, 42)(windows.windows)
        expected = torch.tensor([[101, 11, 102, 12, 102], [101, 13, 102, 0, 0]])
        for special in tokenizer.all_special_ids:
            positions = expected == special
            self.assertTrue(torch.equal(batch["input_ids"][positions], expected[positions]))
            self.assertTrue(bool((batch["labels"][positions] == -100).all()))
        selected = batch["labels"] != -100
        self.assertTrue(bool(selected.any(dim=1).all()))
        torch.testing.assert_close(batch["labels"][selected], expected[selected])
        self.assertTrue(bool((batch["labels"][~batch["attention_mask"].bool()] == -100).all()))
        # Replay uses independent local generators, not global random state.
        replay = tapt.MaskedCollator(tokenizer, .000001, 42)(windows.windows)
        for key in batch:
            torch.testing.assert_close(batch[key], replay[key], rtol=0, atol=0)

    def test_all_overflow_windows_include_tail_and_empty_text_is_explicitly_counted(self):
        text = " ".join(f"token_{value}" for value in range(1, 19))
        dataset = tapt.WindowDataset([text, ""], MaskTokenizer(), 8)
        observed = {value for window in dataset.windows for value in window["input_ids"]}
        self.assertEqual(len(dataset), 3)
        self.assertTrue(set(range(11, 29)).issubset(observed))
        self.assertIn(28, dataset.windows[-1]["input_ids"])
        self.assertEqual(dataset.skipped_empty_windows, 1)

    def test_real_MLM_backward_completes_exact_steps_and_changes_encoder(self):
        texts = ["token_1 token_2", "token_3 token_4", "token_5 token_6", "token_7 token_8"]
        with patch.object(tapt, "prepare_masked_model", side_effect=self.prepare), \
                patch.object(tapt, "encoder_from_masked_model", side_effect=lambda model: deepcopy(model.base_model)):
            encoder, tokenizer, revision, report = tapt.adapt_encoder(texts, config(), self.root / "assets", MLMBudget(), "fit_only")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_optimizer_steps"], 3)
        self.assertEqual(report["pilot_optimizer_steps"], 2)
        self.assertEqual(self.prepared[0].calls, 6)
        self.assertEqual(report["window_draws"], 12)
        self.assertEqual(report["input_text_count"], 4)
        self.assertGreater(report["masked_tokens"], 0)
        self.assertTrue(torch.isfinite(encoder.embedding.weight).all())
        self.assertFalse(torch.equal(self.initial[0], encoder.embedding.weight))
        self.assertEqual(runtime._read(self.root / "tapt_report.json"), report)
        self.assertEqual(revision, "a" * 40)

    def test_partial_microbatch_and_failed_pilot_never_return_completed_encoder(self):
        texts = ["token_1 token_2", "token_3 token_4"]
        for name, budget, expected_steps in (("interrupted", MLMBudget(stop_microbatch=3), 1),
                                              ("pilot_denied", MLMBudget(admit=False), 2)):
            local = self.root / name
            with patch.object(tapt, "prepare_masked_model", side_effect=self.prepare), \
                    patch.object(tapt, "encoder_from_masked_model", side_effect=AssertionError("Partial encoder promoted")):
                with self.assertRaises(runtime.BudgetExceeded):
                    tapt.adapt_encoder(texts, config(), local / "assets", budget, "fit_only")
            report = runtime._read(local / "tapt_report.json")
            self.assertEqual(report["status"], "budget_exhausted")
            self.assertEqual(report["completed_optimizer_steps"], expected_steps)
            self.assertLess(report["completed_optimizer_steps"], report["target_optimizer_steps"])

    def test_public_pretrained_MLM_head_and_revision_are_used(self):
        calls = []
        cfg, tokenizer, model = FakeConfig(), MaskTokenizer(), TinyMaskedModel()
        fake = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: cfg),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs:
                                                (calls.append((args, kwargs)) or model)))
        with patch.dict(sys.modules, {"transformers": fake}):
            actual, _, revision = tapt.prepare_masked_model("fixture/MLM", "main", self.root / "assets")
        self.assertIs(actual, model)
        self.assertEqual(revision, "a" * 40)
        self.assertEqual(calls[0][1]["revision"], "a" * 40)
        self.assertFalse(calls[0][1]["trust_remote_code"])
        self.assertEqual(model.checkpoint_options["gradient_checkpointing_kwargs"], {"use_reentrant": False})

    def test_encoder_transfer_allows_only_unused_pooler_missing_keys(self):
        model = TinyMaskedModel()
        class WithPooler(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.pooler = nn.Module()
                self.pooler.dense = nn.Linear(8, 8)
        fake = SimpleNamespace(AutoModel=SimpleNamespace(from_config=lambda *args, **kwargs: WithPooler()))
        with patch.dict(sys.modules, {"transformers": fake}):
            copied = tapt.encoder_from_masked_model(model)
        torch.testing.assert_close(copied.embedding.weight, model.base_model.embedding.weight, rtol=0, atol=0)
        class IncompleteEncoder(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.extra = nn.Linear(8, 8)
        fake.AutoModel.from_config = lambda *args, **kwargs: IncompleteEncoder()
        with patch.dict(sys.modules, {"transformers": fake}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                tapt.encoder_from_masked_model(model)

    def test_recipe_caps_and_mask_limit_are_strict(self):
        self.assertEqual(tapt.validate_recipe({})["tapt_steps"], 800)
        for override in ({"tapt_steps": True}, {"tapt_mask_probability": .16},
                         {"tapt_fit_time_limit_seconds": 901}, {"tapt_full_time_limit_seconds": 1201}):
            with self.assertRaises(ValueError):
                tapt.validate_recipe(override)


if __name__ == "__main__":
    unittest.main()
