"""Measured batch comparisons, workload parity, reserve, and OOM recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3Model

from . import batching, trainer
from .budget import BudgetExceeded


class RecordingBudget:
    def __init__(self, exhausted=False):
        self.exhausted = exhausted
        self.calls = []

    def check(self, stage, training=False):
        self.calls.append((stage, training))
        if self.exhausted:
            raise BudgetExceeded(stage)
        return 10000.


class BatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        root = Path(__file__).resolve().parents[3] / "outputs" / "mission3_ver25_verification"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.texts = ["hello " * (1 + i % 25) for i in range(180)]
        self.targets = np.asarray([[int((i + j) % 7 == 0) for j in range(9)]
                                   for i in range(len(self.texts))], dtype=np.int8)
        self.fit = np.arange(100)
        self.development = np.arange(100, 180)
        self.settings = {"seed": 42, "control_max_length": 32, "qwen_max_length": 32,
                         "control_batch_size": 8, "control_gradient_accumulation": 2,
                         "qwen_batch_size": 1, "qwen_gradient_accumulation": 16,
                         "warmup_ratio": .05, "control_dropout": 0.}
        self.spec = {"kind": "control", "model_id": "unused", "revision": "a" * 40}
        self.budget = RecordingBudget()

    def run_tune(self, kind="control", budget=None):
        spec = dict(self.spec, kind=kind)
        return batching.autotune(kind, spec, self.texts, self.targets, self.fit,
            self.development, self.settings, "cpu", budget or self.budget, self.root)

    def successful(self, batch, seconds=.1, status="success"):
        return {"status": status, "batch_size": batch,
            "gradient_accumulation": 16 // batch, "effective_batch_size": 16,
            "seconds_per_example": seconds, "examples_per_second": 1 / seconds,
            "seconds_per_microstep": seconds * batch, "seconds_per_prediction": .01,
            "setup_seconds": 1., "peak_vram_bytes": 0, "pilot_weights_discarded": True}

    def test_real_tiny_control_uses_fresh_weights_equal_workloads_and_only_fit_labels(self):
        vocab = self.root / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"]), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        snapshot = self.root / "pretrained_control"
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=12, max_position_embeddings=64,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        BertModel(config, add_pooling_layer=False).save_pretrained(snapshot)
        tokenizer.save_pretrained(snapshot)
        self.spec["model_id"] = str(snapshot)
        original_create = trainer.create_model
        original_weights = trainer.positive_weights
        starts, seen_weights = [], []

        def create(*args, **kwargs):
            model, tokens = original_create(*args, **kwargs)
            starts.append({name: value.detach().clone() for name, value in model.state_dict().items()})
            return model, tokens

        def positive_weights(values):
            seen_weights.append(values.copy())
            return original_weights(values)

        original_settings = deepcopy(self.settings)
        with patch.object(trainer, "create_model", side_effect=create), \
                patch.object(trainer, "positive_weights", side_effect=positive_weights):
            settings, report = self.run_tune()
        self.assertEqual(len(starts), 3)
        for state in starts[1:]:
            for name in starts[0]:
                torch.testing.assert_close(state[name], starts[0][name], rtol=0, atol=0)
        for values in seen_weights:
            np.testing.assert_array_equal(values, self.targets[self.fit])
        self.assertEqual(original_settings, self.settings)
        self.assertEqual(settings["control_batch_size"] * settings["control_gradient_accumulation"], 16)
        self.assertEqual(report["status"], "selected")
        self.assertTrue(all(training for _, training in self.budget.calls))
        self.assertIn("classifier_inference", {stage for stage, _ in self.budget.calls})
        for trial in report["trials"]:
            self.assertEqual(trial["measured_examples"], 64)
            self.assertEqual(trial["warmup_examples"], 16)
            self.assertEqual(trial["stress_examples"], 16)
            self.assertEqual(trial["stress_max_length"], 32)
            self.assertEqual(trial["measured_optimizer_updates"], 4)
            self.assertEqual(trial["prediction_examples"], 64)
            self.assertEqual(trial["measured_indices_digest"], report["measured_indices_digest"])
            self.assertEqual(trial["prediction_indices_digest"], report["prediction_indices_digest"])
            self.assertGreater(trial["seconds_per_example"], 0)
            self.assertGreater(trial["seconds_per_prediction"], 0)
            self.assertTrue(trial["pilot_weights_discarded"])
        self.assertEqual(json.loads((self.root / "control_batch_benchmark.json").read_text())["status"], "selected")
        self.assertEqual(json.loads((self.root / "control_token_lengths.json").read_text())["rows"], len(self.texts))

    def test_oom_drops_traceback_tensor_before_smaller_candidate_and_sanitizes_error(self):
        reference, attempted = [], []

        def trial(*args):
            batch = args[-2]
            attempted.append(batch)
            if batch == 8:
                held = torch.ones(5)
                reference.append(weakref.ref(held))
                raise torch.cuda.OutOfMemoryError("private transcript must never be logged")
            self.assertIsNone(reference[0]())
            return self.successful(batch, {4: .12, 2: .08, 1: .20}[batch])

        with patch.object(batching, "_trial", side_effect=trial):
            settings, report = self.run_tune("qwen")
        self.assertEqual(attempted, [8, 4, 2, 1])
        self.assertEqual(settings["qwen_batch_size"], 2)
        self.assertEqual(settings["qwen_gradient_accumulation"], 8)
        self.assertEqual(report["trials"][0]["status"], "out_of_memory")
        self.assertNotIn("private transcript", (self.root / "qwen_batch_benchmark.json").read_text())

    def test_real_tiny_qwen_adapter_and_head_complete_all_batch_candidates(self):
        vocab = self.root / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"]), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        snapshot = self.root / "pretrained_qwen"
        config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=4, intermediate_size=12,
            max_position_embeddings=64, pad_token_id=0, eos_token_id=3, attention_dropout=0.)
        Qwen3Model(config).save_pretrained(snapshot)
        tokenizer.save_pretrained(snapshot)
        self.spec["model_id"] = str(snapshot)
        self.settings.update(test_only_unquantized=True,
                             base_store_dir=str(self.root / "shared_base"), lora_dropout=0.)
        original_create = trainer.create_model

        def create(spec, settings, device):
            return original_create(spec, settings, device, local_files_only=True)

        with patch.object(trainer, "create_model", side_effect=create) as factory:
            settings, report = self.run_tune("qwen")
        self.assertEqual(factory.call_count, 4)
        self.assertEqual(settings["qwen_batch_size"] * settings["qwen_gradient_accumulation"], 16)
        self.assertEqual([trial["batch_size"] for trial in report["trials"]], [8, 4, 2, 1])
        self.assertTrue(all(trial["status"] == "success" for trial in report["trials"]))
        self.assertTrue(all(trial["measured_optimizer_updates"] == 4 for trial in report["trials"]))
        self.assertTrue((self.root / "qwen_token_lengths.json").exists())

    def test_non_oom_error_propagates_without_retry_or_raw_payload(self):
        with patch.object(batching, "_trial", side_effect=ValueError("private data")) as trial:
            with self.assertRaisesRegex(ValueError, "private data"):
                self.run_tune()
        self.assertEqual(trial.call_count, 1)
        report = json.loads((self.root / "control_batch_benchmark.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["error_type"], "ValueError")
        self.assertNotIn("private data", json.dumps(report))

    def test_all_oom_fails_explicitly_with_incremental_report(self):
        with patch.object(batching, "_trial", side_effect=torch.cuda.OutOfMemoryError("oom")):
            with self.assertRaisesRegex(RuntimeError, "No safe control batch candidate"):
                self.run_tune()
        report = json.loads((self.root / "control_batch_benchmark.json").read_text())
        self.assertEqual(report["status"], "no_safe_candidate")
        self.assertEqual(len(report["trials"]), 3)

    def test_fastest_candidate_with_low_memory_headroom_is_excluded(self):
        def trial(*args):
            batch = args[-2]
            return self.successful(batch, 1 / batch,
                "insufficient_memory_headroom" if batch == 16 else "success")

        with patch.object(batching, "_trial", side_effect=trial):
            settings, report = self.run_tune()
        self.assertEqual(settings["control_batch_size"], 8)
        self.assertEqual(report["trials"][0]["status"], "insufficient_memory_headroom")

    def test_memory_headroom_counts_external_allocations_and_peak_reserved(self):
        with patch.object(trainer, "_synchronize"), \
                patch.object(torch.cuda, "mem_get_info", return_value=(90, 1000)), \
                patch.object(torch.cuda, "memory_reserved", return_value=500), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=540), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=400):
            memory = batching._memory("cuda")
        self.assertFalse(memory["memory_safe"])
        self.assertEqual(memory["headroom_bytes"], 50)
        self.assertEqual(memory["headroom_fraction"], .05)

    def test_exhausted_reserve_stops_before_model_creation(self):
        with patch.object(trainer, "create_model") as create:
            with self.assertRaises(BudgetExceeded):
                self.run_tune(budget=RecordingBudget(exhausted=True))
        create.assert_not_called()
        report = json.loads((self.root / "control_batch_benchmark.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["error_type"], "BudgetExceeded")

    def test_representative_samples_are_unique_deterministic_and_span_lengths(self):
        texts = ["a" * i for i in range(4000)]
        first = batching.representative_indices(texts, np.arange(4000), 80, 42)
        second = batching.representative_indices(texts, np.arange(3999, -1, -1), 80, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 80)
        self.assertLess(min(first), 50)
        self.assertGreater(max(first), 3950)

    def test_overlapping_or_insufficient_sample_sets_fail_before_model_creation(self):
        with patch.object(trainer, "create_model") as create:
            self.development = np.arange(80)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                self.run_tune()
            self.development = np.arange(100, 180)
            self.fit = np.arange(70)
            with self.assertRaisesRegex(ValueError, "at least 80"):
                self.run_tune()
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
