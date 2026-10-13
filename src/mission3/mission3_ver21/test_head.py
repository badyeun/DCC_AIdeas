"""CPU verification of frozen features, whole-fit loss, and head isolation."""
from copy import deepcopy
import re
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import data, head
from .budget import BudgetExceeded
from .neural import DialogueClassifier, EncodeDataset


class OffsetTokenizer:
    pad_token_id = 0
    unk_token_id = 100
    sep_token = "[SEP]"
    is_fast = True
    all_special_ids = [0, 100, 101, 102]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, *, max_length, stride, return_offsets_mapping=False, **kwargs):
        keys = ["input_ids", "attention_mask", "token_type_ids", "overflow_to_sample_mapping"]
        if return_offsets_mapping:
            keys.append("offset_mapping")
        result = {key: [] for key in keys}
        capacity = max_length - 2
        for source, text in enumerate(texts):
            tokens = list(re.finditer(r"\[SEP\]|token_\d+", text))
            ids = [102 if token.group() == "[SEP]" else 10 + int(token.group().split("_")[-1])
                   for token in tokens]
            offsets = [(token.start(), token.end()) for token in tokens]
            start = 0
            while True:
                window = [101] + ids[start:start + capacity] + [102]
                result["input_ids"].append(window)
                result["attention_mask"].append([1] * len(window))
                result["token_type_ids"].append([0] * len(window))
                result["overflow_to_sample_mapping"].append(source)
                if return_offsets_mapping:
                    result["offset_mapping"].append([(0, 0)] + offsets[start:start + capacity] + [(0, 0)])
                if start + capacity >= len(ids):
                    break
                start += capacity - stride
        return result


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(256, 8)
        self.calls = 0

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def synthetic_cache(count=16):
    rng = np.random.default_rng(14)
    targets = rng.integers(0, 2, (count, 9), dtype=np.int8)
    targets[:, 6] = np.arange(count) % 2
    features = np.zeros((count, 256), dtype=np.float32)
    features[:, 0] = 2 * targets[:, 6] - 1
    features[:, 1] = rng.normal(size=count)
    weights = np.zeros((256, 2), dtype=np.float32)
    bias = np.array([.1, 0], dtype=np.float32)
    logits = rng.normal(size=(count, 9, 2)).astype(np.float32)
    logits[:, 6] = bias
    cache = {"global_features": features, "evidence": np.zeros((count, 2), dtype=np.float32),
             "logits": logits, "predictions": logits.argmax(-1).astype(np.int8),
             "initial_weight": weights, "initial_bias": bias}
    return cache, targets


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_feature_hook_reconstructs_real_forward_and_freezes_all_parameters(self):
        torch.manual_seed(43)
        model = DialogueClassifier(TinyEncoder(), utterance_evidence=True, evidence_hidden=12).eval()
        model.encoder_window_batch = 2
        before = {key: value.clone() for key, value in model.state_dict().items()}
        texts = ["token_1\ntoken_2 token_3", " ".join(f"token_{i}" for i in range(1, 20)), ""]
        settings = {"max_length": 8, "overlap": 2, "batch_size": 3}
        cache = head.extract_cache(model, OffsetTokenizer(), texts, settings, "cpu")
        dataset = EncodeDataset(texts, None, OffsetTokenizer(), 8, 2, utterance_evidence=True)
        with torch.inference_mode():
            expected, _ = model(dataset.collate([dataset[i] for i in range(len(dataset))]))
        np.testing.assert_array_equal(cache["logits"], expected.numpy())
        np.testing.assert_array_equal(cache["predictions"], expected.argmax(-1).numpy())
        rebuilt, predicted = head.apply_head(cache, cache["initial_weight"], cache["initial_bias"])
        np.testing.assert_allclose(rebuilt, expected.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(predicted, cache["predictions"])
        self.assertEqual(cache["global_features"].shape, (3, 256))
        self.assertEqual(cache["evidence"].shape, (3, 2))
        self.assertEqual(len(model.dropout._forward_hooks), 0)
        self.assertEqual(len(model.evidence._forward_hooks), 0)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_pilot_rejects_expensive_pass_and_removes_hooks(self):
        class RejectBudget:
            deadline_monotonic = 123
            checks = []

            def check(self, stage, training=False):
                self.checks.append(training)

            def can_start(self, estimate, training=True):
                self.estimate = estimate
                return False

        budget = RejectBudget()
        model = DialogueClassifier(TinyEncoder(), utterance_evidence=True, evidence_hidden=12)
        with self.assertRaises(BudgetExceeded):
            head.extract_cache(model, OffsetTokenizer(), ["token_1"] * 130,
                               {"max_length": 8, "overlap": 2}, "cpu", budget, training=False)
        self.assertEqual(model.encoder.calls, 16)
        self.assertTrue(all(value is False for value in budget.checks))
        self.assertGreater(budget.estimate, 0)
        self.assertEqual(len(model.dropout._forward_hooks), 0)
        self.assertEqual(len(model.evidence._forward_hooks), 0)

    def test_whole_fit_soft_f1_uses_ratio_of_sums_and_correct_gradients(self):
        logits = torch.tensor([[2., -1.], [-.2, .3], [.4, -.1], [-2., 1.]], requires_grad=True)
        truth = torch.tensor([0, 1, 1, 1])
        loss, details = head.head_loss(logits, truth, "head_ce_soft_f1", positive_weight=3, soft_f1_weight=.5)
        expected_logits = logits.detach().clone().requires_grad_()
        probability = expected_logits.softmax(-1)[:, 1]
        ce = (F.cross_entropy(expected_logits, truth, reduction="none")
              * torch.tensor([1., 3., 3., 3.])).mean()
        ratio = (2 * (probability * truth).sum() + 1e-7) / (probability.sum() + truth.sum() + 1e-7)
        expected = ce + .5 * (1 - ratio)
        self.assertAlmostEqual(details["soft_f1"], float(ratio.detach()), places=7)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, expected_logits.grad, rtol=0, atol=0)
        small_batch_average = sum(head.head_loss(logits.detach()[i:i + 2], truth[i:i + 2],
                                                "head_ce_soft_f1")[0] for i in (0, 2)) / 2
        self.assertGreater(abs(float(loss.detach() - small_batch_average)), .001)

    def test_ce_control_omits_soft_f1_and_handles_zero_positive(self):
        logits = torch.tensor([[.1, .2], [.3, .4]], requires_grad=True)
        targets = torch.zeros(2, dtype=torch.long)
        actual, _ = head.head_loss(logits, targets, "head_ce", soft_f1_weight=100)
        torch.testing.assert_close(actual, F.cross_entropy(logits, targets))
        total, _ = head.head_loss(logits, targets, "head_ce_soft_f1")
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            head.head_loss(logits, targets, "unknown")
        with self.assertRaises(ValueError):
            head.head_loss(logits, targets, "head_ce", positive_weight=float("nan"))

    def test_head_optimizer_cannot_touch_cache_or_eight_other_logits(self):
        cache, targets = synthetic_cache()
        cache["global_features"][:, 1] = 0
        original = deepcopy(cache)
        settings = {"max_head_epochs": 10, "head_eval_every": 5, "head_learning_rate": .05}
        records = head.head_candidates(cache, targets, np.arange(8), np.arange(8, 16), settings)
        self.assertEqual([r["candidate_id"] for r in records], ["baseline", *head.RECIPES])
        for record in records[1:]:
            self.assertEqual(record["trainable_parameters"], 514)
            self.assertTrue(record["improved"])
            self.assertEqual(record["epoch"], 5)
            logits, predictions = head.apply_head(cache, record["weight"], record["bias"])
            np.testing.assert_array_equal(logits[:, head.OTHER_LABELS], cache["logits"][:, head.OTHER_LABELS])
            np.testing.assert_array_equal(predictions[:, head.OTHER_LABELS], cache["predictions"][:, head.OTHER_LABELS])
            self.assertTrue(all(step["whole_fit_batch_size"] == 8 for step in record["history"]))
        for key in cache:
            np.testing.assert_array_equal(cache[key], original[key])

    def test_holdout_targets_never_enter_an_optimizer(self):
        cache, targets = synthetic_cache()
        other_targets = targets.copy()
        other_targets[8:] = 1 - other_targets[8:]
        settings = {"max_head_epochs": 1, "head_eval_every": 1}
        first = head.head_candidates(cache, targets, np.arange(8), np.arange(8, 16), settings)
        second = head.head_candidates(cache, other_targets, np.arange(8), np.arange(8, 16), settings)
        for one, two in zip(first[1:], second[1:]):
            np.testing.assert_array_equal(one["weight"], two["weight"])
            np.testing.assert_array_equal(one["bias"], two["bias"])
            self.assertEqual(one["history"][0]["loss"], two["history"][0]["loss"])
        with self.assertRaises(ValueError):
            head.head_candidates(cache, targets, np.arange(8), np.arange(7, 16), settings)

    def test_refit_uses_own_initialization_and_exact_fixed_epoch_count(self):
        cache, targets = synthetic_cache()
        full_cache = deepcopy(cache)
        full_cache["initial_weight"][0] = [-.03, .02]
        full_cache["initial_bias"][:] = [.02, -.02]
        settings = {"max_head_epochs": 3, "head_learning_rate": .001}
        weight, bias, history = head.refit_head(full_cache, targets, "head_ce", 2, settings)
        options = head._settings(settings)
        expected, optimizer, features, evidence, truth = head._new_training(full_cache, targets, np.arange(16), options)
        for _ in range(2):
            head._step(expected, optimizer, features, evidence, truth, "head_ce", options)
        np.testing.assert_array_equal(weight, expected.weight.detach().numpy())
        np.testing.assert_array_equal(bias, expected.bias.detach().numpy())
        self.assertEqual([step["epoch"] for step in history], [1, 2])
        with self.assertRaises(ValueError):
            head.refit_head(cache, targets, "head_ce", 4, settings)

    def test_patch_changes_only_514_nausea_parameters(self):
        torch.manual_seed(23)
        model = DialogueClassifier(TinyEncoder(), utterance_evidence=True, evidence_hidden=12)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        weight = model.class_weights[6].detach().numpy().copy() + np.float32(.1)
        bias = model.class_bias[6].detach().numpy().copy() + np.float32(.2)
        audit = head.set_nausea_head(model, weight, bias)
        self.assertTrue(audit["all_other_tensors_identical"])
        self.assertEqual(audit["learned_parameter_count"], 514)
        changed_elements = sum(int((value != before[name]).sum()) for name, value in model.state_dict().items())
        self.assertEqual(changed_elements, 514)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_budget_failure_never_returns_unevaluated_weights(self):
        class StopBudget:
            def check(self, *args, **kwargs):
                raise BudgetExceeded("test")

        cache, targets = synthetic_cache()
        records = head.head_candidates(cache, targets, np.arange(8), np.arange(8, 16), {}, StopBudget())
        self.assertEqual([r["candidate_id"] for r in records], ["baseline"])
        self.assertEqual(records[0]["search_stop_reason"], "budget")
        self.assertEqual(records[0]["stop_reason"], "unchanged_original_head")
        with self.assertRaises(BudgetExceeded):
            head.refit_head(cache, targets, "head_ce", 1, {}, StopBudget())

    def test_invalid_cache_and_path_escape_are_rejected(self):
        cache, _ = synthetic_cache()
        for key, value in (("global_features", np.zeros((16, 64), dtype=np.float32)),
                           ("initial_bias", np.array([float("nan"), 0], dtype=np.float32)),
                           ("predictions", 1 - cache["predictions"])):
            broken = {**cache, key: value}
            with self.assertRaises(ValueError):
                head._validate_cache(broken)
        for relative in ("../outside.pt", "", "."):
            with self.assertRaises(ValueError):
                head._safe(".", relative)

    def test_selection_prioritizes_joint_improvement_and_ties_remain_equal(self):
        def metric(macro, nausea):
            return {"macro_f1": macro, "per_symptom": {data.TARGET_SYMPTOMS[6]: {"f1": nausea}}}
        baseline = metric(.64, .37)
        self.assertGreater(head._key(metric(.641, .38), baseline), head._key(metric(.65, .36), baseline))
        self.assertEqual(head._key(baseline, baseline), (False, .64))


if __name__ == "__main__":
    unittest.main()
