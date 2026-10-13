"""CPU checks for full-turn caching, fit-only learning and evidence isolation."""
from copy import deepcopy
import re
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from . import evidence
from .budget import BudgetExceeded
from .neural import DialogueClassifier, EncodeDataset, UtteranceEvidence


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
    counts = np.arange(count) % 4
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    turns = rng.normal(0, .1, (int(offsets[-1]), 8)).astype(np.float32)
    for index in range(count):
        turns[offsets[index]:offsets[index + 1], 0] = 2 * targets[index, 6] - 1
    features = np.zeros((count, 256), dtype=np.float32)
    global_logits = np.tile(np.array([.1, 0], dtype=np.float32), (count, 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        module = UtteranceEvidence(8, 256, hidden=12, dropout=.1).eval()
    padded = np.zeros((count, max(1, int(counts.max())), 8), dtype=np.float32)
    mask = np.arange(padded.shape[1])[None] < counts[:, None]
    for index in range(count):
        padded[index, :counts[index]] = turns[offsets[index]:offsets[index + 1]]
    with torch.inference_mode():
        residual = module(torch.from_numpy(padded), torch.from_numpy(mask), torch.from_numpy(features)).numpy()
    logits = rng.normal(size=(count, 9, 2)).astype(np.float32)
    logits[:, 6] = global_logits + residual
    cache = {"turn_features": turns, "turn_offsets": offsets, "global_features": features,
             "global_logits": global_logits, "evidence": residual, "logits": logits,
             "predictions": logits.argmax(-1).astype(np.int8),
             "evidence_dropout": np.asarray(.1, dtype=np.float32)}
    cache.update({evidence.STATE_PREFIX + name: value.detach().numpy().copy()
                  for name, value in module.state_dict().items()})
    return cache, targets


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        torch.manual_seed(43)
        model = DialogueClassifier(TinyEncoder(), utterance_evidence=True, evidence_hidden=12).eval()
        model.encoder_window_batch = 2
        with torch.no_grad():
            model.evidence.output.weight.normal_(0, .1)
            model.evidence.output.bias.copy_(torch.tensor([.03, -.02]))
        return model

    def test_capture_preserves_complete_turns_overlap_and_original_forward(self):
        model = self.model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        texts = ["token_1\ntoken_2 token_3", " ".join(f"token_{i}" for i in range(1, 20)), "",
                 "\n".join(f"token_{i}" for i in range(1, 21))]
        settings = {"max_length": 8, "overlap": 2, "batch_size": 4}
        cache = evidence.extract_cache(model, OffsetTokenizer(), texts, settings, "cpu")
        np.testing.assert_array_equal(np.diff(cache["turn_offsets"]), [2, 1, 0, 20])
        # Context-independent embeddings make overlap weighting independently
        # checkable: repeated window occurrences must not overweight any token.
        embedding = model.encoder.embedding.weight.detach()
        torch.testing.assert_close(torch.from_numpy(cache["turn_features"][2]), embedding[11:30].mean(0))
        torch.testing.assert_close(torch.from_numpy(cache["turn_features"][3:]), embedding[11:31])
        dataset = EncodeDataset(texts, None, OffsetTokenizer(), 8, 2, utterance_evidence=True)
        with torch.inference_mode():
            expected, _ = model(dataset.collate([dataset[i] for i in range(len(dataset))]))
        np.testing.assert_array_equal(cache["logits"], expected.numpy())
        actual, predicted = evidence.apply_evidence(cache, evidence.initial_state(cache), {"evidence_batch_size": 2})
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(predicted, expected.argmax(-1).numpy())
        self.assertEqual(len(model.evidence._forward_pre_hooks), 0)
        self.assertEqual(len(model.evidence._forward_hooks), 0)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_capture_budget_rejection_removes_both_hooks(self):
        class RejectBudget:
            deadline_monotonic = 123.

            def check(self, stage, training=False):
                self.training = training

            def can_start(self, estimate, training=True):
                return False

        model = self.model()
        budget = RejectBudget()
        with self.assertRaises(BudgetExceeded):
            evidence.extract_cache(model, OffsetTokenizer(), ["token_1"] * 130,
                                   {"max_length": 8, "overlap": 2}, "cpu", budget, training=False)
        self.assertFalse(budget.training)
        self.assertEqual(len(model.evidence._forward_pre_hooks), 0)
        self.assertEqual(len(model.evidence._forward_hooks), 0)

    def test_ragged_subset_and_interleaved_merge_keep_row_order_and_empty_turns(self):
        cache, _ = synthetic_cache()
        groups = [np.array([8, 0, 14, 6, 12, 2, 10, 4]), np.arange(1, 16, 2)]
        blocks = [evidence._subset(cache, group) for group in groups]
        merged = evidence.merge_caches(blocks, groups, 16)
        for name, original in cache.items():
            np.testing.assert_array_equal(merged[name], original, err_msg=name)
        with self.assertRaises(ValueError):
            evidence.merge_caches(blocks, [groups[0], groups[0]], 16)
        blocks[1][evidence.STATE_PREFIX + "output.bias"][0] += 1
        with self.assertRaises(ValueError):
            evidence.merge_caches(blocks, groups, 16)

    def test_cache_validation_rejects_nonfinite_offsets_states_and_changed_decisions(self):
        cache, _ = synthetic_cache()
        mutations = [lambda c: c["turn_features"].__setitem__((-1, -1), np.nan),
                     lambda c: c["turn_offsets"].__setitem__(2, -1),
                     lambda c: c["predictions"].__setitem__((0, 6), 1),
                     lambda c: c["global_logits"].__setitem__((0, 0), 10),
                     lambda c: c.__setitem__(evidence.STATE_PREFIX + "attention.bias", np.zeros(2, np.float32))]
        for mutate in mutations:
            broken = deepcopy(cache)
            mutate(broken)
            with self.assertRaises(ValueError):
                evidence._validate_cache(broken)

    def test_weighted_ce_gce_and_warmup_match_original_objective_and_gradients(self):
        logits = torch.tensor([[2., -1.], [-.2, .3], [.4, -.1]], requires_grad=True)
        truth = torch.tensor([0, 1, 1])
        weights = torch.tensor([1., 3., 3.])
        actual, details = evidence.evidence_loss(logits, truth, "evidence_ce_gce", epoch=2)
        compare = logits.detach().clone().requires_grad_()
        ce = F.cross_entropy(compare, truth, reduction="none")
        log_probability = compare.log_softmax(-1).gather(1, truth[:, None]).squeeze(1)
        gce = -torch.expm1(.7 * log_probability) / .7
        expected = (weights * (.5 * ce + .5 * gce)).mean()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, compare.grad, rtol=0, atol=0)
        self.assertEqual(details["gce_mix_applied"], .5)
        warmup, details = evidence.evidence_loss(logits, truth, "evidence_ce_gce", epoch=1)
        control, _ = evidence.evidence_loss(logits, truth, "evidence_ce", epoch=5)
        torch.testing.assert_close(warmup, (weights * ce).mean())
        torch.testing.assert_close(warmup, control)
        self.assertEqual(details["gce_mix_applied"], 0.)

    def test_gce_keeps_finite_gradients_for_extremely_confident_wrong_predictions(self):
        logits = torch.tensor([[1000., -1000.], [-1000., 1000.]], requires_grad=True)
        loss, _ = evidence.evidence_loss(logits, torch.tensor([1, 0]), "evidence_ce_gce", epoch=2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())

    def test_fit_training_updates_evidence_without_altering_cached_features_or_eight_heads(self):
        cache, targets = synthetic_cache(24)
        original = deepcopy(cache)
        options = {"max_evidence_epochs": 3, "evidence_batch_size": 4, "evidence_learning_rate": .01}
        records = evidence.evidence_candidates(cache, targets, np.arange(16), np.arange(16, 24), options)
        self.assertEqual([record["candidate_id"] for record in records], ["baseline", *evidence.RECIPES])
        for record in records[1:]:
            self.assertGreater(record["trainable_parameters"], 514)
            self.assertTrue(any(not np.array_equal(record["state"][name], evidence.initial_state(cache)[name])
                                for name in evidence.STATE_KEYS))
            self.assertTrue(all(row["fitted_samples"] == 16 and row["optimizer_steps"] == 4 for row in record["history"]))
            np.testing.assert_array_equal(record["predictions"][:, evidence.OTHER_LABELS],
                                          cache["predictions"][16:, evidence.OTHER_LABELS])
        for name, value in cache.items():
            np.testing.assert_array_equal(value, original[name], err_msg=name)

    def test_holdout_targets_never_affect_optimizer_or_initialization(self):
        cache, targets = synthetic_cache(24)
        changed = targets.copy()
        changed[16:, 6] = 1 - changed[16:, 6]
        options = {"max_evidence_epochs": 1, "evidence_batch_size": 5}
        first = evidence.evidence_candidates(cache, targets, np.arange(16), np.arange(16, 24), options)
        second = evidence.evidence_candidates(cache, changed, np.arange(16), np.arange(16, 24), options)
        for left, right in zip(first[1:], second[1:]):
            for name in evidence.STATE_KEYS:
                np.testing.assert_array_equal(left["state"][name], right["state"][name])
            np.testing.assert_array_equal(left["predictions"], right["predictions"])

    def test_refit_uses_its_own_full_source_initialization_and_fixed_epoch_count(self):
        cache, targets = synthetic_cache()
        options = {"max_evidence_epochs": 4, "evidence_batch_size": 5}
        state, history = evidence.refit_evidence(cache, targets, "evidence_ce_gce", 2, options)
        self.assertEqual(len(history), 2)
        self.assertEqual([row["gce_mix_applied"] for row in history], [0., .5])
        self.assertTrue(all(row["fitted_samples"] == 16 and row["optimizer_steps"] == 4 for row in history))
        full_cache = deepcopy(cache)
        full_cache[evidence.STATE_PREFIX + "attention.weight"] += .2
        different, _ = evidence.refit_evidence(full_cache, targets, "evidence_ce_gce", 2, options)
        self.assertFalse(np.array_equal(state["attention.weight"], different["attention.weight"]))
        with self.assertRaises(ValueError):
            evidence.refit_evidence(cache, targets, "baseline", 2, options)
        with self.assertRaises(ValueError):
            evidence.refit_evidence(cache, targets, "evidence_ce", 5, options)

    def test_early_stopping_keeps_earliest_epoch_on_ties(self):
        cache, targets = synthetic_cache()
        cache["turn_features"][:] = 0
        cache["turn_offsets"][:] = 0
        cache["turn_features"] = cache["turn_features"][:0]
        options = {"max_evidence_epochs": 8, "early_stopping_patience": 2, "evidence_batch_size": 4}
        records = evidence.evidence_candidates(cache, targets, np.arange(8), np.arange(8, 16), options)
        for record in records[1:]:
            self.assertEqual(record["epoch"], 1)
            self.assertEqual(record["completed_epochs"], 3)
            self.assertEqual(record["stop_reason"], "early_stopping")

    def test_incomplete_epoch_is_not_published_and_budget_is_not_no_improvement(self):
        class StopBudget:
            def __init__(self):
                self.calls = 0

            def check(self, stage, training=False):
                self.calls += 1
                if self.calls == 2:
                    raise BudgetExceeded(stage)

        cache, targets = synthetic_cache()
        records = evidence.evidence_candidates(cache, targets, np.arange(8), np.arange(8, 16),
                                               {"evidence_batch_size": 4}, StopBudget())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["search_stop_reason"], "budget")
        with self.assertRaises(BudgetExceeded):
            evidence.refit_evidence(cache, targets, "evidence_ce", 1, {"evidence_batch_size": 4}, StopBudget())

    def test_measured_epoch_admission_preserves_last_complete_candidate(self):
        class AdmissionBudget:
            deadline_monotonic = 123.

            def check(self, stage, training=False):
                pass

            def can_start(self, estimate, training=True):
                self.estimate = estimate
                return False

        cache, targets = synthetic_cache()
        budget = AdmissionBudget()
        records = evidence.evidence_candidates(cache, targets, np.arange(8), np.arange(8, 16),
                                               {"evidence_batch_size": 4}, budget)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["completed_epochs"], 1)
        self.assertEqual(records[1]["stop_reason"], "budget")
        self.assertEqual(records[0]["search_stop_reason"], "budget")
        self.assertGreater(budget.estimate, 0)
        with self.assertRaises(BudgetExceeded):
            evidence.refit_evidence(cache, targets, "evidence_ce", 2, {"evidence_batch_size": 4}, AdmissionBudget())

    def test_patching_and_reloading_changes_only_evidence_branch_and_nausea_outputs(self):
        model = self.model()
        texts = ["token_1\ntoken_2 token_3", "token_4"]
        cache = evidence.extract_cache(model, OffsetTokenizer(), texts,
                                       {"max_length": 8, "overlap": 2}, "cpu")
        state = evidence.initial_state(cache)
        state["output.bias"] += np.array([-1., 1.], dtype=np.float32)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        report = evidence.set_evidence_state(model, state)
        self.assertEqual(report["changed_state_entries"], ["evidence.output.bias"])
        self.assertTrue(report["all_global_heads_identical"])
        dataset = EncodeDataset(texts, None, OffsetTokenizer(), 8, 2, utterance_evidence=True)
        batch = dataset.collate([dataset[i] for i in range(2)])
        restored = self.model()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            actual, _ = model(batch)
            reloaded, _ = restored(batch)
        expected, _ = evidence.apply_evidence(cache, state)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual, reloaded, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.numpy()[:, evidence.OTHER_LABELS], cache["logits"][:, evidence.OTHER_LABELS])
        for name, value in model.state_dict().items():
            if not name.startswith("evidence."):
                self.assertTrue(torch.equal(before[name], value), name)

    def test_invalid_settings_and_state_are_rejected(self):
        for settings in ({"gce_q": 0}, {"gce_mix": .8}, {"evidence_batch_size": True},
                         {"evidence_learning_rate": float("nan")}, {"positive_weight": -1}):
            with self.assertRaises(ValueError):
                evidence._settings(settings)
        cache, _ = synthetic_cache()
        state = evidence.initial_state(cache)
        state["output.bias"] = state["output.bias"].astype(np.float64)
        with self.assertRaises(ValueError):
            evidence.apply_evidence(cache, state)


if __name__ == "__main__":
    unittest.main()
