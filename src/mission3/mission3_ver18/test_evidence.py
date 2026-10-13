"""CPU checks for complete, ordered, text-only utterance evidence."""
from copy import deepcopy
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .neural import (DialogueClassifier, EncodeDataset, UtteranceEvidence,
                     classification_loss, predictions, validate_evidence_config)
from mission3_ver15.neural import DialogueClassifier as OriginalClassifier
from mission3_ver15.neural import EncodeDataset as OriginalDataset


class OffsetTokenizer:
    pad_token_id = 0
    unk_token_id = 100
    sep_token = "[SEP]"
    is_fast = True
    all_special_ids = [0, 100, 101, 102]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, *, max_length, stride, return_offsets_mapping=False, **kwargs):
        result = {key: [] for key in ("input_ids", "attention_mask", "token_type_ids", "overflow_to_sample_mapping")}
        if return_offsets_mapping:
            result["offset_mapping"] = []
        capacity = max_length - 2
        for source, text in enumerate(texts):
            tokens = list(re.finditer(r"\[SEP\]|token_\d+", text))
            ids = [102 if token.group() == "[SEP]" else 10 + int(token.group().split("_")[-1]) for token in tokens]
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


def paired_models():
    torch.manual_seed(7)
    encoder = TinyEncoder()
    before = torch.get_rng_state()
    baseline = OriginalClassifier(deepcopy(encoder), feature_dim=16)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    evidence = DialogueClassifier(deepcopy(encoder), feature_dim=16, utterance_evidence=True,
                                 evidence_hidden=12, evidence_dropout=0.1)
    return baseline, evidence, after


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def dataset(self, texts, targets=None, **kwargs):
        return EncodeDataset(texts, targets, OffsetTokenizer(), max_length=8, overlap=2,
                             utterance_evidence=True, **kwargs)

    def test_offsets_preserve_every_turn_and_overlap_token_mass(self):
        text = "token_1 token_2\r\ntoken_3 token_4 token_5 token_6 token_7\ntoken_8\n\ntoken_9"
        dataset = self.dataset([text, "", "   "])
        original = OriginalDataset([text, "", "   "], None, OffsetTokenizer(), max_length=8, overlap=2)
        expected_turns = {11: 0, 12: 0, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 2, 19: 3}
        mass = {token: 0.0 for token in expected_turns}
        for actual_doc, original_doc in zip(dataset.documents, original.documents):
            self.assertEqual(len(actual_doc), len(original_doc))
            for actual, expected in zip(actual_doc, original_doc):
                for key in ("input_ids", "attention_mask", "token_type_ids"):
                    self.assertEqual(actual[key], expected[key])
                for token, turn, weight in zip(actual["input_ids"], actual["turn_ids"], actual["token_weights"]):
                    if token in expected_turns:
                        self.assertEqual(turn, expected_turns[token])
                        mass[token] += weight
                    else:
                        self.assertEqual(turn, -1)
                        self.assertEqual(weight, 0)
        self.assertTrue(all(abs(value - 1.0) < 1e-7 for value in mass.values()))
        batch = dataset.collate([dataset[i] for i in range(3)])
        invalid = ~batch["window_mask"].unsqueeze(-1).expand_as(batch["turn_ids"]) | ~batch["attention_mask"].bool()
        self.assertTrue(bool((batch["turn_ids"][invalid] == -1).all()))
        self.assertTrue(bool((batch["token_weights"][invalid] == 0).all()))

    def test_tokenizer_batch_boundary_and_missing_offsets_fail_loudly(self):
        dataset = self.dataset(["token_1\ntoken_2"] * 129)
        self.assertEqual(dataset.documents[0], dataset.documents[-1])

        class MissingOffsets(OffsetTokenizer):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                result.pop("offset_mapping", None)
                return result

        with self.assertRaisesRegex(ValueError, "offset_mapping"):
            EncodeDataset(["token_1"], None, MissingOffsets(), utterance_evidence=True)

    def test_initialization_preserves_original_rng_parameters_and_logits(self):
        baseline, evidence, expected_rng = paired_models()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, evidence.state_dict()[key]), key)
        dataset = self.dataset(["token_1\ntoken_2", "token_3"])
        batch = dataset.collate([dataset[0], dataset[1]])
        for training in (False, True):
            baseline.train(training)
            evidence.train(training)
            torch.manual_seed(81)
            expected, expected_features = baseline(batch)
            torch.manual_seed(81)
            actual, actual_features = evidence(batch)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_features, expected_features))
        # This path is genuinely multilabel: every combination is supported.
        decisions = torch.arange(512).unsqueeze(1).bitwise_right_shift(torch.arange(9)).bitwise_and(1)
        logits = torch.nn.functional.one_hot(decisions, 2).float()
        self.assertTrue(torch.equal(predictions(logits), decisions))

    def test_no_extra_encoder_forward_and_non_nausea_logits_unchanged(self):
        baseline, evidence, _ = paired_models()
        baseline.eval(); evidence.eval()
        baseline.encoder_window_batch = evidence.encoder_window_batch = 2
        nn.init.normal_(evidence.evidence.output.weight)
        nn.init.normal_(evidence.evidence.output.bias)
        dataset = self.dataset(["token_1\ntoken_2\ntoken_3\ntoken_4\ntoken_5", "token_6", ""])
        batch = dataset.collate([dataset[i] for i in range(3)])
        expected, _ = baseline(batch)
        actual, _ = evidence(batch)
        self.assertEqual(baseline.encoder.calls, evidence.encoder.calls)
        other_labels = [i for i in range(9) if i != 6]
        self.assertTrue(torch.equal(actual[:, other_labels], expected[:, other_labels]))
        self.assertTrue(torch.equal(actual[-1], expected[-1]))
        self.assertGreater(float((actual[0, 6] - expected[0, 6]).detach().abs().sum()), 0)

    def test_document_turn_reconstruction_averages_duplicate_occurrences(self):
        _, model, _ = paired_models()
        model.eval()
        dataset = self.dataset(["token_1 token_2 token_3 token_4 token_5\ntoken_6\ntoken_7"])
        batch = dataset.collate([dataset[0]])
        captured = []
        handle = model.evidence.register_forward_pre_hook(lambda module, args: captured.append(args))
        model(batch)
        handle.remove()
        turns, mask, _ = captured[0]
        expected = torch.stack([model.encoder.embedding(torch.tensor([11, 12, 13, 14, 15])).mean(0),
                                model.encoder.embedding(torch.tensor(16)), model.encoder.embedding(torch.tensor(17))])
        torch.testing.assert_close(turns[0], expected)
        self.assertEqual(mask.tolist(), [[True, True, True]])

    def test_ordered_neighbors_cross_windows_and_ignore_padded_turns(self):
        module = UtteranceEvidence(2, 3, hidden=8, dropout=0.0).eval()
        nn.init.normal_(module.output.weight)
        turns = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [100., 100.]]])
        mask = torch.tensor([[True, True, True, False]])
        global_features = torch.tensor([[0.1, 0.2, 0.3]])
        captured = []
        handle = module.projection.register_forward_pre_hook(lambda unused, args: captured.append(args[0].detach().clone()))
        actual = module(turns, mask, global_features)
        handle.remove()
        torch.testing.assert_close(captured[0][0, 1, :6], torch.tensor([1., 2., 3., 4., 5., 6.]))
        torch.testing.assert_close(captured[0][0, 0, :2], torch.zeros(2))
        torch.testing.assert_close(captured[0][0, 2, 4:6], torch.zeros(2))
        reversed_turns = turns.clone()
        reversed_turns[:, :3] = reversed_turns[:, :3].flip(1)
        self.assertFalse(torch.allclose(actual, module(reversed_turns, mask, global_features)))
        turns[:, 3] = -999
        torch.testing.assert_close(actual, module(turns, mask, global_features))
        torch.testing.assert_close(module(turns, torch.zeros_like(mask), global_features), torch.zeros(1, 2))

    def test_two_updates_propagate_to_local_head_attention_and_encoder(self):
        _, model, _ = paired_models()
        model.evidence.projection[-1].p = model.dropout.p = 0.0
        dataset = self.dataset(["token_1 token_2\ntoken_3\ntoken_4", "token_5\ntoken_6\ntoken_7"],
                               [[0, 1, 0, 0, 0, 0, 1, 0, 0], [0] * 9])
        batch = dataset.collate([dataset[0], dataset[1]])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        logits, _ = model(batch)
        classification_loss(logits, batch["targets"], nausea_positive_weight=3.0).backward()
        self.assertGreater(float(model.evidence.output.weight.grad.abs().sum()), 0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch)
        classification_loss(logits, batch["targets"], nausea_positive_weight=3.0).backward()
        for parameter in (model.evidence.projection[0].weight, model.evidence.attention.weight,
                          model.evidence.output.weight, model.encoder.embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_padding_targets_and_strict_checkpoint_reconstruction(self):
        _, model, _ = paired_models()
        model.eval(); nn.init.normal_(model.evidence.output.weight)
        dataset = self.dataset(["token_1\ntoken_2", "token_3\ntoken_4\ntoken_5"], [[0] * 9, [1] * 9])
        batch = dataset.collate([dataset[0], dataset[1]])
        expected, _ = model(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["targets"] = 1 - changed["targets"]
        changed["input_ids"][~changed["attention_mask"].bool()] = 200
        actual, _ = model(changed)
        torch.testing.assert_close(actual, expected)
        single, _ = model(dataset.collate([dataset[0]]))
        torch.testing.assert_close(single[0], expected[0], atol=1e-6, rtol=1e-6)
        restored = DialogueClassifier(TinyEncoder(), feature_dim=16, utterance_evidence=True,
                                      evidence_hidden=12, evidence_dropout=0.1).eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(restored(batch)[0], expected, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            DialogueClassifier(TinyEncoder(), feature_dim=16).load_state_dict(model.state_dict(), strict=True)

    def test_invalid_metadata_and_settings_rejected(self):
        _, model, _ = paired_models()
        dataset = self.dataset(["token_1"])
        batch = dataset.collate([dataset[0]])
        missing = {key: value for key, value in batch.items() if key != "turn_ids"}
        with self.assertRaises(ValueError):
            model(missing)
        wrong = {key: value.clone() for key, value in batch.items()}
        wrong["token_weights"][0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            model(wrong)
        for hidden in (0, 513, True, 1.5):
            with self.assertRaises(ValueError):
                validate_evidence_config(hidden=hidden)
        for dropout in (-0.1, 1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                validate_evidence_config(dropout=dropout)


if __name__ == "__main__":
    unittest.main()
