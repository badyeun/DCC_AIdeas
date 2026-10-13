"""Focused CPU tests without downloading models or importing Transformers."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .neural import (DialogueClassifier, EncodeDataset, classification_loss,
                         predictions, prepare_encoder, seed_everything,
                         binary_supervised_contrastive_loss)
else:
    from neural import (DialogueClassifier, EncodeDataset, classification_loss,
                        predictions, prepare_encoder, seed_everything,
                        binary_supervised_contrastive_loss)


class FakeTokenizer:
    pad_token_id = 0
    sep_token = "[SEP]"
    is_fast = True
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    init_kwargs = {}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, *, max_length, stride, **kwargs):
        result = {key: [] for key in self.model_input_names}
        result["overflow_to_sample_mapping"] = []
        self.seen_texts = texts
        for index, text in enumerate(texts):
            ids = [102 if token == self.sep_token else 10 + int(token.split("_")[-1])
                   for token in text.split()]
            capacity = max_length - 2
            start = 0
            while True:
                window = [101] + ids[start:start + capacity] + [102]
                result["input_ids"].append(window)
                result["attention_mask"].append([1] * len(window))
                result["token_type_ids"].append([0] * len(window))
                result["overflow_to_sample_mapping"].append(index)
                if start + capacity >= len(ids):
                    break
                start += capacity - stride
        return result

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakeConfig:
    hidden_size = 8
    use_cache = True
    _commit_hash = "a" * 40

    def save_pretrained(self, directory):
        Path(directory, "config.json").write_text("{}", encoding="utf-8")


class FakeEncoder(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.embedding = nn.Embedding(300, self.config.hidden_size)
        self.chunk_sizes = []

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.chunk_sizes.append(len(input_ids))
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpoint_options = kwargs


class NeuralTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)

    def dataset(self, texts, targets=None):
        return EncodeDataset(texts, targets, FakeTokenizer(), max_length=8, overlap=2)

    def test_complete_tail_and_utterance_separator(self):
        text = " ".join(f"token_{value}" for value in range(1, 24))
        dataset = self.dataset([text, "token_1\ntoken_2", ""])
        observed = {token for window in dataset.documents[0] for token in window["input_ids"]}
        self.assertTrue(set(range(11, 34)).issubset(observed))
        self.assertIn(33, dataset.documents[0][-1]["input_ids"])
        self.assertIn("[SEP]", dataset.tokenizer.seen_texts[1])
        self.assertEqual(len(dataset.documents[2]), 1)
        with self.assertRaises(ValueError):
            EncodeDataset([text], None, FakeTokenizer(), max_length=8, overlap=6)

    def test_padding_and_batch_composition_do_not_change_prediction(self):
        short, long = "token_1 token_2", " ".join(f"token_{value}" for value in range(1, 20))
        dataset = self.dataset([short, long])
        for attention in (False, True):
            model = DialogueClassifier(FakeEncoder(), label_attention=attention, feature_dim=12).eval()
            model.encoder_window_batch = 2
            single_logits, single_features = model(dataset.collate([dataset[0]]))
            batch = dataset.collate([dataset[0], dataset[1]])
            batch_logits, batch_features = model(batch)
            torch.testing.assert_close(single_logits[0], batch_logits[0], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(single_features[0], batch_features[0], atol=1e-6, rtol=1e-6)
            # Fill fake windows with real-looking tokens: window_mask must still exclude them.
            padded = {key: value.clone() for key, value in batch.items()}
            padded["input_ids"][~padded["window_mask"]] = 299
            padded["attention_mask"][~padded["window_mask"]] = 1
            padded_logits, _ = model(padded)
            torch.testing.assert_close(batch_logits, padded_logits)
            self.assertLessEqual(max(model.encoder.chunk_sizes), 2)

    def test_single_window_has_different_label_token_contexts_and_ignores_padding(self):
        dataset = self.dataset(["token_1 token_2 token_3", "token_4"])
        model = DialogueClassifier(FakeEncoder(), label_attention=True, feature_dim=12).eval()
        projected_inputs = []
        handle = model.projection.register_forward_pre_hook(
            lambda module, args: projected_inputs.append(args[0].detach().clone()))
        single = dataset.collate([dataset[0]])
        model(single)
        context = projected_inputs[0][0, :, model.encoder.config.hidden_size:]
        self.assertEqual(context.shape, (9, 8))
        self.assertGreater(float(context.var(dim=0).sum()), 1e-6)
        handle.remove()
        padded = dataset.collate([dataset[0], dataset[1]])
        expected, _ = model(padded)
        altered = {key: value.clone() for key, value in padded.items()}
        altered["input_ids"][~altered["attention_mask"].bool()] = 299
        actual, _ = model(altered)
        torch.testing.assert_close(expected, actual)

    def test_logits_features_and_complete_encoder_backpropagation(self):
        targets = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1]])
        dataset = self.dataset(["token_1 token_2", "token_3 token_4 token_5"], targets)
        model = DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True, feature_dim=16)
        batch = dataset.collate([dataset[0], dataset[1]])
        logits, features1 = model(batch)
        _, features2 = model(batch)
        self.assertEqual(logits.shape, (2, 9, 2))
        self.assertEqual(features1.shape, (2, 9, 64))
        torch.testing.assert_close(features1.norm(dim=-1), torch.ones(2, 9))
        loss = classification_loss(logits, batch["targets"]) + binary_supervised_contrastive_loss(features1, features2, targets)
        loss.backward()
        self.assertGreater(float(model.encoder.embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.class_weights.grad.abs().sum()), 0)
        self.assertGreater(float(model.feature_weights.grad.abs().sum()), 0)

    def test_class_decisions_and_unweighted_label_symmetric_cross_entropy(self):
        logits = torch.zeros(2, 9, 2)
        logits[0, 3, 1] = 2
        predicted = predictions(logits)
        self.assertEqual(int(predicted.sum()), 1)
        self.assertEqual(int(predicted[0, 3]), 1)
        targets = torch.randint(0, 2, (2, 9))
        loss = classification_loss(logits, targets)
        flipped = classification_loss(logits.flip(-1), 1 - targets)
        torch.testing.assert_close(loss, flipped)
        uniform = classification_loss(torch.zeros_like(logits), targets)
        torch.testing.assert_close(uniform, torch.log(torch.tensor(2.0)))
        with self.assertRaises(FloatingPointError):
            predictions(torch.full_like(logits, float("nan")))

    def test_offline_checkpoint_roundtrip_and_nonfinite_encoder_rejection(self):
        dataset = self.dataset(["token_1 token_2", "token_3"])
        batch = dataset.collate([dataset[0], dataset[1]])
        model = DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True, feature_dim=16).eval()
        expected, _ = model(batch)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "weights.pt")
            torch.save(model.state_dict(), checkpoint)
            restored = DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True, feature_dim=16).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
            actual, _ = restored(batch)
            torch.testing.assert_close(expected, actual)
        with torch.no_grad():
            model.encoder.embedding.weight[11].fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            model(batch)

    def test_real_training_partial_accumulation_checkpoint_and_inference(self):
        from torch.utils.data import DataLoader
        if __package__:
            from . import runtime
        else:
            package_root = str(Path(__file__).resolve().parent.parent)
            with patch.object(sys, "path", [package_root, *sys.path]):
                from mission3_ver14 import runtime

        texts = [" ".join(f"token_{value}" for value in range(index + 1, index + 4 + index % 3))
                 for index in range(10)]
        # The last call has multiple windows; no token is dropped in training.
        texts[-1] = " ".join(f"token_{value}" for value in range(1, 15))
        targets = torch.tensor([[(index + label) % 2 for label in range(9)]
                                for index in range(10)])
        dataset = self.dataset(texts, targets)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate)
        self.assertEqual(len(loader), 5)
        model = DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True, feature_dim=16)
        model.encoder_window_batch = 2
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        settings = {"learning_rate": .01, "head_learning_rate": .02,
                    "max_epochs": 1, "gradient_accumulation": 3,
                    "contrastive_weight": .1, "patience": 1, "log_every": 100}
        step_calls = []
        original_optimizer = torch.optim.AdamW

        class CountingAdamW(original_optimizer):
            def step(self, *args, **kwargs):
                step_calls.append(True)
                return super().step(*args, **kwargs)

        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with patch.object(torch.optim, "AdamW", CountingAdamW):
                epoch, checkpoint = runtime._fit_epochs(
                    model, loader, settings, device, output_dir, "integration", fixed_epochs=1,
                )
            self.assertEqual(epoch, 1)
            # Five batches make one full three-batch group and one two-batch group.
            self.assertEqual(len(step_calls), 2)
            for name in ("encoder.embedding.weight", "projection.0.weight", "class_weights", "feature_weights"):
                self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)
            history = json.loads((output_dir / "integration_history.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["epoch"], 1)
            self.assertTrue(math.isfinite(history[0]["loss"]))
            self.assertGreater(history[0]["loss"], 0)
            self.assertTrue(math.isfinite(history[0]["seconds"]))
            predicted = runtime._predict(model, loader, device)
            self.assertEqual(predicted.shape, (10, 9))
            self.assertTrue(((predicted == 0) | (predicted == 1)).all())
            restored = DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True, feature_dim=16)
            runtime._load_weights(restored, checkpoint)
            roundtrip = runtime._predict(restored, loader, device)
            self.assertEqual(predicted.tobytes(), roundtrip.tobytes())

    def test_contrastive_anchor_classes_are_balanced(self):
        # Different negative and positive anchor costs reveal prevalence weighting.
        view1 = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])[:, None].expand(-1, 9, -1)
        view2 = torch.tensor([[1., .2, 0.], [.2, 1., 0.], [0., .3, 1.]])[:, None].expand(-1, 9, -1)
        targets = torch.tensor([0, 0, 1])[:, None].expand(-1, 9)
        loss = binary_supervised_contrastive_loss(view1, view2, targets, temperature=.5)
        x = F.normalize(torch.cat((view1[:, 0], view2[:, 0])), dim=-1)
        labels = torch.tensor([0, 0, 1, 0, 0, 1])
        costs = []
        for anchor in range(6):
            others = [index for index in range(6) if index != anchor]
            positive = [index for index in others if labels[index] == labels[anchor]]
            scores = torch.stack([x[anchor].dot(x[index]) / .5 for index in others])
            log_denominator = torch.logsumexp(scores, dim=0)
            costs.append(torch.stack([log_denominator - x[anchor].dot(x[index]) / .5
                                      for index in positive]).mean())
        costs = torch.stack(costs)
        expected = (costs[labels == 0].mean() + costs[labels == 1].mean()) / 2
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float((loss - costs.mean()).abs()), 1e-3)
        torch.testing.assert_close(loss, binary_supervised_contrastive_loss(view1, view2, 1 - targets, temperature=.5))

    def test_contrastive_tiny_and_single_class_batches_are_finite(self):
        for count in (1, 4):
            features1 = torch.randn(count, 9, 64, requires_grad=True)
            features2 = torch.randn(count, 9, 64, requires_grad=True)
            for value in (0, 1):
                loss = binary_supervised_contrastive_loss(features1, features2, torch.full((count, 9), value))
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward(retain_graph=True)
        with self.assertRaises(ValueError):
            binary_supervised_contrastive_loss(features1, features2, torch.zeros(4, 9), temperature=0)
        with self.assertRaises(ValueError):
            self.dataset(["token_1"], [[.5] * 9])

    def test_preparation_is_lazy_pinned_and_saves_only_encoder_config(self):
        encoder, tokenizer = FakeEncoder(), FakeTokenizer()
        calls = []
        def load_model(model_id, **options):
            calls.append((model_id, options))
            return encoder
        def load_tokenizer(model_id, **options):
            calls.append((model_id, options))
            return tokenizer
        def load_config(model_id, **options):
            calls.append((model_id, options))
            return encoder.config
        stub = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=load_config),
                               AutoModel=SimpleNamespace(from_pretrained=load_model),
                               AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"transformers": stub}):
            loaded_encoder, loaded_tokenizer, revision = prepare_encoder("official/fake", "main", temporary)
            self.assertIs(loaded_encoder, encoder)
            self.assertIs(loaded_tokenizer, tokenizer)
            self.assertEqual(revision, "a" * 40)
            self.assertTrue(Path(temporary, "encoder", "config.json").is_file())
            self.assertTrue(Path(temporary, "tokenizer", "tokenizer_config.json").is_file())
            self.assertEqual(list(Path(temporary, "encoder").iterdir()), [Path(temporary, "encoder", "config.json")])
        self.assertFalse(encoder.config.use_cache)
        self.assertEqual(encoder.checkpoint_options, {"gradient_checkpointing_kwargs": {"use_reentrant": False}})
        self.assertTrue(all(options["trust_remote_code"] is False for _, options in calls))
        self.assertEqual(calls[0][1]["revision"], "main")
        self.assertTrue(all(options["revision"] == "a" * 40 for _, options in calls[1:]))


class NauseaWeightedLossTests(unittest.TestCase):
    def example(self):
        logits = torch.linspace(-1.3, 2.1, 3 * 9 * 2).reshape(3, 9, 2)
        targets = torch.ones(3, 9, dtype=torch.long)
        targets[1, 6] = 0
        targets[0, 2] = 0
        targets[2, 8] = 0
        return logits, targets

    def test_unit_weight_is_exact_original_cross_entropy(self):
        logits, targets = self.example()
        expected = F.cross_entropy(logits.float().reshape(-1, 2), targets.reshape(-1))
        self.assertTrue(torch.equal(classification_loss(logits, targets), expected))
        self.assertTrue(torch.equal(classification_loss(logits, targets, 1.0), expected))

    def test_weighted_loss_uses_only_positive_nausea_and_original_denominator(self):
        logits, targets = self.example()
        baseline = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        per_item = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction="none").reshape(3, 9)
        positive_nausea = per_item[targets[:, 6] == 1, 6].sum()
        for weight in (0.5, 2.0, 3.0):
            with self.subTest(weight=weight):
                expected = baseline + (weight - 1) * positive_nausea / targets.numel()
                torch.testing.assert_close(classification_loss(logits, targets, weight), expected)

    def test_only_positive_nausea_gradients_are_scaled_and_argmax_is_unchanged(self):
        logits, targets = self.example()
        original = logits.clone()
        before = predictions(logits).clone()
        unweighted = logits.clone().requires_grad_()
        weighted = logits.clone().requires_grad_()
        baseline_gradient, = torch.autograd.grad(classification_loss(unweighted, targets), unweighted)
        actual_gradient, = torch.autograd.grad(classification_loss(weighted, targets, 3.0), weighted)
        nausea_positive = torch.zeros_like(targets, dtype=torch.bool)
        nausea_positive[:, 6] = targets[:, 6] == 1
        torch.testing.assert_close(actual_gradient[nausea_positive], baseline_gradient[nausea_positive] * 3)
        torch.testing.assert_close(actual_gradient[~nausea_positive], baseline_gradient[~nausea_positive])
        self.assertGreater(float(actual_gradient[nausea_positive].abs().sum()),
                           float(baseline_gradient[nausea_positive].abs().sum()))
        self.assertTrue(torch.equal(logits, original))
        self.assertTrue(torch.equal(predictions(weighted.detach()), before))
        self.assertTrue(torch.equal(before, logits.argmax(dim=-1)))

    def test_no_positive_nausea_keeps_other_items_and_gradients_unchanged(self):
        logits, targets = self.example()
        targets[:, 6] = 0
        unweighted = logits.clone().requires_grad_()
        weighted = logits.clone().requires_grad_()
        baseline_loss = classification_loss(unweighted, targets)
        actual_loss = classification_loss(weighted, targets, 3.0)
        torch.testing.assert_close(actual_loss, baseline_loss)
        baseline_gradient, = torch.autograd.grad(baseline_loss, unweighted)
        actual_gradient, = torch.autograd.grad(actual_loss, weighted)
        torch.testing.assert_close(actual_gradient, baseline_gradient)

    def test_invalid_weight_is_rejected_even_without_positive_nausea(self):
        logits, targets = self.example()
        for has_positive in (False, True):
            targets[:, 6] = int(has_positive)
            for weight in (True, False, 0, -1, float("nan"), float("inf"), -float("inf"), "2", None, 1 + 0j):
                with self.subTest(has_positive=has_positive, weight=weight):
                    with self.assertRaisesRegex(ValueError, "finite positive real"):
                        classification_loss(logits, targets, weight)


if __name__ == "__main__":
    unittest.main()
