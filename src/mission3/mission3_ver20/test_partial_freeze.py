"""CPU checks for the frozen RoBERTa prefix and complete-dialogue training."""
import io
import json
from types import SimpleNamespace
import unittest

import torch
from torch import nn

if __package__:
    from .neural import (DialogueClassifier, EncodeDataset, classification_loss,
                         configure_partial_finetuning)
else:
    from neural import (DialogueClassifier, EncodeDataset, classification_loss,
                        configure_partial_finetuning)

try:
    from transformers import RobertaConfig, RobertaModel
except ImportError:
    RobertaConfig = RobertaModel = None


class ToyRoberta(nn.Module):
    """Small structural double; the optional smoke test uses real HF layers."""

    def __init__(self, layers=4):
        super().__init__()
        self.config = SimpleNamespace(model_type="roberta", hidden_size=8,
                                      num_hidden_layers=layers)
        self.embeddings = nn.Sequential(nn.Embedding(256, 8), nn.Dropout(0.4))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Dropout(0.4))
            for _ in range(layers)
        ])
        self.pooler = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.4))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class ToyTokenizer:
    pad_token_id = 0
    sep_token = "[SEP]"

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, texts, *, max_length, stride, **kwargs):
        result = {"input_ids": [], "attention_mask": [],
                  "overflow_to_sample_mapping": []}
        for sample, text in enumerate(texts):
            tokens = [102 if token == self.sep_token else int(token) + 10
                      for token in text.split()]
            capacity = max_length - 2
            start = 0
            while True:
                ids = [101, *tokens[start:start + capacity], 102]
                result["input_ids"].append(ids)
                result["attention_mask"].append([1] * len(ids))
                result["overflow_to_sample_mapping"].append(sample)
                if start + capacity >= len(tokens):
                    break
                start += capacity - stride
        return result


class PartialFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def batch(self):
        targets = [[0, 1, 0, 1, 0, 1, 0, 1, 0],
                   [1, 0, 1, 0, 1, 0, 1, 0, 1]]
        dataset = EncodeDataset(["1 2\n3", " ".join(map(str, range(1, 22)))],
                                targets, ToyTokenizer(), max_length=8, overlap=2)
        self.assertGreater(len(dataset.documents[1]), 1)
        self.assertIn(31, dataset.documents[1][-1]["input_ids"])
        return dataset, dataset.collate([dataset[0], dataset[1]])

    def test_default_plan_freezes_embeddings_first_16_and_pooler(self):
        encoder = ToyRoberta(layers=24)
        audit = configure_partial_finetuning(encoder)
        self.assertEqual(json.loads(json.dumps(audit)), audit)
        self.assertEqual(audit["trainable_layer_indices"], list(range(16, 24)))
        self.assertEqual(audit["frozen_layer_indices"], list(range(16)))
        self.assertEqual(audit["total_parameters"],
                         audit["trainable_parameters"] + audit["frozen_parameters"])
        for name, parameter in encoder.named_parameters():
            expected = any(name.startswith(f"encoder.layer.{index}.")
                           for index in range(16, 24))
            self.assertEqual(parameter.requires_grad, expected, name)

    def test_invalid_counts_and_model_layout_reject_without_mutation(self):
        for trainable, expected in [(True, 4), (0, 4), (-1, 4), (1.5, 4),
                                    (5, 4), (2, True), (2, 0), (2, 3)]:
            encoder = ToyRoberta()
            before = [parameter.requires_grad for parameter in encoder.parameters()]
            with self.subTest(trainable=trainable, expected=expected):
                with self.assertRaises(ValueError):
                    configure_partial_finetuning(encoder, trainable, expected)
                self.assertEqual(before, [p.requires_grad for p in encoder.parameters()])
                self.assertTrue(all(module.training for module in encoder.modules()))
        for mutation in ("model_type", "declared_layers", "missing_embeddings",
                         "plain_list", "shared_parameter", "empty_layer"):
            encoder = ToyRoberta()
            if mutation == "model_type":
                encoder.config.model_type = "bert"
            elif mutation == "declared_layers":
                encoder.config.num_hidden_layers = 3
            elif mutation == "missing_embeddings":
                del encoder.embeddings
            elif mutation == "plain_list":
                del encoder.encoder.layer
                encoder.encoder.layer = [nn.Linear(8, 8) for _ in range(4)]
            elif mutation == "shared_parameter":
                encoder.encoder.layer[2][0].weight = encoder.encoder.layer[0][0].weight
            else:
                encoder.encoder.layer[1] = nn.Identity()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                configure_partial_finetuning(encoder, 2, 4)

    def test_train_eval_cycles_keep_frozen_dropout_off(self):
        encoder = ToyRoberta()
        configure_partial_finetuning(encoder, 2, 4)
        model = DialogueClassifier(encoder, feature_dim=16)
        for _ in range(2):
            model.eval()
            self.assertTrue(all(not module.training for module in model.modules()))
            model.train()
            for module in [encoder.embeddings, *encoder.encoder.layer[:2], encoder.pooler]:
                self.assertTrue(all(not child.training for child in module.modules()))
            for module in encoder.encoder.layer[2:]:
                self.assertTrue(all(child.training for child in module.modules()))
            self.assertTrue(model.dropout.training)
            self.assertTrue(encoder.encoder.training)
        captures = []
        hook = encoder.encoder.layer[1].register_forward_hook(
            lambda module, args, value: captures.append(value.detach().clone()))
        _, batch = self.batch()
        model.encoder_window_batch = 99
        model(batch)
        model(batch)
        hook.remove()
        self.assertTrue(torch.equal(captures[0], captures[1]))

    def _assert_optimizer_step(self, encoder):
        audit = configure_partial_finetuning(encoder, 2, 4)
        model = DialogueClassifier(encoder, feature_dim=16).train()
        model.encoder_window_batch = 2
        _, batch = self.batch()
        frozen_before = {name: parameter.detach().clone()
                         for name, parameter in model.named_parameters()
                         if not parameter.requires_grad}
        trainable_before = {name: parameter.detach().clone()
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad}
        # Including frozen parameters tests that weight decay cannot alter them.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.1)
        logits, _ = model(batch)
        loss = classification_loss(logits, batch["targets"],
                                   positive_class_weights=[1.4, 1.8, 2, 1.2, 1.3, 1, 3, 1.5, 1.3])
        loss.backward()
        self.assertGreater(float(model.class_weights.grad.abs().sum()), 0)
        self.assertGreater(float(model.projection[0].weight.grad.abs().sum()), 0)
        for index in range(2, 4):
            gradients = [parameter.grad for parameter in encoder.encoder.layer[index].parameters()
                         if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen_before:
                self.assertTrue(torch.equal(parameter.detach(), frozen_before[name]), name)
        for index in range(2, 4):
            self.assertTrue(any(
                not torch.equal(parameter.detach(), trainable_before[f"encoder.encoder.layer.{index}.{name}"])
                for name, parameter in encoder.encoder.layer[index].named_parameters()))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(audit["frozen_parameters"], 0)
        return model, batch

    def test_optimizer_updates_only_top_layers_and_classifier(self):
        self._assert_optimizer_step(ToyRoberta())

    def test_offline_inference_does_not_require_freeze_marker(self):
        model, batch = self._assert_optimizer_step(ToyRoberta())
        model.eval()
        expected, _ = model(batch)
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored = DialogueClassifier(ToyRoberta(), feature_dim=16).eval()
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        actual, _ = restored(batch)
        torch.testing.assert_close(expected, actual)
        self.assertFalse(hasattr(restored.encoder, "_mission3_partial_finetuning"))

    def test_overflow_padding_and_chunk_size_preserve_predictions(self):
        encoder = ToyRoberta()
        configure_partial_finetuning(encoder, 2, 4)
        model = DialogueClassifier(encoder, feature_dim=16).eval()
        dataset, batch = self.batch()
        single_logits, _ = model(dataset.collate([dataset[0]]))
        expected, _ = model(batch)
        torch.testing.assert_close(single_logits[0], expected[0])
        model.encoder_window_batch = 1
        actual, _ = model(batch)
        torch.testing.assert_close(expected, actual)
        padded = {key: value.clone() for key, value in batch.items()}
        padded["input_ids"][~padded["window_mask"]] = 250
        padded["attention_mask"][~padded["window_mask"]] = 1
        altered, _ = model(padded)
        torch.testing.assert_close(expected, altered)

    @unittest.skipIf(RobertaModel is None, "Optional Transformers runtime is not installed")
    def test_real_roberta_nonreentrant_checkpointing_and_partial_gradients(self):
        config = RobertaConfig(vocab_size=256, hidden_size=16, intermediate_size=24,
                               num_hidden_layers=4, num_attention_heads=4,
                               max_position_embeddings=32, hidden_dropout_prob=0.2,
                               attention_probs_dropout_prob=0.2)
        encoder = RobertaModel(config)
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self._assert_optimizer_step(encoder)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
