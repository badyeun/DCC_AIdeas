"""Exercise real CPU optimization, dual-view CE, and fresh full-Training refit."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver14 import neural, runtime
    from mission3_ver14.test_neural import FakeEncoder, FakeTokenizer
else:
    from . import neural, runtime
    from .test_neural import FakeEncoder, FakeTokenizer


def examples():
    texts = [f"token_{index + 1} token_{index + 15}" for index in range(6)]
    targets = np.asarray([[(index + label) % 2 for label in range(9)] for index in range(6)], dtype=np.int8)
    targets[:, 6] = [0, 0, 1, 0, 1, 1]
    return texts, targets


def settings(**overrides):
    return {
        "model_id": "fixture/encoder", "revision": "a" * 40, "seed": 21,
        "label_attention": True, "contrastive_weight": 0.0, "ce_views": 2,
        "nausea_positive_weight": 2.0, "batch_size": 2, "encoder_window_batch": 2,
        "max_length": 8, "overlap": 2, "learning_rate": .01,
        "head_learning_rate": .02, "max_epochs": 4, "gradient_accumulation": 2,
        "patience": 2, "allow_cpu": True, "log_every": 100, **overrides,
    }


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        neural.seed_everything(21)
        self.temporary = tempfile.TemporaryDirectory(prefix=".training_test_", dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def loader(self, batch_size=2):
        texts, targets = examples()
        dataset = neural.EncodeDataset(texts, targets, FakeTokenizer(), max_length=8, overlap=2)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)

    def model(self, contrastive=False):
        model = neural.DialogueClassifier(FakeEncoder(), label_attention=True,
                                           contrastive=contrastive, feature_dim=16)
        model.encoder_window_batch = 2
        return model

    def test_two_ce_views_average_real_losses_and_forward_nausea_weights_to_both(self):
        original_ce = neural.classification_loss
        for weight in (2.0, 3.0):
            with self.subTest(weight=weight):
                neural.seed_everything(21)
                model, loader = self.model(), self.loader()
                before = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
                calls, forwards, backward_gradients = [], [], {}

                def observe_ce(logits, targets, nausea_positive_weight=1.0):
                    loss = original_ce(logits, targets, nausea_positive_weight)
                    view_index = len(calls)
                    logits.register_hook(lambda gradient, key=view_index:
                                         backward_gradients.__setitem__(key, gradient.detach().clone()))
                    calls.append((nausea_positive_weight, float(loss.detach())))
                    return loss

                hook = model.register_forward_hook(
                    lambda module, args, output: forwards.append(output[0].detach().clone()))
                try:
                    with patch.object(neural, "classification_loss", side_effect=observe_ce), \
                            patch.object(neural, "binary_supervised_contrastive_loss",
                                         side_effect=AssertionError("CE-only training must not invoke SupCon")):
                        epoch, weights = runtime._fit_epochs(
                            model, loader, settings(nausea_positive_weight=weight), torch.device("cpu"),
                            self.root, f"ce_weight_{int(weight)}", fixed_epochs=1)
                finally:
                    hook.remove()
                self.assertEqual(epoch, 1)
                self.assertTrue(weights.is_file())
                self.assertEqual(len(forwards), 2 * len(loader))
                self.assertEqual([value for value, _ in calls], [weight] * (2 * len(loader)))
                self.assertEqual(set(backward_gradients), set(range(2 * len(loader))))
                self.assertTrue(all(bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0
                                    for gradient in backward_gradients.values()))
                self.assertTrue(any(not torch.equal(forwards[i], forwards[i + 1])
                                    for i in range(0, len(forwards), 2)))
                history = json.loads((self.root / f"ce_weight_{int(weight)}_history.json").read_text(encoding="utf-8-sig"))
                expected_ce = sum((calls[i][1] + calls[i + 1][1]) / 2
                                  for i in range(0, len(calls), 2)) / len(loader)
                self.assertAlmostEqual(history[0]["classification_loss"], expected_ce, places=6)
                self.assertAlmostEqual(history[0]["loss"], expected_ce, places=6)
                self.assertEqual(history[0]["contrastive_loss"], 0.0)
                self.assertEqual(history[0]["ce_views"], 2)
                self.assertEqual(history[0]["nausea_positive_count_per_batch"], {"0": 1, "1": 1, "2": 1})
                self.assertFalse(torch.equal(before["encoder.embedding.weight"], model.encoder.embedding.weight))
                self.assertFalse(torch.equal(before["class_weights"], model.class_weights))
                # CE backpropagates through the encoder/head without training the SupCon projection.
                torch.testing.assert_close(before["feature_weights"], model.feature_weights, rtol=0, atol=0)

    def test_history_reports_actual_ce_supcon_and_nausea_batch_counts(self):
        model, loader = self.model(contrastive=True), self.loader()
        original_ce, original_supcon = neural.classification_loss, neural.binary_supervised_contrastive_loss
        ce_values, supcon_values = [], []
        initial_features = model.feature_weights.detach().clone()

        def observe_ce(logits, targets, nausea_positive_weight=1.0):
            loss = original_ce(logits, targets, nausea_positive_weight)
            ce_values.append(float(loss.detach()))
            return loss

        def observe_supcon(features1, features2, targets):
            loss = original_supcon(features1, features2, targets)
            supcon_values.append(float(loss.detach()))
            return loss

        with patch.object(neural, "classification_loss", side_effect=observe_ce), \
                patch.object(neural, "binary_supervised_contrastive_loss", side_effect=observe_supcon):
            runtime._fit_epochs(model, loader, settings(contrastive_weight=.1), torch.device("cpu"),
                                self.root, "supcon", fixed_epochs=2)
        history = json.loads((self.root / "supcon_history.json").read_text(encoding="utf-8-sig"))
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        self.assertEqual(len(ce_values), 4 * len(loader))
        self.assertEqual(len(supcon_values), 2 * len(loader))
        for index, row in enumerate(history):
            ce_segment = ce_values[index * 2 * len(loader):(index + 1) * 2 * len(loader)]
            supcon_segment = supcon_values[index * len(loader):(index + 1) * len(loader)]
            expected_ce = sum(ce_segment) / len(ce_segment)
            expected_supcon = sum(supcon_segment) / len(supcon_segment)
            self.assertAlmostEqual(row["classification_loss"], expected_ce, places=6)
            self.assertAlmostEqual(row["contrastive_loss"], expected_supcon, places=6)
            self.assertAlmostEqual(row["loss"], expected_ce + .1 * expected_supcon, places=6)
            self.assertGreater(row["contrastive_loss"], 0)
            self.assertTrue(math.isfinite(row["loss"]))
            self.assertEqual(row["nausea_positive_count_per_batch"], {"0": 1, "1": 1, "2": 1})
            self.assertEqual(row["ce_views"], 2)
        self.assertFalse(torch.equal(initial_features, model.feature_weights))

    def test_fixed_selected_epochs_preserve_max_epoch_scheduler_horizon(self):
        model, loader = self.model(), self.loader(batch_size=1)
        original_optimizer = torch.optim.AdamW
        optimizers, learning_rates = [], []

        class ObservedAdamW(original_optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                optimizers.append(self)

            def step(self, *args, **kwargs):
                learning_rates.append([group["lr"] for group in self.param_groups])
                return super().step(*args, **kwargs)

        with patch.object(torch.optim, "AdamW", ObservedAdamW):
            epoch, _ = runtime._fit_epochs(model, loader, settings(ce_views=1), torch.device("cpu"),
                                           self.root, "short_refit", fixed_epochs=1)
        self.assertEqual(epoch, 1)
        self.assertEqual(len(learning_rates), 3)
        # Six microbatches / accumulation 2 = 3 updates, on the original 4-epoch/12-update horizon.
        expected = [[.01, .02], [.01, .02], [.01 * 10 / 11, .02 * 10 / 11]]
        np.testing.assert_allclose(learning_rates, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([group["lr"] for group in optimizers[0].param_groups],
                                   [.01 * 9 / 11, .02 * 9 / 11], rtol=1e-12, atol=1e-12)
        history = json.loads((self.root / "short_refit_history.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["scheduler_horizon_epochs"], 4)
        self.assertEqual(history[0]["ce_views"], 1)

    def test_refit_starts_fresh_trains_all_targets_and_reloaded_probe_matches(self):
        texts, targets = examples()
        config = settings(contrastive_weight=.1, nausea_positive_weight=3.0)
        selected_encoder = FakeEncoder()
        with torch.no_grad():
            selected_encoder.embedding.weight.fill_(99.0)
        stale_weights = self.root / "selection_weights.pt"
        torch.save(selected_encoder.state_dict(), stale_weights)
        prepared, training_tokens, fitted = [], [], []
        original_fit = runtime._fit_epochs
        original_load = runtime._load_weights
        loaded_paths = []

        def prepare_fresh(model_id, revision, asset_dir):
            encoder, tokenizer = FakeEncoder(), FakeTokenizer()
            initial = encoder.embedding.weight.detach().clone()
            prepared.append((model_id, revision, encoder, initial))
            asset_dir.mkdir(parents=True)
            for name, value in (("encoder", encoder.config), ("tokenizer", tokenizer)):
                directory = asset_dir / name
                directory.mkdir()
                value.save_pretrained(directory)
            return encoder, tokenizer, revision

        def observe_fit(model, loader, fit_settings, device, output_dir, prefix, **kwargs):
            fitted.append((len(loader.dataset), loader.dataset.targets.detach().cpu().numpy(), kwargs["fixed_epochs"]))

            def observe_batch(module, arguments):
                if module.training:
                    training_tokens.extend(arguments[0]["input_ids"][:, 0, 1].tolist())
            hook = model.register_forward_pre_hook(observe_batch)
            try:
                return original_fit(model, loader, fit_settings, device, output_dir, prefix, **kwargs)
            finally:
                hook.remove()

        def observe_load(model, path):
            loaded_paths.append(Path(path))
            return original_load(model, path)

        with patch.object(neural, "prepare_encoder", side_effect=prepare_fresh), \
                patch.object(runtime, "_fit_epochs", side_effect=observe_fit), \
                patch.object(runtime, "_load_weights", side_effect=observe_load):
            manifest, expected_probe = runtime.refit_selected(texts, targets, config, 2, self.root)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0][:2], (config["model_id"], config["revision"]))
        self.assertIsNot(prepared[0][2], selected_encoder)
        self.assertFalse(torch.equal(prepared[0][3], selected_encoder.embedding.weight))
        self.assertFalse(torch.equal(prepared[0][3], prepared[0][2].embedding.weight))
        self.assertEqual(len(fitted), 1)
        self.assertEqual((fitted[0][0], fitted[0][2]), (len(targets), 2))
        np.testing.assert_array_equal(fitted[0][1], targets)
        self.assertEqual(Counter(training_tokens), Counter({index + 11: 4 for index in range(6)}))
        self.assertEqual(manifest["selected_epochs"], 2)
        self.assertEqual(manifest["fitted_samples"], len(targets))
        self.assertEqual(manifest["nausea_positive_weight"], 3.0)
        self.assertEqual(loaded_paths, [self.root / manifest["weights"]])
        history = json.loads((self.root / "final_history.json").read_text(encoding="utf-8-sig"))
        self.assertEqual([row["epoch"] for row in history], [1, 2])
        self.assertTrue(all(row["scheduler_horizon_epochs"] == 4 and row["ce_views"] == 2 for row in history))
        restored = neural.DialogueClassifier(FakeEncoder(), label_attention=True, contrastive=True)
        restored.encoder_window_batch = config["encoder_window_batch"]
        runtime._load_weights(restored, self.root / manifest["weights"])
        actual_probe = runtime._predict(restored, runtime._loader(texts, None, FakeTokenizer(), config, False),
                                        torch.device("cpu"))
        self.assertEqual(expected_probe.shape, (len(texts), 9))
        np.testing.assert_array_equal(actual_probe, expected_probe)
        self.assertTrue(np.isin(actual_probe, [0, 1]).all())


if __name__ == "__main__":
    unittest.main()
