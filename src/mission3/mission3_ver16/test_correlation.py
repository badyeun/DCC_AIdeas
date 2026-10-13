"""Train and reload the actual correlation module on nonpersonal CPU fixtures."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver16 import data, neural, runtime
    from mission3_ver16.test_runtime import ControlledBudget, examples, settings
else:
    from . import data, neural, runtime
    from .test_runtime import ControlledBudget, examples, settings
from mission3_ver14.test_neural import FakeConfig, FakeEncoder, FakeTokenizer


class CorrelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        neural.seed_everything(21)
        self.temporary = tempfile.TemporaryDirectory(prefix=".correlation_test_", dir=HERE)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, model_id, revision, assets):
        encoder, tokenizer = FakeEncoder(), FakeTokenizer()
        for name, value in (("encoder", encoder.config), ("tokenizer", tokenizer)):
            directory = assets / name
            directory.mkdir(parents=True)
            value.save_pretrained(directory)
        return encoder, tokenizer, revision

    def transformers_stub(self):
        calls = []

        def config(directory, **options):
            calls.append(("config", options))
            self.assertTrue((Path(directory) / "config.json").is_file())
            return FakeConfig()

        def tokenizer(directory, **options):
            calls.append(("tokenizer", options))
            self.assertTrue((Path(directory) / "tokenizer_config.json").is_file())
            return FakeTokenizer()

        def encoder(config, **options):
            calls.append(("encoder", options))
            return FakeEncoder()

        return SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=config),
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
            AutoModel=SimpleNamespace(from_config=encoder),
        ), calls

    def test_zero_residual_preserves_rng_common_parameters_and_initial_logits(self):
        torch.manual_seed(9)
        before = torch.get_rng_state().clone()
        module = neural.LabelCorrelation()
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        evidence = torch.randn(3, 9)
        for training in (False, True):
            module.train(training)
            torch.testing.assert_close(module(evidence), evidence, rtol=0, atol=0)
        neural.seed_everything(12)
        baseline = neural.DialogueClassifier(FakeEncoder(), feature_dim=16)
        baseline_rng = torch.get_rng_state().clone()
        neural.seed_everything(12)
        enhanced = neural.DialogueClassifier(FakeEncoder(), feature_dim=16, label_correlation=True)
        torch.testing.assert_close(baseline_rng, torch.get_rng_state(), rtol=0, atol=0)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, enhanced.state_dict()[key], rtol=0, atol=0)
        dataset = neural.EncodeDataset(*examples(), FakeTokenizer(), max_length=8, overlap=2)
        batch = dataset.collate([dataset[0], dataset[1]])
        for training in (False, True):
            baseline.train(training)
            enhanced.train(training)
            torch.manual_seed(98)
            expected = baseline(batch)
            torch.manual_seed(98)
            actual = enhanced(batch)
            for first, second in zip(expected, actual):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(neural.predictions(actual[0]), actual[0].argmax(-1), rtol=0, atol=0)

    def test_ce_updates_residual_then_propagates_across_labels(self):
        module = neural.LabelCorrelation(dropout=0.).train()
        evidence = torch.randn(4, 9)
        initial_hidden = module.blocks[0][0].weight.detach().clone()
        optimizer = torch.optim.SGD(module.parameters(), lr=.5)
        corrected = module(evidence)
        # One label's ordinary categorical CE trains its correlation output.
        logits = torch.stack((-corrected[:, 6] / 2, corrected[:, 6] / 2), -1)
        F.cross_entropy(logits, torch.ones(4, dtype=torch.long)).backward()
        decoder = module.blocks[0][-1]
        self.assertGreater(float(decoder.weight.grad[6].abs().sum()), 0.)
        optimizer.step()
        self.assertGreater(float(decoder.weight[6].detach().abs().sum()), 0.)
        # A zero decoder intentionally holds the hidden layer at step one.
        torch.testing.assert_close(module.blocks[0][0].weight, initial_hidden, rtol=0, atol=0)
        optimizer.zero_grad()
        probe = evidence.detach().clone().requires_grad_(True)
        corrected = module(probe)
        corrected[:, 6].sum().backward()
        other_labels = torch.arange(9) != 6
        self.assertGreater(float(probe.grad[:, other_labels].abs().sum()), 0.)
        self.assertGreater(float(module.blocks[0][0].weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(probe.grad).all())

    def test_model_ignores_targets_and_joint_ce_updates_encoder_and_correlation(self):
        model = neural.DialogueClassifier(FakeEncoder(), label_correlation=True)
        dataset = neural.EncodeDataset(*examples(), FakeTokenizer(), max_length=8, overlap=2)
        batch = dataset.collate([dataset[0], dataset[1]])
        model.eval()
        with torch.no_grad():
            expected = model(batch)[0]
            changed = {**batch, "targets": torch.ones(1, 1) * float("nan")}
            torch.testing.assert_close(model(changed)[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(model({key: value for key, value in batch.items()
                                             if key != "targets"})[0], expected, rtol=0, atol=0)
        self.assertEqual(list(inspect.signature(model.correlation.forward).parameters), ["evidence"])
        initial_encoder = model.encoder.embedding.weight.detach().clone()
        model.train()
        logits, _ = model(batch)
        neural.classification_loss(logits, batch["targets"],
                                   positive_class_weights=runtime.POSITIVE_CLASS_WEIGHTS).backward()
        self.assertGreater(float(model.encoder.embedding.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.correlation.blocks[0][-1].weight.grad.abs().sum()), 0.)
        torch.optim.AdamW(model.parameters(), lr=.01).step()
        self.assertFalse(torch.equal(model.encoder.embedding.weight, initial_encoder))
        self.assertGreater(float(model.correlation.blocks[0][-1].weight.detach().abs().sum()), 0.)

    def test_correlations_validate_architecture_and_finite_evidence(self):
        for key, values in (("hidden", (True, 0, -1, 257, 32., "32")),
                            ("depth", (True, 0, -1, 4, 1., "1")),
                            ("dropout", (True, -.1, 1., float("nan"), float("inf"), "0.1"))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    neural.LabelCorrelation(**{key: value})
        module = neural.LabelCorrelation()
        for value in (torch.ones(9), torch.ones(2, 8), torch.ones(2, 9, 1)):
            with self.assertRaises(ValueError):
                module(value)
        with self.assertRaises(FloatingPointError):
            module(torch.full((2, 9), float("nan")))
        with torch.no_grad():
            module.blocks[0][-1].bias.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            module(torch.ones(2, 9))

    def test_matched_candidates_fixed_weights_three_epochs_four_epoch_horizon(self):
        self.assertEqual(runtime.CANDIDATES, (
            {"candidate_id": "baseline_ce", "label_correlation": False},
            {"candidate_id": "cornet", "label_correlation": True},
        ))
        self.assertEqual(runtime.POSITIVE_CLASS_WEIGHTS, [1.4, 1.8, 2., 1.2, 1.3, 1., 3., 1.5, 1.3])
        normalized = runtime.validate_settings(settings(self.root))
        self.assertEqual(normalized["max_epochs"], 3)
        self.assertEqual(normalized["scheduler_horizon_epochs"], 4)
        for key, value in (("max_epochs", 4), ("scheduler_horizon_epochs", 3),
                           ("scheduler_horizon_epochs", True), ("label_correlation", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_settings(settings(self.root, **{key: value}))
        for epoch in (0, 4, True, 1.5):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                runtime.refit_selected(*examples(), normalized, epoch, self.root, ControlledBudget())

    def test_real_selection_fresh_refit_and_offline_reload_of_both_architectures(self):
        texts, targets = examples()
        fit, holdout = np.arange(4), np.arange(4, 6)
        for candidate in runtime.CANDIDATES:
            with self.subTest(candidate=candidate["candidate_id"]):
                config = settings(self.root, **candidate)
                local = self.root / candidate["candidate_id"]
                selection, final = local / "selection", local / "final"
                selection.mkdir(parents=True)
                final.mkdir()
                prepared = []

                def prepare(*args):
                    result = self.prepare(*args)
                    prepared.append(result[0])
                    return result

                with patch.object(neural, "prepare_encoder", side_effect=prepare):
                    selected, result = runtime.train_candidate(texts, targets, fit, holdout, config,
                                                               selection, ControlledBudget())
                    manifest, expected = runtime.refit_selected(texts, targets, config,
                                                                selected["selected_epochs"], final,
                                                                ControlledBudget())
                self.assertEqual(len(prepared), 2)
                self.assertIsNot(prepared[0], prepared[1])
                self.assertEqual(selected["training_scope"], "fit_only")
                self.assertEqual(selected["fitted_samples"], len(fit))
                self.assertEqual(manifest["training_scope"], "all_training")
                self.assertEqual(manifest["fitted_samples"], len(targets))
                self.assertEqual(manifest["selected_epochs"], selected["selected_epochs"])
                self.assertEqual(manifest["label_correlation"], candidate["label_correlation"])
                self.assertEqual(manifest["scheduler_horizon_epochs"], 4)
                self.assertEqual(manifest["positive_class_weights"], runtime.POSITIVE_CLASS_WEIGHTS)
                history = runtime._read(final / "final_history.json")
                self.assertEqual(len(history), manifest["selected_epochs"])
                state = torch.load(final / manifest["weights"], map_location="cpu", weights_only=True)
                correlation_keys = [key for key in state if key.startswith("correlation.")]
                self.assertEqual(bool(correlation_keys), candidate["label_correlation"])
                if candidate["label_correlation"]:
                    self.assertGreater(float(state["correlation.blocks.0.3.weight"].abs().sum()), 0.)
                stub, calls = self.transformers_stub()
                with patch.dict(sys.modules, transformers=stub):
                    predicted = runtime.predict_checkpoint(final / "final_model.ckpt", texts)
                    np.testing.assert_array_equal(predicted, expected)
                    with np.load(selection / "holdout_predictions.npz", allow_pickle=False) as saved:
                        restored = runtime.predict_checkpoint(selection / "candidate_model.ckpt",
                                                               [texts[i] for i in holdout])
                        np.testing.assert_array_equal(restored, saved["predictions"])
                for _, options in calls:
                    self.assertIs(options["trust_remote_code"], False)
                for kind, options in calls:
                    if kind != "encoder":
                        self.assertIs(options["local_files_only"], True)
                # Declaring the wrong architecture cannot silently drop weights.
                incorrect = {**manifest, "label_correlation": not candidate["label_correlation"]}
                data.write_json(final / "wrong_model.ckpt", incorrect)
                with patch.dict(sys.modules, transformers=stub), self.assertRaises(RuntimeError):
                    runtime.predict_checkpoint(final / "wrong_model.ckpt", texts)
                if candidate["label_correlation"]:
                    data.write_json(final / "wrong_model.ckpt", {**manifest, "correlation_hidden": 31})
                    with patch.dict(sys.modules, transformers=stub), self.assertRaises(RuntimeError):
                        runtime.predict_checkpoint(final / "wrong_model.ckpt", texts)
                missing = dict(manifest)
                missing.pop("correlation_hidden")
                data.write_json(final / "wrong_model.ckpt", missing)
                with patch.dict(sys.modules, transformers=stub), self.assertRaisesRegex(ValueError, "declare"):
                    runtime.predict_checkpoint(final / "wrong_model.ckpt", texts)


if __name__ == "__main__":
    unittest.main()
