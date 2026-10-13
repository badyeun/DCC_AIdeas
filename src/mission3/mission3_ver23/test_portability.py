"""Portable path, fixed-combination, publication and FP32 inference contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import adaptation, data, portability
from .budget import BudgetExceeded
from .test_adaptation import new_model, GenerousBudget


HERE = Path(__file__).parent


def manifest():
    return {"format": "mission3_categorical_v1", "kind": "neural", "version": "ver19",
            "labels": list(data.TARGET_SYMPTOMS), "training_scope": "all_training", "fitted_samples": 29200,
            "weights": "weights.pt", "assets": "assets", "batch_size": 4, "encoder_window_batch": 4,
            "max_length": 512, "overlap": 64, "selected_epochs": 3}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class OrderedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.))
        self.forward_flags = []

    def forward(self, batch):
        self.forward_flags.append((self.training, torch.is_grad_enabled(), self.value.dtype))
        values = batch["x"].float().reshape(-1, 1).expand(-1, 9)
        return torch.stack((torch.zeros_like(values), values), -1), None


class PortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_softmax_stable_and_finite_validation(self):
        logits = np.zeros((3, 9, 2), dtype=np.float32)
        logits[0, :, 0], logits[0, :, 1] = 10000., 10001.
        result = portability.probabilities(logits)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result.sum(-1), 1.)
        np.testing.assert_allclose(result[0, :, 1], 1 / (1 + np.exp(-1)))
        for invalid in (np.empty((0, 9, 2)), np.zeros((3, 9)), logits + np.nan,
                        np.zeros((3, 9, 2), dtype=complex)):
            with self.assertRaises(ValueError):
                portability.probabilities(invalid)

    def test_loaded_inference_preserves_order_fp32_and_disables_gradients(self):
        model = OrderedModel().double().train()
        rows = [{"x": torch.tensor([3., -2., 1., 4.])}, {"x": torch.tensor([-5.])}]
        with patch.object(adaptation, "make_loader", return_value=rows) as mocked:
            actual = portability.predict_loaded_logits(model, None, ["sample"] * 5, manifest(), "cpu")
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[:, 0, 1], [3., -2., 1., 4., -5.])
        self.assertTrue(all(flags == (False, False, torch.float32) for flags in model.forward_flags))
        self.assertFalse(mocked.call_args.kwargs["shuffle"])

    def test_inference_rejects_row_count_or_source_batch_changes(self):
        model = OrderedModel()
        with patch.object(adaptation, "make_loader", return_value=[{"x": torch.tensor([1.])}]):
            with self.assertRaises(ValueError):
                portability.predict_loaded_logits(model, None, ["sample"] * 2, manifest(), "cpu")
        wrong = {**manifest(), "batch_size": 8}
        with self.assertRaises(ValueError):
            portability.predict_loaded_logits(model, None, ["sample"], wrong, "cpu")

    def test_neural_checkpoint_returns_raw_argmax(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            write(checkpoint, manifest())
            logits = np.zeros((2, 9, 2), dtype=np.float32)
            logits[0, :, 1] = 2
            logits[1, :, 0] = 3
            with patch.object(portability, "predict_logits", return_value=logits):
                actual = portability.predict_checkpoint(checkpoint, ["one", "two"])
            np.testing.assert_array_equal(actual, logits.argmax(-1))
            self.assertEqual(actual.dtype, np.int8)

    def test_fixed_mean_uses_exact_two_components_and_probabilities(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            checkpoint = root / "ensemble.ckpt"
            ensemble = {"format": portability.ENSEMBLE_FORMAT, "kind": "ensemble", "version": "ver23",
                        "labels": list(data.TARGET_SYMPTOMS), "combination": "equal_probability_mean",
                        "components": ["base.ckpt", "new.ckpt"]}
            write(checkpoint, ensemble)
            write(root / "base.ckpt", manifest())
            write(root / "new.ckpt", {**manifest(), "version": "ver23"})
            first = np.zeros((2, 9, 2), dtype=np.float32)
            second = first.copy()
            first[0, :, 1], first[1, :, 1] = 2., -1.
            second[0, :, 1], second[1, :, 1] = -3., 1.
            with patch.object(portability, "predict_logits", side_effect=[first, second]) as mocked:
                actual = portability.predict_checkpoint(checkpoint, ["one", "two"])
            expected = ((portability.probabilities(first) + portability.probabilities(second)) * .5).argmax(-1)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(mocked.call_count, 2)
            for changed in ({"component_weights": [.5, .5]}, {"weights": [.5, .5]},
                            {"components": ["base.ckpt"]}, {"components": ["base.ckpt", "base.ckpt"]},
                            {"components": ["../base.ckpt", "new.ckpt"]}):
                write(checkpoint, {**ensemble, **changed})
                with self.assertRaises(ValueError):
                    portability.predict_checkpoint(checkpoint, ["one"])
            write(checkpoint, ensemble)
            write(root / "new.ckpt", ensemble)
            with patch.object(portability, "predict_logits") as mocked:
                with self.assertRaises(ValueError):
                    portability.predict_checkpoint(checkpoint, ["one"])
                mocked.assert_not_called()

    def test_safe_paths_reject_windows_and_posix_escape_forms(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            for relative in ("../x", "inside/../x", "..\\x", "/x", "C:\\x", "C:x", "\\server\\x", ""):
                with self.assertRaises(ValueError, msg=relative):
                    portability._safe(root, relative)
            self.assertEqual(portability._safe(root, "final_model/assets"), root.resolve() / "final_model/assets")

    def make_source(self, root):
        original = root / "original"
        original.mkdir()
        (original / "weights.pt").write_bytes(b"verified-source-weights")
        (original / "assets" / "encoder").mkdir(parents=True)
        (original / "assets" / "tokenizer").mkdir(parents=True)
        (original / "assets" / "encoder" / "config.json").write_text("{}", encoding="utf-8")
        (original / "assets" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        checkpoint = original / "model.ckpt"
        write(checkpoint, manifest())
        return {"full_checkpoint": checkpoint}

    def test_baseline_export_is_byte_identical_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            with patch.object(portability.head, "load_model") as mocked:
                path, saved = portability.export(source, {"mode": "baseline", "candidate_id": "baseline"},
                    None, {}, root / "output", GenerousBudget(), "training_reference_not_jointly_improved")
                mocked.assert_not_called()
            self.assertEqual((path.parent / saved["weights"]).read_bytes(), b"verified-source-weights")
            self.assertEqual(saved["added_epochs"], 0)
            self.assertTrue(saved["baseline_reused"])
            self.assertEqual(saved["training_scope"], "all_training")
            self.assertEqual(saved, json.loads(path.read_text(encoding="utf-8")))

    def test_single_and_mean_export_restore_only_complete_trainable_state(self):
        for mode in ("single", "mean"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=HERE) as temporary:
                root = Path(temporary)
                source = self.make_source(root)
                model = new_model()
                frozen = adaptation.frozen_digest(model)
                trained = adaptation.trainable_state(model)
                trained["class_bias"] += .2
                selected = {"mode": mode, "candidate_id": "continue_gce_" + mode,
                            "recipe": "continue_gce", "epoch": 2}
                with patch.object(portability.head, "load_model", return_value=(model, None, manifest())):
                    path, saved = portability.export(source, selected, trained, {}, root / "output", GenerousBudget(), None)
                self.assertEqual(adaptation.frozen_digest(model), frozen)
                self.assertFalse(saved["baseline_reused"])
                self.assertEqual(saved["added_epochs"], 2)
                self.assertEqual(saved["selected_epochs"], 5)
                self.assertEqual(saved["trained_gce_epochs"], 2)
                self.assertEqual(saved["scheduler_horizon_epochs"], 2)
                self.assertEqual([entry["epoch"] for entry in saved["loss_schedule"]], [4, 5])
                state = torch.load(path.parent / "final_model/adapted_weights.pt", weights_only=True)
                self.assertTrue(torch.equal(state["class_bias"], trained["class_bias"]))
                self.assertTrue(saved["parameter_audit"]["frozen_state_unchanged"])
                if mode == "single":
                    self.assertEqual(saved["format"], portability.NEURAL_FORMAT)
                    self.assertFalse((path.parent / "final_model/baseline_weights.pt").exists())
                else:
                    self.assertEqual(saved["format"], portability.ENSEMBLE_FORMAT)
                    self.assertEqual(saved["components"], ["baseline_component.ckpt", "adapted_component.ckpt"])
                    for relative in saved["components"]:
                        component = json.loads((path.parent / relative).read_text(encoding="utf-8"))
                        self.assertTrue((path.parent / component["weights"]).is_file())
                        self.assertEqual(component["assets"], "final_model/assets")
                    self.assertEqual((path.parent / "final_model/baseline_weights.pt").read_bytes(), b"verified-source-weights")

    def test_missing_adaptation_requires_explicit_fallback_and_fit_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            selected = {"mode": "mean", "candidate_id": "continue_gce_mean", "recipe": "continue_gce", "epoch": 1}
            with self.assertRaises(ValueError):
                portability.export(source, selected, None, {}, root / "bad", GenerousBudget(), None)
            _, saved = portability.export(source, selected, None, {}, root / "fallback", GenerousBudget(), "refit_budget")
            self.assertTrue(saved["baseline_reused"])
            self.assertEqual(saved["requested_candidate"], "continue_gce_mean")
            write(source["full_checkpoint"], {**manifest(), "training_scope": "fit_only", "fitted_samples": 23360})
            with self.assertRaises(ValueError):
                portability.export(source, selected, None, {}, root / "bad_fit", GenerousBudget(), "refit_budget")

    def test_budget_failure_does_not_publish_final_manifest(self):
        class StopBudget(GenerousBudget):
            def check(self, stage, training=False):
                raise BudgetExceeded(stage)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            output = root / "output"
            with self.assertRaises(BudgetExceeded):
                portability.export(source, {"mode": "baseline", "candidate_id": "baseline"}, None,
                                   {}, output, StopBudget(), "budget")
            self.assertFalse((output / "mission3_ver23_model.ckpt").exists())


if __name__ == "__main__":
    unittest.main()
