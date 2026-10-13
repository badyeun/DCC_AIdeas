"""Exercise baseline integrity, alignment, and fixed-control comparison gates."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import analysis, baseline, build_notebook, data

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix=".baseline_test_", dir=HERE) as temporary:
        root = Path(temporary)
        previous = root / "ver26"
        previous.mkdir()
        rng = np.random.default_rng(13)
        texts = ["synthetic call " + str(i) for i in range(30)]
        targets = rng.integers(0, 2, size=(30, 9), dtype=np.int8)
        split = {"fit": np.arange(18), "development": np.arange(18, 24), "confirmation": np.arange(24, 30)}
        subset = np.arange(9)
        settings = {**build_notebook.default_settings(), "pilot_samples": 9}
        original = {**settings, "lora_lr": 1e-4, "head_lr": 3e-4, "version": "ver26"}
        truth = targets[split["development"]]
        # Both classes occur for every symptom in this deterministic fixture.
        logits = {kind: rng.normal(size=(len(truth), 9)) for kind in ("qwen", "control")}
        logits["ensemble"] = analysis.ensemble_logits(logits["control"], logits["qwen"])
        scores = {kind: analysis.metrics(truth, value) for kind, value in logits.items()}
        data_digest = data.dataset_digest(texts, targets)
        selection = {"selected_candidate": "ensemble", "training_dataset_digest": data_digest,
                     "selected_epochs": {"qwen": 2, "control": 3}}
        files = {
            "settings.json": original, "effective_settings.json": original,
            "training_data.json": {"digest": data_digest},
            "development_comparison.json": {"metrics": scores},
            "model_specs.json": baseline._read_json(HERE / "model_specs.json"),
            "screening_selection.json": selection, "screening_recipe.json": {"selection": selection},
            "screening_portable_verification.json": {kind: {"samples": 16, "decisions_equal": True}
                                                      for kind in ("qwen", "control")},
        }
        for kind in ("qwen", "control"):
            best = selection["selected_epochs"][kind]
            files[kind + "_history.json"] = [
                {"selected_epochs": i, "macro_f1": scores[kind]["macro_f1"] - (.01 if i != best else 0),
                 "metrics": scores[kind]} for i in (1, 2, 3)]
        for name, value in files.items():
            data.write_json(previous / name, value)
        np.savez_compressed(previous / "split.npz", **split)
        np.savez_compressed(previous / "subset.npz", indices=subset)
        np.savez_compressed(previous / "development_predictions.npz", indices=split["development"], targets=truth,
                            **{kind + "_logits": value for kind, value in logits.items()})
        hashes = {p.name: digest(p) for p in previous.iterdir()}
        marker = {"version": "ver26", "status": "screening_complete", "training_scope": "training_subset",
                  "submission_ready": False, "confirmation_evaluated": False, "validation_evaluated": False,
                  "hashes": hashes}
        data.write_json(previous / "screening_complete.json", marker)
        pin = {"run_id": "fixture_ver26", "completion_sha256": digest(previous / "screening_complete.json"),
               "required_files": hashes, "training_rows": len(texts), "subset_rows": len(subset),
               "development_rows": len(truth), "training_dataset_digest": data_digest,
               "learning_rate_multiplier": .3, "promising_macro_gain": .005}
        output = root / "new_result"
        output.mkdir()
        data.write_json(output / "effective_settings.json", settings)
        with mock.patch.object(baseline, "_pin", return_value=pin):
            yield {"previous": previous, "output": output, "texts": texts, "targets": targets,
                   "split": split, "subset": subset, "settings": settings, "logits": logits,
                   "scores": scores, "records": {kind: {"selected_epochs": 2} for kind in ("qwen", "control")},
                   "pin": pin}


def load(item, **overrides):
    args = {"path": item["previous"], "texts": item["texts"], "targets": item["targets"],
            "split": item["split"], "subset": item["subset"], "settings": item["settings"],
            "output_dir": item["output"]}
    return baseline.load_reference(**{**args, **overrides})


class BaselineTests(unittest.TestCase):
    def test_verified_reference_copies_only_small_artifacts_and_preserves_source(self):
        with fixture() as item:
            before = {p.name: digest(p) for p in item["previous"].iterdir()}
            reference = load(item)
            self.assertEqual(reference["metrics"], item["scores"])
            self.assertEqual(before, {p.name: digest(p) for p in item["previous"].iterdir()})
            self.assertEqual(before, {p.name: digest(p) for p in (item["output"] / "baseline_ver26").iterdir()})
            self.assertFalse(reference["evidence"]["weight_files_required_or_copied"])

    def test_missing_reference_and_tampered_marker_or_npz_fail_before_copy(self):
        for failure in ("missing", "marker", "npz"):
            with self.subTest(failure=failure), fixture() as item:
                if failure == "missing":
                    path = item["previous"] / "absent"
                else:
                    path = item["previous"]
                    target = path / ("screening_complete.json" if failure == "marker" else "development_predictions.npz")
                    target.write_bytes(target.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "baseline"):
                    load(item, path=path)
                self.assertFalse((item["output"] / "baseline_ver26").exists())

    def test_label_split_subset_lr_and_recipe_mismatches_are_rejected(self):
        for failure in ("labels", "split", "subset", "lr", "recipe"):
            with self.subTest(failure=failure), fixture() as item:
                changes = {}
                if failure == "labels":
                    targets = item["targets"].copy()
                    targets[18, 0] = 1 - targets[18, 0]
                    changes["targets"] = targets
                elif failure == "split":
                    changes["split"] = {**item["split"], "development": item["split"]["development"][::-1]}
                elif failure == "subset":
                    changes["subset"] = item["subset"][::-1]
                else:
                    changes["settings"] = {**item["settings"], **({"lora_lr": 1e-4} if failure == "lr" else {"seed": 1})}
                with self.assertRaisesRegex(ValueError, "baseline"):
                    load(item, **changes)
                self.assertFalse((item["output"] / "baseline_ver26").exists())

    def test_identical_qwen_has_no_gain_even_if_fresh_control_is_better(self):
        with fixture() as item:
            reference = load(item)
            logits = {"qwen": item["logits"]["qwen"], "control": np.where(reference["targets"], 5., -5.)}
            scores = {kind: analysis.metrics(reference["targets"], value) for kind, value in logits.items()}
            scores["ensemble"] = analysis.metrics(reference["targets"], analysis.ensemble_logits(logits["control"], logits["qwen"]))
            gate = baseline.compare_reference(reference, logits, scores, item["records"], item["output"])
            self.assertEqual(gate["status"], "no_clear_gain")
            self.assertAlmostEqual(gate["macro_f1_delta"], 0.)
            self.assertFalse(gate["full_training_authorized"])
            stored = baseline._read_json(item["output"] / "historical_comparison.json")
            self.assertGreater(stored["comparisons"]["fresh_ensemble_minus_ver26_ensemble"]["macro_f1_delta"], 0)

    def test_large_matched_improvement_is_only_a_screening_gate(self):
        with fixture() as item:
            reference = load(item)
            logits = {"qwen": np.where(reference["targets"], 20., -20.), "control": item["logits"]["control"]}
            scores = {kind: analysis.metrics(reference["targets"], value) for kind, value in logits.items()}
            scores["ensemble"] = analysis.metrics(reference["targets"], analysis.ensemble_logits(logits["control"], logits["qwen"]))
            gate = baseline.compare_reference(reference, logits, scores, item["records"], item["output"])
            self.assertEqual(gate["status"], "promising_for_larger_fit")
            self.assertFalse(gate["automatic_full_training_recommended"])
            self.assertFalse(gate["submission_ready"])
            self.assertIsNone(gate["target_0_7_achieved_on_official_validation"])
            with np.load(item["output"] / "historical_comparison_predictions.npz", allow_pickle=False) as packed:
                np.testing.assert_array_equal(packed["indices"], reference["indices"])
                np.testing.assert_array_equal(packed["same_control_ensemble_logits"] >= 0, reference["targets"])

    def test_metric_mismatch_fails_without_publishing_gate(self):
        with fixture() as item:
            reference = load(item)
            scores = {**item["scores"], "qwen": {**item["scores"]["qwen"], "macro_f1": 1.}}
            with self.assertRaisesRegex(ValueError, "new metrics"):
                baseline.compare_reference(reference, item["logits"], scores, item["records"], item["output"])
            self.assertFalse((item["output"] / "baseline_gate.json").exists())


if __name__ == "__main__":
    unittest.main()
