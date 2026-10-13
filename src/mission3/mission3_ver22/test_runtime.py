"""Check selection isolation, portable export, budget fallback and resume integrity."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import data, evidence, runtime
from .budget import Budget, BudgetExceeded
from .build_notebook import make_notebook
from .test_evidence import OffsetTokenizer, TinyEncoder, synthetic_cache
from .neural import DialogueClassifier
from .test_notebook import assigned_literal


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = assigned_literal(make_notebook(), "ver22-settings", "SETTINGS")
        self.options.update(output_dir=str(self.root / "results"), work_dir=str(self.root / "work"),
                            train_dir="training", validation_dir="validation", allow_cpu=True)
        self.cache, self.targets = synthetic_cache(20)
        self.initial_state = evidence.initial_state(self.cache)
        self.texts = [f"synthetic example {index}" for index in range(20)]
        self.fit, self.hold = np.arange(16), np.arange(16, 20)
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        runtime._save_npz(self.source_root / "holdout_predictions.npz", indices=self.hold,
                          targets=self.targets[self.hold], predictions=self.cache["predictions"][self.hold])
        self.source = {"root": self.source_root, "fit_checkpoint": self.source_root / "fit.ckpt",
                       "full_checkpoint": self.source_root / "full.ckpt"}
        self.events = []

    def records(self, improved=True):
        baseline = self.cache["predictions"][self.hold].copy()
        candidate = baseline.copy()
        if improved:
            candidate[:, 6] = self.targets[self.hold, 6]
        def record(name, prediction, epoch):
            return {"candidate_id": name, "epoch": epoch,
                    "metrics": data.score_predictions(self.targets[self.hold], prediction),
                    "state": deepcopy(self.initial_state),
                    "history": [], "predictions": prediction, "stop_reason": "max_evidence_epochs"}
        return [record("baseline", baseline, 0), record("evidence_ce", candidate, 5)]

    def setup_run(self, stack, *, improved=True, budget_failure=False, baseline_mismatch=False, confirmation_budget_failure=False):
        def load(directory, include_targets):
            self.assertEqual(directory, "training", "Validation must not enter selection or refit")
            self.events.append("training")
            return [], self.texts, self.targets
        def features(checkpoint, texts, targets, settings, root, role, budget):
            self.events.append(role)
            self.assertEqual(checkpoint, self.source["fit_checkpoint" if role == "selection" else "full_checkpoint"])
            result = deepcopy(self.cache)
            result["_source_role"] = role
            if role == "selection" and baseline_mismatch:
                result["predictions"][self.hold[0], 6] ^= 1
            return result
        def refit(cache, targets, candidate_id, epochs, settings, budget, *, device):
            self.events.append("refit")
            self.assertEqual(device, "cpu")
            self.assertEqual(cache["_source_role"], "full_refit")
            self.assertEqual(epochs, 5)
            self.assertEqual(candidate_id, "evidence_ce")
            np.testing.assert_array_equal(targets, self.targets)
            if budget_failure:
                raise BudgetExceeded("synthetic incomplete full refit")
            return deepcopy(self.initial_state), [{"epoch": 5}]
        def export(source, selected, trained_evidence, settings, root, texts, validation_dir, budget, reason):
            self.events.append("export")
            self.assertEqual(validation_dir, "validation")
            if improved and not budget_failure and not confirmation_budget_failure:
                self.assertIsNotNone(trained_evidence)
                self.assertIsNone(reason)
            else:
                self.assertIsNone(trained_evidence)
            result = {"selected_candidate": selected["candidate_id"] if trained_evidence is not None else "baseline",
                      "baseline_reused": trained_evidence is None}
            data.write_json(root / "mission3_ver22_model.ckpt", result)
            return result
        def confirm(source, records, texts, targets, holdout, settings, budget):
            self.events.append("confirm")
            if confirmation_budget_failure:
                raise BudgetExceeded("synthetic incomplete actual confirmation")
            return runtime._choose(records), {"status": "confirmed", "validation_accessed": False}
        values = {"prepare_data": lambda *args: ("training", "validation"),
                  "_device": lambda *args: "cpu", "_environment": lambda: {"test": True},
                  "_release": lambda: None, "_cached_features": features,
                  "_export_and_evaluate": export, "_confirm_selected": confirm}
        for name, value in values.items():
            stack.enter_context(patch.object(runtime, name, side_effect=value))
        stack.enter_context(patch.object(data, "load_dataset", side_effect=load))
        stack.enter_context(patch.object(data, "make_split", return_value=(self.fit, self.hold)))
        stack.enter_context(patch.object(runtime.source_bundle, "prepare_source", return_value=self.source))
        stack.enter_context(patch.object(runtime.source_bundle, "verify_training", return_value={"verified": True}))
        candidates = stack.enter_context(patch.object(evidence, "evidence_candidates", return_value=self.records(improved)))
        stack.enter_context(patch.object(evidence, "refit_evidence", side_effect=refit))
        return candidates

    def test_accepted_recipe_refits_full_encoder_before_validation_and_resume_checks_hashes(self):
        with ExitStack() as stack:
            self.setup_run(stack)
            result = runtime.run(self.options, {"runtime.py": "synthetic source"})
            self.assertEqual(self.events, ["training", "selection", "confirm", "full_refit", "refit", "export"])
            self.assertEqual(runtime._read(result / "runtime.json")["status"], "complete")
            resumed = {**self.options, "resume_run": str(result)}
            self.events.clear()
            self.assertEqual(runtime.run(resumed, {"runtime.py": "synthetic source"}), result)
            self.assertEqual(self.events, ["training"])
            (result / "final_evidence.npz").write_bytes(b"changed artifact")
            with self.assertRaisesRegex(ValueError, "Saved artifact changed"):
                runtime.run(resumed, {"runtime.py": "synthetic source"})
            # A failed read-only resume must not overwrite previously completed status.
            self.assertEqual(runtime._read(result / "runtime.json")["status"], "complete")

    def test_nonimproving_candidate_preserves_full_baseline_without_refit(self):
        with ExitStack() as stack:
            self.setup_run(stack, improved=False)
            result = runtime.run(self.options, {"runtime.py": "synthetic source"})
        self.assertEqual(self.events, ["training", "selection", "confirm", "export"])
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_not_improved")
        self.assertTrue(status["baseline_reused"])
        self.assertFalse((result / "final_evidence.npz").exists())

    def test_incomplete_refit_discards_evidence_and_exports_baseline(self):
        with ExitStack() as stack:
            self.setup_run(stack, budget_failure=True)
            result = runtime.run(self.options, {"runtime.py": "synthetic source"})
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_budget_limited")
        self.assertEqual(status["exported_candidate"], "baseline")
        self.assertEqual(status["refit_skipped_reason"], "budget_before_complete_evidence_refit")
        self.assertFalse((result / "final_evidence.npz").exists())

    def test_source_prediction_mismatch_stops_before_any_candidate_training(self):
        with ExitStack() as stack:
            candidates = self.setup_run(stack, baseline_mismatch=True)
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                runtime.run(self.options, {"runtime.py": "synthetic source"})
            candidates.assert_not_called()
        result = next((self.root / "results").iterdir())
        self.assertFalse((result / "run_complete.json").exists())
        self.assertEqual(runtime._read(result / "runtime.json")["status"], "failed")

    def test_unevaluated_budget_limited_search_is_not_reported_as_no_improvement(self):
        with ExitStack() as stack:
            candidates = self.setup_run(stack, improved=False)
            only_baseline = self.records(False)[:1]
            only_baseline[0]["search_stop_reason"] = "budget"
            candidates.return_value = only_baseline
            result = runtime.run(self.options, {"runtime.py": "synthetic source"})
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_budget_limited")
        self.assertEqual(status["refit_skipped_reason"], "budget_before_complete_evidence_search")
        self.assertEqual(runtime._read(result / "evidence_artifacts.json")["active_paths"], [])

    def test_unconfirmed_candidate_cannot_refit_or_replace_baseline(self):
        with ExitStack() as stack:
            self.setup_run(stack, confirmation_budget_failure=True)
            result = runtime.run(self.options, {"runtime.py": "synthetic source"})
        self.assertEqual(self.events, ["training", "selection", "confirm", "export"])
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_budget_limited")
        self.assertTrue(status["baseline_reused"])
        self.assertEqual(runtime._read(result / "candidate_confirmation.json")["status"],
                         "budget_before_confirmation")
        self.assertFalse((result / "final_evidence.npz").exists())

    def test_actual_holdout_confirmation_reselects_after_cached_precision_disagreement(self):
        records = self.records(True)
        alternate = deepcopy(records[1])
        alternate["candidate_id"] = "evidence_ce_gce"
        records.append(alternate)
        actual = [records[0]["predictions"].copy(), alternate["predictions"].copy()]
        with patch.object(runtime.head, "load_model", return_value=(object(), None, {})) as load, \
             patch.object(evidence, "set_evidence_state", return_value={}) as update, \
             patch.object(runtime, "_predict_loaded", side_effect=actual) as predict, \
             patch.object(runtime, "_release"), \
             patch.object(data, "load_dataset", side_effect=AssertionError("No Validation read is allowed")):
            selected, audit = runtime._confirm_selected(self.source, records, self.texts, self.targets,
                self.hold, self.options, Budget(time.monotonic() + 300, 30))
        load.assert_called_once_with(self.source["fit_checkpoint"], "cpu")
        self.assertEqual(update.call_count, 2)
        self.assertEqual(predict.call_count, 2)
        self.assertTrue(all(call.kwargs["training"] for call in predict.call_args_list))
        self.assertEqual(predict.call_args.args[2], [self.texts[int(index)] for index in self.hold])
        self.assertEqual(selected["candidate_id"], "evidence_ce_gce")
        self.assertFalse(audit["validation_accessed"])
        self.assertEqual(audit["status"], "confirmed")
        self.assertGreater(audit["candidates"][0]["cached_vs_actual_nausea_disagreements"], 0)
        self.assertEqual(records[1]["metrics"], records[0]["metrics"])

    def test_other_label_changes_rejected_and_exact_tie_keeps_baseline(self):
        before = self.cache["predictions"]
        changed = before.copy()
        changed[0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "frozen symptom"):
            runtime._comparison(self.targets, before, changed)
        self.assertEqual(runtime._choose(self.records(False))["candidate_id"], "baseline")

    def test_cache_keeps_historical_holdout_batch_order_and_validates_reuse(self):
        checkpoint = self.source["fit_checkpoint"]
        (self.source_root / "fit.pt").write_bytes(b"synthetic fit weights")
        manifest = {"weights": "fit.pt", "training_scope": "fit_only", "fitted_samples": 16,
                    "encoder_window_batch": 4}
        data.write_json(checkpoint, manifest)
        destination = self.root / "cache_run"
        destination.mkdir()
        fit, held = np.arange(0, 20, 2), np.arange(1, 20, 2)
        runtime._save_npz(destination / "training_split.npz", fit=fit, holdout=held)
        seen = []
        def extract(model, tokenizer, texts, settings, device, budget):
            indices = np.asarray([self.texts.index(text) for text in texts])
            seen.append(indices)
            return evidence._subset(self.cache, indices)
        budget = Budget(time.monotonic() + 300, 30)
        with patch.object(runtime.head, "load_model", return_value=(object(), None, manifest)) as load, \
             patch.object(evidence, "extract_cache", side_effect=extract), patch.object(runtime, "_release"):
            cached = runtime._cached_features(checkpoint, self.texts, self.targets, self.options, destination, "selection", budget)
            self.assertEqual(len(seen), 2)
            np.testing.assert_array_equal(seen[0], fit)
            np.testing.assert_array_equal(seen[1], held)
            for key, value in self.cache.items():
                np.testing.assert_array_equal(cached[key], value)
            reused = runtime._cached_features(checkpoint, self.texts, self.targets, self.options, destination, "selection", budget)
            self.assertEqual(load.call_count, 1)
            np.testing.assert_array_equal(reused["predictions"], cached["predictions"])
            # Windows does not allow corrupting or deleting an open memory map.
            for value in list(cached.values()) + list(reused.values()):
                mapping = getattr(value, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            (Path(self.options["work_dir"]) / "feature_caches" / destination.name / "selection/global_features.npy").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "Saved artifact changed"):
                runtime._cached_features(checkpoint, self.texts, self.targets, self.options, destination, "selection", budget)

    def test_wrong_encoder_training_scope_is_rejected_before_loading(self):
        checkpoint = self.source["fit_checkpoint"]
        (self.source_root / "fit.pt").write_bytes(b"synthetic fit weights")
        data.write_json(checkpoint, {"weights": "fit.pt", "training_scope": "fit_only",
                                    "fitted_samples": 16, "encoder_window_batch": 4})
        with patch.object(runtime.head, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "training scope"):
                runtime._cached_features(checkpoint, self.texts, self.targets, self.options,
                    self.root / "wrong_scope", "full_refit", Budget(time.monotonic() + 300, 30))
            load.assert_not_called()

    def test_final_export_keeps_source_full_bytes_and_checks_reloaded_predictions(self):
        model = torch.nn.Linear(2, 2)
        checkpoint = self.source["full_checkpoint"]
        assets = self.source_root / "assets"
        assets.mkdir()
        (assets / "config.json").write_text("{}")
        (self.source_root / "full_weights.pt").write_bytes(b"original complete full weights")
        manifest = {"version": "ver19", "weights": "full_weights.pt", "assets": "assets",
                    "training_scope": "all_training", "fitted_samples": 20,
                    "selected_epochs": 3, "batch_size": 4, "encoder_window_batch": 4}
        destination = self.root / "export"
        destination.mkdir()
        settings = {**self.options, "run_validation": False}
        budget = Budget(time.monotonic() + 300, 30)
        with patch.object(runtime.head, "load_model", return_value=(model, None, manifest)), \
             patch.object(evidence, "extract_cache", return_value=self.cache) as extract, \
             patch.object(runtime, "predict_checkpoint", return_value=self.cache["predictions"].copy()) as predict, \
             patch.object(runtime, "_release"):
            result = runtime._export_and_evaluate(self.source, self.records(False)[0], None, settings,
                                                  destination, self.texts, "validation", budget, "not_improved")
        self.assertEqual(extract.call_args.kwargs, {"training": False})
        self.assertEqual((destination / "final_model/weights.pt").read_bytes(), b"original complete full weights")
        self.assertTrue(result["baseline_reused"])
        self.assertEqual(result["selected_epochs"], 3)
        self.assertEqual(result["evidence_epochs"], 0)
        self.assertTrue((destination / result["assets"] / "config.json").is_file())
        predict.assert_called_once()
        self.assertTrue(runtime._read(destination / "preservation_verification.json")["portable_reloaded_decisions_match"])

    def test_updated_export_roundtrips_actual_torch_forward_with_frozen_eight_labels(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(42)
        original_model = DialogueClassifier(TinyEncoder(), utterance_evidence=True, evidence_hidden=12).eval()
        original_model.encoder_window_batch = 4
        original_model.requires_grad_(False)
        original_model.class_weights[6].zero_()
        original_model.class_bias[6].copy_(torch.tensor([0., 1.]))
        assets = self.source_root / "assets"
        assets.mkdir()
        (assets / "config.json").write_text("{}")
        torch.save(original_model.state_dict(), self.source_root / "full.pt")
        manifest = {"version": "ver19", "weights": "full.pt", "assets": "assets", "max_length": 8,
                    "overlap": 2, "batch_size": 4, "encoder_window_batch": 4,
                    "training_scope": "all_training", "fitted_samples": 20, "selected_epochs": 3}
        data.write_json(self.source["full_checkpoint"], manifest)
        def load(checkpoint, device):
            stored = runtime._read(checkpoint)
            restored = deepcopy(original_model)
            restored.load_state_dict(torch.load(Path(checkpoint).parent / stored["weights"], weights_only=True))
            return restored, OffsetTokenizer(), stored
        destination = self.root / "updated_export"
        destination.mkdir()
        texts = ["token_1 token_2\ntoken_3", "token_4"] * 10
        state = {name: value.detach().numpy().copy() for name, value in original_model.evidence.state_dict().items()}
        state["output.bias"] += np.array([2, -2], dtype=np.float32)
        with patch.object(runtime.head, "load_model", side_effect=load):
            result = runtime._export_and_evaluate(self.source, self.records(True)[1], state,
                {**self.options, "run_validation": False}, destination, texts, "validation",
                Budget(time.monotonic() + 300, 30), None)
        self.assertFalse(result["baseline_reused"])
        self.assertEqual(result["evidence_epochs"], 5)
        self.assertEqual(result["evidence_fitted_samples"], 20)
        verification = runtime._read(destination / "preservation_verification.json")
        self.assertTrue(verification["portable_reloaded_decisions_match"])
        self.assertTrue(verification["other_eight_decisions_match_original_full_model"])
        self.assertTrue(verification["parameter_audit"]["all_non_evidence_tensors_identical"])
        self.assertEqual(verification["parameter_audit"]["learned_parameter_count"], sum(value.size for value in state.values()))

        # Validation is descriptive: deliberately worse final labels cannot
        # revert a recipe that the Training-only selection already accepted.
        baseline = evidence.extract_cache(original_model, OffsetTokenizer(), texts, manifest, "cpu")
        self.assertTrue(np.all(baseline["predictions"][:, 6] == 1))
        validation_destination = self.root / "descriptive_validation"
        validation_destination.mkdir()
        validation_paths = [Path(f"synthetic_{index}.json") for index in range(len(texts))]
        with patch.object(runtime.head, "load_model", side_effect=load), \
             patch.object(data, "load_dataset", return_value=(validation_paths, texts, baseline["predictions"])):
            final = runtime._export_and_evaluate(self.source, self.records(True)[1], state,
                {**self.options, "run_validation": True}, validation_destination, texts, "validation",
                Budget(time.monotonic() + 300, 30), None)
        comparison = runtime._read(validation_destination / "validation_comparison.json")
        self.assertLess(comparison["nausea_f1_delta"], 0)
        self.assertFalse(final["baseline_reused"])
        self.assertEqual(final["selected_candidate"], "evidence_ce")


if __name__ == "__main__":
    unittest.main()
