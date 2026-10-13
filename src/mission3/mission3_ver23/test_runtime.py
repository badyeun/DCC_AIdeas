"""Behavioral checks for Training isolation, model publication and safe resume."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from . import data, runtime
from .budget import Budget, BudgetExceeded
from .build_notebook import default_settings


class ArtifactSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_completed_artifact_hash_rejects_missing_and_modified_content(self):
        target = self.root / "model.bin"
        target.write_bytes(b"completed fixture model")
        original = {"model.bin": runtime._hash(target)}
        runtime._verify_hashes(self.root, original)
        target.write_bytes(b"modified fixture model")
        with self.assertRaisesRegex(ValueError, "Saved artifact changed"):
            runtime._verify_hashes(self.root, original)
        with self.assertRaisesRegex(ValueError, "Saved artifact changed"):
            runtime._verify_hashes(self.root, {"missing.bin": original["model.bin"]})
        with self.assertRaisesRegex(ValueError, "Missing completed-artifact hashes"):
            runtime._verify_hashes(self.root, {})

    def test_manifest_artifact_paths_cannot_escape_the_result_directory(self):
        for relative in ("..", "../outside.bin", ".", str(self.root.parent / "outside.bin")):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "escapes"):
                runtime._safe(self.root, relative)
        self.assertEqual(runtime._safe(self.root, "assets/model.bin"), self.root / "assets/model.bin")

    def test_dataset_identity_changes_on_order_text_or_target_change(self):
        texts = ["synthetic first", "synthetic second"]
        labels = np.zeros((2, 9), dtype=np.int8)
        original = runtime.dataset_digest(texts, labels)
        self.assertNotEqual(runtime.dataset_digest(texts[::-1], labels), original)
        self.assertNotEqual(runtime.dataset_digest(["changed", texts[1]], labels), original)
        labels[1, 6] = 1
        self.assertNotEqual(runtime.dataset_digest(texts, labels), original)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.options = {**default_settings(), "output_dir": str(self.root / "results"),
                        "work_dir": str(self.root / "work"), "train_dir": "training",
                        "validation_dir": "validation", "allow_cpu": True}
        self.sources = {"runtime.py": "synthetic runtime source", "requirements.txt": "synthetic requirements"}
        self.texts = [f"synthetic training {index}" for index in range(20)]
        self.targets = np.tile((np.arange(20) % 2)[:, None], (1, 9)).astype(np.int8)
        self.fit, self.holdout = np.arange(16), np.arange(16, 20)
        self.baseline = self.targets[self.holdout].copy()
        self.baseline[-1, :] = 0
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        self.source = {"root": self.source_root, "fit_checkpoint": self.source_root / "fit.ckpt",
                       "full_checkpoint": self.source_root / "full.ckpt"}
        runtime._save_npz(self.source_root / "holdout_predictions.npz", indices=self.holdout,
                          targets=self.targets[self.holdout], predictions=self.baseline)
        self.events = []

    def record(self, name, predictions, *, epoch=1, mode="single"):
        logits = np.stack([1 - predictions, predictions], -1).astype(np.float32)
        if mode == "mean":
            # The synthetic adapted member is more confident than the baseline,
            # so its changed classes also win the actual equal-probability mean.
            logits *= 2
        return {"candidate_id": name, "recipe": "baseline" if name == "baseline" else "continue_gce",
                "mode": "baseline" if name == "baseline" else mode, "epoch": 0 if name == "baseline" else epoch,
                "metrics": data.score_predictions(self.targets[self.holdout], predictions),
                "predictions": predictions.copy(), "logits": logits, "history": [],
                "refit_seconds": 1., "stop_reason": "fixture_completed_epoch"}

    def records(self, improved=True, *, nausea_tie=False, mode="single"):
        candidate = self.targets[self.holdout].copy() if improved else self.baseline.copy()
        if nausea_tie:
            candidate[:, 6] = self.baseline[:, 6]
        name = "continue_gce_mean" if mode == "mean" else "continue_gce"
        return [self.record("baseline", self.baseline), self.record(name, candidate, mode=mode)]

    def setup_run(self, stack, *, improved=True, nausea_tie=False, mode="single", limited=False,
                  refit_failure=False, export_failure=False, real_search=False):
        def load(directory, include_targets):
            self.assertTrue(include_targets)
            self.assertEqual(directory, "training", "Validation must enter only the final export stage")
            self.events.append("training")
            return [], self.texts, self.targets

        def search(source, texts, targets, fit, holdout, settings, root, budget):
            self.events.append("search")
            self.assertEqual(source, self.source)
            np.testing.assert_array_equal(targets, self.targets)
            np.testing.assert_array_equal(fit, self.fit)
            np.testing.assert_array_equal(holdout, self.holdout)
            return self.records(improved, nausea_tie=nausea_tie, mode=mode), limited

        def refit(source, selected, texts, targets, settings, root, budget):
            self.events.append("refit")
            self.assertEqual(source["full_checkpoint"], self.source["full_checkpoint"])
            self.assertEqual(texts, self.texts)
            np.testing.assert_array_equal(targets, self.targets)
            self.assertEqual(selected["epoch"], 1)
            self.assertEqual(selected["mode"], mode)
            if refit_failure:
                data.write_json(root / "final_history.json", [{"epoch": 1, "fixture_incomplete": True}])
                raise BudgetExceeded("synthetic incomplete full refit")
            return {"fixture_trained_parameter": 1}

        def export(source, selected, trained_state, settings, root, texts, validation_dir, budget, reason):
            self.events.append("validation_and_export")
            self.assertEqual(validation_dir, "validation")
            if export_failure:
                raise ValueError("synthetic portable reload mismatch")
            accepted = improved and not refit_failure
            self.assertEqual(trained_state is not None, accepted)
            self.assertEqual(reason is None, accepted)
            manifest = {"selected_candidate": selected["candidate_id"] if accepted else "baseline",
                        "baseline_reused": not accepted, "added_epochs": selected["epoch"] if accepted else 0,
                        "training_scope": "all_training", "fitted_samples": len(self.texts)}
            data.write_json(root / "mission3_ver23_model.ckpt", manifest)
            (root / "mission3.csv").write_text("label file name,symptom\n", encoding="utf-8")
            return manifest

        replacements = {"prepare_data": lambda *args: ("training", "validation"),
                        "_device": lambda *args: "cpu", "_environment": lambda: {"fixture": True},
                        "_release": lambda: None, "_refit": refit, "_export_and_evaluate": export}
        if not real_search:
            replacements["_candidate_search"] = search
        handles = {name: stack.enter_context(patch.object(runtime, name, side_effect=value))
                   for name, value in replacements.items()}
        stack.enter_context(patch.object(data, "load_dataset", side_effect=load))
        stack.enter_context(patch.object(data, "make_split", return_value=(self.fit, self.holdout)))
        handles["prepare_source"] = stack.enter_context(patch.object(runtime.source_bundle, "prepare_source", return_value=self.source))
        stack.enter_context(patch.object(runtime.source_bundle, "verify_training", return_value={"verified": True}))
        return handles

    def test_macro_improvement_with_equal_nausea_accepts_matched_full_refit_before_validation(self):
        with ExitStack() as stack:
            self.setup_run(stack, nausea_tie=True, mode="mean")
            result = runtime.run(self.options, self.sources)
        self.assertEqual(self.events, ["training", "search", "refit", "validation_and_export"])
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["exported_candidate"], "continue_gce_mean")
        self.assertFalse(status["baseline_reused"])
        comparison = runtime._read(result / "training_reference_comparison.json")
        self.assertTrue(comparison["acceptance_gate_passed"])
        self.assertEqual(comparison["nausea_f1_delta"], 0.)
        self.assertFalse(comparison["both_macro_and_nausea_improved"])
        self.assertFalse(comparison["other_eight_predictions_unchanged"])
        self.assertFalse(runtime._read(result / "final_selection.json")["validation_accessed"])
        completion = runtime._read(result / "run_complete.json")
        self.assertIn("mission3.csv", completion["artifact_hashes"])
        runtime._verify_hashes(result, completion["artifact_hashes"])

    def test_nonimproving_candidate_keeps_baseline_without_refit(self):
        with ExitStack() as stack:
            handles = self.setup_run(stack, improved=False)
            result = runtime.run(self.options, self.sources)
            handles["_refit"].assert_not_called()
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_not_improved")
        self.assertEqual(status["exported_candidate"], "baseline")
        self.assertTrue(status["baseline_reused"])
        self.assertEqual(status["added_epochs"], 0)

    def test_incomplete_full_refit_publishes_only_original_full_baseline(self):
        with ExitStack() as stack:
            self.setup_run(stack, refit_failure=True)
            result = runtime.run(self.options, self.sources)
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_budget_limited")
        self.assertEqual(status["selected_candidate"], "continue_gce")
        self.assertEqual(status["exported_candidate"], "baseline")
        self.assertEqual(status["refit_skipped_reason"], "budget_before_complete_refit")
        self.assertEqual(status["added_epochs"], 0)
        self.assertTrue(runtime._read(result / "mission3_ver23_model.ckpt")["baseline_reused"])

    def test_budget_limited_search_is_not_presented_as_completed_negative_result(self):
        with ExitStack() as stack:
            self.setup_run(stack, improved=False, limited=True)
            result = runtime.run(self.options, self.sources)
        status = runtime._read(result / "runtime.json")
        self.assertEqual(status["status"], "complete_budget_limited")
        self.assertTrue(status["search_budget_limited"])
        self.assertEqual(status["refit_skipped_reason"], "budget_before_complete_search")

    def test_failed_export_cannot_write_completion_marker(self):
        with ExitStack() as stack:
            self.setup_run(stack, export_failure=True)
            with self.assertRaisesRegex(ValueError, "portable reload mismatch"):
                runtime.run(self.options, self.sources)
        result = next((self.root / "results").iterdir())
        self.assertFalse((result / "run_complete.json").exists())
        self.assertEqual(runtime._read(result / "runtime.json")["status"], "failed")

    def test_saved_baseline_disagreement_stops_before_any_candidate_load_or_training(self):
        with ExitStack() as stack:
            self.setup_run(stack, real_search=True)
            logits = self.record("baseline", self.baseline)["logits"]
            logits[0, 6, :] = logits[0, 6, ::-1]
            stack.enter_context(patch.object(runtime.portability, "predict_logits", return_value=logits))
            load = stack.enter_context(patch.object(runtime.head, "load_model"))
            train = stack.enter_context(patch.object(runtime.adaptation, "train_epoch"))
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                runtime.run(self.options, self.sources)
            load.assert_not_called()
            train.assert_not_called()
        result = next((self.root / "results").iterdir())
        self.assertFalse((result / "run_complete.json").exists())

    def test_completed_resume_verifies_identity_and_hashes_without_writes_or_learning(self):
        with ExitStack() as stack:
            handles = self.setup_run(stack)
            result = runtime.run(self.options, self.sources)
            before = {p.relative_to(result): p.read_bytes() for p in result.rglob("*") if p.is_file()}
            self.events.clear()
            for handle in handles.values():
                handle.reset_mock()
            resumed = {**self.options, "resume_run": str(result)}
            self.assertEqual(runtime.run(resumed, self.sources), result)
            self.assertEqual(self.events, ["training"])
            handles["_candidate_search"].assert_not_called()
            handles["_refit"].assert_not_called()
            handles["_export_and_evaluate"].assert_not_called()
            handles["prepare_source"].assert_not_called()
            self.assertEqual(before, {p.relative_to(result): p.read_bytes() for p in result.rglob("*") if p.is_file()})
            (result / "mission3_ver23_model.ckpt").write_text("changed", encoding="utf-8")
            original_status = (result / "runtime.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "Saved artifact changed"):
                runtime.run(resumed, self.sources)
            self.assertEqual((result / "runtime.json").read_bytes(), original_status)
            with self.assertRaisesRegex(ValueError, "Resume data, settings or source changed"):
                runtime.run(resumed, {"runtime.py": "different source"})
            self.assertEqual((result / "runtime.json").read_bytes(), original_status)

    def test_incomplete_resume_is_rejected_without_modifying_existing_files(self):
        incomplete = self.root / "results" / "incomplete"
        incomplete.mkdir(parents=True)
        sentinel = incomplete / "runtime.json"
        sentinel.write_bytes(b'{"status":"failed"}')
        with patch.object(runtime, "prepare_data") as prepare:
            with self.assertRaisesRegex(ValueError, "Only a completed run can be resumed"):
                runtime.run({**self.options, "resume_run": str(incomplete)}, self.sources)
            prepare.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b'{"status":"failed"}')

    def test_exact_ties_keep_baseline_or_earlier_eligible_candidate(self):
        equal = self.records(False)
        self.assertIs(runtime._choose(equal), equal[0])
        records = self.records(True)
        later = deepcopy(records[1])
        later["candidate_id"] = "continue_rdrop"
        self.assertIs(runtime._choose([*records, later]), records[1])

    def test_higher_macro_cannot_override_nausea_regression_or_hide_earlier_eligible_epoch(self):
        baseline = self.record("baseline", self.baseline)
        early = self.record("continue_gce", self.baseline.copy(), epoch=1)
        early["predictions"][-1, 0] = 1
        early["metrics"] = data.score_predictions(self.targets[self.holdout], early["predictions"])
        later_prediction = self.targets[self.holdout].copy()
        later_prediction[:, 6] = 0
        later = self.record("continue_gce", later_prediction, epoch=2)
        self.assertGreater(later["metrics"]["macro_f1"], early["metrics"]["macro_f1"])
        self.assertIs(runtime._choose([baseline, early, later]), early)

    def test_actual_search_uses_only_fit_rows_and_discards_incomplete_epoch(self):
        root = self.root / "search"
        root.mkdir()
        model, tokenizer = object(), object()
        baseline_logits = self.record("baseline", self.baseline)["logits"]
        improved_logits = self.record("candidate", self.targets[self.holdout])["logits"]
        seen_fit_texts = []

        def loader_factory(loaded_model, loaded_tokenizer, texts, targets, settings, *, shuffle):
            self.assertIs(loaded_model, model)
            self.assertIs(loaded_tokenizer, tokenizer)
            self.assertTrue(shuffle)
            seen_fit_texts.extend(texts)
            np.testing.assert_array_equal(targets, self.targets[self.fit])
            return [object()] * 4

        def train(*args, **kwargs):
            epoch = args[6]
            if epoch == 2:
                raise BudgetExceeded("synthetic partial epoch")
            return {"epoch": epoch, "samples": len(self.fit), "training_seconds": .1}

        with (patch.object(runtime, "_device", return_value="cpu"), patch.object(runtime, "_release"),
              patch.object(runtime.portability, "predict_logits", return_value=baseline_logits) as baseline_predict,
              patch.object(runtime.portability, "predict_loaded_logits", return_value=improved_logits) as held_predict,
              patch.object(runtime.head, "load_model", return_value=(model, tokenizer, {"training_scope": "fit_only"})) as load,
              patch.object(runtime.adaptation, "configure_adaptation", return_value={}),
              patch.object(runtime.adaptation, "frozen_digest", return_value="unchanged"),
              patch.object(runtime.adaptation, "make_loader", side_effect=loader_factory),
              patch.object(runtime.adaptation, "make_optimizer", return_value=object()),
              patch.object(runtime.adaptation, "train_epoch", side_effect=train)):
            records, limited = runtime._candidate_search(self.source, self.texts, self.targets, self.fit,
                self.holdout, self.options, root, Budget(time.monotonic() + 300, 20))
        self.assertTrue(limited)
        self.assertEqual([r["epoch"] for r in records], [0, 1, 1])
        self.assertEqual([r["mode"] for r in records], ["baseline", "single", "mean"])
        self.assertEqual(seen_fit_texts, self.texts[:16])
        self.assertFalse(set(seen_fit_texts) & set(self.texts[16:]))
        load.assert_called_once_with(self.source["fit_checkpoint"], "cpu")
        self.assertEqual(baseline_predict.call_args.args[1], self.texts[16:])
        self.assertEqual(held_predict.call_args.args[2], self.texts[16:])
        self.assertTrue(held_predict.call_args.kwargs["training"])
        self.assertTrue(all(r["stop_reason"] == "budget" for r in records[1:]))
        self.assertFalse(list(root.rglob("epoch_2_holdout.npz")))
        runtime._save_records(root, records, self.holdout, self.targets)
        with np.load(root / "experiments/continue_gce_mean/holdout_predictions.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["decision_rule"].item(), "equal_probability_mean")
            np.testing.assert_array_equal(saved["predictions"], saved["probabilities"].argmax(-1))
            self.assertFalse(np.array_equal(saved["component_logits"].argmax(-1), saved["predictions"]))
            expected = (runtime.portability.probabilities(baseline_logits)
                        + runtime.portability.probabilities(improved_logits)) / 2
            np.testing.assert_allclose(saved["probabilities"], expected)
            self.assertNotIn("logits", saved.files)

    def test_actual_refit_restarts_from_full_source_and_consumes_all_training_rows(self):
        root = self.root / "refit"
        root.mkdir()
        selected = self.records(True)[1]
        selected["epoch"] = 2
        model, tokenizer = object(), object()
        loader = [object()] * 5
        expected_state = {"fixture": np.asarray([1.], dtype=np.float32)}
        epochs = []

        def train(*args, **kwargs):
            epochs.append(args[6])
            self.assertIs(args[0], model)
            self.assertIs(args[1], loader)
            self.assertTrue(kwargs["pilot_context"]["refit"])
            return {"epoch": args[6], "samples": len(self.texts), "training_seconds": .1}

        with (patch.object(runtime, "_device", return_value="cpu"), patch.object(runtime, "_release"),
              patch.object(runtime.head, "load_model", return_value=(model, tokenizer,
                  {"training_scope": "all_training", "fitted_samples": len(self.texts)})) as load,
              patch.object(runtime.adaptation, "configure_adaptation", return_value={}),
              patch.object(runtime.adaptation, "frozen_digest", return_value="unchanged"),
              patch.object(runtime.adaptation, "make_loader", return_value=loader) as make_loader,
              patch.object(runtime.adaptation, "make_optimizer", return_value=object()),
              patch.object(runtime.adaptation, "train_epoch", side_effect=train),
              patch.object(runtime.adaptation, "trainable_state", return_value=expected_state)):
            state = runtime._refit(self.source, selected, self.texts, self.targets, self.options, root,
                                   Budget(time.monotonic() + 300, 20))
        self.assertIs(state, expected_state)
        self.assertEqual(epochs, [1, 2])
        load.assert_called_once_with(self.source["full_checkpoint"], "cpu")
        self.assertEqual(make_loader.call_args.args[2], self.texts)
        np.testing.assert_array_equal(make_loader.call_args.args[3], self.targets)
        audit = runtime._read(root / "final_parameter_audit.json")
        self.assertTrue(audit["frozen_state_unchanged"])
        self.assertEqual(audit["training_scope"], "all_training")
        self.assertEqual(audit["added_epochs"], 2)

    def test_final_export_writes_submission_csv_from_reloaded_model(self):
        root = self.root / "export"
        root.mkdir()
        paths = [Path(f"synthetic_{index}.json") for index in range(4)]
        predictions = self.baseline.copy()
        logits = self.record("baseline", predictions)["logits"]
        manifest = {"selected_candidate": "baseline", "baseline_reused": True}
        with (patch.object(runtime, "_device", return_value="cpu"),
              patch.object(data, "load_dataset", return_value=(paths, self.texts[-4:], self.targets[-4:])) as load,
              patch.object(runtime.portability, "predict_logits", return_value=logits),
              patch.object(runtime.portability, "export", return_value=(root / "model.ckpt", manifest)),
              patch.object(runtime.portability, "predict_checkpoint", return_value=predictions)):
            saved = runtime._export_and_evaluate(self.source, self.records(False)[0], None, self.options,
                root, self.texts, "validation", Budget(time.monotonic() + 300, 20), "not_improved")
        self.assertEqual(saved, manifest)
        load.assert_called_once_with("validation", include_targets=True)
        self.assertTrue((root / "mission3.csv").is_file())
        self.assertEqual((root / "mission3.csv").read_bytes(), (root / "validation_predictions.csv").read_bytes())
        self.assertTrue(runtime._read(root / "export_verification.json")["all_nine_predictions_match"])
        self.assertEqual(runtime._read(root / "validation_metrics.json"),
                         data.score_predictions(self.targets[-4:], predictions))


if __name__ == "__main__":
    unittest.main()
