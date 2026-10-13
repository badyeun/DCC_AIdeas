"""Protocol tests use synthetic records and bounded trainer adapters, never real score claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mission3_ver14 import data, runtime, runtime_suite as suite
else:
    from . import data, runtime, runtime_suite as suite


def settings(root):
    return {"version": "ver14", "backend": "neural", "model_id": "fixture/encoder", "revision": "a" * 40,
            "input_zip": "", "train_dir": str(root / "train"), "validation_dir": str(root / "validation"),
            "output_dir": str(root / "results"), "work_dir": str(root / "work"), "resume_run": "",
            "run_validation": True, "seed": 42, "holdout_size": .2, "batch_size": 4,
            "gradient_accumulation": 4, "stage3_batch_size": 8, "stage3_gradient_accumulation": 2,
            "max_epochs": 4, "patience": 2, "max_length": 512, "overlap": 64, "encoder_window_batch": 4,
            "learning_rate": 2e-5, "head_learning_rate": 1e-4, "nausea_weight_candidates": [1., 2., 3.],
            "refit_full_training": True, "allow_cpu": True, "log_every": 250}


def synthetic_data(root):
    for split, indices in (("train", range(90)), ("validation", range(100, 112))):
        directory = root / split
        directory.mkdir()
        for i in indices:
            gold = [int((i + 3 * label) % (label + 3) < 2) for label in range(9)]
            text = " ".join(f"feature_{j}_{v}" for j, v in enumerate(gold)) + f" sample_{i}"
            a = {"utterances": [{"text": text}],
                 "symptom": [label for label, flag in zip(data.TARGET_SYMPTOMS, gold) if flag]}
            (directory / f"fixture_{i:04d}.json").write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")


class SuiteTests(unittest.TestCase):
    def test_plans_isolate_weight_attention_and_contrastive_factors(self):
        config = suite.validate_settings(settings(Path("unused")))
        first = suite.stage2_plan(config)
        self.assertEqual([r["nausea_positive_weight"] for r in first], [1, 2, 3])
        self.assertTrue(all(not r["label_attention"] and r["contrastive_weight"] == 0 for r in first))
        last = suite.stage3_plan(config, 2.)
        self.assertEqual([(r["label_attention"], r["contrastive_weight"]) for r in last],
                         [(False, 0), (True, 0), (True, .1)])
        self.assertTrue(all(r["nausea_positive_weight"] == 2 for r in last))
        self.assertTrue(all(r["ce_views"] == 1 for r in first))
        self.assertTrue(all(r["ce_views"] == 2 for r in last))
        for r in first + last:
            self.assertFalse(r["run_validation"])
            self.assertFalse(r["refit_full_training"])
            self.assertEqual(r["batch_size"] * r["gradient_accumulation"], 16)
        for changed in ([2., 3.], [1., float("nan")], [True, 2.], [1., 1.], []):
            with self.assertRaises(ValueError):
                suite.validate_settings({**config, "nausea_weight_candidates": changed})
        with self.assertRaises(ValueError):
            suite.validate_settings({**config, "stage3_gradient_accumulation": 4})

    def test_selection_protects_both_metrics_and_ignores_official_results(self):
        def row(name, macro, nausea):
            return {"candidate_id": name, "metrics": {"macro_f1": macro,
                    "per_symptom": {suite.NAUSEA: {"f1": nausea}}}, "validation_macro_f1": 999}
        baseline = row(suite.BASELINE, .60, .20)
        lower_nausea = row("worse_nausea", .70, .19)
        lower_macro = row("worse_macro", .59, .90)
        better = row("better", .61, .30)
        self.assertIs(suite.select_candidate([baseline, lower_nausea, lower_macro]), baseline)
        self.assertIs(suite.select_candidate([baseline, lower_nausea, lower_macro, better]), better)
        self.assertIs(suite.select_candidate([baseline, row("tie", .60, .20)]), baseline)
        with self.assertRaises(ValueError):
            suite.select_candidate([row("invalid", .6, .2)])
        with self.assertRaises(ValueError):
            suite.select_candidate([baseline, row("invalid", float("nan"), .3)])

    def test_six_candidates_one_refit_final_only_validation_and_verified_resume(self):
        self.exercise_pipeline(interrupt=False)

    def test_interrupted_run_reuses_completed_candidates_and_keeps_partial_attempt(self):
        self.exercise_pipeline(interrupt=True)

    def exercise_pipeline(self, interrupt):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synthetic_data(root)
            config = settings(root)
            calls, events, refits = [], [], []
            real_load = data.load_dataset
            pending_failure = [interrupt]

            def load(directory, include_targets=False):
                if Path(directory) == root / "validation":
                    events.append(("validation_load", include_targets))
                    self.assertTrue(refits, "Validation was used before final model selection and refit")
                    if include_targets:
                        self.assertTrue((run_path[0] / "mission3.csv").is_file())
                return real_load(directory, include_targets=include_targets)

            def artifacts(output, weights):
                (output / "assets" / "encoder").mkdir(parents=True)
                (output / "assets" / "tokenizer").mkdir()
                (output / "assets" / "encoder" / "config.json").write_text("{}")
                (output / "assets" / "tokenizer" / "tokenizer.json").write_text("{}")
                (output / weights).write_bytes(b"synthetic trainer adapter artifact")

            def train(paths, texts, targets, fit, holdout, local, output, cfg):
                calls.append(cfg["candidate_id"])
                events.append(("candidate", cfg["candidate_id"]))
                self.assertFalse(cfg["run_validation"])
                self.assertFalse(cfg["refit_full_training"])
                if cfg["candidate_id"] == "stage2_ce_w3" and pending_failure[0]:
                    pending_failure[0] = False
                    raise RuntimeError("Simulated session interruption")
                pred = targets[holdout].copy()
                if cfg["nausea_positive_weight"] == 1:
                    pred[:4, 6] = 1 - pred[:4, 6]
                if cfg["nausea_positive_weight"] == 3:
                    pred[:5, 0] = 1 - pred[:5, 0]
                report = data.score_predictions(targets[holdout], pred)
                artifacts(output, "selection_weights.pt")
                np.savez_compressed(output / "holdout_predictions.npz", indices=holdout,
                                    targets=targets[holdout], predictions=pred)
                data.write_json(output / "selection_history.json", [{"epoch": 2, "holdout_macro_f1": report["macro_f1"]}])
                return {"kind": "neural", "weights": "selection_weights.pt", "assets": "assets", "selected_epochs": 2,
                        "label_attention": cfg["label_attention"], "contrastive": bool(cfg["contrastive_weight"])}, report

            def refit(texts, targets, cfg, epochs, output):
                events.append(("refit", cfg["candidate_id"]))
                refits.append(cfg)
                self.assertEqual(len(texts), 90)
                self.assertEqual(epochs, 2)
                self.assertEqual(cfg["nausea_positive_weight"], 2.)
                self.assertTrue(cfg["refit_full_training"])
                artifacts(output, "final_weights.pt")
                data.write_json(output / "final_history.json", [{"epoch": 2, "loss": .4}])
                return {"kind": "neural", "weights": "final_weights.pt", "assets": "assets", "selected_epochs": epochs,
                        "fitted_samples": len(targets), "label_attention": False, "contrastive": False}, np.zeros((12, 9), np.int8)

            def predict(checkpoint, texts):
                manifest = json.loads(Path(checkpoint).read_text(encoding="utf-8-sig"))
                self.assertTrue((Path(checkpoint).parent / manifest["weights"]).is_file())
                return np.zeros((len(texts), 9), np.int8)

            def inference(checkpoint, directory, output):
                run_path[0] = Path(output).parent
                paths, texts, _ = load(directory, include_targets=False)
                pred = predict(checkpoint, texts)
                data.write_predictions(paths, pred, output)
                return paths, pred

            run_path = [None]
            with patch.object(runtime, "train_neural", train), patch.object(runtime, "refit_selected", refit), \
                    patch.object(runtime, "predict_checkpoint", predict), patch.object(runtime, "inference", inference), \
                    patch.object(data, "load_dataset", load), patch.object(suite, "_save_environment"):
                sources = {"entry.py": "# immutable test source"}
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "Simulated"):
                        suite.run(config, sources)
                    partial = next((root / "results").iterdir())
                    self.assertFalse(refits)
                    self.assertFalse(any(e[0] == "validation_load" for e in events))
                    config["resume_run"] = str(partial)
                    completed = partial / "experiments" / "stage2_ce_w2" / "completed.json"
                    original = completed.read_bytes()
                    tampered = suite._read(completed)
                    tampered["settings"]["nausea_positive_weight"] = 99.
                    data.write_json(completed, tampered)
                    with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                        suite.run(config, sources)
                    completed.write_bytes(original)
                result = suite.run(config, sources)
                self.assertEqual(len(refits), 1)
                self.assertEqual(calls.count("stage2_ce_w1"), 1)
                self.assertEqual(calls.count("stage2_ce_w2"), 1)
                self.assertEqual(len(set(calls)), 6)
                if interrupt:
                    self.assertEqual(calls.count("stage2_ce_w3"), 2)
                    attempts = [p for p in (result / "experiments" / "stage2_ce_w3").iterdir() if p.is_dir()]
                    self.assertEqual(len(attempts), 2)
                self.assertEqual([e for e in events if e[0] == "validation_load"],
                                 [("validation_load", False), ("validation_load", True)])
                final = suite._read(result / "final_selection.json")
                self.assertEqual(final["candidate_id"], "stage2_ce_w2")
                plan = suite._read(result / "experiment_plan.json")
                self.assertTrue(all(c["nausea_positive_weight"] == 2 for c in plan["stage3"]))
                self.assertTrue((result / "audit_stage.json").is_file())
                self.assertEqual(suite._read(result / "runtime.json")["status"], "complete")
                old_calls, old_events = list(calls), list(events)
                config["resume_run"] = str(result)
                self.assertEqual(suite.run(config, sources), result)
                self.assertEqual(calls, old_calls)
                self.assertEqual(events, old_events)
                old_status = (result / "runtime.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "changed"):
                    suite.run(config, {"entry.py": "# changed source"})
                with self.assertRaisesRegex(ValueError, "changed"):
                    suite.run({**config, "max_epochs": 5}, sources)
                self.assertEqual((result / "runtime.json").read_bytes(), old_status)
                checkpoint = suite._read(result / "mission3_ver14_model.ckpt")
                (result / checkpoint["weights"]).write_bytes(b"damaged synthetic artifact")
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    suite.run(config, sources)


if __name__ == "__main__":
    unittest.main()
