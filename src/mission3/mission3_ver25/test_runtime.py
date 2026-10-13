"""Selection, leakage boundaries, completion states, and offline pipeline tests."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3Model

from . import analysis, data, runtime, trainer
from .budget import Budget, BudgetExceeded
from .build_notebook import default_settings


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_settings_pin_pilot_recipe_and_allow_explicit_longer_budgets(self):
        defaults = runtime.validate_settings(default_settings())
        self.assertEqual(defaults["run_mode"], "pilot")
        self.assertEqual(defaults["pilot_samples"], 4000)
        self.assertEqual(defaults["max_epochs"], 1)
        runtime.validate_settings({**defaults, "time_budget_minutes": 500})
        runtime.validate_settings({**defaults, "time_budget_minutes": None})
        runtime.validate_settings({**defaults, "time_budget_minutes": None, "deadline_monotonic": None,
                                   "invocation_started_monotonic": 100.})
        runtime.validate_settings({**defaults, "run_mode": "full", "source_run": "completed_pilot"})
        for update in ({"threshold": .4}, {"max_epochs": 4}, {"benchmark_examples": 49},
                       {"pilot_samples": 5000}, {"finalization_reserve_seconds": 100},
                       {"qwen_max_length": 512}, {"deadline_monotonic": float("nan")},
                       {"head_lr": .01}, {"lora_r": 8}, {"time_budget_minutes": 0},
                       {"time_budget_minutes": 20}, {"time_budget_minutes": "None"},
                       {"deadline_monotonic": None}, {"invocation_started_monotonic": float("inf")},
                       {"invocation_started_monotonic": True}, {"run_mode": "full"},
                       {"run_mode": "resume"}, {"source_run": "unexpected_pilot_source"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                runtime.validate_settings({**default_settings(), **update})

    def test_macro_selection_has_no_nausea_veto_and_no_search(self):
        records = {"control": {"selected_epochs": 2}, "qwen": {"selected_epochs": 1}}
        scores = {"control": {"macro_f1": .65, "nausea_f1": .5},
                  "qwen": {"macro_f1": .655, "nausea_f1": .2},
                  "ensemble": {"macro_f1": .651, "nausea_f1": .6}}
        chosen = runtime.choose_candidate(scores, records)
        self.assertEqual(chosen["selected_candidate"], "qwen")
        self.assertTrue(chosen["improved"])
        self.assertFalse(chosen["expansion_recommended"])
        scores["ensemble"]["macro_f1"] = .67
        chosen = runtime.choose_candidate(scores, records)
        self.assertEqual(chosen["components"], ["control", "qwen"])
        self.assertTrue(chosen["expansion_recommended"])
        for row in scores.values():
            row["macro_f1"] = .65
        self.assertEqual(runtime.choose_candidate(scores, records)["selected_candidate"], "control")

    def test_pilot_budget_admits_only_subset_epochs_and_never_full_training(self):
        probes = {name: {"seconds_per_example": .1, "seconds_per_prediction": .01,
                          "setup_seconds": 1.} for name in runtime.ORDER}
        settings = default_settings()
        plan = runtime.plan_epochs(probes, 700, 1000, 150, 150, settings,
                                    Budget(10800, 1500, 1.35, clock=lambda: 0))
        self.assertFalse(plan["full_refit_planned"])
        self.assertFalse(plan["confirmation_planned"])
        self.assertEqual(plan["common_epochs"], 1)
        settings = {**settings, "max_epochs": 3}
        plan = runtime.plan_epochs(probes, 700, 1000, 150, 150, settings,
                                    Budget(86400, 1500, 1.35, clock=lambda: 0))
        self.assertEqual(plan["common_epochs"], 3)
        self.assertFalse(plan["full_refit_planned"])
        self.assertFalse(plan["confirmation_planned"])
        smaller = runtime.plan_epochs(probes, 700, 1000, 150, 150, settings,
                                      Budget(2400, 1500, 1.35, clock=lambda: 0))
        self.assertEqual(smaller["common_epochs"], 2)
        with self.assertRaises(BudgetExceeded):
            runtime.plan_epochs(probes, 700, 1000, 150, 150, settings,
                                 Budget(2000, 1500, 1.35, clock=lambda: 0))

    def test_timing_estimates_use_examples_per_second_with_selected_batches(self):
        probes = {name: {"seconds_per_example": .1, "seconds_per_microstep": 999.,
                          "seconds_per_prediction": .01, "setup_seconds": 1.} for name in runtime.ORDER}
        settings = {**default_settings(), "qwen_batch_size": 2, "qwen_gradient_accumulation": 8,
                    "control_batch_size": 4, "control_gradient_accumulation": 4}
        estimates = runtime.timing_estimates(probes, 701, 1001, 150, 150, settings)
        self.assertAlmostEqual(estimates["qwen"]["fit_epoch_seconds"], 70.2)
        self.assertAlmostEqual(estimates["control"]["fit_epoch_seconds"], 70.4)
        self.assertAlmostEqual(estimates["qwen"]["full_epoch_seconds"], 100.2)
        self.assertAlmostEqual(estimates["control"]["full_epoch_seconds"], 100.4)
        self.assertAlmostEqual(estimates["qwen"]["development_seconds"], 1.5)

    def test_unlimited_pilot_plan_keeps_epoch_limit_and_separate_full_run(self):
        probes = {kind: {"seconds_per_example": 1000., "seconds_per_prediction": 1.,
                        "setup_seconds": 10.} for kind in runtime.ORDER}
        settings = {**default_settings(), "time_budget_minutes": None, "max_epochs": 3}
        plan = runtime.plan_epochs(probes, 4000, 29200, 4380, 4380, settings, Budget(None))
        self.assertEqual(plan["common_epochs"], 3)
        self.assertIsNone(plan["remaining_seconds"])
        self.assertFalse(plan["full_refit_planned"])
        self.assertFalse(plan["confirmation_planned"])

    def test_full_budget_recommendation_includes_preparation_without_double_charging_admission(self):
        settings = default_settings()
        estimates = {kind: {"full_epoch_seconds": 60., "setup_seconds": 30.,
                            "confirmation_seconds": 10., "development_seconds": 10.}
                     for kind in runtime.ORDER}
        selection = {"components": ["control", "qwen"], "selected_candidate": "ensemble",
                     "selected_epochs": {"control": 1, "qwen": 1}}
        no_preparation = runtime.estimate_full_run(estimates, selection, settings)
        with_preparation = runtime.estimate_full_run(estimates, selection, settings, preparation_seconds=600)
        self.assertEqual(with_preparation["recommended_budget_minutes"]
                         - no_preparation["recommended_budget_minutes"], 14)
        self.assertEqual(with_preparation["estimated_work_seconds"], no_preparation["estimated_work_seconds"])
        after_preparation = Budget(with_preparation["recommended_budget_minutes"] * 60,
            settings["finalization_reserve_seconds"], settings["budget_safety_factor"], clock=lambda: 600)
        self.assertTrue(after_preparation.can_start(with_preparation["estimated_work_seconds"], training=True))

    def test_internal_evaluation_cannot_consume_export_reserve(self):
        budget = Budget(100, 25, clock=lambda: 80)
        self.assertGreater(budget.check("export"), 0)
        with self.assertRaises(BudgetExceeded):
            runtime._TrainingBudget(budget).check("internal")

    def test_cli_accepts_official_flags_without_reading_audio(self):
        with mock.patch.object(runtime, "inference") as infer:
            runtime.inference_main(["--audio_dir", "nonexistent", "--label_dir", "labels",
                                    "--ckpt_path", "checkpoint", "--output", "result.csv"])
            infer.assert_called_once_with("checkpoint", "labels", "result.csv")

    def _fixture(self, directory):
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sample", "call", "hello", "world"]
        vocabulary += [str(i) for i in range(100)]
        vocab = directory / "vocab.txt"
        vocab.write_text("\n".join(vocabulary), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=24, max_position_embeddings=128,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        control = directory / "control_pretrained"
        BertModel(config).save_pretrained(control)
        tokenizer.save_pretrained(control)
        qwen = directory / "qwen_pretrained"
        Qwen3Model(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8, intermediate_size=24,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=3)).save_pretrained(qwen)
        tokenizer.save_pretrained(qwen)
        for category, count in (("train", 60), ("validation", 10)):
            folder = directory / category
            folder.mkdir()
            for index in range(count):
                annotation = {"utterances": [{"text": f"sample call {index}"}, {"text": "hello world"}],
                    "symptom": [symptom for j, symptom in enumerate(data.TARGET_SYMPTOMS) if (index + j) % 3 == 0]}
                (folder / f"{index:03}.json").write_text(json.dumps(annotation), encoding="utf-8")
        specs = {kind: {"kind": kind, "model_id": str(path), "snapshot_path": str(path), "revision": "a" * 40}
                 for kind, path in (("qwen", qwen), ("control", control))}
        settings = {**default_settings(), "output_dir": str(directory / "results"),
            "work_dir": str(directory / "work"), "train_dir": str(directory / "train"),
            "validation_dir": str(directory / "validation"), "allow_cpu": True, "max_epochs": 1}
        return settings, specs

    @contextlib.contextmanager
    def _tiny_pipeline(self, directory):
        settings, specs = self._fixture(directory)
        original_create, original_train = trainer.create_model, trainer.train_epoch
        original_load, original_subset = data.load_dataset, data.sample_fit_subset
        original_confirmation, original_choose = runtime._confirmation, runtime.choose_candidate
        trace = {"training": [], "validation": [], "confirmation": [], "benchmarks": [],
                 "pilot_root": None, "slow_benchmark": False}

        def create(spec, values, device, **kwargs):
            options = {**values, "test_only_unquantized": True, "control_dropout": 0.,
                       "lora_dropout": 0., "control_max_length": 64, "qwen_max_length": 64}
            return original_create(spec, options, device, local_files_only=True)

        def train(model, loader, optimizer, **kwargs):
            if len(loader.dataset) == 60:
                self.assertEqual(len(trace["confirmation"]), 1)
            trace["training"].append((model.kind, len(loader.dataset), kwargs["pos_weight"].clone(),
                                       loader.batch_size, kwargs["accumulation_steps"]))
            return original_train(model, loader, optimizer, **kwargs)

        def load(path, include_targets=False):
            if Path(path) == directory / "validation" and include_targets:
                completed_models = [folder for folder in (directory / "results").iterdir()
                                    if (folder / "mission3_ver25_model.ckpt").is_file()]
                self.assertEqual(len(completed_models), 1)
                full_root = completed_models[0]
                self.assertTrue((full_root / "final_selection.json").is_file())
                self.assertTrue((full_root / "confirmation_protocol.json").is_file())
                trace["validation"].append(full_root)
            return original_load(path, include_targets)

        def subset(texts, targets, fit, sample_count=4000, seed=42):
            self.assertEqual(sample_count, 4000)
            indices, report = original_subset(texts, targets, fit, sample_count=24, seed=seed)
            return indices, {**report, "requested_samples": 4000}

        def autotune(kind, spec, texts, targets, fit, development, values, device, budget, root):
            self.assertEqual(len(fit), 24)
            self.assertFalse(np.intersect1d(fit, development).size)
            batch = 2 if kind == "qwen" else 4
            trace["benchmarks"].append((kind, len(fit), batch))
            probe = {"seconds_per_example": 1000. if trace["slow_benchmark"] else .0001,
                     "seconds_per_prediction": .0001, "setup_seconds": .001,
                     "batch_size": batch, "gradient_accumulation": 16 // batch,
                     "effective_batch_size": 16, "status": "selected"}
            return {**values, kind + "_batch_size": batch,
                    kind + "_gradient_accumulation": 16 // batch}, probe

        def force_ensemble(scores, records, minimum):
            # Exercise the two-model export route; actual selection rules have
            # a separate known-score test. Keep this policy during source checks.
            return original_choose({"control": {"macro_f1": .1},
                "qwen": {"macro_f1": .2}, "ensemble": {"macro_f1": .3}}, records, minimum)

        def confirm(root, *args, **kwargs):
            self.assertEqual(kwargs["source_root"], trace["pilot_root"])
            self.assertEqual(len(trace["training"]), 2)
            self.assertTrue((root / "final_selection.json").is_file())
            frozen = (root / "final_selection.json").read_bytes()
            self.assertEqual(frozen, (trace["pilot_root"] / "final_selection.json").read_bytes())
            result = original_confirmation(root, *args, **kwargs)
            self.assertEqual(frozen, (root / "final_selection.json").read_bytes())
            trace["confirmation"].append(root)
            return result

        with mock.patch.object(runtime, "resolve_specs", return_value=specs) as downloads, \
             mock.patch.object(runtime, "EXPECTED_TRAINING_SAMPLES", 60), \
             mock.patch.object(runtime, "choose_candidate", side_effect=force_ensemble), \
             mock.patch.object(runtime.batching, "autotune", side_effect=autotune), \
             mock.patch.object(trainer, "create_model", side_effect=create), \
             mock.patch.object(trainer, "train_epoch", side_effect=train), \
             mock.patch.object(data, "load_dataset", side_effect=load), \
             mock.patch.object(data, "sample_fit_subset", side_effect=subset), \
             mock.patch.object(runtime, "_confirmation", side_effect=confirm), \
             contextlib.redirect_stdout(io.StringIO()):
            trace["downloads"] = downloads
            yield settings, trace

    def test_pilot_stops_even_with_long_budget_then_explicit_full_run_reloads_predictions(self):
        evidence_root = Path(__file__).resolve().parents[3] / "outputs" / "mission3_ver25_verification"
        evidence_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence_root) as temporary:
            directory = Path(temporary)
            with self._tiny_pipeline(directory) as (settings, trace):
                before = time.monotonic()
                with mock.patch.object(runtime, "Budget", wraps=Budget) as budget_constructor:
                    pilot_root = runtime.run({**settings, "time_budget_minutes": 500})
                self.assertGreater(budget_constructor.call_args.args[0] - before, 499 * 60)
                trace["pilot_root"] = pilot_root
                pilot = runtime.read_json(pilot_root / "pilot_complete.json")
                self.assertEqual(pilot["status"], "pilot_complete")
                self.assertEqual(pilot["training_scope"], "training_subset")
                self.assertEqual(pilot["fitted_samples"], 24)
                self.assertFalse(pilot["submission_ready"])
                self.assertFalse(pilot["confirmation_evaluated"])
                self.assertFalse(pilot["validation_evaluated"])
                self.assertEqual([row[1] for row in trace["training"]], [24, 24])
                self.assertEqual([(row[3], row[4]) for row in trace["training"]], [(2, 8), (4, 4)])
                self.assertFalse(trace["confirmation"])
                self.assertFalse(trace["validation"])
                for absent in ("run_complete.json", "mission3_ver25_model.ckpt", "confirmation_comparison.json",
                               "confirmation_protocol.json", "validation_metrics.json", "mission3.csv",
                               "final_qwen", "final_control"):
                    self.assertFalse((pilot_root / absent).exists(), absent)
                for name in ("pilot_recipe.json", "subset_manifest.json", "subset.npz",
                             "development_comparison.json", "pilot_portable_verification.json",
                             "full_run_estimate.json"):
                    self.assertIn(name, pilot["hashes"])
                    self.assertEqual(pilot["hashes"][name], runtime._hash(pilot_root / name))
                _, texts, targets = data.load_dataset(directory / "train", True)
                with np.load(pilot_root / "subset.npz") as subset:
                    expected_pilot_weights = trainer.positive_weights(targets[subset["indices"]])
                for row in trace["training"]:
                    torch.testing.assert_close(row[2], expected_pilot_weights)
                pilot_marker = (pilot_root / "pilot_complete.json").read_bytes()
                full_root = runtime.run({**settings, "run_mode": "full", "source_run": str(pilot_root),
                                         "max_epochs": 3})
                complete = runtime.read_json(full_root / "run_complete.json")
                self.assertEqual(complete["training_scope"], "all_training")
                self.assertTrue(complete["validation_evaluated"])
                self.assertEqual([row[1] for row in trace["training"]], [24, 24, 60, 60])
                self.assertEqual(len(trace["confirmation"]), 1)
                self.assertEqual(len(trace["validation"]), 1)
                self.assertEqual((pilot_root / "pilot_complete.json").read_bytes(), pilot_marker)
                self.assertEqual((pilot_root / "final_selection.json").read_bytes(),
                                 (full_root / "final_selection.json").read_bytes())
                self.assertEqual(len(trace["benchmarks"]), 4)
                for row in trace["training"][2:]:
                    torch.testing.assert_close(row[2], trainer.positive_weights(targets))
                self.assertEqual(runtime.read_json(full_root / "split_manifest.json")["data_digest"],
                                 runtime.read_json(full_root / "training_data.json")["digest"])
                offline = directory / "offline.csv"
                runtime.inference_main(["--audio_dir", "unused", "--label_dir", str(directory / "validation"),
                    "--ckpt_path", str(full_root / "mission3_ver25_model.ckpt"), "--output", str(offline)])
                self.assertEqual(offline.read_bytes(), (full_root / "mission3.csv").read_bytes())
                self.assertEqual(complete["hashes"]["mission3.csv"], runtime._hash(offline))

    def test_full_run_rejects_incomplete_changed_and_mismatched_sources_before_downloads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            with self._tiny_pipeline(directory) as (settings, trace):
                incomplete = directory / "incomplete_pilot"
                incomplete.mkdir()
                full = {**settings, "run_mode": "full", "source_run": str(incomplete)}
                with self.assertRaisesRegex(ValueError, "completed pilot_complete"):
                    runtime.run(full)
                trace["downloads"].assert_not_called()
                self.assertFalse(trace["training"])
                pilot_root = runtime.run(settings)
                trace["pilot_root"] = pilot_root
                full["source_run"] = str(pilot_root)
                trace["downloads"].reset_mock()
                manifest = pilot_root / "subset_manifest.json"
                original = manifest.read_bytes()
                try:
                    manifest.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "artifact is missing or changed"):
                        runtime.run(full)
                finally:
                    manifest.write_bytes(original)
                trace["downloads"].assert_not_called()
                training_row = directory / "train" / "000.json"
                original = training_row.read_bytes()
                annotation = json.loads(original.decode("utf-8"))
                annotation["symptom"] = []
                try:
                    training_row.write_text(json.dumps(annotation), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Training data does not match"):
                        runtime.run(full)
                finally:
                    training_row.write_bytes(original)
                trace["downloads"].assert_not_called()
                self.assertEqual([row[1] for row in trace["training"]], [24, 24])
                self.assertFalse(trace["confirmation"])
                self.assertFalse(trace["validation"])
                self.assertFalse(any((path / "run_complete.json").exists()
                                     for path in (directory / "results").iterdir()))

    def test_full_budget_rejection_happens_before_confirmation_or_refit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            with self._tiny_pipeline(directory) as (settings, trace):
                pilot_root = runtime.run(settings)
                trace["pilot_root"] = pilot_root
                trace["slow_benchmark"] = True
                full_root = runtime.run({**settings, "run_mode": "full", "source_run": str(pilot_root)})
                stopped = runtime.read_json(full_root / "diagnostic_complete.json")
                self.assertEqual(stopped["stage"], "full_training_admission")
                self.assertFalse(stopped["submission_ready"])
                self.assertFalse(runtime.read_json(full_root / "budget_plan.json")["admitted"])
                self.assertEqual([row[1] for row in trace["training"]], [24, 24])
                self.assertFalse(trace["confirmation"])
                self.assertFalse(trace["validation"])
                self.assertFalse((full_root / "run_complete.json").exists())
                self.assertFalse((full_root / "mission3_ver25_model.ckpt").exists())

    def test_improving_epochs_save_distinct_best_checkpoints(self):
        evidence_root = Path(__file__).resolve().parents[3] / "outputs" / "mission3_ver25_verification"
        evidence_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=evidence_root) as temporary:
            directory = Path(temporary)
            settings, specs = self._fixture(directory)
            settings = {**settings, "base_store_dir": str(directory / "shared_base")}
            _, texts, targets = data.load_dataset(directory / "train", True)
            fit, development, _, _ = data.three_way_split(texts, targets)
            original_metrics = analysis.metrics
            calls = []

            def score(y, logits):
                result = original_metrics(y, logits)
                calls.append(True)
                result["macro_f1"] = len(calls) / 10
                return result

            with mock.patch.object(analysis, "metrics", side_effect=score), contextlib.redirect_stdout(io.StringIO()):
                record, _ = runtime._fit_candidate("control", specs["control"], texts, targets, fit, development,
                    settings, {"common_epochs": 2}, "cpu", Budget(time.monotonic() + 3600, 1200), directory)
            self.assertEqual(record["selected_epochs"], 2)
            for epoch in (1, 2):
                self.assertTrue((directory / "fit_control" / f"epoch_{epoch}" / "manifest.json").is_file())

    def test_unlimited_full_admits_slow_work_from_bounded_or_unlimited_pilot(self):
        for pilot_minutes in (180, None):
            with self.subTest(pilot_minutes=pilot_minutes), \
                 tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                with self._tiny_pipeline(Path(temporary)) as (settings, trace):
                    pilot_root = runtime.run({**settings, "time_budget_minutes": pilot_minutes})
                    trace["pilot_root"] = pilot_root
                    self.assertTrue((pilot_root / "pilot_complete.json").is_file())
                    self.assertFalse((pilot_root / "run_complete.json").exists())
                    self.assertFalse(trace["confirmation"])
                    self.assertFalse(trace["validation"])
                    trace["slow_benchmark"] = True
                    with mock.patch.object(runtime, "Budget", wraps=Budget) as constructor:
                        full_root = runtime.run({**settings, "run_mode": "full", "source_run": str(pilot_root),
                            "time_budget_minutes": None, "deadline_monotonic": None,
                            "invocation_started_monotonic": time.monotonic() - 600})
                    self.assertIsNone(constructor.call_args.args[0])
                    complete = runtime.read_json(full_root / "run_complete.json")
                    self.assertEqual(complete["training_scope"], "all_training")
                    self.assertTrue(complete["validation_evaluated"])
                    self.assertGreaterEqual(complete["elapsed_invocation_seconds"], 600)
                    self.assertTrue(math.isfinite(complete["elapsed_invocation_seconds"]))
                    plan = runtime.read_json(full_root / "budget_plan.json")
                    self.assertIsNone(plan["remaining_seconds"])
                    self.assertGreater(plan["estimated_work_seconds"], 180 * 60)
                    estimate = runtime.read_json(full_root / "full_run_estimate.json")
                    self.assertGreaterEqual(estimate["estimated_preparation_seconds"], 600)
                    self.assertEqual([row[1] for row in trace["training"]], [24, 24, 60, 60])
                    self.assertEqual(len(trace["confirmation"]), 1)
                    self.assertEqual(len(trace["validation"]), 1)
                    self.assertFalse((full_root / "diagnostic_complete.json").exists())

    def test_budget_stop_keeps_fit_artifacts_without_submission_completion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            checkpoint = root / "fit_qwen" / "epoch_1" / "manifest.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            runtime._diagnostic(root, "budget_not_started", "qwen_fit", time.monotonic())
            report = runtime.read_json(root / "diagnostic_complete.json")
            self.assertFalse(report["submission_ready"])
            self.assertEqual(report["training_scope"], "fit_only")
            self.assertEqual(report["status"], "fit_only_budget")
            self.assertFalse((root / "run_complete.json").exists())
            self.assertFalse((root / "mission3_ver25_model.ckpt").exists())

    def test_partial_training_cannot_start_downloads_or_claim_all_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            settings = {**default_settings(), "allow_cpu": True, "output_dir": str(root)}
            with mock.patch.object(runtime, "prepare_data", return_value=("train", "validation")), \
                 mock.patch.object(data, "load_dataset", return_value=([], ["sample"] * 10, np.zeros((10, 9)))), \
                 mock.patch.object(runtime, "resolve_specs") as download, \
                 self.assertRaisesRegex(ValueError, "29,200"):
                runtime.run(settings)
            download.assert_not_called()
            result = next(root.iterdir())
            self.assertEqual(runtime.read_json(result / "runtime.json")["status"], "failed")
            self.assertFalse((result / "run_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
