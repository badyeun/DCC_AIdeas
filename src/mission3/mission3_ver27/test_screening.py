"""Verify the diagnostic boundary and real tiny-model learning curves."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Qwen3Config, Qwen3Model

from . import analysis, baseline, build_notebook, data, run_notebook, runtime, trainer
from .budget import Budget, BudgetExceeded

HERE = Path(__file__).resolve().parent


def cell(notebook, suffix):
    return "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "ver27-" + suffix))


def literal(notebook, suffix, name):
    for node in ast.parse(cell(notebook, suffix)).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


class ScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_settings_pin_screening_and_accept_optional_finite_budget(self):
        values = runtime.validate_settings(build_notebook.default_settings())
        self.assertEqual((values["run_mode"], values["pilot_samples"], values["max_epochs"]), ("pilot", 4000, 3))
        self.assertFalse(values["run_validation"])
        self.assertEqual(values["lora_lr"], 3e-5)
        self.assertEqual(values["head_lr"], 9e-5)
        self.assertTrue(values["baseline_run"].endswith("ver26/20261011T133609Z_f22b4e55"))
        self.assertEqual(values["time_budget_minutes"], 480)
        for minutes in (180, 479.5):
            runtime.validate_settings({**values, "time_budget_minutes": minutes})
        for update in ({"run_mode": "full", "source_run": "a"}, {"run_validation": True},
                       {"max_epochs": 1}, {"max_epochs": 2}, {"source_run": "pilot"},
                       {"pilot_samples": 20440}, {"time_budget_minutes": True},
                       {"baseline_run": ""}, {"lora_lr": 1e-4}, {"head_lr": 3e-4},
                       {"time_budget_minutes": None}, {"time_budget_minutes": 481},
                       {"time_budget_minutes": float("nan")}, {"time_budget_minutes": float("inf")},
                       {"time_budget_minutes": 0}, {"time_budget_minutes": "None"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                runtime.validate_settings({**values, **update})
        with mock.patch.object(runtime, "prepare_data") as prepare, self.assertRaisesRegex(ValueError, "screening-only"):
            runtime.run({**values, "run_mode": "full"})
        prepare.assert_not_called()

    def test_admission_requires_three_epochs_and_never_starts_full_or_confirmation(self):
        probes = {kind: {"seconds_per_example": .1, "seconds_per_prediction": .01, "setup_seconds": 30.}
                  for kind in runtime.ORDER}
        settings = build_notebook.default_settings()
        plan = runtime.plan_epochs(probes, 4000, 29200, 4380, 4380, settings, Budget(None))
        self.assertEqual(plan["common_epochs"], 3)
        self.assertEqual(plan["scheduler_horizon_epochs"], 3)
        self.assertFalse(plan["full_refit_planned"])
        self.assertFalse(plan["confirmation_planned"])
        self.assertFalse(plan["partial_epoch_plan_allowed"])
        # Enough budget for two epochs is deliberately insufficient for this experiment.
        two_epoch_budget = 1500 + plan["estimated_work_seconds"] * 1.35 * .75
        with self.assertRaises(BudgetExceeded):
            runtime.plan_epochs(probes, 4000, 29200, 4380, 4380, settings,
                                Budget(two_epoch_budget, 1500, 1.35, clock=lambda: 0))

    def test_curve_summary_separates_more_epochs_from_control_gain(self):
        def history(scores):
            return [{"selected_epochs": epoch, "macro_f1": score, "train_mean_loss": 1 / epoch,
                "development_weighted_bce": .8, "metrics": {"macro_average_precision": .6, "nausea_f1": .3}}
                for epoch, score in enumerate(scores, 1)]
        summary = runtime.summarize_learning_curve({"qwen": history([.66, .65, .64]),
                                                   "control": history([.55, .57, .56])})
        self.assertTrue(summary["candidates"]["qwen"]["best_epoch_is_one"])
        self.assertTrue(summary["candidates"]["qwen"]["plateau_or_decline_after_epoch1"])
        self.assertFalse(summary["candidates"]["qwen"]["later_epoch_improved"])
        self.assertEqual(summary["candidates"]["control"]["best_epoch"], 2)
        self.assertAlmostEqual(summary["candidates"]["control"]["best_later_epoch_minus_epoch1_macro_f1"], .02)
        self.assertTrue(summary["full_fit_confirmation_required"])
        self.assertFalse(summary["automatic_full_training_recommended"])
        selection = runtime.choose_candidate({"qwen": {"macro_f1": .66}, "control": {"macro_f1": .57},
            "ensemble": {"macro_f1": .65}}, {"qwen": {"selected_epochs": 1}, "control": {"selected_epochs": 2}})
        self.assertNotIn("expansion_recommended", selection)
        self.assertTrue(selection["control_gain_does_not_prove_more_epochs_help"])

    def test_notebook_is_self_contained_with_current_sources_and_unchanged_dependencies(self):
        notebook = json.loads((HERE.parent / "mission3_ver27.ipynb").read_text(encoding="utf-8"))
        embedded = literal(notebook, "restore", "RUNTIME_SOURCES")
        self.assertEqual(set(embedded), set(build_notebook.SOURCE_FILES))
        self.assertEqual(build_notebook.SOURCE_FILES, run_notebook.SOURCE_FILES)
        for name, source in embedded.items():
            self.assertEqual(source, (HERE / name).read_text(encoding="utf-8"))
            if name.endswith(".py"):
                compile(source, name, "exec")
        for entry in notebook["cells"]:
            if entry["cell_type"] == "code":
                compile("".join(entry["source"]), entry["id"], "exec")
                self.assertEqual(entry["outputs"], [])
        self.assertEqual(literal(notebook, "settings", "SETTINGS"), build_notebook.default_settings())
        self.assertTrue(literal(notebook, "settings", "AUTO_RELEASE_GPU"))
        self.assertEqual(embedded["requirements.txt"],
                         (HERE.parent / "mission3_ver25" / "requirements.txt").read_text(encoding="utf-8"))
        self.assertNotIn("inference.py", embedded)
        self.assertEqual(json.loads(embedded["model_specs.json"])["qwen"]["revision"],
                         "cdbee75f17c01a7cc42f958dc650907174af0554")

    def test_notebook_rejects_unlimited_or_excess_budget_before_setup(self):
        notebook = build_notebook.make_notebook()
        for minutes in (None, 481, True, "480", float("inf"), float("nan")):
            namespace = {"SETTINGS": {**build_notebook.default_settings(), "time_budget_minutes": minutes},
                         "AUTO_RELEASE_GPU": True}
            with self.subTest(minutes=minutes), mock.patch("threading.Timer") as timer, \
                 self.assertRaisesRegex(ValueError, "at most 480"):
                exec(cell(notebook, "budget"), namespace)
            timer.assert_not_called()
            self.assertNotIn("DEADLINE_MONOTONIC", namespace)

    def test_default_eight_hour_budget_starts_before_install(self):
        notebook = build_notebook.make_notebook()
        namespace = {"SETTINGS": build_notebook.default_settings(), "AUTO_RELEASE_GPU": True}
        timer = mock.Mock()
        with mock.patch("time.monotonic", return_value=100), mock.patch("threading.Timer", return_value=timer) as factory, \
             contextlib.redirect_stdout(io.StringIO()):
            exec(cell(notebook, "budget"), namespace)
        self.assertEqual(namespace["DEADLINE_MONOTONIC"], 28900)
        self.assertEqual(namespace["SETTINGS"]["invocation_started_monotonic"], 100)
        self.assertEqual(factory.call_args.args[0], 28800)
        timer.start.assert_called_once_with()
        ids = [entry["id"] for entry in notebook["cells"]]
        self.assertLess(ids.index("ver27-budget"), ids.index("ver27-dependencies"))
        self.assertEqual(namespace["SETTINGS"]["finalization_reserve_seconds"], 1500)

    def test_notebook_result_and_worker_error_both_release_gpu(self):
        notebook = build_notebook.make_notebook()
        with tempfile.TemporaryDirectory(prefix=".screening_notebook_test_", dir=HERE) as temporary:
            directory = Path(temporary)
            (directory / "run_notebook.py").write_text("", encoding="utf-8")
            result = directory / "result"
            result.mkdir()
            marker = {"status": "screening_complete", "submission_ready": False, "scheduler_horizon_epochs": 3,
                      "validation_evaluated": False, "confirmation_evaluated": False, "fitted_samples": 4000,
                      "baseline_comparison_completed": True}
            for name in ("runtime.json", "screening_complete.json"):
                (result / name).write_text(json.dumps(marker), encoding="utf-8")
            for name in ("screening_selection.json", "screening_protocol.json", "screening_portable_verification.json",
                         "learning_curve_summary.json", "historical_comparison.json", "baseline_reference_verified.json"):
                (result / name).write_text("{}", encoding="utf-8")
            (result / "baseline_gate.json").write_text(json.dumps({"status": "no_clear_gain", "macro_f1_delta": -.001}),
                                                      encoding="utf-8")
            for kind in runtime.ORDER:
                (result / (kind + "_history.json")).write_text(json.dumps([
                    {"selected_epochs": epoch, "macro_f1": .5, "train_mean_loss": .8, "development_weighted_bce": .9}
                    for epoch in (1, 2, 3)]), encoding="utf-8")
            supervisor = mock.Mock(side_effect=lambda *a, **k:
                (directory / "notebook_result.json").write_text(json.dumps(str(result)), encoding="utf-8"))
            namespace = {"runtime_dir": directory, "SETTINGS": {**build_notebook.default_settings(),
                         "output_dir": str(directory / "out")}, "budget_module": types.SimpleNamespace(
                         run_supervised=supervisor, BudgetExceeded=BudgetExceeded), "DEADLINE_MONOTONIC": 28900,
                         "BUDGET_STARTED_MONOTONIC": 100, "time": types.SimpleNamespace(monotonic=lambda: 200),
                         "Path": Path, "sys": sys, "release_gpu": mock.Mock(),
                         "RUNTIME_PREPARED": (str(directory), 28900), "DEPENDENCIES_PREPARED": (str(directory), 28900)}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell(notebook, "run"), namespace)
            self.assertIn("3 epoch 학습 곡선 확인 완료", output.getvalue())
            self.assertEqual(supervisor.call_args.kwargs["deadline_monotonic"], 28900)
            namespace["release_gpu"].assert_called_once_with()
            namespace["release_gpu"].reset_mock()
            supervisor.side_effect = subprocess.CalledProcessError(1, ["fixture"])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                exec(cell(notebook, "run"), namespace)
            namespace["release_gpu"].assert_called_once_with()

    def _fixture(self, directory):
        vocab = directory / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sample", "call"]
                                  + [str(i) for i in range(100)]), encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        control = directory / "control_pretrained"
        BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=24, max_position_embeddings=128,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)).save_pretrained(control)
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
                annotation = {"utterances": [{"text": f"sample call {index}"}],
                    "symptom": [label for j, label in enumerate(data.TARGET_SYMPTOMS) if (index + j) % 3 == 0]}
                (folder / f"{index:03}.json").write_text(json.dumps(annotation), encoding="utf-8")
        specs = {kind: {"kind": kind, "model_id": str(path), "snapshot_path": str(path), "revision": "a" * 40}
                 for kind, path in (("qwen", qwen), ("control", control))}
        settings = {**build_notebook.default_settings(), "output_dir": str(directory / "results"),
            "work_dir": str(directory / "work"), "train_dir": str(directory / "train"),
            "validation_dir": str(directory / "validation"), "allow_cpu": True}
        return settings, specs

    def test_missing_or_mismatched_baseline_fails_before_pretrained_download(self):
        with tempfile.TemporaryDirectory(prefix=".screening_baseline_test_", dir=HERE) as temporary:
            directory = Path(temporary)
            settings, _ = self._fixture(directory)
            baseline_folder = directory / "incomplete_reference"
            baseline_folder.mkdir()
            original_subset = data.sample_fit_subset
            for path in (directory / "missing_reference", baseline_folder):
                with self.subTest(path=path.name), mock.patch.object(runtime, "EXPECTED_TRAINING_SAMPLES", 60), \
                     mock.patch.object(data, "sample_fit_subset", side_effect=lambda texts, targets, fit, **kwargs:
                         original_subset(texts, targets, fit, sample_count=24, seed=kwargs["seed"])), \
                     mock.patch.object(runtime, "resolve_specs") as downloads, \
                     mock.patch.object(trainer, "create_model") as create, \
                     self.assertRaisesRegex(ValueError, "ver26 baseline:"):
                    runtime.run({**settings, "baseline_run": str(path)})
                downloads.assert_not_called()
                create.assert_not_called()
            reports = [runtime.read_json(path) for path in (directory / "results").glob("*/runtime.json")]
            self.assertEqual(len(reports), 2)
            self.assertTrue(all(row["status"] == "failed" and row["stage"] == "baseline_verification" for row in reports))

    def test_real_tiny_pipeline_saves_all_epochs_and_reloads_without_validation_or_full(self):
        with tempfile.TemporaryDirectory(prefix=".screening_test_", dir=HERE) as temporary:
            directory = Path(temporary)
            settings, specs = self._fixture(directory)
            original_create, original_train = trainer.create_model, trainer.train_epoch
            original_load, original_subset = data.load_dataset, data.sample_fit_subset
            trace = []
            baseline_events = []

            def create(spec, values, device, **kwargs):
                self.assertEqual(baseline_events, ["verified"])
                values = {**values, "test_only_unquantized": True, "control_dropout": 0.,
                          "lora_dropout": 0., "control_max_length": 64, "qwen_max_length": 64}
                return original_create(spec, values, device, local_files_only=True)

            def train(model, loader, optimizer, **kwargs):
                trace.append((model.kind, len(loader.dataset)))
                self.assertEqual(len(loader.dataset), 24)
                return original_train(model, loader, optimizer, **kwargs)

            def load(path, include_targets=False):
                self.assertNotEqual(Path(path), directory / "validation", "Official Validation must not be loaded")
                return original_load(path, include_targets)

            def subset(texts, targets, fit, sample_count=4000, seed=42):
                self.assertEqual(sample_count, 4000)
                indices, report = original_subset(texts, targets, fit, sample_count=24, seed=seed)
                return indices, {**report, "requested_samples": 4000}

            def autotune(kind, spec, texts, targets, fit, development, values, device, budget, root):
                batch = 2 if kind == "qwen" else 4
                return {**values, kind + "_batch_size": batch, kind + "_gradient_accumulation": 16 // batch}, {
                    "seconds_per_example": .0001, "seconds_per_prediction": .0001, "setup_seconds": .001}

            def reference(path, texts, targets, split, selected_subset, values, output_dir, budget=None):
                # Helper integrity checks use real artifacts in test_baseline. This
                # integration fixture supplies aligned, deterministic old logits.
                self.assertEqual(set(split), {"fit", "development", "confirmation"})
                self.assertEqual(len(selected_subset), 24)
                indices = split["development"]
                truth = targets[indices]
                old_logits = {kind: np.zeros(truth.shape) for kind in ("qwen", "control")}
                old_logits["ensemble"] = analysis.ensemble_logits(old_logits["control"], old_logits["qwen"])
                evidence = {"version": "ver26", "run_id": "fixture"}
                data.write_json(Path(output_dir) / "baseline_reference_verified.json", evidence)
                baseline_events.append("verified")
                return {"evidence": evidence, "indices": indices, "targets": truth, "logits": old_logits,
                        "metrics": {kind: analysis.metrics(truth, logits) for kind, logits in old_logits.items()},
                        "settings": values, "pin": {"promising_macro_gain": .005}}

            with mock.patch.object(runtime, "resolve_specs", return_value=specs), \
                 mock.patch.object(runtime, "EXPECTED_TRAINING_SAMPLES", 60), \
                 mock.patch.object(runtime.baseline, "load_reference", side_effect=reference), \
                 mock.patch.object(runtime.batching, "autotune", side_effect=autotune), \
                 mock.patch.object(trainer, "create_model", side_effect=create), \
                 mock.patch.object(trainer, "train_epoch", side_effect=train), \
                 mock.patch.object(data, "load_dataset", side_effect=load), \
                 mock.patch.object(data, "sample_fit_subset", side_effect=subset), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = runtime.run(settings)
            self.assertEqual(trace, [(kind, 24) for kind in runtime.ORDER for _ in range(3)])
            marker = runtime.read_json(result / "screening_complete.json")
            self.assertEqual(marker["status"], "screening_complete")
            self.assertFalse(marker["submission_ready"])
            self.assertFalse(marker["validation_evaluated"])
            self.assertFalse(marker["confirmation_evaluated"])
            self.assertTrue(marker["baseline_comparison_completed"])
            gate = runtime.read_json(result / "baseline_gate.json")
            self.assertFalse(gate["full_training_authorized"])
            self.assertFalse(gate["automatic_full_training_recommended"])
            self.assertIsNone(gate["target_0_7_achieved_on_official_validation"])
            self.assertEqual(marker["baseline_gate_status"], gate["status"])
            self.assertTrue((result / "historical_comparison.json").is_file())
            self.assertTrue((result / "historical_comparison_predictions.npz").is_file())
            for forbidden in ("pilot_complete.json", "run_complete.json", "mission3.csv", "validation_metrics.json",
                              "confirmation_comparison.json", "final_qwen", "final_control", "submission"):
                self.assertFalse((result / forbidden).exists(), forbidden)
            selection = runtime.read_json(result / "screening_selection.json")
            self.assertTrue(selection["full_fit_confirmation_required"])
            for name, digest in marker["hashes"].items():
                self.assertEqual(runtime._hash(result / name), digest)
            for kind in runtime.ORDER:
                history = runtime.read_json(result / (kind + "_history.json"))
                self.assertEqual([row["selected_epochs"] for row in history], [1, 2, 3])
                for row in history:
                    self.assertEqual(row["scheduler_horizon_epochs"], 3)
                    self.assertGreater(row["train_mean_loss"], 0)
                    self.assertGreater(row["successful_optimizer_updates"], 0)
                    with np.load(result / "learning_curve" / kind /
                                 ("epoch_" + str(row["selected_epochs"]) + "_predictions.npz")) as predictions:
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            torch.from_numpy(predictions["logits"]).double(),
                            torch.from_numpy(predictions["targets"]).double(),
                            pos_weight=torch.tensor(row["positive_weights"], dtype=torch.float64)).item()
                        self.assertAlmostEqual(row["development_weighted_bce"], loss, places=10)
                        np.testing.assert_array_equal(predictions["predictions"], predictions["logits"] >= 0)
                self.assertTrue(all(rate == 0 for rate in history[-1]["learning_rates_end"]))
            verification = runtime.read_json(result / "screening_portable_verification.json")
            self.assertTrue(all(row["decisions_equal"] for row in verification.values()))


if __name__ == "__main__":
    unittest.main()
