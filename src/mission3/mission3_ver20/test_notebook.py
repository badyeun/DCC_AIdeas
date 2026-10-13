"""Verify the standalone notebook's deadlines, release behavior, and worker boundary."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ver20_notebook_builder", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def cell_source(notebook, identifier):
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == identifier))


def assigned_literal(notebook, identifier, variable):
    tree = ast.parse(cell_source(notebook, identifier))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing literal assignment: {variable}")


class NotebookTests(unittest.TestCase):
    def test_saved_notebook_embeds_current_complete_runtime(self):
        saved = json.loads((HERE.parent / "mission3_ver20.ipynb").read_text(encoding="utf-8"))
        embedded = assigned_literal(saved, "ver20-restore", "RUNTIME_SOURCES")
        self.assertEqual(set(embedded), set(builder.SOURCE_FILES))
        for name in builder.SOURCE_FILES:
            self.assertEqual(embedded[name], (HERE / name).read_text(encoding="utf-8"),
                             f"Regenerate mission3_ver20.ipynb after changing {name}")
        self.assertEqual(embedded["requirements.txt"],
                         (HERE.parent / "mission3_ver14" / "requirements.txt").read_text(encoding="utf-8"))

    def test_defaults_use_single_view_training_and_three_hour_cap(self):
        notebook = builder.make_notebook()
        settings = assigned_literal(notebook, "ver20-settings", "SETTINGS")
        self.assertEqual(settings["version"], "ver20")
        self.assertEqual(settings["model_id"], "klue/roberta-large")
        self.assertEqual(settings["revision"], "28d911204e9022eda172571ca8cc61eaffd942f7")
        self.assertEqual(settings["expected_encoder_layers"], 24)
        self.assertEqual(settings["trainable_encoder_layers"], 8)
        self.assertEqual(settings["batch_size"] * settings["gradient_accumulation"], 16)
        self.assertEqual(settings["max_epochs"], 3)
        self.assertEqual(settings["scheduler_horizon_epochs"], 3)
        self.assertNotIn("utterance_evidence", settings)
        self.assertNotIn("gce_mix", settings)
        self.assertEqual(settings["pilot_optimizer_steps"], 10)
        self.assertNotIn("correlation_hidden", settings)
        self.assertEqual(settings["ce_views"], 1)
        self.assertFalse(settings["label_attention"])
        self.assertEqual(settings["contrastive_weight"], 0)
        self.assertEqual(settings["time_budget_minutes"], 180)
        self.assertEqual(settings["finalization_reserve_seconds"], 1500)
        self.assertEqual(settings["budget_safety_factor"], 1.25)
        self.assertTrue(settings["refit_full_training"])
        self.assertTrue(settings["run_validation"])
        self.assertFalse(settings["allow_cpu"])
        self.assertTrue(assigned_literal(notebook, "ver20-settings", "AUTO_RELEASE_GPU"))
        self.assertNotIn("AUTO_RELEASE_GPU", settings)
        self.assertNotIn("deadline_monotonic", settings)

    def test_notebook_is_unexecuted_compiles_and_starts_budget_before_install(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            target = Path(temporary) / "mission3_ver20.ipynb"
            builder.build(target)
            notebook = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")
        sources = assigned_literal(notebook, "ver20-restore", "RUNTIME_SOURCES")
        for name, source in sources.items():
            if name.endswith(".py"):
                compile(source, name, "exec")
        ordered = ["ver20-drive", "ver20-budget", "ver20-restore", "ver20-dependencies", "ver20-run"]
        self.assertEqual(sorted(ordered, key=identifiers.index), ordered)

    def test_deadline_shared_with_daemon_and_release_only_once(self):
        namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": True}
        timer = Mock()
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=Mock())
        with (patch("time.monotonic", return_value=100), patch("threading.Timer", return_value=timer) as factory,
              patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(io.StringIO())):
            exec(cell_source(builder.make_notebook(), "ver20-budget"), namespace)
            self.assertEqual(namespace["SETTINGS"]["deadline_monotonic"], 10900)
            self.assertEqual(factory.call_args.args[0], 10800)
            self.assertTrue(timer.daemon)
            timer.start.assert_called_once_with()
            namespace["release_gpu"]()
            namespace["release_gpu"]()
            colab.runtime.unassign.assert_called_once_with()
            timer.cancel.assert_called_once_with()

    def test_restored_reference_preserves_exact_utf8_bytes_on_every_platform(self):
        notebook = builder.make_notebook()
        sources = assigned_literal(notebook, "ver20-restore", "RUNTIME_SOURCES")
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            release = Mock()
            namespace = {"release_gpu": release}
            with patch("tempfile.mkdtemp", return_value=str(root)), contextlib.redirect_stdout(io.StringIO()):
                exec(cell_source(notebook, "ver20-restore"), namespace)
            self.assertEqual((root / "training_reference.json").read_bytes(),
                             sources["training_reference.json"].encode("utf-8"))
            self.assertEqual((root / "training_reference.json").read_bytes(),
                             (HERE / "training_reference.json").read_bytes())
            release.assert_not_called()

    def test_disabled_release_keeps_time_cap_without_timer(self):
        namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": False}
        with (patch("time.monotonic", return_value=100), patch("threading.Timer") as timer,
              contextlib.redirect_stdout(io.StringIO())):
            exec(cell_source(builder.make_notebook(), "ver20-budget"), namespace)
            namespace["release_gpu"]()
        self.assertEqual(namespace["DEADLINE_MONOTONIC"], 10900)
        timer.assert_not_called()

    def test_release_failure_is_visible_and_has_manual_recovery(self):
        namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": True}
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=Mock(side_effect=RuntimeError("fixture failure")))
        output = io.StringIO()
        with (patch("time.monotonic", return_value=100), patch("threading.Timer"),
              patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(output)):
            exec(cell_source(builder.make_notebook(), "ver20-budget"), namespace)
            namespace["release_gpu"]()
        self.assertIn("자동 반환 요청 실패", output.getvalue())
        self.assertIn("연결 해제 및 런타임 삭제", output.getvalue())

    def test_installer_is_bounded_by_both_remaining_budget_and_twenty_minutes(self):
        source = cell_source(builder.make_notebook(), "ver20-dependencies")
        for deadline, expected in [(10000, 1300), (500, 500)]:
            supervisor = Mock()
            release = Mock()
            namespace = {"DEADLINE_MONOTONIC": deadline, "time": types.SimpleNamespace(monotonic=lambda: 100),
                         "runtime_dir": HERE, "sys": sys, "release_gpu": release,
                         "budget_module": types.SimpleNamespace(run_supervised=supervisor)}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            self.assertEqual(supervisor.call_args.kwargs["deadline_monotonic"], expected)
            self.assertEqual(supervisor.call_args.kwargs["shutdown_margin_s"], 10)
            self.assertEqual(supervisor.call_args.args[0][-1], str(HERE / "requirements.txt"))
            release.assert_not_called()
        supervisor.side_effect = subprocess.CalledProcessError(1, "fixture installer")
        with self.assertRaises(subprocess.CalledProcessError):
            exec(source, namespace)
        release.assert_called_once_with()

    def test_final_cell_displays_saved_results_before_release_and_releases_on_failure(self):
        source = cell_source(builder.make_notebook(), "ver20-run")
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            result = root / "saved_result"
            result.mkdir()
            (result / "runtime.json").write_text(json.dumps({"status": "complete_budget_limited", "skipped_candidates": []}), encoding="utf-8")
            (result / "run_complete.json").write_text("{}", encoding="utf-8")
            (result / "mission3.csv").write_text("label file name,symptom\n", encoding="utf-8")
            (result / "training_reference_comparison.json").write_text(json.dumps({
                "available": True, "reference_version": "ver19", "macro_f1_delta": 0.002,
                "nausea_f1_delta": -0.01, "both_macro_and_nausea_improved": False,
                "status": "macro_only_improved", "per_symptom": {"오심": {
                    "paired": {"fn_to_tp": 3, "tp_to_fn": 7, "fp_to_tn": 10, "tn_to_fp": 2}}}}), encoding="utf-8")
            (result / "pilot_report.json").write_text(json.dumps({
                "device": "fixture GPU", "observed_optimizer_steps": 10,
                "estimated_epoch_seconds": 600, "decision": "fit_only_risk"}), encoding="utf-8")
            (result / "mission3_ver20_model.ckpt").write_text(json.dumps({
                "training_scope": "fit_only", "fitted_samples": 12345, "selected_epochs": 1,
                "refit_skipped_reason": "fixture budget limit"}), encoding="utf-8")
            events = []

            def supervised(command, **kwargs):
                self.assertEqual(kwargs["deadline_monotonic"], 10900)
                settings = json.loads((root / "notebook_settings.json").read_text(encoding="utf-8"))
                self.assertEqual(settings["deadline_monotonic"], 10900)
                (root / "notebook_result.json").write_text(json.dumps(str(result)), encoding="utf-8")

            namespace = {"runtime_dir": root, "SETTINGS": dict(builder.default_settings(), deadline_monotonic=10900),
                         "budget_module": types.SimpleNamespace(run_supervised=supervised),
                         "DEADLINE_MONOTONIC": 10900, "BUDGET_STARTED_MONOTONIC": 100,
                         "time": types.SimpleNamespace(monotonic=lambda: 200), "Path": Path, "sys": sys,
                         "release_gpu": lambda: events.append("released"),
                         "print": lambda *args, **kwargs: events.append(" ".join(map(str, args)))}
            exec(source, namespace)
            self.assertEqual(events[-1], "released")
            self.assertTrue(any("최종 작업 완료" in event for event in events[:-1]))
            self.assertTrue(any("complete" in event for event in events[:-1]))
            self.assertTrue(any("fit_only" in event and "12345" in event for event in events[:-1]))
            self.assertTrue(any("fixture budget limit" in event for event in events[:-1]))
            self.assertTrue(any("동시 개선 조건을 충족하지 못했습니다: macro_only_improved" in event for event in events[:-1]))
            self.assertTrue(any("오심 누락 회복: 3" in event and "새 누락: 7" in event for event in events[:-1]))
            self.assertTrue(any("fixture GPU" in event and "fit_only_risk" in event for event in events[:-1]))
            self.assertTrue(any("자동으로 기존 제출 모델을 교체하지 않습니다" in event for event in events[:-1]))
            events.clear()
            (result / "runtime.json").write_text(json.dumps({"status": "complete_not_improved"}), encoding="utf-8")
            exec(source, namespace)
            self.assertEqual(events[-1], "released")
            self.assertTrue(any("실험 완료 · 동시 개선 미달" in event for event in events))
            self.assertTrue(any("새 추천 제출 모델이 아닙니다" in event for event in events))
            self.assertTrue(any("실험용 예측" in event for event in events))
            self.assertFalse(any("최종 작업 완료" in event for event in events))
            events.clear()
            (result / "training_reference_comparison.json").write_text(json.dumps({
                "available": False, "reason": "fixture missing reference"}), encoding="utf-8")
            (result / "run_complete.json").unlink()
            (result / "runtime.json").write_text(json.dumps({"status": "budget_exhausted"}), encoding="utf-8")
            exec(source, namespace)
            self.assertEqual(events[-1], "released")
            self.assertTrue(any("미완료 실행: budget_exhausted" in event for event in events))
            self.assertFalse(any("최종 작업 완료" in event for event in events))
            self.assertTrue(any("기존 Training 결과 비교 불가: fixture missing reference" in event for event in events))
            events.clear()
            namespace["budget_module"] = types.SimpleNamespace(run_supervised=Mock(side_effect=TimeoutError("fixture deadline")))
            with self.assertRaises(TimeoutError):
                exec(source, namespace)
            self.assertEqual(events, ["released"])

    def test_launcher_uses_runtime_and_passes_deadline_unchanged(self):
        # The fixture exercises package restoration and subprocess boundaries, not ML accuracy.
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            for name in builder.SOURCE_FILES:
                (root / name).write_text("", encoding="utf-8")
            shutil.copyfile(HERE / "run_notebook.py", root / "run_notebook.py")
            (root / "requirements.txt").write_text("fixture requirements\n", encoding="utf-8")
            (root / "training_reference.json").write_text('{"fixture": true}\n', encoding="utf-8")
            (root / "runtime.py").write_text(
                "from pathlib import Path\n"
                "def run(settings, runtime_sources=None):\n"
                "    assert settings['deadline_monotonic'] == 123456.25\n"
                "    assert settings['time_budget_minutes'] == 180\n"
                "    assert settings['resume_run'] == 'existing_run'\n"
                "    assert runtime_sources['requirements.txt'] == 'fixture requirements\\n'\n"
                "    assert 'budget.py' in runtime_sources\n"
                "    assert 'reference.py' in runtime_sources\n"
                "    assert runtime_sources['training_reference.json'] == '{\"fixture\": true}\\n'\n"
                "    assert 'runtime_suite.py' not in runtime_sources\n"
                "    return Path(settings['output_dir']) / 'completed_run'\n", encoding="utf-8")
            settings = dict(builder.default_settings(), resume_run="existing_run", output_dir=str(root / "results"),
                            deadline_monotonic=123456.25)
            settings_file, result_file = root / "settings.json", root / "result.json"
            settings_file.write_text(json.dumps(settings), encoding="utf-8-sig")
            completed = subprocess.run(
                [sys.executable, "-B", str(root / "run_notebook.py"), "--settings", str(settings_file),
                 "--result_file", str(result_file)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(result_file.read_text(encoding="utf-8")), str(root / "results" / "completed_run"))
            self.assertFalse((root / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
