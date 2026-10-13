"""Exercise restoration, budget, worker, and result presentation boundaries."""
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
spec = importlib.util.spec_from_file_location("ver21_notebook_builder", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def cell_source(notebook, identifier):
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == identifier))


def assigned_literal(notebook, identifier, variable):
    for node in ast.parse(cell_source(notebook, identifier)).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing literal assignment: {variable}")


class NotebookTests(unittest.TestCase):
    def prepared_namespace(self, root, *, deadline=10900, now=200):
        prepared = (str(root), deadline)
        return {
            "runtime_dir": root, "SETTINGS": dict(builder.default_settings(), deadline_monotonic=deadline),
            "budget_module": types.SimpleNamespace(run_supervised=Mock()),
            "DEADLINE_MONOTONIC": deadline, "BUDGET_STARTED_MONOTONIC": 100,
            "time": types.SimpleNamespace(monotonic=lambda: now), "Path": Path, "sys": sys,
            "release_gpu": Mock(), "RUNTIME_PREPARED": prepared, "DEPENDENCIES_PREPARED": prepared,
        }

    def test_saved_notebook_embeds_current_sources_and_unchanged_dependencies(self):
        saved = json.loads((HERE.parent / "mission3_ver21.ipynb").read_text(encoding="utf-8"))
        embedded = assigned_literal(saved, "ver21-restore", "RUNTIME_SOURCES")
        self.assertEqual(set(embedded), set(builder.SOURCE_FILES))
        for name in builder.SOURCE_FILES:
            self.assertEqual(embedded[name], (HERE / name).read_text(encoding="utf-8"),
                             f"Regenerate mission3_ver21.ipynb after changing {name}")
        self.assertEqual(embedded["requirements.txt"],
                         (HERE.parent / "mission3_ver19" / "requirements.txt").read_text(encoding="utf-8"))

    def test_defaults_match_head_only_experiment_and_three_hour_cap(self):
        notebook = builder.make_notebook()
        settings = assigned_literal(notebook, "ver21-settings", "SETTINGS")
        self.assertEqual(settings["version"], "ver21")
        self.assertEqual(settings["source_run"], "")
        self.assertEqual(settings["encoder_window_batch"], 4)
        self.assertEqual(settings["max_head_epochs"], 80)
        self.assertEqual(settings["head_eval_every"], 5)
        self.assertEqual(settings["head_learning_rate"], 0.001)
        self.assertEqual(settings["head_weight_decay"], 0.0)
        self.assertEqual(settings["positive_weight"], 3.0)
        self.assertEqual(settings["soft_f1_weight"], 0.5)
        self.assertEqual(settings["time_budget_minutes"], 180)
        self.assertEqual(settings["finalization_reserve_seconds"], 1500)
        self.assertEqual(settings["budget_safety_factor"], 1.25)
        self.assertTrue(settings["refit_full_training"])
        self.assertTrue(settings["run_validation"])
        self.assertFalse(settings["allow_cpu"])
        self.assertNotIn("gradient_accumulation", settings)
        self.assertNotIn("model_id", settings)
        self.assertTrue(assigned_literal(notebook, "ver21-settings", "AUTO_RELEASE_GPU"))
        self.assertNotIn("AUTO_RELEASE_GPU", settings)
        self.assertNotIn("deadline_monotonic", settings)
        prose = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        self.assertIn("입력 ZIP만으로 실행되는 버전이 아닙니다", prose)
        self.assertIn("기존 전체 Training ver19 모델을 그대로 보존한 결과", prose)
        self.assertNotIn("roberta-large", prose)

    def test_all_code_compiles_and_budget_precedes_install(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            target = Path(temporary) / "mission3_ver21.ipynb"
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
        ordered = ["ver21-drive", "ver21-budget", "ver21-restore", "ver21-dependencies", "ver21-run"]
        self.assertEqual(sorted(ordered, key=identifiers.index), ordered)

    def test_shared_deadline_release_idempotence_and_disabled_release(self):
        for enabled in [True, False]:
            namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": enabled}
            timer = Mock()
            colab = types.ModuleType("google.colab")
            colab.runtime = types.SimpleNamespace(unassign=Mock())
            with (patch("time.monotonic", return_value=100), patch("threading.Timer", return_value=timer) as factory,
                  patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(io.StringIO())):
                exec(cell_source(builder.make_notebook(), "ver21-budget"), namespace)
                self.assertEqual(namespace["SETTINGS"]["deadline_monotonic"], 10900)
                namespace["release_gpu"]()
                namespace["release_gpu"]()
            if enabled:
                self.assertEqual(factory.call_args.args[0], 10800)
                self.assertTrue(timer.daemon)
                timer.start.assert_called_once_with()
                colab.runtime.unassign.assert_called_once_with()
                timer.cancel.assert_called_once_with()
            else:
                factory.assert_not_called()
                colab.runtime.unassign.assert_not_called()

    def test_release_failure_is_visible_with_manual_recovery(self):
        namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": True}
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=Mock(side_effect=RuntimeError("fixture failure")))
        output = io.StringIO()
        with (patch("time.monotonic", return_value=100), patch("threading.Timer"),
              patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(output)):
            exec(cell_source(builder.make_notebook(), "ver21-budget"), namespace)
            namespace["release_gpu"]()
        self.assertIn("자동 반환 요청 실패", output.getvalue())
        self.assertIn("연결 해제 및 런타임 삭제", output.getvalue())

    def test_restoration_preserves_reference_bytes_and_reports_failure(self):
        notebook = builder.make_notebook()
        sources = assigned_literal(notebook, "ver21-restore", "RUNTIME_SOURCES")
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            release = Mock()
            namespace = {"release_gpu": release, "DEADLINE_MONOTONIC": 10900,
                         "time": types.SimpleNamespace(monotonic=lambda: 100)}
            with patch("tempfile.mkdtemp", return_value=str(root)), contextlib.redirect_stdout(io.StringIO()):
                exec(cell_source(notebook, "ver21-restore"), namespace)
            for name, source in sources.items():
                self.assertEqual((root / name).read_bytes(), source.encode("utf-8"))
            self.assertEqual((root / "source_reference.json").read_bytes(),
                             (HERE / "source_reference.json").read_bytes())
            self.assertEqual(namespace["RUNTIME_PREPARED"], (str(root), 10900))
            release.assert_not_called()
            namespace["DEPENDENCIES_PREPARED"] = (str(root), 10900)
            with patch("tempfile.mkdtemp", side_effect=OSError("fixture unwritable directory")):
                with self.assertRaises(OSError):
                    exec(cell_source(notebook, "ver21-restore"), namespace)
            self.assertIsNone(namespace["RUNTIME_PREPARED"])
            self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
            release.assert_called_once_with()

    def test_install_is_bounded_by_remaining_budget_and_twenty_minutes(self):
        source = cell_source(builder.make_notebook(), "ver21-dependencies")
        for deadline, expected in [(10000, 1300), (500, 500)]:
            supervisor, release = Mock(), Mock()
            namespace = {"DEADLINE_MONOTONIC": deadline, "time": types.SimpleNamespace(monotonic=lambda: 100),
                         "runtime_dir": HERE, "sys": sys, "release_gpu": release,
                         "budget_module": types.SimpleNamespace(run_supervised=supervisor),
                         "RUNTIME_PREPARED": (str(HERE), deadline)}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            self.assertEqual(supervisor.call_args.kwargs["deadline_monotonic"], expected)
            self.assertEqual(supervisor.call_args.kwargs["shutdown_margin_s"], 10)
            self.assertEqual(supervisor.call_args.args[0][-1], str(HERE / "requirements.txt"))
            self.assertEqual(namespace["DEPENDENCIES_PREPARED"], (str(HERE), deadline))
            release.assert_not_called()
        supervisor.side_effect = subprocess.CalledProcessError(1, "fixture installer")
        with self.assertRaises(subprocess.CalledProcessError):
            exec(source, namespace)
        self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
        release.assert_called_once_with()

    def test_result_status_never_presents_fallback_as_improvement(self):
        source = cell_source(builder.make_notebook(), "ver21-run")
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("# fixture worker\n", encoding="utf-8")
            result = root / "saved_result"
            result.mkdir()
            (result / "run_complete.json").write_text("{}", encoding="utf-8")
            (result / "mission3.csv").write_text("label file name,symptom\n", encoding="utf-8")
            (result / "training_reference_comparison.json").write_text(json.dumps({
                "available": True, "reference_version": "ver19", "macro_f1_delta": 0.002,
                "nausea_f1_delta": 0.018, "both_macro_and_nausea_improved": True,
                "per_symptom": {"오심": {"paired": {
                    "fn_to_tp": 3, "tp_to_fn": 1, "fp_to_tn": 10, "tn_to_fp": 2}}}}), encoding="utf-8")
            (result / "mission3_ver21_model.ckpt").write_text(json.dumps({
                "training_scope": "all_training", "fitted_samples": 29200, "selected_epochs": 3, "head_epochs": 5,
                "refit_skipped_reason": "fixture budget limit"}), encoding="utf-8")
            events = []

            def supervised(command, **kwargs):
                self.assertEqual(kwargs["deadline_monotonic"], 10900)
                self.assertEqual(json.loads((root / "notebook_settings.json").read_text(encoding="utf-8"))[
                    "deadline_monotonic"], 10900)
                (root / "notebook_result.json").write_text(json.dumps(str(result)), encoding="utf-8")

            namespace = {"runtime_dir": root, "SETTINGS": dict(builder.default_settings(), deadline_monotonic=10900),
                         "budget_module": types.SimpleNamespace(run_supervised=supervised),
                         "DEADLINE_MONOTONIC": 10900, "BUDGET_STARTED_MONOTONIC": 100,
                         "time": types.SimpleNamespace(monotonic=lambda: 200), "Path": Path, "sys": sys,
                         "release_gpu": lambda: events.append("released"),
                         "RUNTIME_PREPARED": (str(root), 10900), "DEPENDENCIES_PREPARED": (str(root), 10900),
                         "print": lambda *args, **kwargs: events.append(" ".join(map(str, args)))}
            for status in ["complete", "complete_not_improved", "complete_budget_limited", "budget_exhausted"]:
                events.clear()
                (result / "runtime.json").write_text(json.dumps({"status": status}), encoding="utf-8")
                exec(source, namespace)
                self.assertEqual(events[-1], "released")
                self.assertTrue(any("오심 누락 회복: 3" in event for event in events))
                self.assertTrue(any("오심 판단층 갱신 횟수: 5" in event for event in events))
                if status == "complete":
                    self.assertTrue(any("새 오심 판단층을 적용한 예측" in event for event in events))
                elif status in {"complete_not_improved", "complete_budget_limited"}:
                    self.assertTrue(any("성능 향상 결과가 아닙니다" in event for event in events))
                    self.assertTrue(any("ver19 모델을" in event for event in events))
                    self.assertFalse(any("최종 작업 완료" in event for event in events))
                else:
                    self.assertTrue(any("미완료 실행" in event for event in events))
            events.clear()
            (result / "runtime.json").write_text('{"status":"complete"}', encoding="utf-8")
            (result / "run_complete.json").unlink()
            exec(source, namespace)
            self.assertTrue(any("미완료 실행" in event for event in events))
            self.assertFalse(any("최종 작업 완료" in event for event in events))
            events.clear()
            namespace["budget_module"] = types.SimpleNamespace(run_supervised=Mock(side_effect=TimeoutError("fixture deadline")))
            with self.assertRaises(TimeoutError):
                exec(source, namespace)
            self.assertEqual(events, ["released"])

    def test_unprepared_cells_report_recovery_without_starting_work_or_releasing_gpu(self):
        notebook = builder.make_notebook()
        for identifier in ["ver21-budget", "ver21-restore", "ver21-dependencies", "ver21-run"]:
            for with_cleanup in [False, True]:
                with self.subTest(cell=identifier, with_cleanup=with_cleanup):
                    cleanup = Mock()
                    namespace = {"release_gpu": cleanup} if with_cleanup else {}
                    with (patch("threading.Timer") as timer, patch("tempfile.mkdtemp") as make_directory,
                          patch("subprocess.Popen") as process, self.assertRaises(RuntimeError) as caught):
                        exec(cell_source(notebook, identifier), namespace)
                    self.assertIn("모두 실행", str(caught.exception))
                    self.assertNotIsInstance(caught.exception, NameError)
                    self.assertIsNone(caught.exception.__context__)
                    cleanup.assert_not_called()
                    timer.assert_not_called()
                    make_directory.assert_not_called()
                    process.assert_not_called()

    def test_install_failure_after_success_blocks_run_and_preserves_original_error(self):
        notebook = builder.make_notebook()
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            for name in ["requirements.txt", "run_notebook.py"]:
                (root / name).write_text("", encoding="utf-8")
            namespace = self.prepared_namespace(root)
            supervisor = namespace["budget_module"].run_supervised
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell_source(notebook, "ver21-dependencies"), namespace)
            self.assertEqual(namespace["DEPENDENCIES_PREPARED"], (str(root), 10900))
            failure = subprocess.CalledProcessError(1, "fixture installer")
            supervisor.side_effect = failure
            namespace["release_gpu"].side_effect = RuntimeError("fixture cleanup failure")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as caught:
                exec(cell_source(notebook, "ver21-dependencies"), namespace)
            self.assertIs(caught.exception, failure)
            self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
            supervisor.reset_mock()
            namespace["release_gpu"].reset_mock()
            with self.assertRaisesRegex(RuntimeError, "패키지 준비가 미완료"):
                exec(cell_source(notebook, "ver21-run"), namespace)
            supervisor.assert_not_called()
            namespace["release_gpu"].assert_not_called()
            self.assertFalse((root / "notebook_settings.json").exists())

    def test_missing_runtime_files_and_stale_preparation_block_work(self):
        notebook = builder.make_notebook()
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            for identifier in ["ver21-dependencies", "ver21-run"]:
                with self.subTest(cell=identifier, missing_files=True):
                    namespace = self.prepared_namespace(root)
                    with self.assertRaisesRegex(RuntimeError, "미완료"):
                        exec(cell_source(notebook, identifier), namespace)
                    namespace["budget_module"].run_supervised.assert_not_called()
                    namespace["release_gpu"].assert_not_called()
            for name in ["requirements.txt", "run_notebook.py"]:
                (root / name).write_text("", encoding="utf-8")
            for flag in ["RUNTIME_PREPARED", "DEPENDENCIES_PREPARED"]:
                for stale_context in [(str(root / "old_runtime"), 10900), (str(root), 10800)]:
                    with self.subTest(flag=flag, stale_context=stale_context):
                        namespace = self.prepared_namespace(root)
                        namespace[flag] = stale_context
                        with self.assertRaisesRegex(RuntimeError, "미완료"):
                            exec(cell_source(notebook, "ver21-run"), namespace)
                        namespace["budget_module"].run_supervised.assert_not_called()
                        namespace["release_gpu"].assert_not_called()
            self.assertFalse((root / "notebook_settings.json").exists())

    def test_expired_budget_stops_setup_and_run_without_supervisor(self):
        notebook = builder.make_notebook()
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            for name in ["requirements.txt", "run_notebook.py"]:
                (root / name).write_text("", encoding="utf-8")
            for identifier in ["ver21-restore", "ver21-dependencies", "ver21-run"]:
                with self.subTest(cell=identifier):
                    namespace = self.prepared_namespace(root, deadline=200, now=200)
                    with patch("tempfile.mkdtemp") as make_directory, self.assertRaisesRegex(RuntimeError, "시간 예산이 끝났"):
                        exec(cell_source(notebook, identifier), namespace)
                    make_directory.assert_not_called()
                    namespace["budget_module"].run_supervised.assert_not_called()
                    namespace["release_gpu"].assert_called_once_with()
                    self.assertEqual(namespace["DEADLINE_MONOTONIC"], 200)
            self.assertFalse((root / "notebook_settings.json").exists())

    def test_restore_and_run_cleanup_failures_do_not_replace_original_errors(self):
        notebook = builder.make_notebook()
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("", encoding="utf-8")
            for identifier in ["ver21-restore", "ver21-run"]:
                with self.subTest(cell=identifier):
                    namespace = self.prepared_namespace(root)
                    original = OSError("fixture primary failure")
                    namespace["release_gpu"].side_effect = RuntimeError("fixture cleanup failure")
                    namespace["budget_module"].run_supervised.side_effect = original
                    with (patch("tempfile.mkdtemp", side_effect=original),
                          contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError) as caught):
                        exec(cell_source(notebook, identifier), namespace)
                    self.assertIs(caught.exception, original)
                    namespace["release_gpu"].assert_called_once_with()

    def test_ordered_preparation_in_fresh_namespace_reaches_worker(self):
        notebook = builder.make_notebook()
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=Mock())
        timer = Mock()
        sentinel = RuntimeError("fixture worker reached")
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            namespace = {}
            with (patch("time.monotonic", return_value=100), patch("threading.Timer", return_value=timer),
                  patch("tempfile.mkdtemp", return_value=str(root)), patch("subprocess.Popen") as process,
                  patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(io.StringIO())):
                for identifier in ["ver21-settings", "ver21-budget", "ver21-restore"]:
                    exec(cell_source(notebook, identifier), namespace)
                self.assertEqual(namespace["RUNTIME_PREPARED"], (str(root), 10900))
                self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
                with patch.object(namespace["budget_module"], "run_supervised", side_effect=[0, sentinel]) as supervisor:
                    exec(cell_source(notebook, "ver21-dependencies"), namespace)
                    self.assertEqual(namespace["DEPENDENCIES_PREPARED"], (str(root), 10900))
                    with self.assertRaises(RuntimeError) as caught:
                        exec(cell_source(notebook, "ver21-run"), namespace)
                self.assertIs(caught.exception, sentinel)
                self.assertEqual(supervisor.call_count, 2)
                install, worker = supervisor.call_args_list
                self.assertEqual(install.args[0][1:4], ["-m", "pip", "install"])
                self.assertEqual(install.kwargs["deadline_monotonic"], 1300)
                self.assertEqual(worker.args[0][2], str(root / "run_notebook.py"))
                self.assertEqual(worker.kwargs["deadline_monotonic"], 10900)
                self.assertEqual(json.loads((root / "notebook_settings.json").read_text(encoding="utf-8"))[
                    "deadline_monotonic"], 10900)
                process.assert_not_called()
                colab.runtime.unassign.assert_called_once_with()
                timer.start.assert_called_once_with()
                timer.cancel.assert_called_once_with()

    def test_launcher_passes_sources_and_deadline_to_fresh_runtime(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            for name in builder.SOURCE_FILES:
                (root / name).write_text("", encoding="utf-8")
            shutil.copyfile(HERE / "run_notebook.py", root / "run_notebook.py")
            (root / "requirements.txt").write_text("fixture requirements\n", encoding="utf-8")
            (root / "source_reference.json").write_text('{"fixture": true}\n', encoding="utf-8")
            (root / "runtime.py").write_text(
                "from pathlib import Path\n"
                "def run(settings, runtime_sources=None):\n"
                "    assert settings['deadline_monotonic'] == 123456.25\n"
                "    assert settings['time_budget_minutes'] == 180\n"
                "    assert settings['source_run'] == 'original_ver19_run'\n"
                "    assert settings['resume_run'] == 'existing_run'\n"
                "    assert runtime_sources['requirements.txt'] == 'fixture requirements\\n'\n"
                "    assert 'budget.py' in runtime_sources and 'head.py' in runtime_sources\n"
                "    assert 'source_bundle.py' in runtime_sources\n"
                "    assert runtime_sources['source_reference.json'] == '{\"fixture\": true}\\n'\n"
                "    return Path(settings['output_dir']) / 'completed_run'\n", encoding="utf-8")
            settings = dict(builder.default_settings(), source_run="original_ver19_run", resume_run="existing_run",
                            output_dir=str(root / "results"), deadline_monotonic=123456.25)
            settings_file, result_file = root / "settings.json", root / "result.json"
            settings_file.write_text(json.dumps(settings), encoding="utf-8-sig")
            completed = subprocess.run(
                [sys.executable, "-B", str(root / "run_notebook.py"), "--settings", str(settings_file),
                 "--result_file", str(result_file)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(result_file.read_text(encoding="utf-8")), str(root / "results" / "completed_run"))
            self.assertFalse((root / "__pycache__").exists())

    def test_worker_report_is_displayed_and_saved_before_gpu_release(self):
        notebook = builder.make_notebook()
        report = {"status": "failed", "stage": "runtime_import", "error_type": "ModuleNotFoundError",
                  "category": "dependency_import", "reason": "필요한 패키지를 불러오지 못했습니다.",
                  "frames": [{"file": "runtime.py", "line": 18, "function": "<module>"}]}
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("", encoding="utf-8")
            namespace = self.prepared_namespace(root)
            namespace["SETTINGS"]["output_dir"] = str(root / "results")
            events = []
            namespace["print"] = lambda *args, **kwargs: events.append(" ".join(map(str, args)))

            def fail_worker(*args, **kwargs):
                self.assertEqual(json.loads((root / "notebook_error.json").read_text())["status"], "starting")
                (root / "notebook_error.json").write_text(json.dumps(report), encoding="utf-8")
                raise subprocess.CalledProcessError(1, ["fixture"])

            def release():
                paths = list((root / "results/_diagnostics").glob("ver21_error_*.json"))
                self.assertEqual(len(paths), 1)
                self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8"))["reason"], report["reason"])
                self.assertTrue(any("ModuleNotFoundError" in event for event in events))
                events.append("released")

            namespace["release_gpu"] = release
            namespace["budget_module"].run_supervised.side_effect = fail_worker
            with self.assertRaisesRegex(RuntimeError, "runtime_import / ModuleNotFoundError") as caught:
                exec(cell_source(notebook, "ver21-run"), namespace)
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertEqual(events[-1], "released")
            self.assertTrue(any("runtime.py" in event for event in events))

    def test_worker_without_report_does_not_reuse_stale_failure_or_result(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("", encoding="utf-8")
            (root / "notebook_error.json").write_text(json.dumps({"status": "failed", "reason": "STALE_REASON"}))
            (root / "notebook_result.json").write_text('"stale_success"')
            namespace = self.prepared_namespace(root)
            namespace["SETTINGS"]["output_dir"] = str(root / "results")
            namespace["budget_module"].run_supervised.side_effect = subprocess.CalledProcessError(7, ["fixture"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "WorkerProcessError"):
                exec(cell_source(builder.make_notebook(), "ver21-run"), namespace)
            self.assertNotIn("STALE_REASON", output.getvalue())
            saved = next((root / "results/_diagnostics").glob("*.json"))
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["exit_code"], 7)
            namespace["release_gpu"].assert_called_once_with()

    def test_drive_diagnostic_write_failure_does_not_hide_worker_error(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("", encoding="utf-8")
            namespace = self.prepared_namespace(root)
            namespace["SETTINGS"]["output_dir"] = str(root / "results")
            namespace["budget_module"].run_supervised.side_effect = subprocess.CalledProcessError(1, ["fixture"])
            output = io.StringIO()
            with (patch.object(Path, "mkdir", side_effect=OSError("fixture Drive unavailable")),
                  contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "WorkerProcessError")):
                exec(cell_source(builder.make_notebook(), "ver21-run"), namespace)
            self.assertIn("Drive 오류 보고서 저장 실패", output.getvalue())
            namespace["release_gpu"].assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
