"""Check embedded notebook, setup guards, diagnostics, and worker boundaries."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("ver24_notebook_builder", "build_notebook.py")
worker = load_module("ver24_notebook_worker", "run_notebook.py")
budget = load_module("ver24_notebook_budget", "budget.py")


def cell_source(notebook, suffix):
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == "ver24-" + suffix))


def assigned_literal(notebook, suffix, variable):
    for node in ast.parse(cell_source(notebook, suffix)).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("Missing literal assignment: " + variable)


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = builder.make_notebook()

    def prepared_namespace(self, root, *, deadline=10900, now=200):
        prepared = (str(root), deadline)
        return {
            "runtime_dir": root,
            "SETTINGS": dict(builder.default_settings(), deadline_monotonic=deadline, output_dir=str(root / "out")),
            "budget_module": types.SimpleNamespace(run_supervised=Mock(), BudgetExceeded=budget.BudgetExceeded),
            "DEADLINE_MONOTONIC": deadline, "BUDGET_STARTED_MONOTONIC": 100,
            "time": types.SimpleNamespace(monotonic=lambda: now), "Path": Path, "sys": sys,
            "release_gpu": Mock(), "RUNTIME_PREPARED": prepared, "DEPENDENCIES_PREPARED": prepared,
        }

    def test_saved_notebook_embeds_current_sources_and_unchanged_dependencies(self):
        saved_path = HERE.parent / "mission3_ver24.ipynb"
        self.assertTrue(saved_path.is_file(), "Build mission3_ver24.ipynb before final verification")
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        embedded = assigned_literal(saved, "restore", "RUNTIME_SOURCES")
        self.assertEqual(set(embedded), set(builder.SOURCE_FILES))
        self.assertEqual(builder.SOURCE_FILES, worker.SOURCE_FILES)
        for name in builder.SOURCE_FILES:
            self.assertEqual(embedded[name], (HERE / name).read_text(encoding="utf-8"),
                             "Regenerate mission3_ver24.ipynb after changing " + name)
        self.assertEqual(embedded["requirements.txt"],
                         (HERE.parent / "mission3_ver23" / "requirements.txt").read_text(encoding="utf-8"))

    def test_default_settings_match_comparison_and_three_hour_cap(self):
        settings = assigned_literal(self.notebook, "settings", "SETTINGS")
        self.assertEqual(settings, builder.default_settings())
        expected = {"version": "ver24", "seed": 42, "holdout_size": .2, "batch_size": 8,
                    "max_length": 512, "gradient_accumulation": 2, "max_epochs": 3,
                    "learning_rate": 2e-5, "weight_decay": .01, "dropout": .1,
                    "warmup_ratio": .1, "pilot_optimizer_steps": 10,
                    "time_budget_minutes": 180, "finalization_reserve_seconds": 1200,
                    "budget_safety_factor": 1.35, "allow_cpu": False, "run_validation": True}
        for key, value in expected.items():
            self.assertEqual(settings[key], value)
        for key in ("source_run", "resume_run", "threshold", "deadline_monotonic"):
            self.assertNotIn(key, settings)
        self.assertTrue(assigned_literal(self.notebook, "settings", "AUTO_RELEASE_GPU"))

    def test_all_code_compiles_and_budget_precedes_install(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            target = builder.build(Path(temporary) / "fixture.ipynb")
            notebook = json.loads(target.read_text(encoding="utf-8"))
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        ordered = ["ver24-drive", "ver24-budget", "ver24-restore", "ver24-dependencies", "ver24-run"]
        self.assertEqual(sorted(ordered, key=identifiers.index), ordered)

    def test_shared_deadline_and_idempotent_release(self):
        for enabled in (True, False):
            namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": enabled}
            timer = Mock()
            colab = types.ModuleType("google.colab")
            colab.runtime = types.SimpleNamespace(unassign=Mock())
            with (patch("time.monotonic", return_value=100), patch("threading.Timer", return_value=timer) as factory,
                  patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(io.StringIO())):
                exec(cell_source(self.notebook, "budget"), namespace)
                namespace["release_gpu"]()
                namespace["release_gpu"]()
            self.assertEqual(namespace["SETTINGS"]["deadline_monotonic"], 10900)
            if enabled:
                self.assertEqual(factory.call_args.args[0], 10800)
                self.assertTrue(timer.daemon)
                timer.start.assert_called_once_with()
                timer.cancel.assert_called_once_with()
                colab.runtime.unassign.assert_called_once_with()
            else:
                factory.assert_not_called()
                colab.runtime.unassign.assert_not_called()

    def test_release_failure_explains_manual_recovery(self):
        namespace = {"SETTINGS": builder.default_settings(), "AUTO_RELEASE_GPU": True}
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=Mock(side_effect=RuntimeError("SECRET")))
        output = io.StringIO()
        with (patch("time.monotonic", return_value=100), patch("threading.Timer"),
              patch.dict(sys.modules, {"google.colab": colab}), contextlib.redirect_stdout(output)):
            exec(cell_source(self.notebook, "budget"), namespace)
            namespace["release_gpu"]()
        self.assertIn("자동 반환 요청 실패", output.getvalue())
        self.assertIn("연결 해제 및 런타임 삭제", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())

    def test_restore_preserves_bytes_and_resets_stale_install_on_failure(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            namespace = {"release_gpu": Mock(), "DEADLINE_MONOTONIC": 10900,
                         "time": types.SimpleNamespace(monotonic=lambda: 100)}
            with patch("tempfile.mkdtemp", return_value=str(root)), contextlib.redirect_stdout(io.StringIO()):
                exec(cell_source(self.notebook, "restore"), namespace)
            for name, source in assigned_literal(self.notebook, "restore", "RUNTIME_SOURCES").items():
                self.assertEqual((root / name).read_bytes(), source.encode("utf-8"))
            self.assertEqual(namespace["RUNTIME_PREPARED"], (str(root), 10900))
            namespace["DEPENDENCIES_PREPARED"] = (str(root), 10900)
            with patch("tempfile.mkdtemp", side_effect=OSError("fixture")), self.assertRaises(OSError):
                exec(cell_source(self.notebook, "restore"), namespace)
            self.assertIsNone(namespace["RUNTIME_PREPARED"])
            self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
            namespace["release_gpu"].assert_called_once_with()

    def test_missing_or_stale_preparation_has_actionable_guard(self):
        for suffix in ("restore", "dependencies", "run"):
            with self.subTest(cell=suffix), self.assertRaisesRegex(RuntimeError, "처음.*실행"):
                exec(cell_source(self.notebook, suffix), {})
        namespace = self.prepared_namespace(HERE)
        namespace["DEPENDENCIES_PREPARED"] = (str(HERE), 999)
        with self.assertRaisesRegex(RuntimeError, "준비가 미완료"):
            exec(cell_source(self.notebook, "run"), namespace)
        namespace["budget_module"].run_supervised.assert_not_called()

    def test_install_is_bounded_and_failure_invalidates_preparation(self):
        for deadline, expected in ((10000, 1300), (500, 500)):
            namespace = self.prepared_namespace(HERE, deadline=deadline, now=100)
            supervisor = namespace["budget_module"].run_supervised
            with contextlib.redirect_stdout(io.StringIO()):
                exec(cell_source(self.notebook, "dependencies"), namespace)
            self.assertEqual(supervisor.call_args.kwargs["deadline_monotonic"], expected)
            self.assertEqual(supervisor.call_args.kwargs["shutdown_margin_s"], 10)
            self.assertEqual(supervisor.call_args.args[0][:3], [sys.executable, "-m", "pip"])
            self.assertEqual(namespace["DEPENDENCIES_PREPARED"], (str(HERE), deadline))
            supervisor.side_effect = subprocess.CalledProcessError(1, "fixture")
            with self.assertRaises(subprocess.CalledProcessError):
                exec(cell_source(self.notebook, "dependencies"), namespace)
            self.assertIsNone(namespace["DEPENDENCIES_PREPARED"])
            namespace["release_gpu"].assert_called_once_with()

    def test_completed_result_requires_final_artifacts(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("# fixture\n", encoding="utf-8")
            result = root / "result"
            result.mkdir()
            for filename in ("run_complete.json", "mission3_ver24_model.ckpt"):
                (result / filename).write_text("{}", encoding="utf-8")
            (result / "runtime.json").write_text('{"status":"complete"}', encoding="utf-8")
            (result / "mission3.csv").write_text("label file name,symptom\n", encoding="utf-8")
            (result / "final_model").mkdir()
            namespace = self.prepared_namespace(root)
            namespace["budget_module"].run_supervised.side_effect = lambda *args, **kwargs: (
                root / "notebook_result.json").write_text(json.dumps(str(result)), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source(self.notebook, "run"), namespace)
            self.assertIn("최종 작업 완료", output.getvalue())
            namespace["release_gpu"].assert_called_once_with()
            (result / "run_complete.json").unlink()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source(self.notebook, "run"), namespace)
            self.assertIn("미완료 실행", output.getvalue())
            self.assertNotIn("최종 작업 완료", output.getvalue())
            (result / "run_complete.json").write_text("{}", encoding="utf-8")
            (result / "mission3.csv").unlink()
            namespace["SETTINGS"]["run_validation"] = False
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(cell_source(self.notebook, "run"), namespace)
            self.assertIn("제출 CSV는 만들지 않았습니다", output.getvalue())
            self.assertNotIn("미완료 실행", output.getvalue())

    def test_worker_failure_is_reported_and_saved_before_release(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("# fixture\n", encoding="utf-8")
            namespace = self.prepared_namespace(root)
            events = []
            namespace["print"] = lambda *args, **kwargs: events.append(" ".join(map(str, args)))
            namespace["release_gpu"] = lambda: events.append("released")

            def fail(command, **kwargs):
                (root / "notebook_error.json").write_text(json.dumps({
                    "status": "failed", "stage": "runtime_import", "error_type": "ImportError",
                    "category": "dependency_import_failed", "reason": "fixture safe reason",
                    "module_name": "fixture_module", "frames": [{"file": "trainer.py", "line": 22,
                                                                 "function": "load_encoder"}],
                }), encoding="utf-8")
                raise subprocess.CalledProcessError(1, ["SECRET"])

            namespace["budget_module"].run_supervised.side_effect = fail
            with self.assertRaisesRegex(RuntimeError, "fixture safe reason") as error:
                exec(cell_source(self.notebook, "run"), namespace)
            self.assertNotIn("SECRET", str(error.exception))
            self.assertEqual(events[-1], "released")
            self.assertTrue(any("runtime_import" in item for item in events[:-1]))
            self.assertTrue(any("trainer.py" in item for item in events[:-1]))
            reports = list((root / "out" / "_diagnostics").glob("ver24_error_*.json"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8"))["stage"], "runtime_import")

    def test_supervisor_timeout_diagnostic_precedes_release(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text("# fixture\n", encoding="utf-8")
            namespace = self.prepared_namespace(root)
            namespace["budget_module"].run_supervised.side_effect = budget.BudgetExceeded("SECRET")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "작업 예산"):
                exec(cell_source(self.notebook, "run"), namespace)
            self.assertNotIn("SECRET", output.getvalue())
            reports = list((root / "out" / "_diagnostics").glob("*.json"))
            self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8"))["category"], "time_budget_exhausted")
            namespace["release_gpu"].assert_called_once_with()


class WorkerTests(unittest.TestCase):
    def test_failure_report_never_copies_exception_payload(self):
        for error, category in ((ValueError("SECRET patient payload /private/file.json"), "unclassified"),
                                (RuntimeError("CUDA out of memory SECRET"), "out_of_memory"),
                                (RuntimeError("Colab에서 GPU 런타임을 선택하세요 SECRET"), "gpu_unavailable"),
                                (FileNotFoundError("Input ZIP is missing: SECRET"), "input_zip_missing"),
                                (budget.BudgetExceeded("SECRET"), "time_budget_exhausted")):
            try:
                raise error
            except Exception as caught:
                report = worker._failure_report(caught, "runtime_run")
            self.assertEqual(report["category"], category)
            self.assertNotIn("SECRET", json.dumps(report))
            self.assertEqual(report["frames"][-1]["file"], "test_notebook.py")

    def test_import_module_name_is_validated(self):
        safe = worker._failure_report(ModuleNotFoundError("SECRET", name="numpy.core"), "runtime_import")
        self.assertEqual(safe["module_name"], "numpy.core")
        unsafe = worker._failure_report(ModuleNotFoundError("SECRET", name="/private/file.json"), "runtime_import")
        self.assertNotIn("module_name", unsafe)

    def test_worker_invalid_settings_creates_sanitized_report(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text((HERE / "run_notebook.py").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "settings.json").write_text("SECRET invalid JSON", encoding="utf-8")
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(root / "run_notebook.py"),
                                     "--settings", str(root / "settings.json"),
                                     "--result_file", str(root / "result.json")],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("SECRET", result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            report = json.loads((root / "notebook_error.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stage"], "settings_load")
            self.assertEqual(report["category"], "settings_unreadable")
            self.assertFalse((root / "result.json").exists())

    def test_worker_import_failure_is_recorded_without_traceback_payload(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            (root / "run_notebook.py").write_text((HERE / "run_notebook.py").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "settings.json").write_text("{}", encoding="utf-8")
            (root / "__init__.py").write_text("# fixture\n", encoding="utf-8")
            (root / "runtime.py").write_text('raise ImportError("SECRET")\n', encoding="utf-8")
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(root / "run_notebook.py"),
                                     "--settings", str(root / "settings.json"),
                                     "--result_file", str(root / "result.json")],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("SECRET", result.stdout + result.stderr)
            report = json.loads((root / "notebook_error.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stage"], "runtime_import")
            self.assertEqual(report["error_type"], "ImportError")


if __name__ == "__main__":
    unittest.main()
