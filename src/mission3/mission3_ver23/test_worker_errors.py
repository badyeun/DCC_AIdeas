"""Exercise safe worker failure records using fresh real subprocesses."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ver23_worker_error_fixture", HERE / "run_notebook.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
PRIVATE = "PATIENT_SAMPLE_987654_PRIVATE_DIALOGUE"


class WorkerErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The Windows venv launcher may emit its own startup diagnostic before
        # any Python code runs; the worker must add no raw traceback to it.
        startup = subprocess.run([sys.executable, "-B", "-c", "pass"], capture_output=True,
                                 text=True, encoding="utf-8", timeout=30,
                                 env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if startup.returncode or startup.stdout:
            raise AssertionError("Fixture interpreter startup failed")
        cls.startup_stderr = startup.stderr

    def fixture(self, root, runtime_source):
        for name in launcher.SOURCE_FILES:
            (root / name).write_text("", encoding="utf-8")
        shutil.copyfile(HERE / "run_notebook.py", root / "run_notebook.py")
        (root / "runtime.py").write_text(runtime_source, encoding="utf-8")
        (root / "settings.json").write_text('{"deadline_monotonic":123456.25}', encoding="utf-8")
        (root / "notebook_error.json").write_text(
            json.dumps({"status": "failed", "reason": PRIVATE}), encoding="utf-8")

    def execute(self, root, result_file=None):
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run(
            [sys.executable, "-B", str(root / "run_notebook.py"),
             "--settings", str(root / "settings.json"),
             "--result_file", str(result_file or root / "result.json")],
            capture_output=True, text=True, encoding="utf-8", env=environment, timeout=30)

    def failed_report(self, root, completed, stage, error_type, category):
        self.assertEqual(completed.returncode, 1, completed.stderr)
        saved = (root / "notebook_error.json").read_text(encoding="utf-8")
        report = json.loads(saved)
        self.assertEqual(json.loads(completed.stdout), report)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], stage)
        self.assertEqual(report["error_type"], error_type)
        self.assertEqual(report["category"], category)
        self.assertTrue(report["reason"])
        self.assertTrue(report["frames"])
        for frame in report["frames"]:
            self.assertEqual(set(frame), {"file", "line", "function"})
            self.assertEqual(Path(frame["file"]).name, frame["file"])
            self.assertIsInstance(frame["line"], int)
        self.assertNotIn(PRIVATE, saved + completed.stdout + completed.stderr)
        self.assertNotIn(str(root), saved + completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, self.startup_stderr)
        return report

    def test_import_failure_is_recorded_before_runtime_starts(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, f"raise ImportError({PRIVATE!r})\n")
            report = self.failed_report(root, self.execute(root), "runtime_import", "ImportError",
                                        "dependency_import_failed")
            self.assertEqual(report["frames"][-1]["file"], "runtime.py")
            self.assertFalse((root / "result.json").exists())

    def test_runtime_failure_does_not_echo_exception_values(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, f"def run(settings, runtime_sources=None):\n    raise ValueError({PRIVATE!r})\n")
            report = self.failed_report(root, self.execute(root), "runtime_run", "ValueError", "unclassified")
            self.assertEqual(report["frames"][-1], {"file": "runtime.py", "line": 2, "function": "run"})

    def test_missing_module_name_is_reported_but_import_path_is_not(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, "import missing_ver23_fixture_dependency\n")
            report = self.failed_report(root, self.execute(root), "runtime_import", "ModuleNotFoundError",
                                        "dependency_import_failed")
            self.assertEqual(report["module_name"], "missing_ver23_fixture_dependency")
            self.fixture(root, f"raise ImportError({PRIVATE!r}, name={'private/' + PRIVATE!r}, path={PRIVATE!r})\n")
            report = self.failed_report(root, self.execute(root), "runtime_import", "ImportError",
                                        "dependency_import_failed")
            self.assertNotIn("module_name", report)

    def test_bad_settings_are_recorded_without_contents(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, f"raise RuntimeError({PRIVATE!r})\n")
            (root / "settings.json").write_text('{"private": "' + PRIVATE, encoding="utf-8")
            self.failed_report(root, self.execute(root), "settings_load", "JSONDecodeError", "settings_unreadable")

    def test_missing_embedded_file_is_recorded(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, "def run(settings, runtime_sources=None):\n    return 'unused'\n")
            (root / "README.md").unlink()
            self.failed_report(root, self.execute(root), "source_read", "FileNotFoundError",
                               "runtime_sources_unreadable")

    def test_result_write_failure_is_recorded_without_full_path(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root, "def run(settings, runtime_sources=None):\n    return 'completed_run'\n")
            destination = root / PRIVATE / "result.json"
            report = self.failed_report(root, self.execute(root, destination), "result_write", "FileNotFoundError",
                                        "result_record_failed")
            self.assertEqual(report["errno"], 2)

    def test_known_reasons_are_static_and_do_not_echo_sample_identifiers(self):
        cases = [
            ("Colab에서 GPU 런타임을 선택하세요.", "gpu_unavailable"),
            ("CUDA out of memory", "out_of_memory"),
            ("Input ZIP is missing:", "input_zip_missing"),
            ("Required ver19 result not found.", "ver19_source_missing"),
            ("ver19 source mismatch:", "ver19_source_verification_failed"),
            ("Current fit-only encoder does not reproduce the pinned ver19 holdout predictions", "baseline_prediction_mismatch"),
        ]
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            for message, category in cases:
                with self.subTest(category=category):
                    self.fixture(root, "def run(settings, runtime_sources=None):\n"
                                 f"    raise RuntimeError({(message + ' ' + PRIVATE)!r})\n")
                    self.failed_report(root, self.execute(root), "runtime_run", "RuntimeError", category)

    def test_success_overwrites_stale_failure_and_preserves_result_interface(self):
        with tempfile.TemporaryDirectory(prefix=".worker_errors_", dir=HERE) as directory:
            root = Path(directory)
            self.fixture(root,
                "import json\nfrom pathlib import Path\n"
                "def run(settings, runtime_sources=None):\n"
                "    root = Path(__file__).parent\n"
                "    assert json.loads((root / 'notebook_error.json').read_text()) == {'status': 'starting', 'stage': 'initialize'}\n"
                "    assert settings['deadline_monotonic'] == 123456.25\n"
                "    assert runtime_sources['requirements.txt'] == ''\n"
                "    assert runtime_sources['runtime.py'] == Path(__file__).read_text()\n"
                "    assert 'source_reference.json' in runtime_sources\n"
                "    return root / 'completed_run'\n")
            completed = self.execute(root)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertEqual(json.loads((root / "result.json").read_text()), str(root / "completed_run"))
            self.assertEqual(json.loads((root / "notebook_error.json").read_text()),
                             {"status": "complete", "stage": "complete"})
            self.assertEqual(completed.stderr, self.startup_stderr)
            self.assertEqual(completed.stdout, "")
            self.assertFalse((root / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
