"""CPU-only checks for Modal persistence, supervision and notebook delivery."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from . import build_notebook, support

# Match the standalone notebook's top-level package layout without importing ML.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission3_ver29.budget import BudgetExceeded
from mission3_ver29.settings import default_settings, recipe_settings


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".modal_test_", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.volume = self.root / "volume"
        self.volume.mkdir()
        self.bundle = self.volume / "mission3_ver29_inputs.zip"
        self.bundle.write_bytes(b"test bundle")
        self.digest = support.file_hash(self.bundle)

    def make_run(self, name="run-1", status="paused", mode="screening"):
        run = self.volume / "results/mission3/ver29" / name
        run.mkdir(parents=True)
        support.write_json(run / "run_state.json", {"status": status, "run_mode": mode})
        return run

    def prepare(self, **updates):
        settings = default_settings()
        settings.update(updates)
        return support.prepare(settings, self.volume, self.root / "scratch", expected_sha=self.digest)

    def test_missing_volume_is_not_created(self):
        path = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            support.volume_path(path)
        self.assertFalse(path.exists())

    def test_writable_path_rejects_escape_or_link_components(self):
        with self.assertRaises(ValueError):
            support.writable_dir(self.volume, "../outside")
        with patch.object(Path, "is_symlink", lambda p: p.name == "results"):
            with self.assertRaises(ValueError):
                support.writable_dir(self.volume, "results/mission3/ver29")
        self.assertFalse((self.volume / "results").exists())

    def test_missing_or_changed_input_fails_before_records(self):
        with self.assertRaises(ValueError):
            support.input_path(self.volume)
        self.bundle.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse((self.volume / "invocations").exists())

    def test_input_outside_volume_is_rejected(self):
        outside = self.root / "outside.zip"
        outside.write_bytes(b"test bundle")
        with self.assertRaises(FileNotFoundError):
            support.input_path(self.volume, str(outside), self.digest)

    def test_prepare_preserves_recipe_and_separates_persistent_paths(self):
        settings, invocation = self.prepare()
        self.assertEqual(recipe_settings(settings), recipe_settings(default_settings()))
        for key in ("output_dir", "cache_dir", "input_zip"):
            self.assertTrue(Path(settings[key]).is_relative_to(self.volume))
        self.assertFalse(Path(settings["work_dir"]).is_relative_to(self.volume))
        self.assertEqual(Path(settings["input_zip"]), self.bundle)
        self.assertFalse(Path(settings["output_dir"]).is_relative_to(invocation))
        self.assertEqual(json.loads((invocation / "settings.json").read_text()), settings)

    def test_invocations_are_unique_and_previous_records_preserved(self):
        _, first = self.prepare()
        original = (first / "settings.json").read_bytes()
        _, second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual((first / "settings.json").read_bytes(), original)

    def test_full_requires_completed_screening_exact_path(self):
        with self.assertRaises(ValueError):
            self.prepare(run_mode="full")
        run = self.make_run(status="screening_complete")
        settings, _ = self.prepare(run_mode="full", source_run=str(run))
        self.assertTrue(settings["run_validation"])
        self.assertEqual(settings["source_run"], str(run))
        with self.assertRaises(ValueError):
            self.prepare(run_mode="full", source_run=str(run.parent))

    def test_resume_requires_same_phase_and_paused(self):
        run = self.make_run(mode="full")
        with self.assertRaises(ValueError):
            self.prepare(resume_run=str(run))
        settings, _ = self.prepare(run_mode="full", resume_run=str(run))
        self.assertEqual(settings["source_run"], "")
        self.assertTrue(settings["run_validation"])
        support.write_json(run / "run_state.json", {"status": "interrupted", "run_mode": "full"})
        with self.assertRaises(ValueError):
            self.prepare(run_mode="full", resume_run=str(run))

    def test_cannot_combine_full_source_and_resume(self):
        source = self.make_run("screen", status="screening_complete")
        resume = self.make_run("full", mode="full")
        with self.assertRaises(ValueError):
            self.prepare(run_mode="full", source_run=str(source), resume_run=str(resume))

    def test_uv_preserves_exact_cuda_torch_constraint(self):
        constraint = self.root / "constraints.txt"
        with patch.object(support.shutil, "which", return_value="/usr/local/bin/uv"):
            command = support.install_command("requirements.txt", constraint, "2.8.0+cu128", "/python")
        self.assertEqual(command[:5], ["/usr/local/bin/uv", "pip", "install", "--python", "/python"])
        self.assertEqual(constraint.read_text(), "torch==2.8.0+cu128\n")
        self.assertIn(str(constraint), command)

    def test_installer_pip_fallback_and_unsupported_torch(self):
        with patch.object(support.shutil, "which", return_value=None), \
                patch.object(support.importlib.util, "find_spec", return_value=object()):
            command = support.install_command("req.txt", self.root / "pin", "2.6.0")
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        for version in ("2.5.1", "3.0.0", "unknown"):
            with self.assertRaises(RuntimeError):
                support.install_command("req.txt", self.root / "pin", version)

    def test_no_installer_fails_explicitly(self):
        with patch.object(support.shutil, "which", return_value=None), \
                patch.object(support.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(RuntimeError):
                support.install_command("req.txt", self.root / "pin", "2.6.0")

    def test_real_child_logs_utf8_and_nonzero_exit_is_not_hidden(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            support.run_logged([sys.executable, "-u", "-c",
                                "import sys; sys.stdout.buffer.write('ready \\uac00\\ub098\\ub2e4'.encode('utf-8'))"],
                               self.root / "child.log", time.monotonic() + 10, margin=1)
        self.assertTrue(output.getvalue().endswith("ready 가나다"))
        self.assertEqual((self.root / "child.log").read_text(encoding="utf-8"), output.getvalue())
        with self.assertRaises(subprocess.CalledProcessError):
            support.run_logged([sys.executable, "-c", "raise SystemExit(3)"], self.root / "fail.log",
                               time.monotonic() + 10, margin=1)

    def test_deadline_stops_child_and_log_reexecution_is_rejected(self):
        log = self.root / "timeout.log"
        with self.assertRaises(BudgetExceeded):
            support.run_logged([sys.executable, "-c", "import time; time.sleep(10)"], log,
                               time.monotonic() + 0.6, margin=0.2)
        with self.assertRaises(FileExistsError):
            support.run_logged([sys.executable, "-c", "pass"], log, time.monotonic() + 10, margin=1)

    def test_keyboard_interrupt_stops_owned_child(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   start_new_session=support.os.name == "posix")
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        with patch.object(support.subprocess, "Popen", return_value=process), \
                patch.object(support.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                support.run_logged([sys.executable, "-c", "pass"], self.root / "cancel.log",
                                   time.monotonic() + 10, margin=1)
        self.assertIsNotNone(process.poll())

    def test_old_pointer_never_reported_as_current_success(self):
        old = self.make_run("old", status="screening_complete")
        before = set(old.parent.glob("*/run_state.json"))
        fresh = self.make_run("fresh", status="starting")
        pointer = self.root / "result.json"
        support.write_json(pointer, {"status": "complete", "submission_ready": True, "run_dir": str(old)})
        result = support.collect_result(pointer, old.parent, before, "BudgetExceeded")
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse(result["submission_ready"])
        self.assertEqual(result["run_dir"], str(fresh))

    def test_ambiguous_new_runs_are_not_guessed(self):
        first = self.make_run("first")
        self.make_run("second")
        result = support.collect_result(self.root / "missing.json", first.parent, set())
        self.assertEqual(result["status"], "interrupted")
        self.assertNotIn("run_dir", result)

    def test_valid_screening_result_is_never_submission_ready(self):
        run = self.make_run(status="screening_complete")
        pointer = self.root / "result.json"
        support.write_json(pointer, {"status": "screening_complete", "submission_ready": True, "run_dir": str(run)})
        result = support.collect_result(pointer, run.parent, set())
        self.assertEqual(result["status"], "screening_complete")
        self.assertFalse(result["submission_ready"])

    def test_export_omits_models_without_modifying_run(self):
        run = self.make_run()
        (run / "selection.json").write_text('{"selected_recipe":"klue_asl"}')
        (run / "model.safetensors").write_bytes(b"weights")
        (run / "optimizer.pt").write_bytes(b"checkpoint")
        (run / "best_model").mkdir()
        (run / "best_model/config.json").write_text("{}")
        (run / "predictions.npz").write_bytes(b"predictions")
        before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        archive = support.export_analysis(run, self.volume / "exports")
        with zipfile.ZipFile(archive) as reader:
            self.assertEqual(set(reader.namelist()), {"run-1/run_state.json", "run-1/selection.json",
                                                     "run-1/predictions.npz"})
        self.assertEqual(before, {p.relative_to(run).as_posix(): p.read_bytes()
                                  for p in run.rglob("*") if p.is_file()})

    def test_execute_exports_on_failed_child_and_raises(self):
        settings, invocation = self.prepare()

        def failed_child(*args, **kwargs):
            run = self.make_run("failed", status="failed")
            support.write_json(invocation / "result.json", {
                "status": "failed", "submission_ready": False, "run_dir": str(run), "error_type": "ValueError"})
            raise subprocess.CalledProcessError(1, ["child"])

        with patch.object(support, "run_logged", side_effect=failed_child), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "CalledProcessError"):
                support.execute(settings, invocation, self.root, {}, time.monotonic() + 10)
        result = json.loads((invocation / "launcher.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["analysis_zip"]).is_file())

    def test_notebook_embeds_current_core_and_compiles_all_cells(self):
        notebook = build_notebook.make_notebook()
        assignments = {}
        count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec")
            count += 1
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ("RUNTIME_SOURCES", "SUPPORT_SOURCE", "VOLUME_DIR", "RUN_MODE"):
                        assignments[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(count, 4)
        self.assertEqual(assignments["VOLUME_DIR"], "/mnt/mission3-ver29-data")
        self.assertEqual(assignments["RUN_MODE"], "screening")
        self.assertEqual(assignments["RUNTIME_SOURCES"], {
            name: (build_notebook.CORE / name).read_text(encoding="utf-8")
            for name in build_notebook.source_files()})
        self.assertEqual(assignments["SUPPORT_SOURCE"], Path(support.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
