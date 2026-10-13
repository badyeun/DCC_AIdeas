"""Verify the standalone notebook and fresh-process launcher without GPU training."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ver14_notebook_builder", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def assigned_literal(notebook, identifier, variable):
    cell = next(cell for cell in notebook["cells"] if cell["id"] == identifier)
    tree = ast.parse("".join(cell["source"]))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing literal assignment: {variable}")


class NotebookTests(unittest.TestCase):
    def test_saved_standalone_notebook_contains_current_runtime_sources(self):
        saved = json.loads((HERE.parent / "mission3_ver14.ipynb").read_text(encoding="utf-8"))
        embedded = assigned_literal(saved, "ver14-restore", "RUNTIME_SOURCES")
        for name in builder.SOURCE_FILES:
            self.assertEqual(embedded[name], (HERE / name).read_text(encoding="utf-8"),
                             f"Regenerate mission3_ver14.ipynb after changing {name}")

    def test_embedded_sources_are_complete_current_and_compile(self):
        notebook = builder.make_notebook()
        sources = assigned_literal(notebook, "ver14-restore", "RUNTIME_SOURCES")
        self.assertEqual(set(sources), set(builder.SOURCE_FILES))
        self.assertIn("runtime_suite.py", sources)
        self.assertIn("audit.py", sources)
        for name, source in sources.items():
            self.assertEqual(source, (HERE / name).read_text(encoding="utf-8"))
            if name.endswith(".py"):
                compile(source, name, "exec")
        previous = HERE.parent / "mission3_threshold_free" / "requirements.txt"
        self.assertEqual(sources["requirements.txt"], previous.read_text(encoding="utf-8"))

    def test_notebook_settings_support_the_complete_matched_workflow(self):
        notebook = builder.make_notebook()
        settings = assigned_literal(notebook, "ver14-settings", "SETTINGS")
        self.assertEqual(settings["version"], "ver14")
        self.assertEqual(settings["model_id"], "klue/roberta-base")
        self.assertEqual(settings["revision"], "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da")
        self.assertEqual(settings["nausea_weight_candidates"], [1.0, 2.0, 3.0])
        self.assertEqual(settings["batch_size"] * settings["gradient_accumulation"], 16)
        self.assertEqual(settings["stage3_batch_size"], 8)
        self.assertEqual(settings["stage3_batch_size"] * settings["stage3_gradient_accumulation"], 16)
        self.assertEqual(settings["seed"], 42)
        self.assertEqual(settings["holdout_size"], .2)
        self.assertTrue(settings["refit_full_training"])
        self.assertTrue(settings["run_validation"])
        self.assertFalse(settings["allow_cpu"])
        self.assertEqual(settings["resume_run"], "")
        self.assertNotIn("max_hours", settings)
        markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                             if cell["cell_type"] == "markdown")
        for name in ("attention_off_ce", "attention_on_ce", "attention_on_supcon",
                     "experiment_plan.json", "candidate_summary.csv", "final_selection.json"):
            self.assertIn(name, markdown)

    def test_notebook_is_valid_unexecuted_and_ready_for_colab(self):
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            output = Path(temporary) / "mission3_ver14.ipynb"
            builder.build(output)
            notebook = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), cell["id"], "exec")
        self.assertLess(identifiers.index("ver14-restore"), identifiers.index("ver14-dependencies"))
        self.assertLess(identifiers.index("ver14-dependencies"), identifiers.index("ver14-run"))

    def test_launcher_imports_embedded_suite_and_preserves_resume_settings(self):
        # A lightweight fixture exercises the launcher boundary, not model training.
        with tempfile.TemporaryDirectory(prefix=".notebook_test_", dir=HERE) as temporary:
            root = Path(temporary)
            shutil.copyfile(HERE / "run_notebook.py", root / "run_notebook.py")
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "requirements.txt").write_text("fixture requirements\n", encoding="utf-8")
            (root / "runtime_suite.py").write_text(
                "from pathlib import Path\n"
                "def run(settings, runtime_sources=None):\n"
                "    assert settings['resume_run'] == 'existing_run'\n"
                "    assert settings['nausea_weight_candidates'] == [1.0, 2.0, 3.0]\n"
                "    assert runtime_sources['requirements.txt'] == 'fixture requirements\\n'\n"
                "    assert 'run_notebook.py' in runtime_sources\n"
                "    return Path(settings['output_dir']) / 'completed_run'\n", encoding="utf-8")
            settings = builder.default_settings()
            settings.update(resume_run="existing_run", output_dir=str(root / "results"))
            settings_file, result_file = root / "settings.json", root / "result.json"
            settings_file.write_text(json.dumps(settings), encoding="utf-8-sig")
            completed = subprocess.run(
                [sys.executable, "-B", str(root / "run_notebook.py"), "--settings", str(settings_file),
                 "--result_file", str(result_file)], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(result_file.read_text(encoding="utf-8")),
                             str(root / "results" / "completed_run"))
            self.assertFalse((root / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
