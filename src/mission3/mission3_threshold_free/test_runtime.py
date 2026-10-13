"""Portable sparse artifacts and notebook integrity with small offline fixtures."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

if __package__:
    from . import data, runtime
else:
    package_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_root))
    from mission3_threshold_free import data, runtime


class RuntimeTests(unittest.TestCase):
    def annotation(self, directory, index):
        flags = [int((index + 3 * label) % (label + 3) < 2) for label in range(9)]
        text = " ".join(f"feature{label}_{'present' if flag else 'absent'}"
                        for label, flag in enumerate(flags))
        annotation = {
            "utterances": [{"text": text, "speaker": "unused private metadata"},
                           {"text": f"unique_dialogue_{index}"}],
            "symptom": [name for name, flag in zip(data.TARGET_SYMPTOMS, flags) if flag],
        }
        path = directory / f"dialogue_{index:04d}.json"
        path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
        return flags

    def read_predictions(self, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        predictions = []
        for row in rows:
            symptoms = ast.literal_eval(row["symptom"])
            self.assertIsInstance(symptoms, list)
            self.assertEqual(len(symptoms), len(set(symptoms)))
            self.assertTrue(set(symptoms) <= set(data.TARGET_SYMPTOMS))
            predictions.append([int(name in symptoms) for name in data.TARGET_SYMPTOMS])
        return rows, np.asarray(predictions, dtype=np.int8)

    def test_sparse_run_reload_poisoned_targets_and_portable_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = root / "training", root / "validation"
            train.mkdir()
            validation.mkdir()
            for index in range(90):
                self.annotation(train, index)
            truth = np.asarray([self.annotation(validation, index)
                                for index in range(100, 112)], dtype=np.int8)
            settings = {
                "version": "ver10", "backend": "sparse", "input_zip": "",
                "train_dir": str(train), "validation_dir": str(validation),
                "work_dir": str(root / "work"), "output_dir": str(root / "results"),
                "run_validation": True, "seed": 42, "holdout_size": .2,
                "sparse_candidates": ["svm_c1"], "char_features": 2000,
                "word_features": 2000,
            }
            sources = {name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                       for name in ("__init__.py", "data.py", "bootstrap.py", "neural.py",
                                    "runtime.py", "inference.py", "requirements.txt")}
            result = runtime.run(settings, runtime_sources=sources)
            checkpoint = result / "mission3_ver10_model.ckpt"
            manifest = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
            self.assertEqual(manifest["format"], "mission3_categorical_v1")
            self.assertEqual(manifest["kind"], "sparse")
            self.assertEqual(manifest["selected_name"], "svm_c1")
            self.assertEqual(manifest["decision_rule"], "categorical predict/argmax")
            status = json.loads((result / "runtime.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(status["status"], "complete")
            identity = json.loads((result / "run_identity.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(identity["training_samples"], 90)
            self.assertEqual(identity["fit_samples"] + identity["holdout_samples"], 90)
            self.assertEqual(identity["model_selection"], "Training internal holdout only")
            self.assertTrue((result / "training_errors" / "errors_오심.csv").is_file())
            with np.load(result / "training_split.npz", allow_pickle=False) as split:
                self.assertFalse(set(split["fit"]) & set(split["holdout"]))
                self.assertEqual(set(split["fit"]) | set(split["holdout"]), set(range(90)))
            paths, texts, _ = data.load_dataset(validation, include_targets=False)
            rows, original = self.read_predictions(result / "mission3.csv")
            self.assertEqual(list(rows[0]), ["label file name", "symptom"])
            self.assertEqual([row["label file name"] for row in rows], [path.name for path in paths])
            self.assertEqual(original.shape, (12, 9))
            self.assertGreater(data.score_predictions(truth, original)["macro_f1"], .8)
            np.testing.assert_array_equal(runtime.predict_checkpoint(checkpoint, texts), original)

            # Invalid targets would fail evaluation, but must never affect inference.
            for path in paths:
                annotation = json.loads(path.read_text(encoding="utf-8"))
                annotation["symptom"] = {"invalid": "inference must ignore this"}
                path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
            _, poisoned = runtime.inference(checkpoint, validation, root / "poisoned.csv")
            np.testing.assert_array_equal(poisoned, original)
            with self.assertRaisesRegex(ValueError, "symptom"):
                data.load_dataset(validation, include_targets=True)

            # Move the full result folder and invoke its entry point from an unrelated cwd.
            portable = root / "relocated artifact"
            self.assertTrue(result.resolve().is_relative_to(root.resolve()))
            self.assertTrue(portable.resolve().is_relative_to(root.resolve()))
            result.rename(portable)
            cold_output = root / "cold_cli.csv"
            completed = subprocess.run(
                [sys.executable, str(portable / "submission" / "inference.py"),
                 "--audio_dir", str(root / "unused_audio"), "--label_dir", str(validation),
                 "--ckpt_path", str(portable / checkpoint.name), "--output", str(cold_output)],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cold_rows, cold = self.read_predictions(cold_output)
            self.assertEqual([row["label file name"] for row in cold_rows], [path.name for path in paths])
            np.testing.assert_array_equal(cold, original)

    def test_generated_notebooks_are_standalone_current_sources(self):
        source_root = Path(__file__).resolve().parent
        notebooks = [source_root.parent / f"mission3_ver{version}.ipynb" for version in range(10, 14)]
        if not all(path.exists() for path in notebooks):
            self.skipTest("Generate ver10-ver13 notebooks before testing embedded sources")
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            code = ["".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code"]
            embedded = None
            for source in code:
                module = ast.parse(source, filename=path.name)
                for node in module.body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "RUNTIME_SOURCES"
                            for target in node.targets):
                        embedded = ast.literal_eval(node.value)
            self.assertIsNotNone(embedded)
            self.assertTrue({"__init__.py", "data.py", "bootstrap.py", "runtime.py",
                             "neural.py", "inference.py", "requirements.txt"} <= set(embedded))
            for name, contents in embedded.items():
                self.assertEqual(contents, (source_root / name).read_text(encoding="utf-8"),
                                 f"Stale embedded source: {path.name} {name}")
                if name.endswith(".py"):
                    ast.parse(contents, filename=name)


if __name__ == "__main__":
    unittest.main()
