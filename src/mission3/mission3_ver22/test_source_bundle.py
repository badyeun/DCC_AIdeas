"""Pinned source, safe ZIP import, cache integrity, and Training leakage checks."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mission3_ver22 import data, source_bundle as source
else:
    from . import data, source_bundle as source


class SourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".source_test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "original" / "run"
        self.work = self.base / "work"
        self.root.mkdir(parents=True)
        self.texts = [f"synthetic example {row}" for row in range(10)]
        self.targets = np.array([[(row + label) % 2 for label in range(9)] for row in range(10)], dtype=np.int8)
        self.fit, self.holdout = np.arange(6), np.arange(6, 10)
        self.predictions = self.targets[self.holdout].copy()
        self.predictions[0, 6] = 1 - self.predictions[0, 6]
        self.ref = {"format": "mission3_ver21_pinned_source_v1", "version": "ver19", "run_id": "run",
                    "labels": list(data.TARGET_SYMPTOMS), "decision_rule": "categorical predict/argmax",
                    "training_samples": 10, "fit_samples": 6, "holdout_samples": 4,
                    "training_digest": source.dataset_digest(self.texts, self.targets),
                    "fit_checkpoint": "experiments/candidate_model.ckpt", "full_checkpoint": "final_model/final_model.ckpt",
                    "provenance": {"validation_accessed": False},
                    "metrics": data.score_predictions(self.targets[self.holdout], self.predictions),
                    "array_hashes": {
                        "fit_indices": source.array_digest(self.fit, "<i8"),
                        "holdout_indices": source.array_digest(self.holdout, "<i8"),
                        "training_targets": source.array_digest(self.targets, "i1"),
                        "holdout_targets": source.array_digest(self.targets[self.holdout], "i1"),
                        "predictions": source.array_digest(self.predictions, "i1"),
                    }}
        for key, scope, count in (("fit_checkpoint", "fit_only", 6), ("full_checkpoint", "all_training", 10)):
            path = self.root / self.ref[key]
            path.parent.mkdir(parents=True)
            manifest = {"version": "ver19", "training_scope": scope, "fitted_samples": count,
                        "labels": self.ref["labels"], "decision_rule": self.ref["decision_rule"],
                        "model_id": "klue/roberta-base", "resolved_revision": "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da",
                        "utterance_evidence": True, "loss_mode": "nausea_ce_gce", "weights": "weights.pt", "assets": "assets"}
            path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            (path.parent / "weights.pt").write_bytes(b"synthetic weights " + scope.encode())
            assets = path.parent / "assets"
            assets.mkdir()
            (assets / "config.json").write_text("{}", encoding="utf-8")
        (self.root / "run_identity.json").write_text(json.dumps({"training_digest": self.ref["training_digest"]}), encoding="utf-8")
        np.savez_compressed(self.root / "training_split.npz", fit=self.fit, holdout=self.holdout)
        np.savez_compressed(self.root / "holdout_predictions.npz", indices=self.holdout,
                            targets=self.targets[self.holdout], predictions=self.predictions)
        self.ref["files"] = {p.relative_to(self.root).as_posix(): {"size": p.stat().st_size,
                             "sha256": source._hash_file(p)} for p in sorted(self.root.rglob("*")) if p.is_file()}
        self.ref_hash = "a" * 64

    def prepare(self, location=None, **kwargs):
        return source._prepare_source(str(self.root) if location is None else location,
                                      self.work, self.ref, self.ref_hash, **kwargs)

    def archive(self, additions=(), prefix="wrapper/ver19/run/"):
        path = self.base / ("fixture_" + str(len(list(self.base.glob("*.zip")))) + ".zip")
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in self.ref["files"]:
                archive.write(self.root / name, prefix + name)
            for name, payload in additions:
                archive.writestr(name, payload)
        return path

    def verify(self, prepared=None, **kwargs):
        prepared = self.prepare() if prepared is None else prepared
        arguments = {"texts": self.texts, "targets": self.targets, "fit": self.fit,
                     "holdout": self.holdout, "source": prepared, **kwargs}
        with patch.object(source, "_load_reference", return_value=copy.deepcopy(self.ref)):
            return source.verify_training(**arguments)

    def test_directory_whitelist_and_scopes(self):
        (self.root / "validation_metrics.json").write_text("unwanted", encoding="utf-8")
        result = self.prepare()
        self.assertTrue(result["fit_checkpoint"].is_file())
        self.assertTrue(result["full_checkpoint"].is_file())
        self.assertFalse((result["root"] / "validation_metrics.json").exists())
        self.assertEqual(len(result["source_hashes_verified"]), len(self.ref["files"]))
        report = self.verify(result)
        self.assertTrue(report["verified"])
        self.assertFalse(report["validation_accessed"])
        self.assertEqual(report["selection_source_scope"], "fit_only")
        json.dumps(report, allow_nan=False)

    def test_wrapped_zip_only_extracts_whitelist(self):
        archive = self.archive([("wrapper/ver19/run/validation.csv", b"do not extract")])
        result = self.prepare(str(archive))
        self.assertEqual(set(p.relative_to(result["root"]).as_posix() for p in result["root"].rglob("*") if p.is_file()),
                         set(self.ref["files"]) | {"source_complete.json"})
        self.assertEqual(self.verify(result)["baseline_metrics"], self.ref["metrics"])

    def test_zip_rejects_all_unsafe_members_even_unselected(self):
        for name in ("../escape", "/absolute", "C:/drive", "back\\slash", "folder//double", "folder/./dot", "trailing "):
            with self.subTest(name=name):
                archive = self.archive([(name.replace("\\", "/"), b"unused")])
                if "\\" in name:
                    archive.write_bytes(archive.read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
                with self.assertRaises(ValueError):
                    self.prepare(str(archive))

    def test_zip_duplicate_and_file_directory_collision_rejected(self):
        for extra in [[("duplicate", b"a"), ("DUPLICATE", b"b")],
                      [("collision", b"file"), ("collision/nested", b"child")]]:
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.prepare(str(self.archive(extra)))

    def test_zip_symlink_and_encrypted_members_rejected(self):
        archive = self.archive()
        link = zipfile.ZipInfo("unsafe-link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "a") as output:
            output.writestr(link, "../outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare(str(archive))
        archive = self.archive()
        with zipfile.ZipFile(archive) as handle:
            handle.infolist()[0].flag_bits |= 1
            with self.assertRaisesRegex(ValueError, "encrypted"):
                source._zip_inventory(handle)

    def test_zip_wrong_exact_size_and_checksum_rejected(self):
        for corruption in (b"bad", b"synthetic weights fit_onlX"):
            path = self.root / "experiments/weights.pt"
            path.write_bytes(corruption)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.prepare(str(self.archive()))

    def test_zip_multiple_matching_runs_rejected(self):
        extras = [("second/run/" + name, (self.root / name).read_bytes()) for name in self.ref["files"]]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.prepare(str(self.archive(extras)))

    def test_completed_cache_is_rehashed_not_trusted_by_size(self):
        first = self.prepare()
        second = self.prepare("", search_roots=[])
        self.assertEqual(first["root"], second["root"])
        weight = first["fit_checkpoint"].parent / "weights.pt"
        weight.write_bytes(b"x" * weight.stat().st_size)
        third = self.prepare("", search_roots=[self.root.parent])
        self.assertNotEqual(first["root"], third["root"])
        self.assertEqual(source._hash_file(third["fit_checkpoint"].parent / "weights.pt"),
                         self.ref["files"]["experiments/weights.pt"]["sha256"])

    def test_timeout_leaves_no_complete_cache_and_is_not_swallowed(self):
        class Expired:
            def check(self, stage):
                raise TimeoutError("expired")
        with self.assertRaises(TimeoutError):
            self.prepare(budget=Expired())
        self.assertFalse(list(self.work.rglob("source_complete.json")))
        self.prepare()
        with self.assertRaises(TimeoutError):
            self.prepare("", budget=Expired(), search_roots=[])

    def test_missing_source_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "source_run"):
            self.prepare("", search_roots=[])

    def test_full_model_cannot_be_disguised_as_fit_model(self):
        path = self.root / self.ref["fit_checkpoint"]
        document = json.loads(path.read_text(encoding="utf-8"))
        document["training_scope"] = "all_training"
        document["fitted_samples"] = 10
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        # Even a hypothetical matching file checksum cannot bypass the scope check.
        self.ref["files"][self.ref["fit_checkpoint"]] = {"size": path.stat().st_size, "sha256": source._hash_file(path)}
        with self.assertRaisesRegex(ValueError, "training_scope"):
            self.prepare()

    def test_training_text_target_and_order_must_match(self):
        prepared = self.prepare()
        texts = self.texts.copy()
        texts[0] += " modified"
        labels = self.targets.copy()
        labels[0, 0] = 1 - labels[0, 0]
        for changes in ({"texts": texts}, {"targets": labels}, {"fit": self.fit[::-1]},
                        {"holdout": self.holdout[::-1]}, {"texts": self.texts[:-1]}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.verify(prepared, **changes)

    def test_invalid_split_and_mutated_saved_arrays_rejected(self):
        prepared = self.prepare()
        for fit in (np.array([0, 0, 2, 3, 4, 5]), np.arange(6, dtype=float), np.arange(4, 10)):
            with self.subTest(fit=fit.tolist()), self.assertRaises(ValueError):
                self.verify(prepared, fit=fit)
        path = prepared["root"] / "holdout_predictions.npz"
        np.savez_compressed(path, indices=self.holdout, targets=self.targets[self.holdout], predictions=1 - self.predictions)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.verify(prepared)

    def test_production_reference_is_pinned_and_training_only(self):
        reference = source._load_reference()
        self.assertEqual(reference["training_samples"], 29200)
        self.assertEqual(reference["fit_samples"], 23360)
        self.assertEqual(reference["holdout_samples"], 5840)
        self.assertEqual(len(reference["files"]), 19)
        self.assertFalse(reference["provenance"]["validation_accessed"])
        self.assertFalse(any("validation" in name.lower() for name in reference["files"]))
        self.assertFalse(set(reference) & {"texts", "filenames", "sample_ids", "probabilities"})
        with patch.object(source, "REFERENCE_FILE_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "checksum"):
            source._load_reference()


if __name__ == "__main__":
    unittest.main()
