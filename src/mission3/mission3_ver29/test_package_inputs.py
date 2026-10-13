"""Verify private Kaggle packaging preserves labels and excludes old assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from .bootstrap import TRAIN_PREFIX, VALID_PREFIX, prepare_data
from .package_inputs import package_inputs, verify_package


class PackageInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.zip"
        self.output = self.root / "compact.zip"
        self.assets = {
            TRAIN_PREFIX + "train.json": b'{"utterances": [], "symptom": []}\r\n',
            VALID_PREFIX + "valid.json": b' {"utterances": [], "symptom": []}\n',
            "old_model.ckpt": b"old weights must not be copied",
        }

    def make_reference(self, status="screening_complete"):
        reference = self.root / "ver28_analysis_test.zip"
        with zipfile.ZipFile(reference, "w") as archive:
            archive.writestr("run/run_state.json", json.dumps({"format": "mission3_ver28_v1",
                                                              "run_mode": "screening", "status": status}))
            for name in ("split_indices.npz", "split_manifest.json", "development_predictions.npz",
                         "confirmation_predictions.npz", "selection.json", "artifact_manifest.json"):
                archive.writestr("run/" + name, b"fixture; reference runtime verifies semantic contents")
        return reference

    def make_source(self, *, bad_hash=False, extra=None):
        files = [{"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                 for name, raw in self.assets.items()]
        if bad_hash:
            files[0]["sha256"] = "0" * 64
        manifest = {"format": "mission3_ver5_inputs_v1", "training_count": 1,
                    "validation_count": 1, "files": files}
        with zipfile.ZipFile(self.source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in self.assets.items():
                archive.writestr(name, raw)
            archive.writestr("bundle_manifest.json", json.dumps(manifest))
            if extra is not None:
                if isinstance(extra, str) and "\\" in extra:
                    # Prevent Windows ZipInfo from normalizing this malicious
                    # stored name before the reader under test can inspect it.
                    info = zipfile.ZipInfo(extra)
                    info.filename = extra
                    extra = info
                archive.writestr(extra, b"unexpected")

    def test_exact_bytes_manifest_and_existing_bootstrap_roundtrip(self):
        self.make_source()
        source_bytes = self.source.read_bytes()
        source_stat = self.source.stat()
        report = package_inputs(self.source, self.output, (1, 1))
        self.assertEqual(report["label_file_count"], 2)
        self.assertFalse(report["old_models_included"])
        self.assertEqual(report["output_archive_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), {TRAIN_PREFIX + "train.json", VALID_PREFIX + "valid.json",
                                                      "bundle_manifest.json"})
            for name in (TRAIN_PREFIX + "train.json", VALID_PREFIX + "valid.json"):
                self.assertEqual(archive.read(name), self.assets[name])
        train, valid = prepare_data(self.output, self.root / "restored")
        self.assertEqual((train / "train.json").read_bytes(), self.assets[TRAIN_PREFIX + "train.json"])
        self.assertEqual((valid / "valid.json").read_bytes(), self.assets[VALID_PREFIX + "valid.json"])
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.source.stat().st_mtime_ns, source_stat.st_mtime_ns)

    def test_changed_source_hash_never_publishes_output(self):
        self.make_source(bad_hash=True)
        with self.assertRaisesRegex(ValueError, "Source label content"):
            package_inputs(self.source, self.output, (1, 1))
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_unsafe_archive_path_is_rejected(self):
        for name in ("../outside.json", "/absolute.json", "C:/absolute.json", "folder\\file.json"):
            with self.subTest(name=name):
                self.make_source(extra=name)
                with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
                    package_inputs(self.source, self.output, (1, 1))
                self.assertFalse(self.output.exists())

    def test_link_and_duplicate_names_are_rejected(self):
        link = zipfile.ZipInfo("linked")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_source(extra=link)
        with self.assertRaisesRegex(ValueError, "Links"):
            package_inputs(self.source, self.output, (1, 1))
        self.make_source(extra=(TRAIN_PREFIX + "train.json").upper())
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP"):
            package_inputs(self.source, self.output, (1, 1))

    def test_wrong_counts_and_existing_outputs_are_preserved(self):
        self.make_source()
        with self.assertRaisesRegex(ValueError, "Label counts differ"):
            package_inputs(self.source, self.output, (2, 1))
        self.output.write_bytes(b"keep this output")
        with self.assertRaises(FileExistsError):
            package_inputs(self.source, self.output, (1, 1))
        self.assertEqual(self.output.read_bytes(), b"keep this output")
        before = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            package_inputs(self.source, self.source, (1, 1))
        self.assertEqual(self.source.read_bytes(), before)

    def test_verify_rejects_corrupted_content_and_non_label_payload(self):
        self.make_source()
        package_inputs(self.source, self.output, (1, 1))
        damaged = self.root / "damaged.zip"
        with zipfile.ZipFile(self.output) as original, zipfile.ZipFile(damaged, "w") as archive:
            for name in original.namelist():
                raw = original.read(name)
                if name == TRAIN_PREFIX + "train.json":
                    raw = b"x" * len(raw)
                archive.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, "Label content differs"):
            verify_package(damaged, (1, 1))
        with zipfile.ZipFile(self.output, "a") as archive:
            archive.writestr("old.ckpt", b"weights")
        with self.assertRaisesRegex(ValueError, "only labels"):
            verify_package(self.output, (1, 1))

    def test_reference_bytes_preserved_and_bootstrap_ignores_reference(self):
        self.make_source()
        reference = self.make_reference()
        raw_reference = reference.read_bytes()
        report = package_inputs(self.source, self.output, (1, 1), reference_analysis=reference)
        self.assertTrue(report["reference_archive_bytes_preserved"])
        self.assertEqual(report["reference_analysis"]["sha256"], hashlib.sha256(raw_reference).hexdigest())
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read(report["reference_analysis"]["path"]), raw_reference)
            self.assertEqual(len(json.loads(archive.read("bundle_manifest.json"))["files"]), 2)
        train, valid = prepare_data(self.output, self.root / "restored")
        self.assertTrue((train / "train.json").is_file())
        self.assertTrue((valid / "valid.json").is_file())
        self.assertFalse(list((self.root / "restored").rglob("*.zip")))

    def test_incomplete_reference_does_not_publish(self):
        self.make_source()
        reference = self.make_reference(status="paused")
        with self.assertRaisesRegex(ValueError, "completed ver28 screening"):
            package_inputs(self.source, self.output, (1, 1), reference_analysis=reference)
        self.assertFalse(self.output.exists())

    def test_tampered_reference_is_rejected(self):
        self.make_source()
        reference = self.make_reference()
        report = package_inputs(self.source, self.output, (1, 1), reference_analysis=reference)
        damaged = self.root / "changed_reference.zip"
        with zipfile.ZipFile(self.output) as original, zipfile.ZipFile(damaged, "w") as archive:
            for name in original.namelist():
                raw = original.read(name)
                if name == report["reference_analysis"]["path"]:
                    raw = b"x" * len(raw)
                archive.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, "Reference analysis content differs"):
            verify_package(damaged, (1, 1))


if __name__ == "__main__":
    unittest.main()
