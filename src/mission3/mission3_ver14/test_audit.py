"""Verify gold preservation, partition integrity, and private-content-free exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .audit import CSV_FIELDS, REVIEW_STATUS, TARGET_SYMPTOMS, write_audit
except ImportError:
    from audit import CSV_FIELDS, REVIEW_STATUS, TARGET_SYMPTOMS, write_audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = [self.root / f"PRIVATE_RECORD_IDENTIFIER_{index}.json" for index in range(5)]
        self.texts = [
            "PRIVATE_NAME PRIVATE_ADDRESS 속이 메스껍고 토할 것 같아요.",
            "메스껍지 않아요.\n아니요.",
            "배가 아프고 어지러워요.",
            "여기는 구역입니다.",
            "울렁거리나요?\n아니요.",
        ]
        self.targets = np.zeros((5, 9), dtype=np.int8)
        self.targets[[1, 2], TARGET_SYMPTOMS.index("오심")] = 1
        self.fit, self.holdout = np.array([0, 1, 2]), np.array([3, 4])

    def tearDown(self):
        self.temporary.cleanup()

    def run_audit(self, **changes):
        inputs = dict(paths=self.paths, texts=self.texts, targets=self.targets,
                      fit=self.fit, holdout=self.holdout, output_dir=self.root / "out")
        inputs.update(changes)
        return write_audit(**inputs)

    def candidates(self, summary):
        with (Path(summary["output_dir"]) / summary["files"]["candidates"]).open(
                encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_text_only_cues_never_read_source_files_or_use_filename_hints(self):
        # Source paths deliberately do not exist; only supplied text is inspected.
        self.assertTrue(all(not path.exists() for path in self.paths))
        summary = self.run_audit()
        counts = summary["counts"]["training"]
        self.assertEqual(counts["strong_cue_samples"], 3)
        self.assertEqual(counts["strong_cue_label_positive"], 1)
        self.assertEqual(counts["strong_cue_label_negative"], 2)
        self.assertEqual(counts["no_strong_cue_label_positive"], 1)
        # Broad "구역" is intentionally not the narrower clinical cue "구역질".
        self.assertNotIn("3", {row["original_index"] for row in self.candidates(summary)})
        renamed = [self.root / f"메스꺼움_{index}.json" for index in range(5)]
        self.assertEqual(counts, self.run_audit(paths=renamed)["counts"]["training"])

    def test_negation_flags_do_not_flip_gold_or_establish_review_judgments(self):
        summary = self.run_audit()
        rows = {row["original_index"]: row for row in self.candidates(summary)}
        self.assertEqual(rows["1"]["has_negation_expression"], "1")
        self.assertEqual(json.loads(rows["1"]["labels"]), ["오심"])
        self.assertEqual(rows["1"]["candidate_reason"], "strong_cue_label_positive")
        self.assertEqual(rows["4"]["has_question_expression"], "1")
        self.assertEqual(json.loads(rows["4"]["labels"]), [])
        self.assertTrue(all(row["review_status"] == REVIEW_STATUS and row["review_note"] == ""
                            for row in rows.values()))
        self.assertFalse(summary["targets_changed"])
        self.assertEqual(summary["rows_removed"], 0)

    def test_targets_and_split_inputs_are_immutable_including_read_only_arrays(self):
        expected = self.targets.copy()
        original_fit, original_holdout = self.fit.copy(), self.holdout.copy()
        self.targets.setflags(write=False)
        self.fit.setflags(write=False)
        self.holdout.setflags(write=False)
        self.run_audit()
        np.testing.assert_array_equal(self.targets, expected)
        np.testing.assert_array_equal(self.fit, original_fit)
        np.testing.assert_array_equal(self.holdout, original_holdout)

    def test_split_rejects_overlap_duplicates_out_of_range_and_missing_rows(self):
        invalid = [
            {"holdout": np.array([2, 3, 4])},
            {"fit": np.array([0, 0, 1, 2])},
            {"holdout": np.array([3, 5])},
            {"holdout": np.array([3])},
            {"fit": np.array([0.0, 1.0, 2.0])},
        ]
        for changes in invalid:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.run_audit(**changes)
        self.assertFalse((self.root / "out").exists())

    def test_candidate_csv_and_json_never_export_text_source_identifiers_or_private_fields(self):
        summary = self.run_audit()
        root = Path(summary["output_dir"])
        outputs = "\n".join(path.read_text(encoding="utf-8-sig") for path in root.iterdir())
        for sentinel in ("PRIVATE_RECORD_IDENTIFIER", "PRIVATE_NAME", "PRIVATE_ADDRESS"):
            self.assertNotIn(sentinel, outputs)
        rows = self.candidates(summary)
        self.assertEqual(tuple(rows[0]), CSV_FIELDS)
        self.assertNotIn("text", rows[0])
        self.assertNotIn("label file name", rows[0])
        self.assertEqual({row["split"] for row in rows}, {"fit", "holdout"})
        self.assertEqual(len(rows), summary["counts"]["training"]["review_candidates"])

    def test_repeat_calls_preserve_existing_artifacts(self):
        first = self.run_audit()
        root = Path(first["output_dir"])
        original = {path.name: path.read_bytes() for path in root.iterdir()}
        second = self.run_audit()
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(original, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_invalid_targets_and_alignment_are_explicit_errors(self):
        for changes in [{"paths": self.paths[:-1]}, {"texts": [None] * 5},
                        {"targets": np.ones((5, 8), dtype=np.int8)},
                        {"targets": np.full((5, 9), 2, dtype=np.int8)}]:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.run_audit(**changes)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
