"""Fail-closed historical comparison checks on anonymous synthetic arrays."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parent))
    from mission3_ver20 import data, reference
else:
    from . import data, reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".reference_test_", dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "reference.json"
        self.y = np.array([[(row + label) % 2 for label in range(9)] for row in range(10)], dtype=np.int8)
        self.fit = np.arange(6)
        self.holdout = np.arange(6, 10)
        self.predicted = self.y[self.holdout].copy()
        self.predicted[1, 6] = 0
        self.predicted[0, 6] = 1
        self.predicted[1, 0] = 0
        self.document = {
            "format": "mission3_training_reference_v1", "version": "fixture", "candidate_id": "fixture",
            "labels": list(data.TARGET_SYMPTOMS), "training_scope": "fit_only",
            "decision_rule": "categorical predict/argmax", "training_digest": "fixture-digest",
            "training_samples": 10, "fit_samples": 6, "holdout_samples": 4, "split_seed": 42, "holdout_size": .2,
            "array_hashes": {
                "fit_indices": reference.array_digest(self.fit, "<i8"),
                "holdout_indices": reference.array_digest(self.holdout, "<i8"),
                "training_targets": reference.array_digest(self.y, "i1"),
                "holdout_targets": reference.array_digest(self.y[self.holdout], "i1"),
                "predictions": reference.array_digest(self.predicted, "i1"),
            },
            "metrics": data.score_predictions(self.y[self.holdout], self.predicted),
            "predictions_payload": self.payload(self.predicted), "limitations": ["Synthetic fixture only"],
        }

    def payload(self, values):
        values = np.asarray(values, dtype=np.int8)
        return {"encoding": "zlib+base64/int8-c-order", "shape": list(values.shape),
                "sha256": reference.array_digest(values, "i1"),
                "data": base64.b64encode(zlib.compress(values.tobytes())).decode("ascii")}

    def verify(self, document=None, **kwargs):
        document = self.document if document is None else document
        raw = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.path.write_bytes(raw)
        arguments = {"training_digest_value": "fixture-digest", "targets": self.y,
                     "fit": self.fit, "holdout": self.holdout, "path": self.path, **kwargs}
        with patch.object(reference, "REFERENCE_FILE_SHA256", hashlib.sha256(raw).hexdigest()):
            return reference.verify_training_reference(**arguments)

    def test_valid_reference_and_serializable_summary(self):
        verified = self.verify()
        np.testing.assert_array_equal(verified["predictions"], self.predicted)
        summary = reference.reference_summary(verified)
        self.assertNotIn("predictions", summary)
        self.assertNotIn("predictions_payload", summary)
        json.dumps(summary, allow_nan=False)

    def test_changed_file_fails_checksum_before_parsing(self):
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference file checksum"):
            reference.verify_training_reference("fixture-digest", self.y, self.fit, self.holdout, path=self.path)

    def test_data_digest_and_label_order_are_required(self):
        with self.assertRaisesRegex(ValueError, "Training data digest"):
            self.verify(training_digest_value="changed")
        changed = copy.deepcopy(self.document)
        changed["labels"] = changed["labels"][::-1]
        with self.assertRaisesRegex(ValueError, "label order"):
            self.verify(changed)

    def test_scope_and_decision_policy_cannot_change(self):
        for key, value, message in (("training_scope", "all_training", "historical training scope"),
                                    ("decision_rule", "threshold", "decision rule")):
            changed = copy.deepcopy(self.document)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.verify(changed)

    def test_split_settings_and_row_order_must_match(self):
        for kwargs, message in (({"split_seed": 43}, "split seed"),
                                ({"holdout_size": .3}, "holdout proportion"),
                                ({"fit": self.fit[::-1]}, "fit order"),
                                ({"holdout": self.holdout[::-1]}, "holdout order")):
            with self.subTest(kwargs=list(kwargs)), self.assertRaisesRegex(ValueError, message):
                self.verify(**kwargs)

    def test_indices_reject_duplicate_overlap_float_and_out_of_range(self):
        for invalid in ([0, 0, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6],
                        [0., 1., 2., 3., 4., 5.], [0, 1, 2, 3, 4, 10]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.verify(fit=np.asarray(invalid))

    def test_changed_target_in_fit_or_holdout_fails(self):
        for row in (0, 9):
            changed = self.y.copy()
            changed[row, 0] = 1 - changed[row, 0]
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Training targets"):
                self.verify(targets=changed)

    def test_prediction_payload_shape_corruption_and_nonbinary_rejected(self):
        for kind in ("shape", "nonbinary", "base64", "too_many_bytes", "trailing_stream"):
            changed = copy.deepcopy(self.document)
            if kind == "shape":
                changed["predictions_payload"]["shape"] = [5, 9]
            elif kind == "nonbinary":
                bad = self.predicted.copy()
                bad[0, 0] = 2
                changed["predictions_payload"] = self.payload(bad)
            elif kind == "base64":
                changed["predictions_payload"]["data"] = "###"
            else:
                raw = self.predicted.tobytes()
                compressed = zlib.compress(raw + b"\0") if kind == "too_many_bytes" else zlib.compress(raw) + b"extra"
                changed["predictions_payload"]["data"] = base64.b64encode(compressed).decode("ascii")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.verify(changed)

    def test_hashes_and_stored_scores_are_verified(self):
        for kind in ("payload", "array", "metrics"):
            changed = copy.deepcopy(self.document)
            if kind == "payload":
                changed["predictions_payload"]["sha256"] = "0" * 64
            elif kind == "array":
                changed["array_hashes"]["predictions"] = "0" * 64
            else:
                changed["metrics"]["macro_f1"] += .001
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.verify(changed)

    def test_paired_transitions_and_joint_improvement(self):
        result = reference.compare_reference(self.verify(), self.y[self.holdout], self.y, self.holdout)
        self.assertTrue(result["both_macro_and_nausea_improved"])
        self.assertEqual(result["status"], "both_improved")
        paired = result["per_symptom"]["오심"]["paired"]
        self.assertEqual(paired, {"fn_to_tp": 1, "tp_to_fn": 0, "fp_to_tn": 1,
                                  "tn_to_fp": 0, "persistent_fn": 0, "persistent_fp": 0})
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_macro_improvement_does_not_claim_nausea_improvement(self):
        new = self.predicted.copy()
        new[1, 0] = self.y[self.holdout][1, 0]
        result = reference.compare_reference(self.verify(), new, self.y, self.holdout)
        self.assertGreater(result["macro_f1_delta"], 0)
        self.assertEqual(result["nausea_f1_delta"], 0)
        self.assertFalse(result["both_macro_and_nausea_improved"])
        self.assertEqual(result["status"], "macro_only_improved")

    def test_identical_predictions_not_claimed_as_improvement(self):
        result = reference.compare_reference(self.verify(), self.predicted, self.y, self.holdout)
        self.assertEqual(result["macro_f1_delta"], 0)
        self.assertEqual(result["nausea_f1_delta"], 0)
        self.assertEqual(result["status"], "neither_improved")

    def test_comparison_rechecks_inputs_and_reference(self):
        verified = self.verify()
        with self.assertRaises(ValueError):
            reference.compare_reference(verified, self.predicted, self.y, self.holdout[::-1])
        with self.assertRaises(ValueError):
            reference.compare_reference(verified, np.full((4, 9), .5), self.y, self.holdout)
        changed_y = self.y.copy()
        changed_y[0, 0] = 1 - changed_y[0, 0]
        with self.assertRaises(ValueError):
            reference.compare_reference(verified, self.predicted, changed_y, self.holdout)
        verified["predictions"][0, 0] = 1 - verified["predictions"][0, 0]
        with self.assertRaisesRegex(ValueError, "reference predictions"):
            reference.compare_reference(verified, self.predicted, self.y, self.holdout)

    def test_embedded_artifact_checksum_and_no_identifiers(self):
        raw = reference.REFERENCE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), reference.REFERENCE_FILE_SHA256)
        saved = json.loads(raw)
        self.assertEqual(saved["training_scope"], "fit_only")
        self.assertEqual(saved["training_samples"], 29200)
        self.assertEqual(saved["fit_samples"], 23360)
        self.assertEqual(saved["holdout_samples"], 5840)
        self.assertFalse(saved["provenance"]["validation_accessed"])
        self.assertFalse(set(saved) & {"texts", "filenames", "sample_ids", "probabilities"})
        predicted = reference._decode_predictions(saved["predictions_payload"], saved["holdout_samples"])
        self.assertEqual(predicted.shape, (5840, 9))


if __name__ == "__main__":
    unittest.main()
