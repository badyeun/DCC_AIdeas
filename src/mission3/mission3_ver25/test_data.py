"""Integrity and split leakage checks using synthetic transcripts only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import data


class DataTests(unittest.TestCase):
    def test_input_uses_only_exact_text_in_array_order(self):
        annotation = {"utterances": [{"text": " first  turn ", "speaker": "secret", "startAt": 20},
                                      {"text": "", "startAt": 0}, {"text": "last\tturn"}],
                      "utterences": [{"text": "wrong spelling must not override canonical"}],
                      "gender": "metadata", "address": "metadata", "symptom": ["고열"]}
        self.assertEqual(data.dialogue_text(annotation), " first  turn \n\nlast\tturn")
        self.assertEqual(data.dialogue_text({"utterences": [{"text": "legacy"}]}), "legacy")
        for invalid in ({"utterances": None}, {"utterances": ["bad"]}, {"utterances": [{"text": 1}]}):
            with self.assertRaises(ValueError):
                data.dialogue_text(invalid)

    def test_outside_labels_and_empty_dialogues_keep_original_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            folder = Path(temporary)
            annotations = [
                {"utterances": [], "symptom": ["outside"]},
                {"utterances": [{"text": "synthetic"}], "symptom": ["오심", "고열", "outside"]},
                {"utterences": [{"text": "legacy"}], "symptom": None},
            ]
            for index, annotation in enumerate(annotations):
                (folder / f"synthetic_{index}.json").write_text(json.dumps(annotation), encoding="utf-8")
            paths, texts, targets = data.load_dataset(folder, include_targets=True)
            self.assertEqual(len(paths), 3)
            self.assertEqual(texts, ["", "synthetic", "legacy"])
            np.testing.assert_array_equal(targets[0], np.zeros(9))
            np.testing.assert_array_equal(targets[1], [1, 0, 0, 0, 0, 0, 1, 0, 0])
            np.testing.assert_array_equal(targets[2], np.zeros(9))
            annotations[1]["symptom"] = {"malformed": "ignored at inference"}
            (folder / "synthetic_1.json").write_text(json.dumps(annotations[1]), encoding="utf-8")
            self.assertIsNone(data.load_dataset(folder, include_targets=False)[2])
            with self.assertRaises(ValueError):
                data.load_dataset(folder, include_targets=True)

    def test_three_way_split_preserves_rows_conflicts_and_group_isolation(self):
        rng = np.random.default_rng(17)
        originals = [f"synthetic call {index}" for index in range(500)]
        texts = originals + ["  synthetic\ncall " + str(index) for index in range(35)]
        targets = (rng.random((len(texts), 9)) < np.linspace(.12, .4, 9)).astype(np.int8)
        targets[500] = 1 - targets[0]
        untouched = targets.copy()
        fit, development, confirmation, report = data.three_way_split(texts, targets)
        splits = (fit, development, confirmation)
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(len(texts)))
        np.testing.assert_array_equal(targets, untouched)
        group_sets = [{data.text_group(texts[index]) for index in ids} for ids in splits]
        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(group_sets[first] & group_sets[second])
        self.assertGreater(report["conflicting_label_groups"], 0)
        self.assertEqual(report["text_groups"], 500)
        self.assertEqual(report["duplicate_extra_samples"], 35)
        for ids, expected in zip(splits, (.7, .15, .15)):
            self.assertAlmostEqual(len(ids) / len(texts), expected, delta=.02)
            np.testing.assert_allclose(targets[ids].mean(axis=0), targets.mean(axis=0), atol=.035)
        repeated = data.three_way_split(texts, targets)
        for actual, expected in zip(repeated[:3], splits):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(repeated[3], report)
        self.assertNotIn("synthetic call", json.dumps(report))
        self.assertEqual(report["coverage_warnings"], [])
        changed = targets.copy()
        changed[0, 0] = 1 - changed[0, 0]
        self.assertNotEqual(report["data_digest"], data.dataset_digest(texts, changed))

    def test_impossible_label_coverage_is_reported_without_dropping_rows(self):
        texts = ["synthetic one", "synthetic two", "synthetic three"]
        targets = np.zeros((3, 9), dtype=np.int8)
        targets[0, 0] = 1
        fit, development, confirmation, report = data.three_way_split(texts, targets)
        self.assertEqual([len(part) for part in (fit, development, confirmation)], [1, 1, 1])
        self.assertTrue(report["coverage_warnings"])
        with self.assertRaisesRegex(ValueError, "three distinct"):
            data.three_way_split(["same", " same ", "other"], targets)

    def test_pilot_subset_has_exact_size_preserves_labels_and_uses_only_fit(self):
        rng = np.random.default_rng(13)
        texts = [f"private synthetic transcript {index}" for index in range(6000)]
        targets = (rng.random((6000, 9)) < np.linspace(.01, .45, 9)).astype(np.int8)
        fit, development, confirmation, split_report = data.three_way_split(texts, targets)
        original_targets = targets.copy()
        original_fit = fit.copy()
        selected, report = data.sample_fit_subset(texts, targets, fit)
        self.assertEqual(len(selected), 4000)
        self.assertEqual(len(np.unique(selected)), 4000)
        self.assertTrue(set(selected) <= set(fit))
        self.assertFalse(set(selected) & set(development))
        self.assertFalse(set(selected) & set(confirmation))
        np.testing.assert_array_equal(targets, original_targets)
        np.testing.assert_array_equal(fit, original_fit)
        np.testing.assert_allclose(targets[selected].mean(axis=0), targets[fit].mean(axis=0), atol=.003)
        self.assertEqual(report["requested_samples"], 4000)
        self.assertEqual(report["actual_samples"], 4000)
        self.assertEqual(report["unfilled_samples"], 0)
        self.assertTrue(report["original_targets_preserved"])
        self.assertEqual(report["data_digest"], split_report["data_digest"])
        self.assertEqual(report["parent_fit_digest"], split_report["splits"]["fit"]["indices_digest"])
        self.assertEqual(report["coverage_warnings"], [])
        self.assertNotIn("private synthetic", json.dumps(report))
        repeated, repeated_report = data.sample_fit_subset(texts, targets, fit[::-1])
        np.testing.assert_array_equal(selected, repeated)
        self.assertEqual(report, repeated_report)

    def test_pilot_subset_keeps_duplicate_groups_and_reports_largest_feasible_size(self):
        texts = [f"synthetic group {group}" if copy == 0 else f" synthetic\ngroup {group} "
                 for group in range(60) for copy in range(3)]
        rng = np.random.default_rng(8)
        targets = (rng.random((180, 9)) < np.linspace(.04, .4, 9)).astype(np.int8)
        # Conflicting original labels remain intact within an indivisible group.
        targets[1] = 1 - targets[0]
        untouched = targets.copy()
        fit = np.arange(150)
        selected, report = data.sample_fit_subset(texts, targets, fit, sample_count=20)
        self.assertEqual(len(selected), 18)
        self.assertEqual(report["unfilled_samples"], 2)
        self.assertFalse(report["exact_sample_count"])
        self.assertEqual(report["selected_duplicate_groups"], 6)
        for group in range(60):
            membership = np.isin(np.arange(group * 3, group * 3 + 3), selected)
            self.assertTrue(membership.all() or not membership.any())
        np.testing.assert_array_equal(targets, untouched)
        repeated, repeated_report = data.sample_fit_subset(texts, targets, fit, sample_count=20)
        np.testing.assert_array_equal(selected, repeated)
        self.assertEqual(report, repeated_report)

    def test_pilot_subset_exact_capacity_can_recombine_large_groups(self):
        sizes = [6, 4, 3, 3]
        texts = [f"synthetic group {group}" for group, size in enumerate(sizes) for _ in range(size)]
        targets = np.zeros((len(texts), 9), dtype=np.int8)
        selected, report = data.sample_fit_subset(texts, targets, np.arange(16), sample_count=7)
        self.assertEqual(len(selected), 7)
        self.assertEqual(report["selected_text_groups"], 2)
        self.assertEqual(report["coverage_warnings"], [])
        self.assertEqual(report["coverage"]["고열"], {"positive": False, "negative": True})

    def test_pilot_subset_allocation_ignores_non_fit_labels(self):
        texts = [f"synthetic transcript {index}" for index in range(300)]
        rng = np.random.default_rng(7)
        targets = (rng.random((300, 9)) < .2).astype(np.int8)
        fit = np.arange(200)
        selected, report = data.sample_fit_subset(texts, targets, fit, sample_count=80)
        changed = targets.copy()
        changed[200:] = 1 - changed[200:]
        repeated, changed_report = data.sample_fit_subset(texts, changed, fit, sample_count=80)
        np.testing.assert_array_equal(selected, repeated)
        self.assertEqual(report["indices_digest"], changed_report["indices_digest"])
        self.assertNotEqual(report["data_digest"], changed_report["data_digest"])
        self.assertNotEqual(report["subset_digest"], changed_report["subset_digest"])

    def test_pilot_subset_rejects_invalid_requests_and_partial_fit_groups(self):
        texts = [f"synthetic transcript {index}" for index in range(10)]
        targets = np.zeros((10, 9), dtype=np.int8)
        for invalid in (0, -1, True, 1.5, 11):
            with self.subTest(sample_count=invalid), self.assertRaises(ValueError):
                data.sample_fit_subset(texts, targets, np.arange(10), sample_count=invalid)
        for invalid_fit in ([0, 0], [-1, 0], [0, 10], [0., 1.], [[0, 1]], [True, False]):
            with self.subTest(fit=invalid_fit), self.assertRaises(ValueError):
                data.sample_fit_subset(texts, targets, invalid_fit, sample_count=1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            data.sample_fit_subset(texts, targets, np.arange(10), sample_count=1, seed=-1)
        duplicates = ["same", " same ", "third"]
        with self.assertRaisesRegex(ValueError, "complete normalized text groups"):
            data.sample_fit_subset(duplicates, targets[:3], [0, 2], sample_count=1)
        with self.assertRaisesRegex(ValueError, "No complete fit text group"):
            data.sample_fit_subset(["same", " same "], targets[:2], [0, 1], sample_count=1)

    def test_submission_identifiers_are_output_only_and_scores_must_be_binary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary) / "mission3.csv"
            flags = np.zeros((1, 9), dtype=np.int8)
            flags[0, 6] = 1
            data.write_predictions(["synthetic.json"], flags, output)
            self.assertIn("['오심']", output.read_text(encoding="utf-8-sig"))
            with self.assertRaises(ValueError):
                data.write_predictions(["synthetic.json"], flags * .5, output)


if __name__ == "__main__":
    unittest.main()
