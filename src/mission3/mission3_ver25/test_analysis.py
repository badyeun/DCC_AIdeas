"""Synthetic fixed-threshold, complementarity, AP, and paired bootstrap tests."""
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from . import analysis


def repeat_label(column):
    return np.repeat(np.asarray(column)[:, None], 9, axis=1)


class AnalysisTests(unittest.TestCase):
    def test_metrics_match_hand_computed_counts_and_zero_tie_is_positive(self):
        truth = repeat_label([1, 1, 0, 0])
        logits = repeat_label([0., -2., 1., -3.])
        report = analysis.metrics(truth, logits)
        row = report["per_symptom"]["오심"]
        self.assertEqual(row["true_positive"], 1)
        self.assertEqual(row["false_positive"], 1)
        self.assertEqual(row["false_negative"], 1)
        self.assertEqual(row["true_negative"], 1)
        self.assertEqual((row["f1"], row["precision"], row["recall"]), (.5, .5, .5))
        self.assertEqual(report["macro_f1"], .5)
        self.assertAlmostEqual(row["average_precision"], 7 / 12)
        self.assertEqual(analysis.metrics(np.zeros((4, 9)), logits)["macro_average_precision"], None)
        with self.assertRaises(ValueError):
            analysis.metrics(truth, np.full((4, 9), np.nan))

    def test_probability_mean_has_mathematically_stable_symmetric_ties(self):
        left = repeat_label([1000., 50., .3, -50., -1000., 4., 2., -3.])
        right = repeat_label([-1000., -50., -.3, 50., 1000., -3., -3., 2.])
        mixed = analysis.ensemble_logits(left, right)
        self.assertTrue(np.isfinite(mixed).all())
        np.testing.assert_array_equal(mixed[:5], np.zeros((5, 9)))
        np.testing.assert_array_equal(mixed >= 0, left + right >= 0)
        left = repeat_label([5., 1., -2., 3.])
        right = repeat_label([-3., 1., -1., -2.])
        probability_mean = .5 / (1 + np.exp(-left)) + .5 / (1 + np.exp(-right))
        mixed = analysis.ensemble_logits(left, right)
        np.testing.assert_allclose(1 / (1 + np.exp(-mixed)), probability_mean)
        truth = repeat_label([1, 0, 1, 0])
        expected = average_precision_score(truth[:, 0], probability_mean[:, 0])
        self.assertEqual(analysis.metrics(truth, mixed)["per_symptom"]["오심"]["average_precision"], expected)
        # Equal logit sums are different probability means: preserve AP ranking.
        self.assertEqual((left + right)[0, 0], (left + right)[1, 0])
        self.assertLess(mixed[0, 0], mixed[1, 0])
        smallest = np.nextafter(0., 1.)
        extreme_left = repeat_label([-smallest, 1e308, -1e308])
        extreme_right = repeat_label([0., 1e308, -1e308])
        extreme_mix = analysis.ensemble_logits(extreme_left, extreme_right)
        np.testing.assert_array_equal(extreme_mix[:, 0] >= 0, [False, True, False])

    def test_paired_error_accounting_distinguishes_fixes_new_and_common_errors(self):
        truth = repeat_label([1, 1, 1, 0, 0, 0])
        old = repeat_label([-1, -1, 1, 1, 1, -1])
        new = repeat_label([-1, 1, -1, 1, -1, 1])
        row = analysis.paired_error_changes(truth, old, new)["per_symptom"]["오심"]
        for key in ("common_false_negative", "common_false_positive", "fixed_false_negative",
                    "fixed_false_positive", "new_false_negative", "new_false_positive"):
            self.assertEqual(row[key], 1)
        self.assertEqual(row["control_false_negative"], row["common_false_negative"] + row["fixed_false_negative"])
        self.assertEqual(row["candidate_false_positive"], row["common_false_positive"] + row["new_false_positive"])

    def test_bootstrap_is_paired_reproducible_and_resamples_complete_calls(self):
        truth = repeat_label([1, 1, 1, 0, 0, 0])
        old = repeat_label([-1, -1, 1, 1, 1, -1])
        perfect = (truth * 2 - 1).astype(float)
        identical = analysis.paired_bootstrap_ci(truth, old, old, iterations=80)
        self.assertEqual(identical["macro_f1_delta"], {"estimate": 0., "lower": 0., "upper": 0.})
        report = analysis.paired_bootstrap_ci(truth, old, perfect, iterations=80, seed=12)
        self.assertEqual(report, analysis.paired_bootstrap_ci(truth, old, perfect, iterations=80, seed=12))
        self.assertAlmostEqual(report["macro_f1_delta"]["estimate"], 2 / 3)
        self.assertGreaterEqual(report["macro_f1_delta"]["lower"], 0)
        for interval in report["per_symptom_f1_delta"].values():
            for key in ("estimate", "lower", "upper"):
                self.assertAlmostEqual(interval[key], report["macro_f1_delta"][key])
        with self.assertRaises(ValueError):
            analysis.paired_bootstrap_ci(truth, old, perfect, iterations=1)


if __name__ == "__main__":
    unittest.main()
