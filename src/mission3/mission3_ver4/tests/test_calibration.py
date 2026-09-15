import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver4.calibration import (
    _best_threshold, _grouped_splits, apply_calibrator, calibrate_crossfit, fit_calibrator,
)


def test_exact_threshold_matches_exhaustive_including_ties():
    rng = np.random.default_rng(18)
    for _ in range(40):
        truth = rng.integers(0, 2, size=35)
        probabilities = np.round(rng.uniform(size=35), 1)
        threshold, reported = _best_threshold(truth, probabilities)
        candidates = [0.0, *np.unique(probabilities), np.nextafter(1.0, np.inf)]
        expected = max(f1_score(truth, probabilities >= t, zero_division=0) for t in candidates)
        assert reported == pytest.approx(expected)
        assert f1_score(truth, probabilities >= threshold, zero_division=0) == pytest.approx(expected)


def test_constant_labels_and_tied_probabilities():
    truth = np.tile([0, 1, 0, 1, 0, 1, 0, 1, 0], (8, 1))
    probabilities = np.ones_like(truth, dtype=float)
    calibrator = fit_calibrator(truth, probabilities, probabilities)
    predicted = apply_calibrator(probabilities, probabilities, calibrator) >= calibrator["thresholds"]
    np.testing.assert_array_equal(predicted, truth)
    assert calibrator["thresholds"][0] > 1
    assert calibrator["thresholds"][1] == 0
    json.dumps(calibrator, allow_nan=False)
    threshold, score = _best_threshold(np.array([1, 0, 1, 0]), np.full(4, 0.5))
    assert threshold <= 0.5
    assert score == pytest.approx(2 / 3)


def test_weight_endpoints_and_probability_validation():
    a = np.full((12, 9), 0.9)
    b = np.full((12, 9), 0.1)
    for weight, expected in [(0, b), (1, a)]:
        result = apply_calibrator(a, b, {"neural_weights": [weight] * 9, "thresholds": [0.5] * 9})
        np.testing.assert_array_equal(result, expected)
    for invalid in [np.full((12, 9), np.nan), np.full((12, 9), np.inf), a + 1, a[:, :8], a[:0]]:
        with pytest.raises(ValueError):
            fit_calibrator(np.zeros((12, 9)), invalid, b)
    with pytest.raises(ValueError):
        fit_calibrator(np.full((12, 9), 0.5), a, b)


def test_fit_can_choose_either_pure_model():
    truth = np.tile([0, 1], 20)[:, None].repeat(9, axis=1)
    good = 0.49 + truth * 0.02
    adversarial = 0.99 - truth * 0.98
    assert fit_calibrator(truth, good, adversarial)["neural_weights"] == [1.0] * 9
    assert fit_calibrator(truth, adversarial, good)["neural_weights"] == [0.0] * 9


def test_adjacent_float_threshold_keeps_greater_equal_decision_correct():
    lower = 0.5
    upper = np.nextafter(lower, 1.0)
    threshold, score = _best_threshold(np.array([0, 1]), np.array([lower, upper]))
    assert score == 1.0
    assert lower < threshold <= upper


def test_crossfit_keeps_duplicate_groups_and_scores_every_row():
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(20), 3)
    truth = rng.integers(0, 2, size=(60, 9), dtype=np.int8)
    seen = []
    for train, test in _grouped_splits(truth, groups, seed=42, n_splits=5):
        assert set(groups[train]).isdisjoint(groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(60))
    neural = 0.2 + truth * 0.6
    sparse = rng.uniform(size=(60, 9))
    final, report = calibrate_crossfit(truth, neural, sparse, groups)
    assert report["calibration_crossfit_macro_f1"] == pytest.approx(1.0)
    assert report["calibration_samples"] == 60
    assert report["calibration_groups"] == 20
    assert report["crossfit_splits"] == 5
    assert sum(f["score_samples"] for f in report["crossfit_fold_counts"]) == 60
    json.dumps({"calibrator": final, "report": report}, allow_nan=False)
    with pytest.raises(ValueError, match="two distinct groups"):
        calibrate_crossfit(truth, neural, sparse, np.zeros(60))
