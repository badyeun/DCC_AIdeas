import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver4.calibration import (
    FIXED_THRESHOLD, _best_threshold, _grouped_splits, apply_calibrator, calibrate_crossfit,
    fit_calibrator,
)


def test_threshold_is_always_the_mandated_half_and_score_matches_it():
    """The rules forbid per-symptom thresholds; 0.5 decides every label."""
    rng = np.random.default_rng(18)
    for _ in range(40):
        truth = rng.integers(0, 2, size=35)
        probabilities = np.round(rng.uniform(size=35), 1)
        threshold, reported = _best_threshold(truth, probabilities)
        assert threshold == FIXED_THRESHOLD == 0.5
        expected = f1_score(truth, probabilities >= 0.5, zero_division=0)
        assert reported == pytest.approx(expected)


def test_fitted_calibrator_stores_one_shared_threshold():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, size=(120, 9), dtype=np.int8)
    neural = rng.uniform(size=(120, 9))
    sparse = rng.uniform(size=(120, 9))
    calibrator = fit_calibrator(truth, neural, sparse)
    assert calibrator["thresholds"] == [0.5] * 9
    json.dumps(calibrator, allow_nan=False)


def test_constant_labels_and_tied_probabilities():
    truth = np.tile([0, 1, 0, 1, 0, 1, 0, 1, 0], (8, 1))
    probabilities = np.ones_like(truth, dtype=float)
    calibrator = fit_calibrator(truth, probabilities, probabilities)
    # Every probability is 1.0, so a 0.5 threshold must fire for every label --
    # a constant-negative label can no longer be switched off by its threshold.
    predicted = apply_calibrator(probabilities, probabilities, calibrator) >= calibrator["thresholds"]
    np.testing.assert_array_equal(predicted, np.ones_like(truth))
    assert calibrator["thresholds"] == [0.5] * 9
    json.dumps(calibrator, allow_nan=False)
    threshold, score = _best_threshold(np.array([1, 0, 1, 0]), np.full(4, 0.5))
    assert threshold == 0.5
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


def test_greater_equal_decision_includes_a_probability_exactly_at_half():
    lower = np.nextafter(0.5, 0.0)
    threshold, score = _best_threshold(np.array([0, 1]), np.array([lower, 0.5]))
    assert threshold == 0.5
    # 0.5 itself is predicted positive (>=), the value just below it is not.
    assert score == 1.0


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
