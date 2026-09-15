import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver5 import calibration as module
from mission3_ver5.calibration import _fit_mixture, apply_calibrator, calibrate_crossfit
from mission3_ver5.ver4_runtime.calibration import (
    _grouped_splits, apply_calibrator as apply_legacy,
    calibrate_crossfit as crossfit_legacy, fit_calibrator as fit_legacy,
)


def test_beta_zero_reproduces_legacy_and_ties_prefer_it():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 2, size=(50, 9))
    base = rng.uniform(size=(50, 9))
    sparse = base.copy()
    old = fit_legacy(truth, base, sparse)
    new = _fit_mixture(truth, base, sparse, base)
    assert new["candidate_weights"] == [0.0] * 9
    assert new["neural_weights"] == old["neural_weights"]
    assert new["thresholds"] == old["thresholds"]
    np.testing.assert_array_equal(apply_calibrator(base, sparse, None, new), apply_legacy(base, sparse, old))


def test_mixture_search_matches_independent_exhaustive_f1():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, size=(28, 9))
    arrays = [np.round(rng.uniform(size=(28, 9)), 1) for _ in range(3)]
    fitted = _fit_mixture(truth, *arrays)
    predictions = apply_calibrator(*arrays, fitted) >= fitted["thresholds"]
    for column in (0, 5, 8):
        best = 0.0
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            for beta in (0, 0.25, 0.5, 0.75, 1):
                p = (1 - beta) * (alpha * arrays[0][:, column] + (1 - alpha) * arrays[1][:, column]) + beta * arrays[2][:, column]
                best = max(best, *(f1_score(truth[:, column], p >= threshold, zero_division=0)
                                   for threshold in [0.0, *np.unique(p), np.nextafter(1.0, np.inf)]))
        assert f1_score(truth[:, column], predictions[:, column]) == pytest.approx(best)


def test_constants_and_saturated_probabilities_remain_valid():
    truth = np.tile([0, 1, 0, 1, 0, 1, 0, 1, 0], (12, 1))
    probabilities = np.ones((12, 9), dtype=np.float32)
    fitted = _fit_mixture(truth, probabilities, probabilities, probabilities)
    assert fitted["thresholds"][0] > 1
    assert fitted["thresholds"][1] == 0
    result = apply_calibrator(probabilities, probabilities, probabilities, fitted)
    np.testing.assert_array_equal(result, np.ones((12, 9)))
    np.testing.assert_array_equal(result >= fitted["thresholds"], truth)
    for alpha in (0, 0.25, 0.5, 0.75, 1):
        for beta in (0, 0.25, 0.5, 0.75, 1):
            saturated = apply_calibrator(probabilities, probabilities, probabilities,
                                         {**fitted, "neural_weights": [alpha] * 9, "candidate_weights": [beta] * 9})
            np.testing.assert_array_equal(saturated, np.ones((12, 9)))
    json.dumps(fitted, allow_nan=False)


def test_candidate_inputs_and_weights_are_validated_even_for_zero_beta():
    valid = np.full((12, 9), 0.5)
    fitted = {"neural_weights": [1.0] * 9, "candidate_weights": [0.0] * 9, "thresholds": [0.5] * 9}
    for invalid in [np.full((12, 9), np.nan), np.full((12, 9), np.inf), valid + 1, valid[:, :8], valid[:11]]:
        with pytest.raises(ValueError):
            apply_calibrator(valid, valid, invalid, fitted)
    for weights in [[0.0] * 8, [np.nan] * 9, [-0.1] * 9, [1.1] * 9]:
        with pytest.raises(ValueError):
            apply_calibrator(valid, valid, valid, {**fitted, "candidate_weights": weights})
    with pytest.raises(ValueError, match="required"):
        apply_calibrator(valid, valid, None, {**fitted, "candidate_weights": [0.25] * 9})


def test_crossfit_group_separation_and_candidate_success():
    rng = np.random.default_rng(9)
    truth = rng.integers(0, 2, size=(90, 9))
    groups = np.repeat(np.arange(30), 3)
    # Unique row IDs in probabilities let the spy verify which rows each fit saw.
    base = np.full((90, 9), 0.5) + np.arange(90)[:, None] * 1e-6
    candidate = 0.2 + truth * 0.6
    expected = list(_grouped_splits(truth, groups, 44, 5))
    calls = []
    actual_fit = module._fit_mixture

    def spy(y, b, s, c):
        calls.append(b[:, 0].copy())
        return actual_fit(y, b, s, c)

    with patch.object(module, "_fit_mixture", side_effect=spy):
        fitted, report = calibrate_crossfit(truth, base, base, candidate, groups)
    for observed, (train, test) in zip(calls[:5], expected):
        np.testing.assert_array_equal(observed, base[train, 0])
        assert set(groups[train]).isdisjoint(groups[test])
    assert len(calls) == 6  # Five fold-training fits, then one deployment fit.
    assert report["selected_name"] == "candidate_mixture"
    assert report["selected_crossfit_macro_f1"] == pytest.approx(1.0)
    assert all(beta > 0 for beta in fitted["candidate_weights"])
    assert sum(f["score_samples"] for f in report["crossfit_fold_counts"]) == 90
    json.dumps(report, allow_nan=False)


def test_equal_procedure_score_falls_back_to_original_baseline():
    truth = np.tile([0, 1], 40)[:, None].repeat(9, axis=1)
    base = 0.2 + truth * 0.6
    candidate = 1.0 - base
    fitted, report = calibrate_crossfit(truth, base, base, candidate, np.arange(80))
    assert report["baseline_crossfit_macro_f1"] == pytest.approx(1.0)
    assert report["selected_name"] == "ver4_baseline"
    assert report["fallback_reason"] is not None
    assert fitted["candidate_weights"] == [0.0] * 9
    old = fit_legacy(truth, base, base)
    assert fitted["neural_weights"] == old["neural_weights"]
    assert fitted["thresholds"] == old["thresholds"]


def test_overfit_candidate_falls_back_and_samefold_baseline_matches_legacy():
    rng = np.random.default_rng(18)
    truth = rng.integers(0, 2, size=(50, 9))
    base = 0.2 + 0.5 * rng.uniform(size=(50, 9)) + 0.1 * truth
    candidate = rng.uniform(size=(50, 9))
    groups = np.arange(50)
    fitted, report = calibrate_crossfit(truth, base, base, candidate, groups)
    old, old_report = crossfit_legacy(truth, base, base, groups, seed=44, n_splits=5)
    assert report["candidate_crossfit_macro_f1"] < report["baseline_crossfit_macro_f1"]
    assert report["baseline_crossfit_macro_f1"] == old_report["calibration_crossfit_macro_f1"]
    assert report["selected_name"] == "ver4_baseline"
    assert fitted["candidate_weights"] == [0.0] * 9
    assert fitted["neural_weights"] == old["neural_weights"]
    assert fitted["thresholds"] == old["thresholds"]
