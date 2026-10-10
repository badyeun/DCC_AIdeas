import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver6 import calibration as module
from mission3_ver6.calibration import _fit_mixture, apply_calibrator, calibrate_crossfit
from mission3_ver6.ver4_runtime.calibration import (
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


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_mixture_search_matches_independent_exhaustive_f1(blend_space):
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, size=(28, 9))
    arrays = [np.round(rng.uniform(size=(28, 9)), 1) for _ in range(3)]
    fitted = _fit_mixture(truth, *arrays, blend_space=blend_space)
    predictions = apply_calibrator(*arrays, fitted) >= fitted["thresholds"]
    values = arrays
    if blend_space == "logit":
        values = [np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                  for p in arrays]
    for column in (0, 5, 8):
        best = 0.0
        for alpha in (0, 0.25, 0.5, 0.75, 1):
            for beta in (0, 0.25, 0.5, 0.75, 1):
                p = (1 - beta) * (alpha * values[0][:, column] + (1 - alpha) * values[1][:, column]) + beta * values[2][:, column]
                if blend_space == "logit":
                    p = 1 / (1 + np.exp(-p))
                # Thresholds are fixed at 0.5, so only the blend is searched.
                best = max(best, f1_score(truth[:, column], p >= 0.5, zero_division=0))
        assert fitted["thresholds"] == [0.5] * 9
        assert f1_score(truth[:, column], predictions[:, column]) == pytest.approx(best)


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_constants_and_saturated_probabilities_remain_valid(blend_space):
    truth = np.tile([0, 1, 0, 1, 0, 1, 0, 1, 0], (12, 1))
    probabilities = np.ones((12, 9), dtype=np.float32)
    fitted = _fit_mixture(truth, probabilities, probabilities, probabilities, blend_space=blend_space)
    assert fitted["thresholds"] == [0.5] * 9
    result = apply_calibrator(probabilities, probabilities, probabilities, fitted)
    expected = np.full((12, 9), 1 - 1e-6 if blend_space == "logit" else 1)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-16)
    # Saturated probabilities all clear 0.5; a constant-negative label can no
    # longer be silenced by giving it its own threshold above 1.
    np.testing.assert_array_equal(result >= fitted["thresholds"], np.ones_like(truth))
    for alpha in (0, 0.25, 0.5, 0.75, 1):
        for beta in (0, 0.25, 0.5, 0.75, 1):
            saturated = apply_calibrator(probabilities, probabilities, probabilities,
                                         {**fitted, "neural_weights": [alpha] * 9, "candidate_weights": [beta] * 9})
            np.testing.assert_allclose(saturated, expected, rtol=0, atol=2e-16)
    json.dumps(fitted, allow_nan=False)


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_candidate_inputs_and_weights_are_validated_even_for_zero_beta(blend_space):
    valid = np.full((12, 9), 0.5)
    fitted = {"neural_weights": [1.0] * 9, "candidate_weights": [0.0] * 9,
              "thresholds": [0.5] * 9, "blend_space": blend_space}
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

    def spy(y, b, s, c, blend_space="raw"):
        calls.append((blend_space, b[:, 0].copy()))
        return actual_fit(y, b, s, c, blend_space=blend_space)

    with patch.object(module, "_fit_mixture", side_effect=spy):
        fitted, report = calibrate_crossfit(truth, base, base, candidate, groups)
    for fold, (train, test) in enumerate(expected):
        for offset, space in enumerate(("raw", "logit")):
            observed_space, observed = calls[2 * fold + offset]
            assert observed_space == space
            np.testing.assert_array_equal(observed, base[train, 0])
        assert set(groups[train]).isdisjoint(groups[test])
    assert len(calls) == 11  # Both families in each fold, then one deployment fit.
    assert calls[-1][0] == "raw"  # Candidate-family ties prefer the original raw blend.
    np.testing.assert_array_equal(calls[-1][1], base[:, 0])
    assert report["selected_name"] == "candidate_mixture"
    assert report["selected_crossfit_macro_f1"] == pytest.approx(1.0)
    assert report["candidate_raw_crossfit_macro_f1"] == pytest.approx(1.0)
    assert report["candidate_logit_crossfit_macro_f1"] == pytest.approx(1.0)
    assert report["candidate_blend_space"] == report["selected_blend_space"] == "raw"
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
    assert report["selected_blend_space"] == fitted["blend_space"] == "raw"
    assert fitted["candidate_weights"] == [0.0] * 9
    old = fit_legacy(truth, base, base)
    assert fitted["neural_weights"] == old["neural_weights"]
    assert fitted["thresholds"] == old["thresholds"]


def test_logit_mixture_matches_weighted_odds_and_roundtrips_for_inference():
    base = np.tile([0, 1, .2, .4, .6, .8, .3, .5, .7], (4, 1))
    sparse = np.flip(base, axis=1).copy()
    candidate = np.full_like(base, .37)
    alpha = np.linspace(0, 1, 9)
    beta = np.linspace(1, 0, 9)
    fitted = {"neural_weights": alpha.tolist(), "candidate_weights": beta.tolist(),
              "thresholds": [.5] * 9, "blend_space": "logit"}
    odds = []
    for probabilities in (base, sparse, candidate):
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        odds.append(probabilities / (1 - probabilities))
    expected_odds = odds[0] ** ((1 - beta) * alpha) * odds[1] ** ((1 - beta) * (1 - alpha)) * odds[2] ** beta
    expected = expected_odds / (1 + expected_odds)
    restored = json.loads(json.dumps(fitted, allow_nan=False))
    actual = apply_calibrator(base, sparse, candidate, restored)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("alpha,beta,unused", [(0, .25, "base"), (1, .25, "sparse"), (.5, 1, "both")])
def test_logit_zero_coefficients_allow_skipped_inference_components(alpha, beta, unused):
    rng = np.random.default_rng(31)
    base, sparse, candidate = [rng.uniform(size=(12, 9)) for _ in range(3)]
    fitted = {"neural_weights": [alpha] * 9, "candidate_weights": [beta] * 9,
              "thresholds": [.5] * 9, "blend_space": "logit"}
    expected = apply_calibrator(base, sparse, candidate, fitted)
    if unused in ("base", "both"):
        base = np.zeros_like(base)
    if unused in ("sparse", "both"):
        sparse = np.ones_like(sparse)
    np.testing.assert_array_equal(apply_calibrator(base, sparse, candidate, fitted), expected)


def test_logit_zero_beta_needs_no_candidate_and_missing_space_keeps_legacy():
    rng = np.random.default_rng(32)
    base, sparse, candidate = [rng.uniform(size=(12, 9)) for _ in range(3)]
    fitted = {"neural_weights": [.25] * 9, "candidate_weights": [0.] * 9, "thresholds": [.5] * 9}
    np.testing.assert_array_equal(apply_calibrator(base, sparse, None, fitted), apply_legacy(base, sparse, fitted))
    fitted["blend_space"] = "logit"
    np.testing.assert_array_equal(apply_calibrator(base, sparse, None, fitted), apply_calibrator(base, sparse, candidate, fitted))
    with pytest.raises(ValueError, match="blend_space"):
        apply_calibrator(base, sparse, candidate, {**fitted, "blend_space": "unknown"})


def test_crossfit_can_select_logit_family_globally_with_transparent_scores():
    # At equal blend weights the two spaces agree on a 0.5 decision, so the
    # families only diverge when one component is saturated and the chosen
    # weights are uneven. A saturated sparse model produces exactly that.
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, (180, 9))
    truth = (signal > .4).astype(np.int8)
    base = 1 / (1 + np.exp(-(1.2 * signal + rng.normal(0, .8, (180, 9)))))
    sparse = 1 / (1 + np.exp(-(4.0 * signal + rng.normal(0, 2.5, (180, 9)))))
    candidate = 1 / (1 + np.exp(-(2.0 * signal + rng.normal(0, 1.2, (180, 9)))))
    fitted, report = calibrate_crossfit(truth, base, sparse, candidate, np.arange(len(truth)))
    assert report["candidate_logit_crossfit_macro_f1"] > report["candidate_raw_crossfit_macro_f1"]
    assert report["candidate_crossfit_macro_f1"] == report["candidate_logit_crossfit_macro_f1"]
    assert report["selected_crossfit_macro_f1"] > report["baseline_crossfit_macro_f1"]
    assert report["selected_name"] == "candidate_mixture"
    assert report["candidate_blend_space"] == report["selected_blend_space"] == fitted["blend_space"] == "logit"
    assert report["per_symptom_crossfit"]["selected"] == report["per_symptom_crossfit"]["candidate_logit"]
    assert fitted["thresholds"] == [0.5] * 9


def test_samefold_baseline_matches_legacy_exactly():
    """The ver4 branch must stay bit-identical to the standalone ver4 routine.

    Adoption/fallback behaviour is covered deterministically by the
    ``_controlled_five_fold_gate`` tests below; this one pins the baseline path.
    """
    rng = np.random.default_rng(18)
    truth = rng.integers(0, 2, size=(50, 9))
    base = 0.2 + 0.5 * rng.uniform(size=(50, 9)) + 0.1 * truth
    candidate = rng.uniform(size=(50, 9))
    groups = np.arange(50)
    _, report = calibrate_crossfit(truth, base, base, candidate, groups)
    old, old_report = crossfit_legacy(truth, base, base, groups, seed=44, n_splits=5)
    assert report["baseline_crossfit_macro_f1"] == old_report["calibration_crossfit_macro_f1"]
    assert report["per_symptom_crossfit"]["baseline"] == old_report["per_symptom_crossfit"]
    assert old["thresholds"] == [0.5] * 9


def test_fallback_keeps_the_legacy_baseline_calibrator_unchanged():
    """A candidate that cannot beat the gate leaves ver4's calibrator in place."""
    rng = np.random.default_rng(7)
    truth = rng.integers(0, 2, size=(60, 9))
    base = 0.2 + 0.5 * rng.uniform(size=(60, 9)) + 0.3 * truth
    candidate = np.full((60, 9), 0.5)  # No information: cannot improve any fold.
    groups = np.arange(60)
    fitted, report = calibrate_crossfit(truth, base, base, candidate, groups)
    old, _ = crossfit_legacy(truth, base, base, groups, seed=44, n_splits=5)
    assert report["selected_name"] == "ver4_baseline"
    assert fitted["candidate_weights"] == [0.0] * 9
    assert fitted["neural_weights"] == old["neural_weights"]
    assert fitted["thresholds"] == old["thresholds"] == [0.5] * 9


def _controlled_five_fold_gate(monkeypatch, improved_folds, **gate):
    """Control fitted procedures; use real predictions/F1 to test adoption rules.

    Each balanced fold has ten rows. Baseline predicts all-positive; candidate
    is perfect in selected folds and exactly baseline elsewhere. Mock only the
    fitting/split machinery so calibration search noise cannot change the gate
    boundary being tested.
    """
    truth = np.tile([0, 1], 25)[:, None].repeat(9, axis=1)
    base = np.full(truth.shape, 0.9)
    candidate = base.copy()
    for fold in improved_folds:
        rows = slice(fold * 10, (fold + 1) * 10)
        candidate[rows] = 0.1 + 0.8 * truth[rows]
    indices = np.arange(len(truth))
    folds = [(np.setdiff1d(indices, indices[fold * 10:(fold + 1) * 10]),
              indices[fold * 10:(fold + 1) * 10]) for fold in range(5)]
    baseline = {"neural_weights": [1.0] * 9, "thresholds": [0.5] * 9}
    monkeypatch.setattr(module, "_grouped_splits", lambda *args: iter(folds))
    monkeypatch.setattr(module, "_fit_baseline", lambda *args: dict(baseline))

    def fit_candidate(*args, blend_space="raw"):
        return {**baseline, "candidate_weights": [1.0] * 9, "blend_space": blend_space}

    monkeypatch.setattr(module, "_fit_mixture", fit_candidate)
    fitted, report = calibrate_crossfit(truth, base, base, candidate, np.repeat(np.arange(5), 10), **gate)
    expected_gain = (f1_score(truth, candidate >= 0.5, average="macro", zero_division=0)
                     - f1_score(truth, base >= 0.5, average="macro", zero_division=0))
    assert report["selection_gain"] == pytest.approx(expected_gain, abs=1e-15)
    assert report["candidate_improved_folds"] == len(improved_folds)
    assert len(report["candidate_fold_gains"]) == 5
    return fitted, report, expected_gain


def test_gain_guard_rejects_positive_but_insufficient_improvement(monkeypatch):
    fitted, report, gain = _controlled_five_fold_gate(monkeypatch, range(5), min_gain=0.34)
    assert 0 < gain < report["minimum_gain"]
    assert report["candidate_improved_folds"] == 5
    assert report["selected_name"] == "ver4_baseline"
    assert report["fallback_reason"] == "candidate_gain_below_minimum"
    assert fitted["candidate_weights"] == [0.0] * 9
    assert report["selected_crossfit_macro_f1"] == report["baseline_crossfit_macro_f1"]


def test_gain_guard_accepts_exact_minimum_and_rejects_next_float(monkeypatch):
    _, initial, gain = _controlled_five_fold_gate(monkeypatch, range(5))
    assert initial["selected_name"] == "candidate_mixture"
    fitted, accepted, _ = _controlled_five_fold_gate(monkeypatch, range(5), min_gain=gain)
    assert accepted["selected_name"] == "candidate_mixture"
    assert fitted["candidate_weights"] == [1.0] * 9
    _, rejected, _ = _controlled_five_fold_gate(monkeypatch, range(5), min_gain=np.nextafter(gain, np.inf))
    assert rejected["selected_name"] == "ver4_baseline"
    assert rejected["fallback_reason"] == "candidate_gain_below_minimum"


def test_two_improved_folds_rejected_despite_material_aggregate_gain(monkeypatch):
    fitted, report, gain = _controlled_five_fold_gate(monkeypatch, [0, 1])
    assert gain >= 0.002
    assert report["minimum_improved_folds"] == 3
    assert report["candidate_fold_gains"][2:] == [0.0, 0.0, 0.0]
    assert report["selected_name"] == "ver4_baseline"
    assert report["fallback_reason"] == "candidate_gain_not_consistent_across_folds"
    assert fitted["candidate_weights"] == [0.0] * 9


def test_exactly_three_improved_folds_are_accepted(monkeypatch):
    fitted, report, gain = _controlled_five_fold_gate(monkeypatch, [0, 2, 4])
    assert gain >= 0.002
    assert report["candidate_improved_folds"] == report["minimum_improved_folds"] == 3
    assert report["selected_name"] == "candidate_mixture"
    assert report["fallback_reason"] is None
    assert fitted["candidate_weights"] == [1.0] * 9
    assert report["selected_crossfit_macro_f1"] == report["candidate_crossfit_macro_f1"]
    assert "not a statistical significance test" in report["selection_rule"]


@pytest.mark.parametrize("gate", [
    {"min_gain": -0.001}, {"min_gain": np.nan}, {"min_gain": np.inf},
    {"min_improved_folds": 0}, {"min_improved_folds": -1}, {"min_improved_folds": True},
])
def test_invalid_adoption_gate_fails_before_fitting(monkeypatch, gate):
    monkeypatch.setattr(module, "_fit_baseline", lambda *args: pytest.fail("invalid gate reached fitting"))
    truth = np.tile([0, 1], 10)[:, None].repeat(9, axis=1)
    probabilities = truth * 0.6 + 0.2
    with pytest.raises(ValueError):
        calibrate_crossfit(truth, probabilities, probabilities, probabilities, np.arange(len(truth)), **gate)
