import json
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mission3_ver9 import calibration as module
from mission3_ver9.calibration import (
    _fit_mixture, apply_selected, compare_candidates, fit_candidate_only_crossfit,
)
from mission3_ver9.ver4_runtime.calibration import (
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
    np.testing.assert_array_equal(apply_selected(base, sparse, None, new), apply_legacy(base, sparse, old))


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_mixture_search_matches_independent_exhaustive_f1(blend_space):
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, size=(28, 9))
    arrays = [np.round(rng.uniform(size=(28, 9)), 1) for _ in range(3)]
    fitted = _fit_mixture(truth, *arrays, blend_space=blend_space)
    predictions = apply_selected(*arrays, fitted) >= fitted["thresholds"]
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
                best = max(best, *(f1_score(truth[:, column], p >= threshold, zero_division=0)
                                   for threshold in [0.0, *np.unique(p), np.nextafter(1.0, np.inf)]))
        assert f1_score(truth[:, column], predictions[:, column]) == pytest.approx(best)


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_constants_and_saturated_probabilities_remain_valid(blend_space):
    truth = np.tile([0, 1, 0, 1, 0, 1, 0, 1, 0], (12, 1))
    probabilities = np.ones((12, 9), dtype=np.float32)
    fitted = _fit_mixture(truth, probabilities, probabilities, probabilities, blend_space=blend_space)
    assert fitted["thresholds"][0] > 1
    assert fitted["thresholds"][1] == 0
    result = apply_selected(probabilities, probabilities, probabilities, fitted)
    expected = np.full((12, 9), 1 - 1e-6 if blend_space == "logit" else 1)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(result >= fitted["thresholds"], truth)
    for alpha in (0, 0.25, 0.5, 0.75, 1):
        for beta in (0, 0.25, 0.5, 0.75, 1):
            saturated = apply_selected(probabilities, probabilities, probabilities,
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
            apply_selected(valid, valid, invalid, fitted)
    for weights in [[0.0] * 8, [np.nan] * 9, [-0.1] * 9, [1.1] * 9]:
        with pytest.raises(ValueError):
            apply_selected(valid, valid, valid, {**fitted, "candidate_weights": weights})
    with pytest.raises(ValueError, match="required"):
        apply_selected(valid, valid, None, {**fitted, "candidate_weights": [0.25] * 9})


def test_crossfit_group_separation_and_candidate_success():
    rng = np.random.default_rng(9)
    truth = rng.integers(0, 2, size=(90, 9))
    groups = np.repeat(np.arange(30), 3)
    # Unique row IDs in probabilities let the spy verify which rows each fit saw.
    base = np.full((90, 9), 0.5) + np.arange(90)[:, None] * 1e-6
    candidate = 0.2 + truth * 0.6
    expected = list(_grouped_splits(truth, groups, 42, 5))
    calls = []
    actual_fit = module._fit_mixture

    def spy(y, b, s, c, blend_space="raw"):
        calls.append((blend_space, b[:, 0].copy()))
        return actual_fit(y, b, s, c, blend_space=blend_space)

    with patch.object(module, "_fit_mixture", side_effect=spy):
        fitted, report = compare_candidates(truth, base, base, candidate, groups)
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
    fitted, report = compare_candidates(truth, base, base, candidate, np.arange(80))
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
    actual = apply_selected(base, sparse, candidate, restored)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("alpha,beta,unused", [(0, .25, "base"), (1, .25, "sparse"), (.5, 1, "both")])
def test_logit_zero_coefficients_allow_skipped_inference_components(alpha, beta, unused):
    rng = np.random.default_rng(31)
    base, sparse, candidate = [rng.uniform(size=(12, 9)) for _ in range(3)]
    fitted = {"neural_weights": [alpha] * 9, "candidate_weights": [beta] * 9,
              "thresholds": [.5] * 9, "blend_space": "logit"}
    expected = apply_selected(base, sparse, candidate, fitted)
    if unused in ("base", "both"):
        base = np.zeros_like(base)
    if unused in ("sparse", "both"):
        sparse = np.ones_like(sparse)
    np.testing.assert_array_equal(apply_selected(base, sparse, candidate, fitted), expected)


def test_logit_zero_beta_needs_no_candidate_and_missing_space_keeps_legacy():
    rng = np.random.default_rng(32)
    base, sparse, candidate = [rng.uniform(size=(12, 9)) for _ in range(3)]
    fitted = {"neural_weights": [.25] * 9, "candidate_weights": [0.] * 9, "thresholds": [.5] * 9}
    np.testing.assert_array_equal(apply_selected(base, sparse, None, fitted), apply_legacy(base, sparse, fitted))
    fitted["blend_space"] = "logit"
    np.testing.assert_array_equal(apply_selected(base, sparse, None, fitted), apply_selected(base, sparse, candidate, fitted))
    with pytest.raises(ValueError, match="blend_space"):
        apply_selected(base, sparse, candidate, {**fitted, "blend_space": "unknown"})


def test_crossfit_can_select_logit_family_globally_with_transparent_scores():
    rng = np.random.default_rng(45)
    neural_logits, sparse_logits = [rng.normal(0, 3, (180, 9)) for _ in range(2)]
    truth = ((neural_logits + sparse_logits) / 2 > .9).astype(np.int8)
    base, sparse = [1 / (1 + np.exp(-x)) for x in (neural_logits, sparse_logits)]
    candidate = np.full_like(base, .5)
    fitted, report = compare_candidates(truth, base, sparse, candidate, np.arange(len(truth)))
    assert report["candidate_logit_crossfit_macro_f1"] > report["candidate_raw_crossfit_macro_f1"]
    assert report["candidate_crossfit_macro_f1"] == report["candidate_logit_crossfit_macro_f1"]
    assert report["selected_crossfit_macro_f1"] > report["baseline_crossfit_macro_f1"]
    assert report["selected_name"] == "candidate_mixture"
    assert report["candidate_blend_space"] == report["selected_blend_space"] == fitted["blend_space"] == "logit"
    assert report["per_symptom_crossfit"]["selected"] == report["per_symptom_crossfit"]["candidate_logit"]


def test_overfit_candidate_falls_back_and_samefold_baseline_matches_legacy():
    rng = np.random.default_rng(18)
    truth = rng.integers(0, 2, size=(50, 9))
    base = 0.2 + 0.5 * rng.uniform(size=(50, 9)) + 0.1 * truth
    candidate = rng.uniform(size=(50, 9))
    groups = np.arange(50)
    fitted, report = compare_candidates(truth, base, base, candidate, groups)
    old, old_report = crossfit_legacy(truth, base, base, groups, seed=42, n_splits=5)
    assert report["candidate_crossfit_macro_f1"] < report["baseline_crossfit_macro_f1"]
    assert report["baseline_crossfit_macro_f1"] == old_report["calibration_crossfit_macro_f1"]
    assert report["selected_name"] == "ver4_baseline"
    assert fitted["candidate_weights"] == [0.0] * 9
    assert fitted["neural_weights"] == old["neural_weights"]
    assert fitted["thresholds"] == old["thresholds"]


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
    fitted, report = compare_candidates(truth, base, base, candidate, np.repeat(np.arange(5), 10), **gate)
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
        compare_candidates(truth, probabilities, probabilities, probabilities, np.arange(len(truth)), **gate)


def test_standalone_thresholds_use_other_groups_and_report_heldout_predictions(monkeypatch):
    # Group-specific score offsets make leakage observable: a full-data threshold
    # scores .8, whereas thresholds learned on the other group score .5 overall.
    truth = np.array([0, 1, 0, 1])[:, None].repeat(9, axis=1)
    candidate = np.array([.1, .2, .8, .9])[:, None].repeat(9, axis=1)
    groups = np.array(["low", "low", "high", "high"])
    expected = list(_grouped_splits(truth, groups, 42, 2))
    calls = []
    original = module._best_threshold

    def record_fit(y, probabilities):
        calls.append((y.copy(), probabilities.copy()))
        return original(y, probabilities)

    monkeypatch.setattr(module, "_best_threshold", record_fit)
    monkeypatch.setattr(module, "_fit_mixture", lambda *a, **k: pytest.fail("diagnostic reached mixture fitting"))
    monkeypatch.setattr(module, "_fit_baseline", lambda *a, **k: pytest.fail("diagnostic reached baseline fitting"))
    fitted, report = fit_candidate_only_crossfit(truth, candidate, groups, n_splits=2)

    assert len(calls) == 27  # Nine labels in two excluded-group fits, then final fit.
    for fold, (train, test) in enumerate(expected):
        assert set(groups[train]).isdisjoint(groups[test])
        for column in range(9):
            observed_truth, observed_probabilities = calls[fold * 9 + column]
            np.testing.assert_array_equal(observed_truth, truth[train, column])
            np.testing.assert_array_equal(observed_probabilities, candidate[train, column])
    for column in range(9):
        np.testing.assert_array_equal(calls[18 + column][0], truth[:, column])
        np.testing.assert_array_equal(calls[18 + column][1], candidate[:, column])
    assert set(fitted) == {"thresholds"}
    np.testing.assert_allclose(fitted["thresholds"], [.15] * 9)
    assert report["candidate_standalone_crossfit_macro_f1"] == pytest.approx(.5)
    assert report["candidate_standalone_fit_macro_f1"] == pytest.approx(.8)
    assert report["candidate_standalone_crossfit_splits"] == 2
    for metrics in report["candidate_standalone_per_symptom_crossfit"].values():
        assert metrics == {"precision": .5, "recall": .5, "f1": .5, "support": 2}
    json.dumps((fitted, report), allow_nan=False)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_standalone_all_negative_targets_are_safe_at_probability_boundaries(probability):
    truth = np.zeros((12, 9), dtype=np.int8)
    candidate = np.full(truth.shape, probability)
    fitted, report = fit_candidate_only_crossfit(truth, candidate, np.repeat(np.arange(3), 4))
    assert report["candidate_standalone_crossfit_splits"] == 3
    assert all(threshold > 1 for threshold in fitted["thresholds"])
    assert not (candidate >= fitted["thresholds"]).any()
    assert report["candidate_standalone_crossfit_macro_f1"] == 0.0
    assert report["candidate_standalone_fit_macro_f1"] == 0.0
    assert all(metrics["support"] == 0
               for metrics in report["candidate_standalone_per_symptom_crossfit"].values())
    json.dumps((fitted, report), allow_nan=False)


@pytest.mark.parametrize("invalid", [
    np.full((12, 9), np.nan), np.full((12, 9), np.inf),
    np.full((12, 9), -.1), np.full((12, 9), 1.1),
    np.full((12, 8), .5), np.full((11, 9), .5), np.empty((0, 9)),
])
def test_standalone_invalid_probabilities_fail_before_fitting(monkeypatch, invalid):
    monkeypatch.setattr(module, "_best_threshold", lambda *a: pytest.fail("invalid input reached fitting"))
    with pytest.raises(ValueError):
        fit_candidate_only_crossfit(np.zeros((12, 9)), invalid, np.arange(12))


@pytest.mark.parametrize("invalid", [
    np.full((12, 9), .5), np.full((12, 9), -1),
    np.full((12, 9), np.nan), np.zeros((12, 8)),
])
def test_standalone_invalid_targets_fail_before_fitting(monkeypatch, invalid):
    monkeypatch.setattr(module, "_best_threshold", lambda *a: pytest.fail("invalid target reached fitting"))
    with pytest.raises(ValueError):
        fit_candidate_only_crossfit(invalid, np.full((12, 9), .5), np.arange(12))


@pytest.mark.parametrize("groups,n_splits", [
    (np.zeros(12), 5), (np.arange(11), 5), (np.arange(12)[:, None], 5),
    (np.arange(12), 1), (np.arange(12), 2.5),
])
def test_standalone_invalid_groups_or_folds_fail_before_fitting(monkeypatch, groups, n_splits):
    monkeypatch.setattr(module, "_best_threshold", lambda *a: pytest.fail("invalid split reached fitting"))
    with pytest.raises(ValueError):
        fit_candidate_only_crossfit(np.zeros((12, 9)), np.full((12, 9), .5), groups, n_splits=n_splits)


@pytest.mark.parametrize("tests", [([0, 1], [1, 2, 3]), ([0, 1], [2]), ([0, 0, 1], [2, 3])])
def test_standalone_rejects_missing_or_repeated_scored_rows(monkeypatch, tests):
    folds = [(np.array([2, 3]), np.array(tests[0])),
             (np.array([0, 1]), np.array(tests[1]))]
    monkeypatch.setattr(module, "_grouped_splits", lambda *a: iter(folds))
    with pytest.raises(RuntimeError, match="exactly once"):
        fit_candidate_only_crossfit(np.zeros((4, 9)), np.full((4, 9), .5), np.arange(4))


def test_requested_improved_fold_count_is_not_silently_reduced():
    truth = np.tile([0, 1], 20)[:, None].repeat(9, axis=1)
    base = np.full(truth.shape, .5)
    candidate = .1 + .8 * truth
    fitted, report = compare_candidates(
        truth, base, base, candidate, np.repeat(np.arange(4), 10),
        n_splits=2, min_improved_folds=3,
    )
    assert report["candidate_improved_folds"] == 2
    assert report["minimum_improved_folds"] == 3
    assert report["selected_name"] == "ver4_baseline"
    assert report["fallback_reason"] == "candidate_gain_not_consistent_across_folds"
    assert fitted["candidate_weights"] == [0.0] * 9


@pytest.mark.parametrize("blend_space", ["raw", "logit"])
def test_saved_selection_reproduces_apply_exactly(tmp_path, blend_space):
    rng = np.random.default_rng(104)
    arrays = [rng.uniform(.01, .99, size=(7, 9)) for _ in range(3)]
    selection = {
        "neural_weights": [0.0, .25, .5, .75, 1.0, .25, .5, .75, 1.0],
        "candidate_weights": [0.0, .25, .5, .75, 1.0, 1.0, .75, .5, .25],
        "thresholds": [.5] * 9,
        "blend_space": blend_space,
    }
    destination = tmp_path / "selection.json"
    destination.write_text(json.dumps(selection, allow_nan=False))
    restored = json.loads(destination.read_text())
    np.testing.assert_array_equal(apply_selected(*arrays, restored), apply_selected(*arrays, selection))
