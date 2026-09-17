"""Training-only selection between ver4 and two bounded three-way mixtures."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

if __package__:
    from .ver4_runtime.calibration import (
        WEIGHT_GRID, _best_threshold, _grouped_splits, _inputs, _probabilities,
        apply_calibrator as _apply_baseline, fit_calibrator as _fit_baseline,
    )
    from .ver4_runtime.data import TARGET_SYMPTOMS
else:
    from ver4_runtime.calibration import (
        WEIGHT_GRID, _best_threshold, _grouped_splits, _inputs, _probabilities,
        apply_calibrator as _apply_baseline, fit_calibrator as _fit_baseline,
    )
    from ver4_runtime.data import TARGET_SYMPTOMS

CANDIDATE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LOGIT_EPSILON = 1e-6


def _blend_space(value):
    if value not in ("raw", "logit"):
        raise ValueError("blend_space must be 'raw' or 'logit'")
    return value


def _logits(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64),
                            LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(probabilities) - np.log1p(-probabilities)


def _sigmoid(logits):
    # Convex mixtures of clipped logits stay within finite, safe exp bounds.
    return 1.0 / (1.0 + np.exp(-logits))


def _validated_inputs(y_true, base_neural, sparse, candidate):
    truth, base, sparse = _inputs(y_true, base_neural, sparse)
    candidate = _probabilities(candidate, "candidate")
    if candidate.shape != base.shape:
        raise ValueError("candidate and base_neural must have identical shapes")
    return truth, base, sparse, candidate


def _with_zero_candidate(calibrator):
    return {**calibrator, "candidate_weights": [0.0] * len(TARGET_SYMPTOMS),
            "blend_space": "raw"}


def _fit_mixture(y_true, base_neural, sparse, candidate, blend_space="raw"):
    """Exact F1 selection per label; equal scores prefer zero candidate weight."""
    blend_space = _blend_space(blend_space)
    truth, base, sparse, candidate = _validated_inputs(y_true, base_neural, sparse, candidate)
    if blend_space == "logit":
        base, sparse, candidate = map(_logits, (base, sparse, candidate))
    alphas, betas, thresholds = [], [], []
    for column in range(len(TARGET_SYMPTOMS)):
        best = None
        for alpha in WEIGHT_GRID:
            baseline = alpha * base[:, column] + (1.0 - alpha) * sparse[:, column]
            for beta in CANDIDATE_GRID:
                probabilities = (1.0 - beta) * baseline + beta * candidate[:, column]
                if blend_space == "logit":
                    probabilities = _sigmoid(probabilities)
                threshold, score = _best_threshold(truth[:, column], probabilities)
                # Keep ver4's alpha tie convention after preferring smaller beta.
                choice = (score, -beta, -abs(alpha - 0.5), alpha, threshold)
                if best is None or choice > best:
                    best = choice
        _, negative_beta, _, alpha, threshold = best
        alphas.append(float(alpha))
        betas.append(float(-negative_beta))
        thresholds.append(float(threshold))
    return {"neural_weights": alphas, "candidate_weights": betas,
            "thresholds": thresholds, "blend_space": blend_space}


def apply_calibrator(base_neural, sparse, candidate, calibrator):
    """Apply the stored space; older manifests default to the exact raw blend."""
    blend_space = _blend_space(calibrator.get("blend_space", "raw"))
    # Reuse ver4's shape, range, alpha and threshold validation in both spaces.
    baseline = _apply_baseline(base_neural, sparse, calibrator)
    betas = np.asarray(calibrator["candidate_weights"], dtype=np.float64)
    if betas.shape != (len(TARGET_SYMPTOMS),) or not np.isfinite(betas).all():
        raise ValueError("candidate_weights must contain one finite weight per symptom")
    if np.any((betas < 0) | (betas > 1)):
        raise ValueError("candidate_weights must be in [0, 1]")
    if blend_space == "logit":
        alphas = np.asarray(calibrator["neural_weights"], dtype=np.float64)
        baseline = alphas * _logits(base_neural) + (1.0 - alphas) * _logits(sparse)
    if candidate is None:
        if np.any(betas != 0):
            raise ValueError("candidate probabilities are required for nonzero candidate_weights")
        return _sigmoid(baseline) if blend_space == "logit" else baseline
    candidate = _probabilities(candidate, "candidate")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and base_neural must have identical shapes")
    if blend_space == "logit":
        candidate = _logits(candidate)
    mixture = (1.0 - betas) * baseline + betas * candidate
    return _sigmoid(mixture) if blend_space == "logit" else mixture


def _metrics(truth, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0,
    )
    return {
        symptom: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for symptom, p, r, f, s in zip(TARGET_SYMPTOMS, precision, recall, f1, support)
    }


def fit_candidate_only_crossfit(y_true, candidate, groups, seed=44, n_splits=5):
    """Diagnose a standalone candidate using grouped Training calibration only.

    The candidate model must exclude all supplied samples from model fitting and
    epoch selection. Each scored fold uses thresholds fitted on the other
    groups; the returned thresholds are then fitted on all Training calibration
    rows. Never supply official Validation. This diagnostic does not participate
    in the baseline/mixture selection procedure or its adoption guard.
    """
    candidate = _probabilities(candidate, "candidate")
    truth, candidate, _ = _inputs(y_true, candidate, candidate)

    def fit_thresholds(rows):
        return [
            float(_best_threshold(truth[rows, column], candidate[rows, column])[0])
            for column in range(len(TARGET_SYMPTOMS))
        ]

    predictions = np.zeros_like(truth)
    seen = np.zeros(len(truth), dtype=np.int64)
    fold_count = 0
    for train, test in _grouped_splits(truth, groups, seed, n_splits):
        thresholds = np.asarray(fit_thresholds(train), dtype=np.float64)
        predictions[test] = candidate[test] >= thresholds
        np.add.at(seen, test, 1)
        fold_count += 1
    if not np.all(seen == 1):
        raise RuntimeError("every calibration sample must be scored exactly once")

    calibrator = {"thresholds": fit_thresholds(np.arange(len(truth)))}
    fitted = candidate >= np.asarray(calibrator["thresholds"], dtype=np.float64)
    report = {
        "candidate_standalone_crossfit_macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)),
        "candidate_standalone_fit_macro_f1": float(
            f1_score(truth, fitted, average="macro", zero_division=0)),
        "candidate_standalone_per_symptom_crossfit": _metrics(truth, predictions),
        "candidate_standalone_crossfit_splits": fold_count,
    }
    return calibrator, report


def calibrate_crossfit(y_true, base_neural, sparse, candidate, groups, seed=44, n_splits=5,
                       min_gain=0.002, min_improved_folds=3):
    """Tune on calibration-fold training rows, compare on excluded groups.

    All supplied probabilities must originate from base/candidate models that
    excluded this entire Training calibration set from model fitting and epoch
    selection. These arrays must never contain official Validation samples.
    Compare only the original baseline and fixed raw/logit mixture families.
    The global procedure choice uses the resulting cross-fit scores, so its
    selected score is a selection estimate, not an unbiased performance guarantee.
    """
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and nonnegative")
    if type(min_improved_folds) is not int or min_improved_folds < 1:
        raise ValueError("min_improved_folds must be a positive integer")
    truth, base, sparse, candidate = _validated_inputs(y_true, base_neural, sparse, candidate)
    baseline_predictions = np.zeros_like(truth)
    mixture_predictions = {space: np.zeros_like(truth) for space in ("raw", "logit")}
    seen = np.zeros(len(truth), dtype=np.int8)
    fold_counts = []
    fold_gains = {space: [] for space in ("raw", "logit")}
    for train, test in _grouped_splits(truth, groups, seed, n_splits):
        baseline_calibrator = _fit_baseline(truth[train], base[train], sparse[train])
        baseline_predictions[test] = (
            _apply_baseline(base[test], sparse[test], baseline_calibrator)
            >= np.asarray(baseline_calibrator["thresholds"])
        )
        fold_baseline_score = f1_score(truth[test], baseline_predictions[test], average="macro", zero_division=0)
        for space, predictions in mixture_predictions.items():
            mixture_calibrator = _fit_mixture(
                truth[train], base[train], sparse[train], candidate[train], blend_space=space,
            )
            predictions[test] = (
                apply_calibrator(base[test], sparse[test], candidate[test], mixture_calibrator)
                >= np.asarray(mixture_calibrator["thresholds"])
            )
            fold_gains[space].append(float(f1_score(truth[test], predictions[test], average="macro", zero_division=0)
                                          - fold_baseline_score))
        seen[test] += 1
        fold_counts.append({"fit_samples": int(len(train)), "score_samples": int(len(test))})
    if not np.all(seen == 1):
        raise RuntimeError("every calibration sample must be scored exactly once")
    baseline_score = float(f1_score(truth, baseline_predictions, average="macro", zero_division=0))
    mixture_scores = {
        space: float(f1_score(truth, predictions, average="macro", zero_division=0))
        for space, predictions in mixture_predictions.items()
    }
    # A single global family choice avoids per-symptom selection on scored folds.
    candidate_space = "logit" if mixture_scores["logit"] > mixture_scores["raw"] else "raw"
    candidate_score = mixture_scores[candidate_space]
    candidate_predictions = mixture_predictions[candidate_space]
    baseline_final = _with_zero_candidate(_fit_baseline(truth, base, sparse))
    gain = candidate_score - baseline_score
    required_folds = min(min_improved_folds, len(fold_counts))
    improved_folds = sum(value > 0 for value in fold_gains[candidate_space])
    material = candidate_score > baseline_score and gain >= min_gain
    consistent = improved_folds >= required_folds
    use_candidate = material and consistent
    selected = (_fit_mixture(truth, base, sparse, candidate, blend_space=candidate_space)
                if use_candidate else baseline_final)
    selected_predictions = candidate_predictions if use_candidate else baseline_predictions
    fitted = apply_calibrator(base, sparse, candidate, selected) >= np.asarray(selected["thresholds"])
    report = {
        "score_scope": "Training-only calibration. Procedure selection reuses cross-fit scores; selected score is a selection estimate, not an unbiased guarantee or official Validation.",
        "baseline_crossfit_macro_f1": baseline_score,
        "candidate_crossfit_macro_f1": candidate_score,
        "candidate_raw_crossfit_macro_f1": mixture_scores["raw"],
        "candidate_logit_crossfit_macro_f1": mixture_scores["logit"],
        "candidate_blend_space": candidate_space,
        "selected_blend_space": selected["blend_space"],
        "selected_crossfit_macro_f1": candidate_score if use_candidate else baseline_score,
        "selected_fit_macro_f1": float(f1_score(truth, fitted, average="macro", zero_division=0)),
        "selected_name": "candidate_mixture" if use_candidate else "ver4_baseline",
        "fallback_reason": (None if use_candidate else
                            "candidate_gain_below_minimum" if not material else "candidate_gain_not_consistent_across_folds"),
        "selection_gain": float(gain), "minimum_gain": float(min_gain),
        "candidate_improved_folds": improved_folds, "minimum_improved_folds": required_folds,
        "candidate_fold_gains": fold_gains[candidate_space], "all_candidate_fold_gains": fold_gains,
        "per_symptom_crossfit": {
            "baseline": _metrics(truth, baseline_predictions),
            "candidate": _metrics(truth, candidate_predictions),
            "candidate_raw": _metrics(truth, mixture_predictions["raw"]),
            "candidate_logit": _metrics(truth, mixture_predictions["logit"]),
            "selected": _metrics(truth, selected_predictions),
        },
        "calibration_samples": int(len(truth)),
        "calibration_groups": int(len(np.unique(groups))),
        "crossfit_splits": len(fold_counts),
        "crossfit_fold_counts": fold_counts,
        "selection_rule": "Compare fixed raw/logit families globally; ties prefer raw. Require strictly positive aggregate gain at least minimum_gain and improvements on minimum_improved_folds; otherwise use ver4. Training-only selection guard, not a statistical significance test or Validation guarantee.",
    }
    return selected, report
