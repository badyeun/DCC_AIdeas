"""Training-only selection between ver4 and a calibrated three-way mixture."""
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


def _validated_inputs(y_true, base_neural, sparse, candidate):
    truth, base, sparse = _inputs(y_true, base_neural, sparse)
    candidate = _probabilities(candidate, "candidate")
    if candidate.shape != base.shape:
        raise ValueError("candidate and base_neural must have identical shapes")
    return truth, base, sparse, candidate


def _with_zero_candidate(calibrator):
    return {**calibrator, "candidate_weights": [0.0] * len(TARGET_SYMPTOMS)}


def _fit_mixture(y_true, base_neural, sparse, candidate):
    """Exact F1 selection per label; equal scores prefer zero candidate weight."""
    truth, base, sparse, candidate = _validated_inputs(y_true, base_neural, sparse, candidate)
    alphas, betas, thresholds = [], [], []
    for column in range(len(TARGET_SYMPTOMS)):
        best = None
        for alpha in WEIGHT_GRID:
            baseline = alpha * base[:, column] + (1.0 - alpha) * sparse[:, column]
            for beta in CANDIDATE_GRID:
                probabilities = (1.0 - beta) * baseline + beta * candidate[:, column]
                threshold, score = _best_threshold(truth[:, column], probabilities)
                # Keep ver4's alpha tie convention after preferring smaller beta.
                choice = (score, -beta, -abs(alpha - 0.5), alpha, threshold)
                if best is None or choice > best:
                    best = choice
        _, negative_beta, _, alpha, threshold = best
        alphas.append(float(alpha))
        betas.append(float(-negative_beta))
        thresholds.append(float(threshold))
    return {"neural_weights": alphas, "candidate_weights": betas, "thresholds": thresholds}


def apply_calibrator(base_neural, sparse, candidate, calibrator):
    """Return mixture probabilities. A zero-beta fallback needs no candidate."""
    baseline = _apply_baseline(base_neural, sparse, calibrator)
    betas = np.asarray(calibrator["candidate_weights"], dtype=np.float64)
    if betas.shape != (len(TARGET_SYMPTOMS),) or not np.isfinite(betas).all():
        raise ValueError("candidate_weights must contain one finite weight per symptom")
    if np.any((betas < 0) | (betas > 1)):
        raise ValueError("candidate_weights must be in [0, 1]")
    if candidate is None:
        if np.any(betas != 0):
            raise ValueError("candidate probabilities are required for nonzero candidate_weights")
        return baseline
    candidate = _probabilities(candidate, "candidate")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and base_neural must have identical shapes")
    return (1.0 - betas) * baseline + betas * candidate


def _metrics(truth, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0,
    )
    return {
        symptom: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for symptom, p, r, f, s in zip(TARGET_SYMPTOMS, precision, recall, f1, support)
    }


def calibrate_crossfit(y_true, base_neural, sparse, candidate, groups, seed=44, n_splits=5):
    """Tune on calibration-fold training rows, compare on excluded groups.

    All supplied probabilities must originate from base/candidate models that
    excluded this entire Training calibration set from model fitting and epoch
    selection. These arrays must never contain official Validation samples.
    The global procedure choice uses the resulting cross-fit scores, so its
    selected score is a selection estimate, not an unbiased performance guarantee.
    """
    truth, base, sparse, candidate = _validated_inputs(y_true, base_neural, sparse, candidate)
    baseline_predictions = np.zeros_like(truth)
    candidate_predictions = np.zeros_like(truth)
    seen = np.zeros(len(truth), dtype=np.int8)
    fold_counts = []
    for train, test in _grouped_splits(truth, groups, seed, n_splits):
        baseline_calibrator = _fit_baseline(truth[train], base[train], sparse[train])
        mixture_calibrator = _fit_mixture(truth[train], base[train], sparse[train], candidate[train])
        baseline_predictions[test] = (
            _apply_baseline(base[test], sparse[test], baseline_calibrator)
            >= np.asarray(baseline_calibrator["thresholds"])
        )
        candidate_predictions[test] = (
            apply_calibrator(base[test], sparse[test], candidate[test], mixture_calibrator)
            >= np.asarray(mixture_calibrator["thresholds"])
        )
        seen[test] += 1
        fold_counts.append({"fit_samples": int(len(train)), "score_samples": int(len(test))})
    if not np.all(seen == 1):
        raise RuntimeError("every calibration sample must be scored exactly once")
    baseline_score = float(f1_score(truth, baseline_predictions, average="macro", zero_division=0))
    candidate_score = float(f1_score(truth, candidate_predictions, average="macro", zero_division=0))
    baseline_final = _with_zero_candidate(_fit_baseline(truth, base, sparse))
    candidate_final = _fit_mixture(truth, base, sparse, candidate)
    use_candidate = candidate_score > baseline_score
    selected = candidate_final if use_candidate else baseline_final
    selected_predictions = candidate_predictions if use_candidate else baseline_predictions
    fitted = apply_calibrator(base, sparse, candidate, selected) >= np.asarray(selected["thresholds"])
    report = {
        "score_scope": "Training-only calibration. Procedure selection reuses cross-fit scores; selected score is a selection estimate, not an unbiased guarantee or official Validation.",
        "baseline_crossfit_macro_f1": baseline_score,
        "candidate_crossfit_macro_f1": candidate_score,
        "selected_crossfit_macro_f1": candidate_score if use_candidate else baseline_score,
        "selected_fit_macro_f1": float(f1_score(truth, fitted, average="macro", zero_division=0)),
        "selected_name": "candidate_mixture" if use_candidate else "ver4_baseline",
        "fallback_reason": None if use_candidate else "candidate_crossfit_not_better_than_baseline",
        "per_symptom_crossfit": {
            "baseline": _metrics(truth, baseline_predictions),
            "candidate": _metrics(truth, candidate_predictions),
            "selected": _metrics(truth, selected_predictions),
        },
        "calibration_samples": int(len(truth)),
        "calibration_groups": int(len(np.unique(groups))),
        "crossfit_splits": len(fold_counts),
        "crossfit_fold_counts": fold_counts,
        "selection_rule": "Use candidate procedure only if its aggregate cross-fit macro F1 strictly exceeds the same-fold ver4 baseline; no per-label gate.",
    }
    return selected, report
