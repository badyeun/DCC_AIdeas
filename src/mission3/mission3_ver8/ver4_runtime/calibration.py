"""Training-only grouped calibration for nine independent symptom labels."""
from __future__ import annotations

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support

if __package__:
    from .data import TARGET_SYMPTOMS
else:
    from data import TARGET_SYMPTOMS

WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALL_NEGATIVE_THRESHOLD = float(np.nextafter(1.0, np.inf))
# The organisers require the same 0.5 decision threshold for every symptom.
# Per-symptom thresholds are not permitted, so none are searched or stored.
FIXED_THRESHOLD = 0.5


def _probabilities(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(TARGET_SYMPTOMS):
        raise ValueError(f"{name} must have shape (samples, {len(TARGET_SYMPTOMS)})")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return values


def _inputs(y_true, neural_probs, sparse_probs):
    neural = _probabilities(neural_probs, "neural_probs")
    sparse = _probabilities(sparse_probs, "sparse_probs")
    truth = np.asarray(y_true)
    if truth.shape != neural.shape or sparse.shape != neural.shape:
        raise ValueError("truth, neural_probs and sparse_probs must have identical shapes")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels")
    return truth.astype(np.int8), neural, sparse


def _best_threshold(truth, probabilities):
    """Score a label at the mandated 0.5 threshold; never search for a better cut.

    The returned threshold is always ``FIXED_THRESHOLD``. Only the score varies,
    so callers still choose blend weights by comparing scores -- but they now
    compare the F1 each blend actually achieves at 0.5.
    """
    truth = np.asarray(truth)
    predicted = np.asarray(probabilities) >= FIXED_THRESHOLD
    denominator = int(truth.sum()) + int(predicted.sum())
    if denominator == 0:
        return FIXED_THRESHOLD, 0.0
    true_positives = int(np.count_nonzero(predicted & (truth == 1)))
    return FIXED_THRESHOLD, 2.0 * true_positives / denominator


def fit_calibrator(y_true, neural_probs, sparse_probs):
    """Select each label's blend and threshold using supplied training labels."""
    truth, neural, sparse = _inputs(y_true, neural_probs, sparse_probs)
    weights, thresholds = [], []
    for column in range(len(TARGET_SYMPTOMS)):
        candidates = []
        for weight in WEIGHT_GRID:
            blended = weight * neural[:, column] + (1.0 - weight) * sparse[:, column]
            threshold, score = _best_threshold(truth[:, column], blended)
            candidates.append((score, -abs(weight - 0.5), weight, threshold))
        _, _, weight, threshold = max(candidates)
        weights.append(float(weight))
        thresholds.append(float(threshold))
    return {"neural_weights": weights, "thresholds": thresholds}


def apply_calibrator(neural_probs, sparse_probs, calibrator):
    """Return blended probabilities; use >= stored thresholds for labels."""
    neural = _probabilities(neural_probs, "neural_probs")
    sparse = _probabilities(sparse_probs, "sparse_probs")
    if neural.shape != sparse.shape:
        raise ValueError("neural_probs and sparse_probs must have identical shapes")
    weights = np.asarray(calibrator["neural_weights"], dtype=np.float64)
    thresholds = np.asarray(calibrator["thresholds"], dtype=np.float64)
    expected = (len(TARGET_SYMPTOMS),)
    if weights.shape != expected or thresholds.shape != expected:
        raise ValueError("calibrator must contain one weight and threshold per symptom")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("calibrator weights must be finite and in [0, 1]")
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > ALL_NEGATIVE_THRESHOLD)):
        raise ValueError("calibrator thresholds must be finite and in [0, nextafter(1, inf)]")
    return weights * neural + (1.0 - weights) * sparse


def _grouped_splits(truth, groups, seed, n_splits):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(truth):
        raise ValueError("groups must contain one group ID per sample")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("cross-fit calibration requires at least two distinct groups")
    n_splits = min(n_splits, len(unique_groups))
    group_targets = np.zeros((len(unique_groups), truth.shape[1]), dtype=np.int8)
    np.maximum.at(group_targets, inverse, truth)
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_groups, test_groups in splitter.split(np.zeros(len(unique_groups)), group_targets):
        train = np.flatnonzero(np.isin(inverse, train_groups))
        test = np.flatnonzero(np.isin(inverse, test_groups))
        if not len(train) or not len(test):
            raise ValueError("grouped calibration produced an empty fold")
        yield train, test


def calibrate_crossfit(y_true, neural_probs, sparse_probs, groups, seed=42, n_splits=5):
    """Cross-fit within Training calibration data, then fit deployment settings.

    Base-model predictions must themselves come from samples excluded from
    base-model training and model selection. Never pass official Validation.
    """
    truth, neural, sparse = _inputs(y_true, neural_probs, sparse_probs)
    crossfit_predictions = np.zeros_like(truth)
    seen = np.zeros(len(truth), dtype=np.int8)
    fold_counts = []
    for train, test in _grouped_splits(truth, groups, seed, n_splits):
        calibrator = fit_calibrator(truth[train], neural[train], sparse[train])
        probabilities = apply_calibrator(neural[test], sparse[test], calibrator)
        crossfit_predictions[test] = probabilities >= np.asarray(calibrator["thresholds"])
        seen[test] += 1
        fold_counts.append({"fit_samples": int(len(train)), "score_samples": int(len(test))})
    if not np.all(seen == 1):
        raise RuntimeError("each calibration sample must be scored exactly once")
    final_calibrator = fit_calibrator(truth, neural, sparse)
    fitted = apply_calibrator(neural, sparse, final_calibrator) >= np.asarray(final_calibrator["thresholds"])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, crossfit_predictions, average=None, zero_division=0
    )
    report = {
        "score_scope": "Training-only calibration; cross-fit estimates calibration generalization, not official Validation.",
        "calibration_fit_macro_f1": float(f1_score(truth, fitted, average="macro", zero_division=0)),
        "calibration_crossfit_macro_f1": float(f1.mean()),
        "per_symptom_crossfit": {
            symptom: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for symptom, p, r, f, s in zip(TARGET_SYMPTOMS, precision, recall, f1, support)
        },
        "calibration_samples": int(len(truth)),
        "calibration_groups": int(len(np.unique(groups))),
        "crossfit_splits": len(fold_counts),
        "crossfit_fold_counts": fold_counts,
    }
    return final_calibrator, report
