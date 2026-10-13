"""Fixed-decision multilabel scores and paired call-level uncertainty."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .data import TARGET_SYMPTOMS, _binary, score_predictions


def _logits(values, rows=None):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != len(TARGET_SYMPTOMS)
            or (rows is not None and len(values) != rows) or not np.isfinite(values).all()):
        raise ValueError("logits must be a finite (samples, 9) array aligned with targets")
    return values


def metrics(targets, logits):
    """Every logit >= 0 is positive, including exact zero ties."""
    targets = _binary(targets, name="targets")
    logits = _logits(logits, len(targets))
    predicted = logits >= 0
    result = score_predictions(targets, predicted)
    ap_values = []
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        truth, prediction = targets[:, index].astype(bool), predicted[:, index]
        row = result["per_symptom"][symptom]
        # AP is undefined without positive targets; JSON null makes this explicit.
        ap = float(average_precision_score(truth, logits[:, index])) if truth.any() else None
        row.update({"true_positive": int((truth & prediction).sum()),
                    "true_negative": int((~truth & ~prediction).sum()), "average_precision": ap})
        if ap is not None:
            ap_values.append(ap)
    result["macro_average_precision"] = float(np.mean(ap_values)) if ap_values else None
    result["average_precision_valid_labels"] = len(ap_values)
    result["nausea_f1"] = result["per_symptom"]["오심"]["f1"]
    result["decision_rule"] = "independent logits >= 0 (sigmoid >= 0.5); zero ties are positive"
    return result


def ensemble_logits(control_logits, candidate_logits):
    """Return logit of the fixed 1:1 sigmoid probability mean.

    Compute the probability mean in log space, preserving its ranking for AP.
    Its mathematical decision is a+b >= 0. Roundoff in the log-space
    subtraction is repaired with that identity, without a fitted offset.
    Same-sign inputs use their signs directly; opposite-sign addition cannot
    overflow and retains the decision even for subnormal floating-point inputs.
    """
    left = _logits(control_logits)
    right = _logits(candidate_logits, len(left))
    log_positive = np.logaddexp(-np.logaddexp(0., -left), -np.logaddexp(0., -right))
    log_negative = np.logaddexp(-np.logaddexp(0., left), -np.logaddexp(0., right))
    mixed = log_positive - log_negative
    same_sign = np.signbit(left) == np.signbit(right)
    decision = np.sign(left) + np.sign(right)
    decision[~same_sign] = left[~same_sign] + right[~same_sign]
    nonzero = decision != 0
    mixed[nonzero] = np.copysign(np.maximum(np.abs(mixed[nonzero]), np.nextafter(0., 1.)), decision[nonzero])
    mixed[~nonzero] = 0.
    return mixed


def equal_mean_logits(members):
    """Fixed equal probability mean for the registered 1/2/4-member recipes.

    Four members use two equal pairs and an equal pair of pairs. This is
    algebraically the equal four-model probability mean and matches the
    preserved ver28 baseline pair during screening and offline inference.
    """
    values = list(members)
    if len(values) not in (1, 2, 4):
        raise ValueError("Only the registered one, two, or four model mean is supported")
    if len(values) == 1:
        return _logits(values[0]).copy()
    if len(values) == 2:
        return ensemble_logits(*values)
    return ensemble_logits(ensemble_logits(*values[:2]), ensemble_logits(*values[2:]))


def paired_error_changes(targets, control_logits, candidate_logits):
    """Compare paired rows; common errors identify remaining complementarity."""
    targets = _binary(targets, name="targets").astype(bool)
    control = _logits(control_logits, len(targets)) >= 0
    candidate = _logits(candidate_logits, len(targets)) >= 0
    if not len(targets):
        raise ValueError("Cannot compare an empty dataset")
    result = {"samples": len(targets), "per_symptom": {}}
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        truth, old, new = targets[:, index], control[:, index], candidate[:, index]
        result["per_symptom"][symptom] = {
            "common_false_negative": int((truth & ~old & ~new).sum()),
            "common_false_positive": int((~truth & old & new).sum()),
            "fixed_false_negative": int((truth & ~old & new).sum()),
            "fixed_false_positive": int((~truth & old & ~new).sum()),
            "new_false_negative": int((truth & old & ~new).sum()),
            "new_false_positive": int((~truth & ~old & new).sum()),
            "control_false_negative": int((truth & ~old).sum()),
            "control_false_positive": int((~truth & old).sum()),
            "candidate_false_negative": int((truth & ~new).sum()),
            "candidate_false_positive": int((~truth & new).sum()),
        }
    return result


def _f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator != 0)


def paired_bootstrap_ci(targets, control_logits, candidate_logits, iterations: int = 500,
                        seed: int = 42, confidence: float = .95):
    """Percentile CIs for candidate-minus-control F1, resampling entire calls.

    Both models and all nine labels use the identical resampled row indices.
    This estimates uncertainty conditional on these trained models and split;
    it does not include training seed or repeated selection uncertainty.
    """
    truth = _binary(targets, name="targets").astype(bool)
    old = _logits(control_logits, len(truth)) >= 0
    new = _logits(candidate_logits, len(truth)) >= 0
    if not len(truth):
        raise ValueError("Cannot bootstrap an empty dataset")
    if type(iterations) is not int or iterations < 2:
        raise ValueError("iterations must be an integer >= 2")
    if type(seed) is not int or seed < 0 or not 0 < confidence < 1:
        raise ValueError("seed must be nonnegative and confidence must be in (0, 1)")
    # One row contains every class and both predictions: a call is the unit.
    counts = np.stack([np.stack([truth & prediction, ~truth & prediction, truth & ~prediction], axis=1)
                       for prediction in (old, new)], axis=1).astype(np.int8)
    point = _f1_from_counts(*counts.sum(axis=0).transpose(1, 0, 2))
    point_delta = point[1] - point[0]
    rng = np.random.default_rng(seed)
    delta = np.empty((iterations, len(TARGET_SYMPTOMS)), dtype=np.float64)
    for iteration in range(iterations):
        sampled = counts[rng.integers(0, len(truth), size=len(truth))].sum(axis=0)
        f1 = _f1_from_counts(*sampled.transpose(1, 0, 2))
        delta[iteration] = f1[1] - f1[0]
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]

    def interval(values, estimate):
        lower, upper = np.quantile(values, quantiles)
        return {"estimate": float(estimate), "lower": float(lower), "upper": float(upper)}

    return {"samples": len(truth), "iterations": iterations, "seed": seed, "confidence": confidence,
            "unit": "call", "comparison": "candidate minus control", "method": "paired percentile bootstrap",
            "macro_f1_delta": interval(delta.mean(axis=1), point_delta.mean()),
            "per_symptom_f1_delta": {name: interval(delta[:, index], point_delta[index])
                                     for index, name in enumerate(TARGET_SYMPTOMS)},
            "limitations": "Conditional on trained models and this split; repeated selection and training-seed "
                           "uncertainty are not included. Duplicate calls may still be statistically dependent."}
