"""Grouped threshold cross-fitting on the original fold's stopping split.

This is model/epoch selection on Training stop. It is not model-level OOF and
its repeatedly selected score is not an unbiased final performance estimate.
No official Validation or model-calibration rows belong here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

if __package__:
    from .ver4_runtime.calibration import _best_threshold, _grouped_splits
else:
    from ver4_runtime.calibration import _best_threshold, _grouped_splits


class StopSelector:
    def __init__(self, targets, groups, seed=44, n_splits=3):
        self.targets = np.asarray(targets, dtype=np.float32)
        if self.targets.ndim != 2 or not len(self.targets) or not np.isin(self.targets, [0, 1]).all():
            raise ValueError("Stop targets must be a nonempty binary matrix")
        self.groups = np.asarray(groups)
        self.folds = list(_grouped_splits(self.targets, self.groups, seed, n_splits))

    def score(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != self.targets.shape or not np.isfinite(probabilities).all():
            raise ValueError("Stop probabilities must be finite and match targets")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Stop probabilities must lie in [0, 1]")
        predictions = np.zeros_like(self.targets, dtype=np.int8)
        seen = np.zeros(len(self.targets), dtype=np.int8)
        thresholds = []
        for fit, score in self.folds:
            threshold = [_best_threshold(self.targets[fit, label], probabilities[fit, label])[0]
                         for label in range(self.targets.shape[1])]
            predictions[score] = probabilities[score] >= np.asarray(threshold)
            seen[score] += 1
            thresholds.append(threshold)
        if not np.all(seen == 1):
            raise RuntimeError("Every stop row must be scored once")
        present = self.targets.sum(0) > 0
        ap = float(average_precision_score(self.targets[:, present], probabilities[:, present], average="macro")) if present.any() else 0.0
        return {"stop_macro_f1": float(f1_score(self.targets, predictions, average="macro", zero_division=0)),
                "stop_macro_ap": ap, "stop_labels_with_positives": int(present.sum()),
                "threshold_crossfit_splits": len(self.folds), "thresholds_by_fold": thresholds}


def better(score, best, tolerance=1e-6):
    """F1 selects; AP is only a deterministic secondary criterion for F1 ties."""
    if best is None:
        return True
    delta = score["stop_macro_f1"] - best["stop_macro_f1"]
    return delta > tolerance or (abs(delta) <= tolerance and score["stop_macro_ap"] > best["stop_macro_ap"] + tolerance)
