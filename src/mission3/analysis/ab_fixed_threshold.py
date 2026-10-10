"""A/B: per-symptom thresholds (the original rule) vs the mandated fixed 0.5.

Both paths run on the same saved Training calibration predictions -- the exact
data the pipeline uses to choose its calibrator, and the only place where both
model components' raw probabilities survive on disk.

Official Validation Macro F1 cannot be recomputed here: the five neural fold
checkpoints are gitignored and absent from this clone, so the blended
Validation probabilities do not exist locally.

Run:  PYTHONPATH=src/mission3 python src/mission3/analysis/ab_fixed_threshold.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mission3_ver4 import calibration as cal

T = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
RESULTS = Path(__file__).resolve().parents[3] / "results" / "mission3" / "ver4"


def original_best_threshold(truth, probabilities):
    """The pre-2026-09-20 rule: exact F1 maximum over every candidate cut.

    Verbatim copy of the function this project used before the organisers
    required a single shared 0.5 threshold. Kept here only to quantify what the
    rule change costs; nothing in the pipeline calls it.
    """
    positives = int(truth.sum())
    if positives == 0:
        return cal.ALL_NEGATIVE_THRESHOLD, 0.0
    if positives == len(truth):
        return 0.0, 1.0
    order = np.argsort(-probabilities, kind="stable")
    sorted_probs = probabilities[order]
    boundaries = np.r_[np.flatnonzero(sorted_probs[:-1] != sorted_probs[1:]), len(truth) - 1]
    true_positives = np.cumsum(truth[order])[boundaries]
    scores = 2.0 * true_positives / (positives + boundaries + 1)
    thresholds = np.zeros(len(boundaries), dtype=np.float64)
    has_next = boundaries < len(truth) - 1
    upper = sorted_probs[boundaries[has_next]]
    lower = sorted_probs[boundaries[has_next] + 1]
    midpoint = lower + (upper - lower) / 2
    thresholds[has_next] = np.where(midpoint > lower, midpoint, upper)
    best = np.flatnonzero(scores == scores.max())
    winner = min(best, key=lambda i: (abs(thresholds[i] - 0.5), -thresholds[i]))
    return float(thresholds[winner]), float(scores[winner])


def run(patched):
    """Fit and cross-fit with either threshold rule, on identical inputs."""
    keep = cal._best_threshold
    if patched is not None:
        cal._best_threshold = patched
    try:
        return cal.calibrate_crossfit(Y, NEURAL, SPARSE, GROUPS, seed=44, n_splits=5)
    finally:
        cal._best_threshold = keep


data = np.load(RESULTS / "calibration_predictions.npz", allow_pickle=True)
Y, NEURAL, SPARSE = data["targets"], data["neural"], data["sparse"]
GROUPS = np.array([str(name) for name in data["filenames"]])
stored = json.loads((RESULTS / "mission3_ver4_metrics.json").read_text(encoding="utf-8"))

print(f"보정 데이터: 통화 {len(Y):,}개 / 그룹 {len(set(GROUPS)):,}개")
print("Validation은 신경망 fold 가중치가 없어 로컬에서 재계산 불가\n")

old_cal, old_rep = run(original_best_threshold)
new_cal, new_rep = run(None)

print("=" * 76)
print("1. 옛 코드 재현 확인 — 저장된 ver4 기록과 같아야 함")
print("=" * 76)
saved_f1 = stored["calibration_crossfit_macro_f1"]
saved = stored["calibrator"]
print(f"  재현한 교차검증 Macro F1 : {old_rep['calibration_crossfit_macro_f1']:.9f}")
print(f"  저장된 ver4 기록         : {saved_f1:.9f}")
print(f"  일치                     : {abs(old_rep['calibration_crossfit_macro_f1'] - saved_f1) < 1e-12}")
print(f"  임계값 9개 일치          : {np.allclose(old_cal['thresholds'], saved['thresholds'])}")
print(f"  혼합비율 9개 일치        : {old_cal['neural_weights'] == saved['neural_weights']}")

print("\n" + "=" * 76)
print("2. 교차검증 Macro F1")
print("=" * 76)
o = old_rep["calibration_crossfit_macro_f1"]
n = new_rep["calibration_crossfit_macro_f1"]
print(f"  증상별 임계값 (기존 규칙) : {o:.6f}")
print(f"  전부 0.5   (새 규칙)      : {n:.6f}")
print(f"  차이                      : {n - o:+.6f}   ({(n - o) * 100:+.4f}%p)")

print("\n" + "=" * 76)
print("3. 증상별")
print("=" * 76)
op, np_ = old_rep["per_symptom_crossfit"], new_rep["per_symptom_crossfit"]
print(f"{'증상':<10}{'기존F1':>9}{'새F1':>9}{'차이':>10}  {'기존임계값':>10}{'혼합 기존→새':>14}")
for i, s in enumerate(T):
    a, b = op[s]["f1"], np_[s]["f1"]
    mark = "  *" if old_cal["neural_weights"][i] != new_cal["neural_weights"][i] else ""
    print(f"{s:<10}{a:>9.4f}{b:>9.4f}{b - a:>+10.4f}  {old_cal['thresholds'][i]:>10.4f}"
          f"{old_cal['neural_weights'][i]:>8.2f} → {new_cal['neural_weights'][i]:<4.2f}{mark}")
print("  * = 임계값이 고정되자 혼합 비율이 스스로 옮겨간 증상")

print("\n" + "=" * 76)
print("4. 정밀도 / 재현율")
print("=" * 76)
print(f"{'증상':<10}{'정밀도':>24}{'재현율':>24}")
for s in T:
    print(f"{s:<10}{op[s]['precision']:>11.3f} → {np_[s]['precision']:<11.3f}"
          f"{op[s]['recall']:>11.3f} → {np_[s]['recall']:<11.3f}")

print("\n" + "=" * 76)
print("5. 혼합 비율을 다시 맞추지 않았다면 (임계값만 0.5로 바꿨을 때)")
print("=" * 76)


def macro_at(weights, thresholds):
    w = np.asarray(weights, dtype=float)
    p = w * NEURAL + (1.0 - w) * SPARSE
    scores = []
    for i in range(9):
        pred = p[:, i] >= thresholds[i]
        tp = int(np.count_nonzero(pred & (Y[:, i] == 1)))
        fp = int(np.count_nonzero(pred & (Y[:, i] == 0)))
        fn = int(np.count_nonzero(~pred & (Y[:, i] == 1)))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


ow = old_cal["neural_weights"]
print(f"  기존 비율 + 기존 임계값 : {macro_at(ow, old_cal['thresholds']):.6f}   (같은 데이터에 맞춘 값이라 낙관적)")
print(f"  기존 비율 + 임계값 0.5  : {macro_at(ow, [0.5] * 9):.6f}")
print(f"  새 비율   + 임계값 0.5  : {macro_at(new_cal['neural_weights'], [0.5] * 9):.6f}")
print("  → 비율을 다시 맞추는 것이 손실의 대부분을 흡수함")

print("\n" + "=" * 76)
print("6. 배포용 보정값")
print("=" * 76)
print(f"  기존 임계값 : {[round(x, 4) for x in old_cal['thresholds']]}")
print(f"  새   임계값 : {new_cal['thresholds']}")
print(f"  기존 혼합   : {ow}")
print(f"  새   혼합   : {new_cal['neural_weights']}")
