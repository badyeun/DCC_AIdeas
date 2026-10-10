"""Is the fixed-0.5 vs per-symptom-threshold gap larger than split noise?

The cross-fit score depends on how the 2,880 calibration calls are split. If
re-seeding the split moves the score by more than the rule change does, the
rule change is not measurable on this sample.

Run:  PYTHONPATH=src/mission3 python src/mission3/analysis/seed_spread.py
"""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np

from mission3_ver4 import calibration as cal
from ab_fixed_threshold import original_best_threshold

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "ver4"
d = np.load(R / "calibration_predictions.npz", allow_pickle=True)
Y, NEURAL, SPARSE = d["targets"], d["neural"], d["sparse"]
GROUPS = np.array([str(f) for f in d["filenames"]])

SEEDS = (44, 43, 142, 7, 2026, 1, 99, 314, 1234, 57)


def score(seed, patched):
    keep = cal._best_threshold
    if patched is not None:
        cal._best_threshold = patched
    try:
        _, rep = cal.calibrate_crossfit(Y, NEURAL, SPARSE, GROUPS, seed=seed, n_splits=5)
        return rep["calibration_crossfit_macro_f1"]
    finally:
        cal._best_threshold = keep


print(f"보정 통화 2,880개 · 5-fold 교차검증 · seed {len(SEEDS)}개\n")
print(f"{'seed':>6}{'증상별 임계값':>16}{'전부 0.5':>12}{'차이':>12}")
old_all, new_all, diffs = [], [], []
for s in SEEDS:
    o, n = score(s, original_best_threshold), score(s, None)
    old_all.append(o)
    new_all.append(n)
    diffs.append(n - o)
    print(f"{s:>6}{o:>16.6f}{n:>12.6f}{n - o:>+12.6f}")

print()
print(f"{'평균':>6}{statistics.mean(old_all):>16.6f}{statistics.mean(new_all):>12.6f}"
      f"{statistics.mean(diffs):>+12.6f}")
print(f"{'표준편차':>6}{statistics.stdev(old_all):>14.6f}{statistics.stdev(new_all):>12.6f}"
      f"{statistics.stdev(diffs):>12.6f}")
print(f"{'최소':>6}{min(old_all):>16.6f}{min(new_all):>12.6f}{min(diffs):>+12.6f}")
print(f"{'최대':>6}{max(old_all):>16.6f}{max(new_all):>12.6f}{max(diffs):>+12.6f}")

print()
print(f"seed만 바꿔서 생기는 흔들림 (기존 규칙) : {max(old_all) - min(old_all):.6f}")
print(f"규칙 변경으로 생기는 평균 차이           : {abs(statistics.mean(diffs)):.6f}")
wins = sum(1 for x in diffs if x > 0)
print(f"0.5 고정이 더 높았던 seed                : {wins} / {len(SEEDS)}")
print()
if abs(statistics.mean(diffs)) < (max(old_all) - min(old_all)):
    print("→ 규칙 변경의 효과가 분할 잡음보다 작습니다. 이 표본으로는 차이를 구분할 수 없습니다.")
else:
    print("→ 규칙 변경의 효과가 분할 잡음을 넘습니다.")
