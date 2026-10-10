"""Is the TF-IDF blend real, or did we just read the holdout twice?

The weight is picked on one half of the holdout and scored on the other, with
the halves swapped, over many random splits. A weight that only looked good
because we searched for it on the scoring data will not survive this.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
E = np.load(R / "_baseline_ens_probs.npy")
T = np.load(R / "exp01_best_probs.npy")
GRID = np.arange(0.0, 0.45, 0.05)


def ap(P, idx):
    return np.mean([average_precision_score(Y[idx, i], P[idx, i]) for i in range(9)
                    if Y[idx, i].any()])


def f1(P, idx):
    return macro_f1(Y[idx], (P[idx] >= .5).astype(np.int8))


rng = np.random.default_rng(0)
picked, gain_ap, gain_f1 = [], [], []
for rep in range(40):
    order = rng.permutation(len(Y))
    for a, b in ((order[:len(Y)//2], order[len(Y)//2:]), (order[len(Y)//2:], order[:len(Y)//2])):
        w = GRID[int(np.argmax([ap((1-g)*E + g*T, a) for g in GRID]))]
        picked.append(w)
        gain_ap.append(ap((1-w)*E + w*T, b) - ap(E, b))
        gain_f1.append(f1((1-w)*E + w*T, b) - f1(E, b))

picked = np.array(picked); gain_ap = np.array(gain_ap); gain_f1 = np.array(gain_f1)
print(f"반분 검증 {len(picked)}회 (가중치는 한쪽에서 고르고 반대쪽에서 채점)")
print(f"  고른 가중치: 중앙값 {np.median(picked):.2f} · "
      + " ".join(f"{g:.2f}→{int((picked==g).sum())}회" for g in GRID if (picked == g).any()))
print(f"  ΔAP  평균 {gain_ap.mean():+.5f} · 표준오차 {gain_ap.std(ddof=1)/np.sqrt(len(gain_ap)):.5f}"
      f" · 이긴 비율 {(gain_ap > 0).mean():.0%}")
print(f"  ΔF1  평균 {gain_f1.mean():+.5f} · 표준오차 {gain_f1.std(ddof=1)/np.sqrt(len(gain_f1)):.5f}"
      f" · 이긴 비율 {(gain_f1 > 0).mean():.0%}")

print("\n가중치를 고정했을 때 (탐색 없음, 전체 holdout):")
print(f"  {'w':<8}{'F1@0.5':>10}{'ΔF1':>10}{'평균 AP':>11}{'ΔAP':>10}")
b0, a0 = f1(E, slice(None)), ap(E, slice(None))
for w in (0.0, 0.10, 0.15, 0.20, 0.25, 0.30):
    P = (1 - w) * E + w * T
    print(f"  {w:<8.2f}{f1(P, slice(None)):>10.5f}{f1(P, slice(None))-b0:>+10.5f}"
          f"{ap(P, slice(None)):>11.5f}{ap(P, slice(None))-a0:>+10.5f}")
