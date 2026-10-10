"""Does a per-symptom blend weight beat one global weight?

Nine weights fitted on a holdout will always look better on that holdout, so
each is fitted on one half and scored on the other, with the halves swapped.
If per-symptom weights are genuinely better they beat the global weight on the
half they were not fitted on; if they are just absorbing noise, they lose.

NOTE: per-symptom MIXING weights are the subject of the open question in
docs/멘토_질문.md. Nothing here is folded into the submission until that is
answered -- this run only measures what the answer would be worth.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
E = np.load(R / "_baseline_ens_probs.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
GRID = np.arange(0.0, 0.45, 0.05)


def ap_label(P, idx, i):
    return average_precision_score(Y[idx, i], P[idx, i]) if Y[idx, i].any() else 0.0


def ap(P, idx):
    return np.mean([ap_label(P, idx, i) for i in range(9)])


def f1(P, idx):
    return macro_f1(Y[idx], (P[idx] >= .5).astype(np.int8))


rng = np.random.default_rng(1)
res = {"global": [[], []], "per-label": [[], []]}
chosen = np.zeros((0, 9))
for rep in range(40):
    order = rng.permutation(len(Y))
    for a, b in ((order[:len(Y)//2], order[len(Y)//2:]), (order[len(Y)//2:], order[:len(Y)//2])):
        g = GRID[int(np.argmax([ap((1-w)*E + w*T, a) for w in GRID]))]
        Pg = (1-g)*E + g*T
        res["global"][0].append(ap(Pg, b) - ap(E, b))
        res["global"][1].append(f1(Pg, b) - f1(E, b))

        ws = np.array([GRID[int(np.argmax([ap_label((1-w)*E + w*T, a, i) for w in GRID]))]
                       for i in range(9)])
        chosen = np.vstack([chosen, ws])
        Pp = (1-ws)*E + ws*T
        res["per-label"][0].append(ap(Pp, b) - ap(E, b))
        res["per-label"][1].append(f1(Pp, b) - f1(E, b))

print("반분 검증 80회 — 혼합 비율은 한쪽에서 고르고 반대쪽에서 채점\n")
print(f"{'방식':<12}{'ΔAP 평균':>11}{'표준오차':>10}{'승률':>7}   "
      f"{'ΔF1 평균':>11}{'표준오차':>10}{'승률':>7}")
for k, (a, f) in res.items():
    a, f = np.array(a), np.array(f)
    print(f"{k:<12}{a.mean():>+11.5f}{a.std(ddof=1)/np.sqrt(len(a)):>10.5f}{(a>0).mean():>7.0%}   "
          f"{f.mean():>+11.5f}{f.std(ddof=1)/np.sqrt(len(f)):>10.5f}{(f>0).mean():>7.0%}")

d = np.array(res["per-label"][0]) - np.array(res["global"][0])
print(f"\n증상별 − 전역 (같은 분할에서 짝지음): ΔAP {d.mean():+.5f} ± "
      f"{d.std(ddof=1)/np.sqrt(len(d)):.5f} · 증상별이 이긴 비율 {(d>0).mean():.0%}")
print(f"\n{'증상':<8}{'고른 가중치 중앙값':>18}{'변동(표준편차)':>15}")
for i, t in enumerate(TARGETS):
    print(f"{t:<8}{np.median(chosen[:, i]):>18.2f}{chosen[:, i].std():>15.3f}")
