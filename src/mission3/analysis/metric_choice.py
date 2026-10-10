"""F1@0.5의 조용한 대리지표로 무엇을 써야 하는가.

AP를 쓰자고 했던 근거는 "seed 편차가 1/5.9"였는데, 조용한 것만으로는
부족합니다. 재려는 것과 **같이 움직여야** 합니다. AP는 순위 전체(꼬리 포함)를
재고 F1@0.5는 임계선 근처만 보기 때문에 둘이 어긋납니다.

후보를 셋 더 놓고, 조용함과 F1@0.5와의 상관을 함께 봅니다.
  R-precision : 상위 k개(k=양성 수)의 적중률 — 임계선 근처를 직접 봄
  순위 상한   : 증상별 최적 임계에서의 F1 — 그 순위가 낼 수 있는 최대
  F1@0.5      : 목표 그 자체
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")


def f1_at(p, t=0.5):
    return macro_f1(Y, (p >= t).astype(np.int8))


def rprec(p):
    out = []
    for i in range(9):
        k = int(Y[:, i].sum())
        out.append(Y[np.argsort(-p[:, i])[:k], i].mean())
    return float(np.mean(out))


def ceiling(p):
    out = []
    for i in range(9):
        best = 0.0
        for t in np.arange(.03, .97, .01):
            pr = p[:, i] >= t
            tp = int((pr & (Y[:, i] == 1)).sum()); d = int(Y[:, i].sum()) + int(pr.sum())
            best = max(best, 2 * tp / d if d else 0.0)
        out.append(best)
    return float(np.mean(out))


def ap(p):
    return float(np.mean([average_precision_score(Y[:, i], p[:, i]) for i in range(9)]))


METRICS = {"F1@0.5": f1_at, "R-precision": rprec, "순위 상한": ceiling, "평균 AP": ap}

# seed 잡음: 같은 설정을 seed만 바꾼 3개 묶음 세 개
GRP = {"mean lr3e-5": ["raw_s42", "raw_s1", "raw_s7"],
       "attn lr3e-5": ["attn_raw_s42", "attn_raw_s1", "attn_raw_s7"],
       "mean lr2e-5": ["raw_s42_lr2e-5e3", "raw_s1_lr2e-5e3", "raw_s7_lr2e-5e3"]}
print("=== seed 잡음 (같은 설정, seed만 다름 — 작을수록 조용함) ===")
print(f"{'지표':<14}" + "".join(f"{g:>14}" for g in GRP) + f"{'평균 편차':>11}")
noise = {}
vals = {m: {g: [fn(np.load(R / f"probs_roberta-base_{t}.npy")) for t in v]
            for g, v in GRP.items()} for m, fn in METRICS.items()}
for m in METRICS:
    sp = [max(vals[m][g]) - min(vals[m][g]) for g in GRP]
    noise[m] = float(np.mean(sp))
    print(f"{m:<14}" + "".join(f"{s:>14.5f}" for s in sp) + f"{noise[m]:>11.5f}")

# F1@0.5와의 상관: 쌓인 단일 실행 전부
runs = sorted(R.glob("probs_roberta-base_*.npy"))
tab = {m: [] for m in METRICS}
for p in runs:
    q = np.load(p)
    for m, fn in METRICS.items():
        tab[m].append(fn(q))
tab = {m: np.array(v) for m, v in tab.items()}
print(f"\n=== F1@0.5와의 상관 (단일 실행 {len(runs)}개) ===")
print(f"{'지표':<14}{'상관':>9}{'seed 잡음':>11}{'유용도 = 상관/잡음':>20}")
for m in METRICS:
    c = np.corrcoef(tab[m], tab["F1@0.5"])[0, 1]
    print(f"{m:<14}{c:>9.3f}{noise[m]:>11.5f}{abs(c)/noise[m]:>20.0f}")
print("\n'유용도'는 같은 확신을 얻는 데 드는 실행 수에 반비례합니다.")
print("상관이 낮으면 아무리 조용해도 쓸모가 없습니다.")
