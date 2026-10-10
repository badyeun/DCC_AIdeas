"""AP가 좋아지면 F1@0.5도 좋아지는가 — 지표 교체의 전제 검증.

attention 헤드는 AP가 mean보다 0.0053 높은데 F1@0.5는 같습니다. AP가 순위
전체를 보는 반면 F1@0.5는 임계선 근처만 보기 때문일 수 있습니다. 그렇다면
AP로 실험을 판정하는 것은 틀린 기준이 됩니다.

세 가지를 봅니다:
  1. 증상별 최적 임계값에서의 F1 (= 그 순위로 낼 수 있는 최대)
  2. AP 우위가 순위의 어느 구간에서 나오는가
  3. 기존 실험들에서 ΔAP와 ΔF1이 실제로 같이 움직였는가
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
GRP = {"mean lr3e-5": ["raw_s42", "raw_s1", "raw_s7"],
       "attn lr3e-5": ["attn_raw_s42", "attn_raw_s1", "attn_raw_s7"],
       "mean lr2e-5": ["raw_s42_lr2e-5e3", "raw_s1_lr2e-5e3", "raw_s7_lr2e-5e3"]}
P = {k: np.mean([np.load(R / f"probs_roberta-base_{t}.npy") for t in v], axis=0)
     for k, v in GRP.items()}


def ceiling(p):
    """증상별로 최적 임계값을 고른 F1 — 그 순위가 낼 수 있는 최대치."""
    s = []
    for i in range(9):
        best = 0.0
        for t in np.arange(.03, .97, .01):
            pr = p[:, i] >= t
            tp = int((pr & (Y[:, i] == 1)).sum()); d = int(Y[:, i].sum()) + int(pr.sum())
            best = max(best, 2 * tp / d if d else 0.0)
        s.append(best)
    return float(np.mean(s)), s


print("=== 1. 순위가 낼 수 있는 최대 F1 (증상별 최적 임계) ===")
print(f"{'구성':<14}{'F1@0.5':>10}{'순위 상한':>11}{'평균 AP':>10}")
for k, p in P.items():
    c, _ = ceiling(p)
    ap = np.mean([average_precision_score(Y[:, i], p[:, i]) for i in range(9)])
    print(f"{k:<14}{macro_f1(Y, (p >= .5).astype(np.int8)):>10.5f}{c:>11.5f}{ap:>10.5f}")
print("→ AP가 높은 쪽이 '순위 상한'도 높으면 AP는 올바른 대리지표입니다.")

print("\n=== 2. attention의 AP 우위는 순위의 어느 구간에서 나오나 ===")
print(f"{'증상':<8}{'mean AP':>9}{'attn AP':>9}{'차이':>8}   "
      f"{'상위 k=양성수 안 적중률 (mean→attn)':<10}")
for i, t in enumerate(TARGETS):
    a = average_precision_score(Y[:, i], P["mean lr3e-5"][:, i])
    b = average_precision_score(Y[:, i], P["attn lr3e-5"][:, i])
    k = int(Y[:, i].sum())
    pa = Y[np.argsort(-P["mean lr3e-5"][:, i])[:k], i].mean()
    pb = Y[np.argsort(-P["attn lr3e-5"][:, i])[:k], i].mean()
    print(f"{t:<8}{a:>9.3f}{b:>9.3f}{b-a:>+8.3f}   {pa:.3f} → {pb:.3f}  ({pb-pa:+.3f})")

print("\n=== 3. 지금까지 쌓인 실행들에서 ΔAP와 ΔF1은 같이 움직였나 ===")
runs = sorted(R.glob("probs_roberta-base_*.npy"))
xs, ys = [], []
base = np.load(R / "probs_roberta-base_raw_s42_lr2e-5e3.npy")
bf, ba = (macro_f1(Y, (base >= .5).astype(np.int8)),
          np.mean([average_precision_score(Y[:, i], base[:, i]) for i in range(9)]))
for p in runs:
    q = np.load(p)
    xs.append(np.mean([average_precision_score(Y[:, i], q[:, i]) for i in range(9)]) - ba)
    ys.append(macro_f1(Y, (q >= .5).astype(np.int8)) - bf)
xs, ys = np.array(xs), np.array(ys)
print(f"  단일 실행 {len(xs)}개 · ΔAP와 ΔF1의 상관 = {np.corrcoef(xs, ys)[0,1]:.3f}")
keep = np.abs(xs) < 0.03
print(f"  |ΔAP| < 0.03인 {int(keep.sum())}개만: 상관 {np.corrcoef(xs[keep], ys[keep])[0,1]:.3f}")
slope = np.polyfit(xs, ys, 1)[0]
print(f"  회귀 기울기: ΔAP 1단위당 ΔF1 {slope:.2f}배")
