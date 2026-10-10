"""세 번째 계열(정규화 텍스트 모델)을 앙상블에 넣으면 이득인가.

1라운드는 "정규화가 단독 성능을 올리는가"만 물었고 답은 −0.0003이었습니다.
**앙상블 재료로서의 가치는 물은 적이 없습니다.** 헤드 혼합이 성공한 조건은
"비슷하게 좋으면서 다름"이었는데, 정규화 모델이 바로 그 조건을 만족합니다:
단독 0.63742(기준 0.63848, 차이 잡음 안), 불일치율 0.0208(전 계열 최고).

겉보기 차이가 0.001 수준이라 눈으로는 판단할 수 없습니다. 같은 부트스트랩
표본에서 짝지어 채점합니다. 비교는 **미리 하나만** 정합니다 —
"현 제출 구성" 대 "거기에 정규화 계열을 더한 것".
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
F = {
    "mean2e5": [np.load(R / f"probs_roberta-base_raw_s{s}_lr2e-5e3.npy") for s in (42, 1, 7)],
    "attn3e5": [np.load(R / f"probs_roberta-base_attn_raw_s{s}.npy") for s in (42, 1, 7)],
    "attn2e5": [np.load(R / f"probs_roberta-base_attn_raw_s{s}_lr2e5_LS.npy") for s in (42, 1, 7)],
    "norm":    [np.load(R / f"probs_roberta-base_synonym+filler_s{s}.npy") for s in (42, 1, 7)],
}


def blend(keys, w=0.10):
    mem = [m for k in keys for m in F[k]]
    return (1 - w) * np.mean(mem, axis=0) + w * T


PAIRS = [
    ("현 제출 (mean2e5+attn2e5)", ["mean2e5", "attn2e5"],
     "+ 정규화 3개", ["mean2e5", "attn2e5", "norm"]),
    ("mean2e5+attn3e5", ["mean2e5", "attn3e5"],
     "+ 정규화 3개", ["mean2e5", "attn3e5", "norm"]),
]
rng = np.random.default_rng(31)
B = 800
for la, ka, lb, kb in PAIRS:
    A, Bp = blend(ka), blend(kb)
    fa, fb = macro_f1(Y, (A >= .5).astype(np.int8)), macro_f1(Y, (Bp >= .5).astype(np.int8))
    d = []
    for _ in range(B):
        i = rng.integers(0, len(Y), len(Y))
        yb = Y[i]
        d.append(macro_f1(yb, (Bp[i] >= .5).astype(np.int8))
                 - macro_f1(yb, (A[i] >= .5).astype(np.int8)))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "○ 이득 실재" if lo > 0 else "✗ 손해 실재" if hi < 0 else "− 구분 불가"
    print(f"{la}  →  {lb}")
    print(f"  실측 {fa:.5f} → {fb:.5f}  ({fb-fa:+.5f})")
    print(f"  부트스트랩 {B}회: 평균 {d.mean():+.5f} · 95% [{lo:+.5f}, {hi:+.5f}]"
          f" · 이긴 비율 {(d>0).mean():.0%}")
    print(f"  판정: {verdict}\n")

print("주의: 부트스트랩은 '이 홀드아웃에서의 차이'만 봅니다. 정규화 모델 3개를")
print("다른 seed로 다시 뽑았을 때의 흔들림은 포함되지 않습니다.")
