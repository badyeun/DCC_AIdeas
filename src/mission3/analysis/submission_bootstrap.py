"""제출 후보들의 차이가 진짜인가 — 홀드아웃 부트스트랩.

후보 네 개가 0.643~0.647 안에 모여 있습니다. 그중 최고를 그냥 고르면
홀드아웃에 맞춘 값을 고르는 것이라, 제출 성적이 그만큼 안 나옵니다.
홀드아웃 통화를 복원추출로 다시 뽑아 가며 각 후보를 채점하고, 차이의
분포를 봅니다. 0을 포함하면 구분 불가입니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
G = {"mean3e5": ["raw_s42", "raw_s1", "raw_s7"],
     "attn3e5": ["attn_raw_s42", "attn_raw_s1", "attn_raw_s7"],
     "mean2e5": ["raw_s42_lr2e-5e3", "raw_s1_lr2e-5e3", "raw_s7_lr2e-5e3"],
     "attn2e5": ["attn_raw_s42_lr2e5_LS", "attn_raw_s1_lr2e5_LS", "attn_raw_s7_lr2e5_LS"]}
load = {k: [np.load(R / f"probs_roberta-base_{t}.npy") for t in v] for k, v in G.items()}
CAND = {
    "A mean2e5+attn3e5": ["mean2e5", "attn3e5"],
    "B mean2e5+attn2e5 (현 제출)": ["mean2e5", "attn2e5"],
    "C mean3e5+attn3e5": ["mean3e5", "attn3e5"],
    "D 네 그룹 12개": ["mean3e5", "attn3e5", "mean2e5", "attn2e5"],
}
P = {k: 0.9 * np.mean([m for g in v for m in load[g]], axis=0) + 0.1 * T
     for k, v in CAND.items()}

rng = np.random.default_rng(7)
N, B = len(Y), 400
draws = {k: [] for k in CAND}
for _ in range(B):
    idx = rng.integers(0, N, N)
    yb = Y[idx]
    for k, p in P.items():
        draws[k].append(macro_f1(yb, (p[idx] >= .5).astype(np.int8)))
draws = {k: np.array(v) for k, v in draws.items()}

print(f"부트스트랩 {B}회 (홀드아웃 {N:,}통화 복원추출)\n")
print(f"{'후보':<28}{'실측':>9}{'부트 평균':>11}{'표준편차':>10}")
for k, p in P.items():
    print(f"{k:<28}{macro_f1(Y, (p >= .5).astype(np.int8)):>9.5f}"
          f"{draws[k].mean():>11.5f}{draws[k].std():>10.5f}")

print(f"\n짝지은 차이 (같은 부트스트랩 표본에서 비교)")
print(f"{'비교':<40}{'평균 차이':>11}{'95% 구간':>22}{'유의':>7}")
keys = list(CAND)
best = max(keys, key=lambda k: macro_f1(Y, (P[k] >= .5).astype(np.int8)))
for k in keys:
    if k == best:
        continue
    d = draws[best] - draws[k]
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "O" if lo > 0 or hi < 0 else "X"
    print(f"{best[:18]+' − '+k[:18]:<40}{d.mean():>+11.5f}{f'[{lo:+.5f}, {hi:+.5f}]':>22}{sig:>7}")
print("\n'유의 X'는 그 두 구성을 홀드아웃으로 구분할 수 없다는 뜻입니다.")
