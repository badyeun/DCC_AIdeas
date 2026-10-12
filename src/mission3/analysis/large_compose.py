"""large 계열을 몇 개, 어느 조합으로 넣을 것인가 — 정직하게 고르기.

탐욕 선택은 이미 -0.0021로 기각됐다(홀드아웃에 과적합). 그래서 조합을
미리 몇 개만 지정해 두고, **결정을 A반에서 하고 측정을 B반에서** 한다.
A반에서 고른 조합이 B반에서도 이득이면 그 조합이 실제로 일반화된다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy"); T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
E = [L(v) for v in
     ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
      "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
      "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
      "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
      "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
      "roberta-base_cls_raw_s42_LS"]]
P = {"s1": "roberta-large_raw_s1_LGC", "s7": "roberta-large_raw_s7_LGC",
     "attn": "roberta-large_attn_raw_s42_LGC", "tapt": "klue-roberta-large-tapt_raw_s42_CAND",
     "norm": "roberta-large_synonym+filler_s42", "old42": "roberta-large_raw_s42"}
P = {k: L(v) for k, v in P.items()}

# 미리 지정한 조합만 비교한다 (사후 탐색 금지)
PLANS = {
    "현 11개 (추가 없음)": [],
    "+ large3 (s1,s7,attn)": ["s1", "s7", "attn"],
    "+ large3 + tapt": ["s1", "s7", "attn", "tapt"],
    "+ large3 + norm": ["s1", "s7", "attn", "norm"],
    "+ large3 + tapt + norm": ["s1", "s7", "attn", "tapt", "norm"],
    "+ large2 (s1,attn)": ["s1", "attn"],
    "+ large2 + tapt": ["s1", "attn", "tapt"],
    "+ large5 (전부)": ["s1", "s7", "attn", "tapt", "norm"],
}
f1 = lambda p, i: macro_f1(Y[i], p[i] >= .5)
blend = lambda ms, i: f1(.9 * np.mean(ms, axis=0) + .1 * T, i)
ALL = np.arange(len(Y))
base = blend(E, ALL)

print(f"{'구성':<26}{'멤버':>5}{'홀드아웃':>11}{'Δ':>10}")
print("-" * 52)
for name, keys in PLANS.items():
    m = E + [P[k] for k in keys]
    s = blend(m, ALL)
    print(f"{name:<26}{len(m):>5}{s:>11.5f}{s-base:>+10.5f}")

print("\n=== 정직한 선택: A반에서 고르고 B반에서 측정 (300회) ===")
rng = np.random.default_rng(11); n = len(Y)
names = list(PLANS)
chosen, gains = {k: 0 for k in names}, {k: [] for k in names}
bgain = []
for _ in range(300):
    p = rng.permutation(n); A, B = p[:n//2], p[n//2:]
    scores = {k: blend(E + [P[x] for x in v], A) for k, v in PLANS.items()}
    best = max(scores, key=scores.get)
    chosen[best] += 1
    g = blend(E + [P[x] for x in PLANS[best]], B) - blend(E, B)
    gains[best].append(g); bgain.append(g)
print(f"{'구성':<26}{'A반에서 선택된 횟수':>20}{'그때 B반 Δ':>13}")
for k in names:
    c = chosen[k]
    g = f"{np.mean(gains[k]):+.5f}" if c else "-"
    print(f"{k:<26}{c:>20}{g:>13}")
bg = np.array(bgain)
print(f"\n  '반쪽 데이터로 고르기' 전략의 기대 이득: {bg.mean():+.5f} "
      f"± {bg.std(ddof=1)/np.sqrt(len(bg)):.5f}  (양수 {100*(bg>0).mean():.0f}%)")

print("\n=== 고정 선택을 B반에서만 재서 비교 (선택 편향 없음) ===")
rng2 = np.random.default_rng(23)
print(f"{'구성':<26}{'반분 Δ':>10}{'±SE':>9}{'승률':>7}")
halves = []
for _ in range(240):
    p = rng2.permutation(n); halves += [p[:n//2]]
for name, keys in PLANS.items():
    if not keys: continue
    m = E + [P[k] for k in keys]
    d = np.array([blend(m, i) - blend(E, i) for i in halves])
    print(f"{name:<26}{d.mean():>+10.5f}{d.std(ddof=1)/np.sqrt(len(d)):>9.5f}"
          f"{(d>0).mean():>7.0%}")
