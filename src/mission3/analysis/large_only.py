"""large 만으로 가면 몇 개가 최적인가.

large5 만(0.65075)이 16개(0.65038)를 이겼다. 그런데 멤버 5개는 앙상블
평균의 이점이 약하다 — 크기 곡선은 10개 부근에서 포화했고, seed 표준편차는
단일 모델에서 0.0023 이다. 5개로 줄이면 홀드아웃 숫자가 운일 가능성이 커진다.

반분 검증은 '평가 잡음'만 잡고 'seed 잡음'은 못 잡는다. 그래서 여기서는
   (1) 사용 가능한 large 급 전부로 크기 곡선을 다시 그리고
   (2) 같은 크기에서 조합을 무작위로 뽑아 분산을 함께 본다.
"""
import sys, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy"); T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
BASE = [L(v) for v in
        ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
         "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
         "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
         "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
         "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
         "roberta-base_cls_raw_s42_LS"]]
# 사용 가능한 large 급 전부. 설정이 다른 것도 '다양성'이므로 모두 후보.
LGN = {"mean s1 (1.5e-5)": "roberta-large_raw_s1_LGC",
       "mean s7 (1.5e-5)": "roberta-large_raw_s7_LGC",
       "attn s42 (1.5e-5)": "roberta-large_attn_raw_s42_LGC",
       "tapt s42 (2e-5)": "klue-roberta-large-tapt_raw_s42_CAND",
       "norm s42 (1.5e-5)": "roberta-large_synonym+filler_s42",
       "mean s42 (구설정)": "roberta-large_raw_s42",
       "mean s42 LS": "roberta-large_raw_s42_LS",
       "lr1e-5 s42": "roberta-large_lr1e-5_raw_s42_LS"}
LGN = {k: v for k, v in LGN.items() if (R / f"probs_{v}.npy").exists()}
LG = {k: L(v) for k, v in LGN.items()}

f1 = lambda p, i: macro_f1(Y[i], p[i] >= .5)
sc = lambda ms, i: f1(.9 * np.mean(ms, axis=0) + .1 * T, i)
ALL = np.arange(len(Y))
s11 = sc(BASE, ALL)
print(f"현재 11개 base: {s11:.5f}\n")
print(f"사용 가능한 large 급 {len(LG)}개, 단독 점수:")
for k, p in LG.items():
    print(f"  {k:<22}{f1(p, ALL):.5f}")

print("\n=== large 만으로, 크기별 (같은 크기 조합을 전부 또는 60개 표본) ===")
keys = list(LG)
rng = np.random.default_rng(5)
print(f"{'멤버 수':>7}{'조합 수':>8}{'평균':>10}{'최저':>10}{'최고':>10}{'표준편차':>10}")
curve = {}
for k in range(1, len(keys) + 1):
    combos = list(itertools.combinations(keys, k))
    if len(combos) > 60:
        combos = [tuple(rng.choice(keys, k, replace=False)) for _ in range(60)]
    v = np.array([sc([LG[x] for x in c], ALL) for c in combos])
    curve[k] = v
    print(f"{k:>7}{len(combos):>8}{v.mean():>10.5f}{v.min():>10.5f}"
          f"{v.max():>10.5f}{v.std():>10.5f}")

print("\n=== large 와 base 를 섞는 비율 (large 전부 + base 상위 m개) ===")
print(f"{'구성':<26}{'멤버':>5}{'홀드아웃':>11}{'Δ':>10}{'비용배수':>9}")
cost = lambda nl, nb: (nl * 3.5 + nb) / 11
for m in (0, 2, 4, 6, 8, 11):
    ms = list(LG.values()) + BASE[:m]
    s = sc(ms, ALL)
    print(f"{f'large {len(LG)} + base {m}':<26}{len(ms):>5}{s:>11.5f}"
          f"{s-s11:>+10.5f}{cost(len(LG), m):>8.1f}x")

print("\n=== 정직한 검증: A반에서 구성을 고르고 B반에서 측정 (240회) ===")
n = len(Y); rng2 = np.random.default_rng(77)
HAL = [(lambda p: (p[:n//2], p[n//2:]))(rng2.permutation(n)) for _ in range(240)]
def honest(ms, label):
    d = np.array([sc(ms, B) - sc(BASE, B) for _, B in HAL])
    print(f"{label:<30}{d.mean():>+10.5f}{d.std(ddof=1)/np.sqrt(len(d)):>9.5f}"
          f"{(d>0).mean():>7.0%}")
print(f"{'구성':<30}{'B반 Δ':>10}{'±SE':>9}{'승률':>7}")
honest(list(LG.values()), f"large {len(LG)}개 전부")
honest([LG[k] for k in list(LG)[:5]], "large 5개 (어제 확정분)")
honest(list(LG.values()) + BASE, f"large {len(LG)} + base 11")
honest(BASE + [LG[k] for k in list(LG)[:5]], "base 11 + large 5 (= 16개)")
