"""내가 고른 large 5개가 '규칙'인가 '홀드아웃 최댓값'인가.

크기 곡선에서 k=5 의 평균은 0.64724, 최댓값이 0.65075 였다. 내가 보고한
구성의 점수가 바로 그 최댓값이다. 조합 56개 중 1등을 홀드아웃에서 집은
것이라면, 반분 검증이 99% 승을 줘도 소용없다 — 반분은 '평가 잡음'만 다시
뽑고 '멤버 선택'은 이미 전체 홀드아웃에서 끝났기 때문이다.

그래서 선택까지 A반 안으로 넣는다. A반에서 고르고 B반에서만 재면, 선택이
이득인지 손해인지 정직하게 나온다.
"""
import sys, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy"); T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
BN = ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
      "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
      "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
      "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
      "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
      "roberta-base_cls_raw_s42_LS"]
LN = {"s1": "roberta-large_raw_s1_LGC", "s7": "roberta-large_raw_s7_LGC",
      "attn": "roberta-large_attn_raw_s42_LGC",
      "tapt": "klue-roberta-large-tapt_raw_s42_CAND",
      "norm": "roberta-large_synonym+filler_s42",
      "old42": "roberta-large_raw_s42", "s42LS": "roberta-large_raw_s42_LS",
      "lr1e5": "roberta-large_lr1e-5_raw_s42_LS"}
B = [L(v) for v in BN]; LG = {k: L(v) for k, v in LN.items()}
RECENT = ["s1", "s7", "attn", "tapt", "norm"]      # 어제 큐에서 나온 것 = 사전 규칙

f1 = lambda p, i: macro_f1(Y[i], p[i] >= .5)
sc = lambda ms, i: f1(.9 * np.mean(ms, axis=0) + .1 * T, i)
ALL = np.arange(len(Y)); n = len(Y)
base = sc(B, ALL)

# 내가 보고한 구성이 정말 56개 중 1등인가
k5 = [(sc([LG[x] for x in c], ALL), c) for c in itertools.combinations(LN, 5)]
k5.sort(reverse=True)
print(f"large 5개 조합 56개 중 상위 5개:")
for s, c in k5[:5]:
    tag = "  <- 내가 보고한 구성" if set(c) == set(RECENT) else ""
    print(f"  {s:.5f}  {'+'.join(c)}{tag}")
rank = 1 + [set(c) for _, c in k5].index(set(RECENT))
print(f"\n  내 구성의 순위: 56개 중 {rank}등")
print(f"  k=5 평균 {np.mean([s for s, _ in k5]):.5f} / 내 구성 {k5[rank-1][0]:.5f}")

rng = np.random.default_rng(101)
HAL = [(lambda p: (p[:n//2], p[n//2:]))(rng.permutation(n)) for _ in range(300)]

def run(build, label):
    d = np.array([sc(build(A), Bh) - sc(B, Bh) for A, Bh in HAL])
    print(f"{label:<40}{d.mean():>+10.5f}{d.std(ddof=1)/np.sqrt(len(d)):>9.5f}"
          f"{(d>0).mean():>7.0%}")
    return d.mean()

def pick_best_k(A, k, pool):
    best = max(itertools.combinations(pool, k),
               key=lambda c: sc([LG[x] for x in c], A))
    return list(best)

print("\n=== 선택까지 A반 안에서 하고 B반에서만 측정 (300회) ===")
print(f"{'전략':<40}{'B반 Δ':>10}{'±SE':>9}{'승률':>7}")
print("-- 선택 없음 (사전에 정한 구성) " + "-" * 25)
run(lambda A: B, "현재 11개 base (기준선 0)")
run(lambda A: B + [LG[k] for k in RECENT], "base11 + large5(최근 큐) = 16개")
run(lambda A: B + list(LG.values()), "base11 + large8(전부) = 19개")
run(lambda A: list(LG.values()), "large8(전부) 만")
run(lambda A: [LG[k] for k in RECENT], "large5(최근 큐) 만")
print("-- A반에서 골라서 (선택 편향 노출) " + "-" * 22)
run(lambda A: pick_best_k(A, 5, list(LN)), "large 5개를 A반에서 최적 선택")
run(lambda A: B + pick_best_k(A, 5, list(LN)), "base11 + A반 최적 large5")
run(lambda A: pick_best_k(A, 3, list(LN)), "large 3개를 A반에서 최적 선택")
