"""large 계열 +0.0019 가 진짜인가, 그냥 '멤버 늘리면 오른다'인가.

s1+attn 조합은 7개 조합 중 홀드아웃에서 최고인 것을 내가 고른 것이므로,
선택 편향이 있다. 선택 없는 선택지(세 개 전부)와, 통제군을 함께 본다.

통제군: 기존 계열에서 같은 개수를 추가(중복)했을 때. 여기서 오르면
'멤버 수 효과'이고, 오르지 않으면 large 계열 고유의 기여다.
"""
import sys, itertools, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy"); T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")

SHIP = ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
        "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
        "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
        "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
        "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
        "roberta-base_cls_raw_s42_LS"]
E = [L(v) for v in SHIP]
LG = [L("roberta-large_raw_s1_LGC"), L("roberta-large_raw_s7_LGC"),
      L("roberta-large_attn_raw_s42_LGC")]
print("모양 확인:", Y.shape, E[0].shape, LG[0].shape,
      "(세 large 가 출하 멤버와 같은 홀드아웃인가)")

f1 = lambda P, idx: macro_f1(Y[idx], P[idx] >= .5)
blend = lambda ms, idx: f1(.9 * np.mean(ms, axis=0) + .1 * T, idx)
rng = np.random.default_rng(0); n = len(Y)
HALVES = [h for _ in range(120) for h in
          (lambda p: ((p[:n//2],), (p[n//2:],)))(rng.permutation(n))]
IDX = [h[0] for h in HALVES]
ALL = np.arange(n)

def stat(extra):
    d = np.array([blend(E + extra, i) - blend(E, i) for i in IDX])
    return blend(E + extra, ALL) - blend(E, ALL), d.mean(), (d > 0).mean()

print(f"\n{'추가 멤버':<40}{'Δ 홀드아웃':>12}{'반분 Δ':>10}{'승률':>7}")
print("-" * 69)
for k in (1, 2, 3):
    # large: 선택 없이 전부 / 통제: 기존 멤버 중복 (품질 상위 k개)
    if k == 3:
        dh, dm, w = stat(LG)
        print(f"{'large 세 개 전부 (선택 없음)':<40}{dh:>+12.5f}{dm:>+10.5f}{w:>7.0%}")
    else:
        best = max(itertools.combinations(range(3), k),
                   key=lambda c: blend(E + [LG[i] for i in c], ALL))
        dh, dm, w = stat([LG[i] for i in best])
        print(f"{f'large {k}개 (홀드아웃 최적 조합 = 선택 편향)':<40}"
              f"{dh:>+12.5f}{dm:>+10.5f}{w:>7.0%}")
for k in (1, 2, 3):
    dh, dm, w = stat(E[:k])
    print(f"{f'통제: 기존 멤버 {k}개 중복':<40}{dh:>+12.5f}{dm:>+10.5f}{w:>7.0%}")

print("\n=== 정직한 검증: 결정을 A반에서 하고 B반에서 잰다 ===")
# A반에서 'large 계열을 넣는 게 낫다'가 보이는가 → B반에서 실제로 이득인가
rng2 = np.random.default_rng(7); picks, gains = 0, []
for _ in range(200):
    p = rng2.permutation(n); A, B = p[:n//2], p[n//2:]
    if blend(E + LG, A) > blend(E, A):          # A반의 판단
        picks += 1
        gains.append(blend(E + LG, B) - blend(E, B))   # B반의 실측
gains = np.array(gains)
print(f"  A반에서 '추가가 낫다'고 판단한 비율: {picks/200:.0%}")
print(f"  그 경우 B반 실측 Δ: {gains.mean():+.5f} ± {gains.std(ddof=1)/np.sqrt(len(gains)):.5f}"
      f"   (양수 {100*(gains>0).mean():.0f}%)")

print("\n=== large 멤버는 기존과 얼마나 다른 예측을 하는가 ===")
base_mean = np.mean(E, axis=0)
for nm, p in [("출하 멤버 평균 상호 불일치", None), ("large vs 출하", None)]:
    pass
dis = lambda a, b: np.mean((a >= .5) != (b >= .5))
pair = [dis(E[i], E[j]) for i in range(11) for j in range(i+1, 11)]
cross = [dis(l, e) for l in LG for e in E]
inner = [dis(LG[i], LG[j]) for i in range(3) for j in range(i+1, 3)]
print(f"  출하 11개 내부 평균 불일치: {np.mean(pair):.4f}")
print(f"  large ↔ 출하 평균 불일치:   {np.mean(cross):.4f}")
print(f"  large 내부 평균 불일치:     {np.mean(inner):.4f}")
