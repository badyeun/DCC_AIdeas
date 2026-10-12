"""large 계열이 통하니, large 급 모델을 더 넣으면 더 오르는가.

large-tapt는 단독 추가로 +0.00028(승률 68%)였다 — 기준 미달이지만 부호는
양수였다. large 계열을 먼저 넣은 뒤에도 여전히 더해지는지 본다. 더해지면
'large 급'이라는 축이 열려 있다는 뜻이고, 사라지면 large 세 개로 포화다.
"""
import sys, numpy as np
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
TAPT = L("klue-roberta-large-tapt_raw_s42_CAND")
# 과거에 잰 large 변형들도 있다 (다른 설정이지만 large 급)
EXTRA = {"large lr1e-5": "roberta-large_lr1e-5_raw_s42_LS",
         "large s42 LS": "roberta-large_raw_s42_LS",
         "large s42 (구)": "roberta-large_raw_s42",
         "large norm": "roberta-large_synonym+filler_s42"}

f1 = lambda P, idx: macro_f1(Y[idx], P[idx] >= .5)
blend = lambda ms, idx: f1(.9 * np.mean(ms, axis=0) + .1 * T, idx)
rng = np.random.default_rng(0); n = len(Y); ALL = np.arange(n)
IDX = []
for _ in range(120):
    p = rng.permutation(n); IDX += [p[:n//2], p[n//2:]]

def delta(base, extra):
    d = np.array([blend(base + extra, i) - blend(base, i) for i in IDX])
    return blend(base + extra, ALL) - blend(base, ALL), d.mean(), (d > 0).mean()

B11, B14 = E, E + LG
print(f"기준선  11개: {blend(B11, ALL):.5f}")
print(f"기준선  14개 (= 11 + large 3): {blend(B14, ALL):.5f}\n")
print(f"{'추가 후보':<20}{'단독':>9}{'11개 위':>20}{'14개 위':>20}")
print(f"{'':<20}{'':>9}{'Δ':>10}{'승률':>10}{'Δ':>10}{'승률':>10}")
print("-" * 69)
cands = [("large-tapt", TAPT)] + [(k, L(v)) for k, v in EXTRA.items()
                                  if (R / f"probs_{v}.npy").exists()]
for name, p in cands:
    solo = f1(p, ALL)
    _, d1, w1 = delta(B11, [p])
    _, d4, w4 = delta(B14, [p])
    print(f"{name:<20}{solo:>9.5f}{d1:>+10.5f}{w1:>10.0%}{d4:>+10.5f}{w4:>10.0%}")

print("\n=== large 급 전체를 한꺼번에 (선택 없음) ===")
pool = [p for _, p in cands]
for k, lab in ((len(pool), "후보 전부"),):
    dh, dm, w = delta(B14, pool)
    print(f"  14개 + large급 {len(pool)}개: Δ {dh:+.5f}  반분 {dm:+.5f}  승률 {w:.0%}")
dh, dm, w = delta(B11, LG + [TAPT])
print(f"  11개 + large3 + tapt:   Δ {dh:+.5f}  반분 {dm:+.5f}  승률 {w:.0%}")
dh, dm, w = delta(B11, LG)
print(f"  11개 + large3 (현 계획): Δ {dh:+.5f}  반분 {dm:+.5f}  승률 {w:.0%}")
