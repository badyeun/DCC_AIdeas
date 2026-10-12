"""개수가 문제인가, 아니면 교체하면 다를까.

두 가지 가설을 가린다.
  (1) 멤버가 많아서 떨어지는가 → 크기 곡선과 '같은 품질 멤버 추가'로 확인
  (2) 추가 대신 교체하면 다를까 → 가장 기여 작은 멤버를 후보로 바꿔 본다

교체 대상을 홀드아웃에서 고르면 거기 과적합하므로, 마지막에 반분 검증으로
'풀에서 최적 11개를 고르는 것'이 고정 11개보다 나은지까지 본다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
f1 = lambda P: macro_f1(Y, (P >= .5).astype(np.int8))

SHIP = [("mean2e5 s42", "roberta-base_raw_s42_lr2e-5e3"),
        ("mean2e5 s1", "roberta-base_raw_s1_lr2e-5e3"),
        ("attn2e5 s42", "roberta-base_attn_raw_s42_lr2e5_LS"),
        ("attn2e5 s1", "roberta-base_attn_raw_s1_lr2e5_LS"),
        ("mean3e5", "roberta-base_raw_s42"),
        ("attn3e5", "roberta-base_attn_raw_s7"),
        ("norm s42", "roberta-base_synonym+filler_s42"),
        ("norm s1", "roberta-base_synonym+filler_s1"),
        ("dapt mean", "domain_adapted_raw_s42_DAPT_LS"),
        ("dapt attn", "domain_adapted_attn_raw_s42_DAPT_LS"),
        ("cls", "roberta-base_cls_raw_s42_LS")]
CAND = [("km-bert", "km-bert_raw_s42_CAND"),
        ("bert-kor-base", "bert-kor-base_raw_s42_CAND"),
        ("KoSimCSE", "KoSimCSE-roberta_raw_s42_CAND"),
        ("roberta-small", "roberta-small_raw_s42_CAND"),
        ("klue/bert-base", "bert-base_raw_s42"),
        ("KcELECTRA", "KcELECTRA-base-v2022_raw_s42"),
        ("roberta-large", "roberta-large_raw_s42")]
E = {k: L(v) for k, v in SHIP}
C = {k: L(v) for k, v in CAND if (R / f"probs_{v}.npy").exists()}
blend = lambda ms: f1(.9 * np.mean(ms, axis=0) + .1 * T)
names = list(E)
full = blend([E[k] for k in names])

print("=== 가설 1: 멤버가 많아서 떨어지는가 ===")
print(f"  11개: {full:.5f}")
dup = blend([E[k] for k in names] + [E["mean2e5 s42"]])      # 같은 품질 멤버 하나 더
print(f"  12개 (같은 품질 멤버 중복 추가): {dup:.5f}   Δ {dup-full:+.5f}")
dup2 = blend([E[k] for k in names] + [E["mean2e5 s42"], E["attn3e5"]])
print(f"  13개 (둘 더): {dup2:.5f}   Δ {dup2-full:+.5f}")
print("  → 개수를 늘려도 떨어지지 않는다. 떨어뜨린 건 '낮은 품질 멤버'였다.\n")

print("=== 가설 2: 추가 대신 교체 ===")
loo = {k: full - blend([E[j] for j in names if j != k]) for k in names}
weak = sorted(loo, key=loo.get)[:3]
print(f"  기여 최하 3개: " + ", ".join(f"{k}({loo[k]:+.5f})" for k in weak))
print(f"\n{'후보':<16}{'추가(12개)':>12}{'최하1개 교체':>14}{'최하2개 교체':>14}")
for ck, cp in C.items():
    add = blend([E[k] for k in names] + [cp])
    r1 = blend([E[k] for k in names if k != weak[0]] + [cp])
    r2 = blend([E[k] for k in names if k not in weak[:2]] + [cp, cp])
    print(f"{ck:<16}{add-full:>+12.5f}{r1-full:>+14.5f}{r2-full:>+14.5f}")

print("\n=== 가설 2 정직한 검정: 풀에서 최적 11개를 고르면? ===")
pool = [E[k] for k in names] + list(C.values())
rng = np.random.default_rng(99)
gains = []
for rep in range(40):
    o = rng.permutation(len(Y))
    for a, b in ((o[:len(Y)//2], o[len(Y)//2:]), (o[len(Y)//2:], o[:len(Y)//2])):
        def s(ms, idx):
            P = .9 * np.mean(ms, axis=0) + .1 * T
            return macro_f1(Y[idx], (P[idx] >= .5).astype(np.int8))
        # a 에서 탐욕적으로 11개 고르기
        cur, used = [], set()
        for _ in range(11):
            best = max((s(cur + [pool[i]], a), i) for i in range(len(pool)) if i not in used)
            used.add(best[1]); cur.append(pool[best[1]])
        fixed = s([E[k] for k in names], b)
        gains.append(s(cur, b) - fixed)
gains = np.array(gains)
print(f"  반분 검증 {len(gains)}회: Δ {gains.mean():+.5f} ± "
      f"{gains.std(ddof=1)/np.sqrt(len(gains)):.5f} · 이긴 비율 {(gains>0).mean():.0%}")
print(f"  판정: {'○ 교체가 낫다' if gains.mean() > 0.001 else '✗ 고정 11개가 낫거나 같다'}")
