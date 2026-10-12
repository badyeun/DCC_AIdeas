"""약한 멤버를 덜어내고 large 를 넣으면 더 오르는가.

'자유도를 늘리면 진다'가 이 과제의 반복된 교훈이다(탐욕 선택 -0.0021,
멤버 가중치 -0.0013, 증상별 임계값 -0.0004). 덜어낼 멤버를 홀드아웃에서
고르는 것도 같은 종류의 자유도다. 그래서 두 가지를 분리해서 본다.

  (1) 16개에서 기여 낮은 멤버를 덜어내면 오르는가  (홀드아웃 = 낙관적)
  (2) 그 '덜어내기' 전략이 못 본 데이터에서도 이기는가  (A반 결정 / B반 측정)

(1)만 보면 항상 올라 보인다. 채택 여부는 (2)로만 정한다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy"); T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")

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
LARGE = [("LG mean s1", "roberta-large_raw_s1_LGC"),
         ("LG mean s7", "roberta-large_raw_s7_LGC"),
         ("LG attn s42", "roberta-large_attn_raw_s42_LGC"),
         ("LG tapt", "klue-roberta-large-tapt_raw_s42_CAND"),
         ("LG norm", "roberta-large_synonym+filler_s42")]
P = {k: L(v) for k, v in SHIP + LARGE}
BASE_K = [k for k, _ in SHIP]
LG_K = [k for k, _ in LARGE]
A16 = BASE_K + LG_K

f1 = lambda p, i: macro_f1(Y[i], p[i] >= .5)
sc = lambda keys, i: f1(.9 * np.mean([P[k] for k in keys], axis=0) + .1 * T, i)
ALL = np.arange(len(Y))
# large 는 base 의 3.5배 비용
cost = lambda keys: sum(3.5 if k.startswith("LG") else 1.0 for k in keys)

s11, s16 = sc(BASE_K, ALL), sc(A16, ALL)
print(f"현재 11개:  {s11:.5f}   (비용 {cost(BASE_K):.0f} 단위)")
print(f"16개:       {s16:.5f}   (비용 {cost(A16):.0f} 단위)\n")

print("=== 16개 안에서 각 멤버를 빼면 (leave-one-out) ===")
loo = {k: s16 - sc([j for j in A16 if j != k], ALL) for k in A16}
print(f"{'멤버':<14}{'기여':>10}   {'멤버':<14}{'기여':>10}")
order = sorted(A16, key=loo.get)
for a, b in zip(order[::2], order[1::2] + [None]):
    r = f"{a:<14}{loo[a]:>+10.5f}"
    if b: r += f"   {b:<14}{loo[b]:>+10.5f}"
    print(r)
print("  (음수 = 빼면 오름)")

rng = np.random.default_rng(31); n = len(Y)
HALVES = []
for _ in range(240):
    p = rng.permutation(n); HALVES.append((p[:n//2], p[n//2:]))

def honest(build, label):
    """build(A반 점수함수) -> 멤버목록.  A반에서 짜고 B반에서 잰다."""
    d, sizes = [], []
    for A, B in HALVES:
        keys = build(A)
        d.append(sc(keys, B) - sc(BASE_K, B))
        sizes.append(len(keys))
    d = np.array(d)
    print(f"{label:<34}{d.mean():>+10.5f}{d.std(ddof=1)/np.sqrt(len(d)):>9.5f}"
          f"{(d>0).mean():>7.0%}{np.mean(sizes):>8.1f}")
    return d.mean()

print("\n=== 정직한 비교: 전략을 A반에서 실행하고 B반에서 측정 (현재 11개 기준) ===")
print(f"{'전략':<34}{'B반 Δ':>10}{'±SE':>9}{'승률':>7}{'멤버수':>8}")
honest(lambda A: A16, "16개 고정 (덜어내기 없음)")
for k in (1, 2, 3, 4, 5):
    def build(A, k=k):
        s = sc(A16, A)
        c = {m: s - sc([j for j in A16 if j != m], A) for m in A16}
        drop = sorted(c, key=c.get)[:k]
        return [m for m in A16 if m not in drop]
    honest(build, f"16개에서 기여 최하 {k}개 제거")
honest(lambda A: LG_K + BASE_K[:6], "large 5 + base 상위 6 (고정)")
honest(lambda A: LG_K, "large 5 만")
honest(lambda A: LG_K + BASE_K[:1], "large 5 + base 1")

print("\n=== 고정 구성들을 홀드아웃에서 직접 (선택 없음, 참고용) ===")
FIX = {"현재 11개": BASE_K, "16개": A16,
       "large5 + base6": LG_K + BASE_K[:6],
       "large5 + base4": LG_K + BASE_K[:4],
       "large5 만": LG_K, "large3 + base11": BASE_K + LG_K[:3]}
print(f"{'구성':<22}{'멤버':>5}{'홀드아웃':>11}{'Δ(11개)':>10}{'비용':>7}{'배수':>7}")
for nm, keys in FIX.items():
    s = sc(keys, ALL)
    print(f"{nm:<22}{len(keys):>5}{s:>11.5f}{s-s11:>+10.5f}"
          f"{cost(keys):>7.0f}{cost(keys)/cost(BASE_K):>7.1f}x")
