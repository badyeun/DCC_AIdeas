"""1파도·2파도·large 확인 큐의 판정.

단독 점수는 앙상블 기여를 예측하지 못한다(상관 +0.22). 그래서 모든 후보를
**출하 11개 앙상블에 더했을 때의 Δ**로만 판정한다. 그리고 홀드아웃 한 번의
Δ는 평가 잡음(0.0015)에 묻히므로, 반분 검증으로 부호의 안정성까지 본다.

사전 등록 기준: 2-seed 평균 기여 > +0.001 이면 채택.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
f1 = lambda P, idx=None: macro_f1(Y[idx] if idx is not None else Y,
                                  (P[idx] if idx is not None else P) >= .5)

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
E = [L(v) for _, v in SHIP]

# 이번 큐에서 새로 나온 후보 전부
NEW = [
    ("1파도", "large-tapt",      ["klue-roberta-large-tapt_raw_s42_CAND"]),
    ("1파도", "km-bert",         ["km-bert_raw_s42_CAND"]),
    ("1파도", "bert-kor-base",   ["bert-kor-base_raw_s42_CAND"]),
    ("1파도", "KoSimCSE",        ["KoSimCSE-roberta_raw_s42_CAND"]),
    ("1파도", "roberta-small",   ["roberta-small_raw_s42_CAND"]),
    ("1파도", "kobigbird",       ["kobigbird-bert-base_raw_s42_CAND"]),
    ("2파도", "meanmax head",    ["roberta-base_meanmax_raw_s42_W2"]),
    ("2파도", "pool layer 10",   ["roberta-base_L10_raw_s42_W2"]),
    ("2파도", "pool layer 8",    ["roberta-base_L8_raw_s42_W2"]),
    ("2파도", "freeze 4",        ["roberta-base_fz4_raw_s42_W2"]),
    ("2파도", "freeze 6",        ["roberta-base_fz6_raw_s42_W2"]),
    ("2파도", "attn + layer 10", ["roberta-base_attn_L10_raw_s42_W2"]),
    ("large", "large mean s1",   ["roberta-large_raw_s1_LGC"]),
    ("large", "large mean s7",   ["roberta-large_raw_s7_LGC"]),
    ("large", "large attn s42",  ["roberta-large_attn_raw_s42_LGC"]),
]

blend = lambda ms, idx=None: f1(.9 * np.mean(ms, axis=0) + .1 * T, idx)
full = blend(E)

rng = np.random.default_rng(0)
n = len(Y)
HALVES = []
for _ in range(120):
    p = rng.permutation(n)
    HALVES.append((p[:n // 2], p[n // 2:]))

def verdict(members):
    """홀드아웃 전체 Δ, 반분 Δ 평균±표준오차, 승률."""
    cand_full = blend(E + members)
    d = []
    for a, b in HALVES:
        for idx in (a, b):
            d.append(blend(E + members, idx) - blend(E, idx))
    d = np.array(d)
    return cand_full - full, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), (d > 0).mean()

print(f"출하 11개 기준선: {full:.5f}   (홀드아웃 5,840통화)\n")
print(f"{'파도':<7}{'후보':<17}{'단독':>9}{'12번째 Δ':>11}{'반분 Δ':>10}{'±SE':>9}{'승률':>7}")
print("-" * 70)
rows = []
for wave, name, files in NEW:
    m = [L(f) for f in files]
    solo = f1(np.mean(m, axis=0))
    dh, dm, se, win = verdict(m)
    rows.append((wave, name, solo, dh, dm, se, win, files))
    print(f"{wave:<7}{name:<17}{solo:>9.5f}{dh:>+11.5f}{dm:>+10.5f}{se:>9.5f}{win:>7.0%}")

print("\n=== large를 '계열'로 넣으면 (여러 멤버 동시) ===")
LG = {"s1": "roberta-large_raw_s1_LGC", "s7": "roberta-large_raw_s7_LGC",
      "attn s42": "roberta-large_attn_raw_s42_LGC"}
import itertools
print(f"{'구성':<28}{'Δ 홀드아웃':>12}{'반분 Δ':>10}{'±SE':>9}{'승률':>7}")
for k in (1, 2, 3):
    for combo in itertools.combinations(LG, k):
        m = [L(LG[c]) for c in combo]
        dh, dm, se, win = verdict(m)
        print(f"{'+'.join(combo):<28}{dh:>+12.5f}{dm:>+10.5f}{se:>9.5f}{win:>7.0%}")

print("\n=== 사전 등록 기준 (2-seed 평균 기여 > +0.001) ===")
two = [L(LG["s1"]), L(LG["s7"])]
solo_each = [f1(p) for p in two]
dh1, dm1, se1, w1 = verdict([two[0]])
dh2, dm2, se2, w2_ = verdict([two[1]])
print(f"  large mean s1  단독 {solo_each[0]:.5f}  기여 {dm1:+.5f}")
print(f"  large mean s7  단독 {solo_each[1]:.5f}  기여 {dm2:+.5f}")
print(f"  2-seed 평균 기여: {(dm1+dm2)/2:+.5f}  →  "
      f"{'채택' if (dm1+dm2)/2 > .001 else '기준 미달'}")
