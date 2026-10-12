"""TF-IDF 쪽은 아직 단일 모델입니다 — 거기도 앙상블할 수 있습니다.

신경망은 11개를 평균하는데 TF-IDF 는 하나뿐인데, 그 하나가 10% 비중으로
+0.0012 를 냅니다. 설정이 다른 TF-IDF 를 몇 개 섞으면 그 10% 구간이
더 안정될 수 있습니다. CPU 로 몇 분이고 신경망 재학습이 필요 없습니다.

판정은 제출 11개 앙상블에 섞었을 때의 F1@0.5 로 합니다.
"""
import sys, numpy as np
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load, split, macro_f1
from normalize import VARIANTS, normalize

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
f1 = lambda P: macro_f1(Y, (P >= .5).astype(np.int8))
SHIP = ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
        "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
        "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
        "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
        "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
        "roberta-base_cls_raw_s42_LS"]
NEURAL = np.mean([L(n) for n in SHIP], axis=0)

names, texts, Yall = load(TRAIN_DIR, "train")
tr, te = split(texts, Yall, seed=2026)

# 각 변형: (이름, 전처리, char ngram, word ngram, C)
SETTINGS = [
    ("기존 char2-5+word1-2 C4 norm", "synonym+filler", (2, 5), (1, 2), 4.0),
    ("원문 텍스트",                  "raw",            (2, 5), (1, 2), 4.0),
    ("char 만 (2-5)",               "synonym+filler", (2, 5), None,   4.0),
    ("word 만 (1-2)",               "synonym+filler", None,   (1, 2), 4.0),
    ("char 3-6 더 긴 n-gram",        "synonym+filler", (3, 6), (1, 2), 4.0),
    ("C=1 더 강한 정규화",            "synonym+filler", (2, 5), (1, 2), 1.0),
    ("C=16 약한 정규화",              "synonym+filler", (2, 5), (1, 2), 16.0),
]


def fit_predict(variant, char_ng, word_ng, C):
    docs = [normalize(t, **VARIANTS[variant]) for t in texts]
    parts_tr, parts_te = [], []
    for ng, analyzer, mx in ((char_ng, "char", 120_000), (word_ng, "word", 40_000)):
        if ng is None:
            continue
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ng, min_df=2,
                            max_features=mx, sublinear_tf=True, dtype=np.float32)
        parts_tr.append(v.fit_transform([docs[i] for i in tr]))
        parts_te.append(v.transform([docs[i] for i in te]))
    Xtr = hstack(parts_tr).tocsr() if len(parts_tr) > 1 else parts_tr[0]
    Xte = hstack(parts_te).tocsr() if len(parts_te) > 1 else parts_te[0]
    P = np.zeros((Xte.shape[0], 9))
    for i in range(9):
        clf = LogisticRegression(C=C, max_iter=1500, class_weight="balanced",
                                 solver="liblinear")
        clf.fit(Xtr, Yall[tr][:, i])
        P[:, i] = clf.predict_proba(Xte)[:, 1]
    return P


print(f"신경망 11개만: {f1(NEURAL):.5f}\n")
print(f"{'TF-IDF 변형':<30}{'단독':>9}{'혼합 0.10':>11}{'기존 대비':>11}")
probs = {}
base = None
for lab, variant, cn, wn, C in SETTINGS:
    P = fit_predict(variant, cn, wn, C)
    probs[lab] = P
    mixed = f1(.9 * NEURAL + .1 * P)
    if base is None:
        base = mixed
    print(f"{lab:<30}{f1(P):>9.5f}{mixed:>11.5f}{mixed-base:>+11.5f}")
    np.save(R / f"_tfv_{len(probs)}.npy", P)

print("\n여러 TF-IDF를 평균해 섞으면:")
keys = list(probs)
for sel, lab in (([0], "기존 1개"), ([0, 1], "기존+원문"), ([0, 2, 3], "기존+char만+word만"),
                 ([0, 1, 2, 3], "4개"), (list(range(len(keys))), f"{len(keys)}개 전부")):
    S = np.mean([probs[keys[i]] for i in sel], axis=0)
    best = max(((f1(.9 - 0 * w + 0 * w) if False else f1((1 - w) * NEURAL + w * S)), w)
               for w in (.05, .10, .15, .20))
    print(f"  {lab:<28} 혼합0.10 {f1(.9*NEURAL+.1*S):.5f} · 최적 w={best[1]:.2f} {best[0]:.5f}")
