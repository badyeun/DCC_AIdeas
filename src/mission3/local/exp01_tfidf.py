"""Experiment 1 -- a local TF-IDF baseline, then normalisation and post-processing A/B.

Everything is measured on a Training-internal grouped holdout. Official
Validation is not read. Thresholds are 0.5 for every symptom.

Run:  python src/mission3/local/exp01_tfidf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, TRAIN_DIR, load, macro_f1, split          # noqa: E402
from normalize import VARIANTS, normalize                                    # noqa: E402

LOG = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local" / "exp01_tfidf.md"


def out(line=""):
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fit_predict(train_texts, train_Y, test_texts, C=4.0, char_max=120_000, word_max=40_000):
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                               max_features=char_max, sublinear_tf=True, dtype=np.float32)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                               max_features=word_max, sublinear_tf=True, dtype=np.float32)
    Xtr = hstack([char_vec.fit_transform(train_texts), word_vec.fit_transform(train_texts)]).tocsr()
    Xte = hstack([char_vec.transform(test_texts), word_vec.transform(test_texts)]).tocsr()
    P = np.zeros((Xte.shape[0], len(TARGETS)), dtype=np.float64)
    for i in range(len(TARGETS)):
        clf = LogisticRegression(C=C, max_iter=1500, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, train_Y[:, i])
        P[:, i] = clf.predict_proba(Xte)[:, 1]
    return P


def score(P, Y, force_one=False):
    pred = (P >= 0.5).astype(np.int8)
    empties = int((~pred.any(axis=1)).sum())
    if force_one:
        blank = ~pred.any(axis=1)
        if blank.any():
            pred[blank, P[blank].argmax(axis=1)] = 1
    m, per = macro_f1(Y, pred, per_symptom=True)
    return m, per, empties


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    out("# 실험 1 — 로컬 TF-IDF 기준선 + 정규화/후처리 A/B\n")
    out(f"실행 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    names, texts, Y = load(TRAIN_DIR, "train")
    tr, te = split(texts, Y)
    out(f"Training 29,200개를 그룹 분할 → 학습 {len(tr):,} / 자체검증 {len(te):,}")
    out("공식 Validation은 읽지 않음. 임계값은 전 증상 0.5.\n")

    out("## 정규화 변형 비교\n")
    out("| 전처리 | Macro F1 | 최소1개 강제 | 빈 예측 |")
    out("|---|---:|---:|---:|")
    results = {}
    for tag, opts in VARIANTS.items():
        t0 = time.time()
        norm = [normalize(t, **opts) for t in texts]
        P = fit_predict([norm[i] for i in tr], Y[tr], [norm[i] for i in te])
        base, per, empties = score(P, Y[te])
        forced, _, _ = score(P, Y[te], force_one=True)
        results[tag] = (base, forced, per, P)
        out(f"| {tag} | {base:.5f} | {forced:.5f} | {empties} |")
        print(f"   ({time.time()-t0:.0f}초)", flush=True)

    best = max(results, key=lambda k: results[k][1])
    out(f"\n최고: **{best}** — 최소1개 강제 시 {results[best][1]:.5f}")
    raw_b, raw_f = results["raw"][0], results["raw"][1]
    out(f"원문 대비: 정규화 {results[best][0]-raw_b:+.5f}, 후처리까지 {results[best][1]-raw_f:+.5f}")

    out("\n## 증상별 (원문 vs 최고 전처리)\n")
    out("| 증상 | 원문 | " + best + " | 차이 |")
    out("|---|---:|---:|---:|")
    for i, s in enumerate(TARGETS):
        a, b = results["raw"][2][i], results[best][2][i]
        out(f"| {s} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")

    np.save(LOG.parent / "exp01_best_probs.npy", results[best][3])
    np.save(LOG.parent / "exp01_holdout_Y.npy", Y[te])
    out(f"\n확률 저장: results/mission3/local/exp01_best_probs.npy")
