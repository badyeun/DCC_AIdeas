"""The TF-IDF side of the submission ensemble.

Blending a plain linear bag-of-ngrams into the transformer ensemble at 10%
lifts holdout AP by 0.0026 and Macro F1 by 0.0011. Split-half validation (the
weight chosen on one half, scored on the other, 80 times) picked 0.10 in 62 of
80 runs and won 98% of them, so the gain survives the search that found it.

Why it helps: the encoder blurs rare explicit wording, while a linear model
keys on it directly. The biggest AP gains are 호흡곤란 (+0.0087), 두통
(+0.0049) and 어지러움 (+0.0035).

Trains in seconds on CPU. Decisions stay at 0.5 for all nine symptoms; this
module only supplies probabilities to average.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS                                            # noqa: E402
from normalize import VARIANTS, normalize                                   # noqa: E402

VARIANT = "synonym+filler"      # the variant that won the TF-IDF A/B
BLEND_WEIGHT = 0.10


def prepare(texts):
    return [normalize(t, **VARIANTS[VARIANT]) for t in texts]


def fit(texts, Y, C=4.0):
    """Fit on raw transcripts; normalisation is applied inside."""
    texts = prepare(texts)
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                               max_features=120_000, sublinear_tf=True, dtype=np.float32)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                               max_features=40_000, sublinear_tf=True, dtype=np.float32)
    X = hstack([char_vec.fit_transform(texts), word_vec.fit_transform(texts)]).tocsr()
    models = []
    for i in range(len(TARGETS)):
        clf = LogisticRegression(C=C, max_iter=1500, class_weight="balanced", solver="liblinear")
        clf.fit(X, Y[:, i])
        models.append(clf)
    return {"char": char_vec, "word": word_vec, "models": models,
            "targets": list(TARGETS), "variant": VARIANT}


def predict(bundle, texts):
    texts = prepare(texts)
    X = hstack([bundle["char"].transform(texts), bundle["word"].transform(texts)]).tocsr()
    P = np.zeros((X.shape[0], len(bundle["models"])), dtype=np.float64)
    for i, clf in enumerate(bundle["models"]):
        P[:, i] = clf.predict_proba(X)[:, 1]
    return P


def save(bundle, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from common_local import TRAIN_DIR, load as load_calls
    out = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local" \
        / "submission_models" / "tfidf.pkl"
    names, texts, Y = load_calls(TRAIN_DIR, "train")
    print(f"Training {len(texts):,}개 전부로 학습 중...", flush=True)
    bundle = fit(texts, Y)
    save(bundle, out)
    print(f"저장: {out}  ({out.stat().st_size/1e6:.1f} MB)")
