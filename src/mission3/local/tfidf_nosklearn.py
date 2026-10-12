"""sklearn 없이 tfidf.pkl 의 예측을 재현한다 — 로컬 측정 전용.

왜 필요한가: 이 PC 의 Smart App Control 이 sklearn 의 서명 없는 .pyd 를
정책으로 차단한다(CodeIntegrity 3077). 끄는 것은 되돌릴 수 없는 변경이므로,
측정만 sklearn 없이 할 수 있게 한다. **제출물은 그대로 sklearn 을 쓴다** --
채점 환경은 이 정책과 무관하고, transformers 자체도 sklearn 을 import 한다.

재현 대상(저장된 설정에서 읽음): TfidfVectorizer char(2,5) / word(1,2),
sublinear_tf, smooth_idf, norm l2, lowercase, 그리고 LogisticRegression
(liblinear, 이진) 의 predict_proba.

정확성은 추측하지 않는다 -- 홀드아웃에서 sklearn 으로 만든
tfidf_holdout_probs.npy 를 재현하는지로 검증한다(verify() 참조).
"""
from __future__ import annotations

import io
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from normalize import VARIANTS, normalize                                  # noqa: E402

WHITE = re.compile(r"\s\s+")                 # sklearn CountVectorizer._white_spaces
TOKEN = re.compile(r"(?u)\b\w\w+\b")         # 기본 token_pattern


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {})


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("sklearn", "scipy")):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def load(path):
    """sklearn 을 import 하지 않고 번들을 읽는다."""
    return _U(io.BytesIO(Path(path).read_bytes())).load()


def _char_ngrams(doc, lo, hi):
    doc = WHITE.sub(" ", doc)
    n_len = len(doc)
    out = list(doc) if lo == 1 else []
    start = 2 if lo == 1 else lo
    for n in range(start, min(hi + 1, n_len + 1)):
        for i in range(n_len - n + 1):
            out.append(doc[i:i + n])
    return out


def _word_ngrams(doc, lo, hi):
    toks = TOKEN.findall(doc)
    if hi == 1:
        return toks
    out = list(toks) if lo == 1 else []
    start = 2 if lo == 1 else lo
    n_orig = len(toks)
    for n in range(start, min(hi + 1, n_orig + 1)):
        for i in range(n_orig - n + 1):
            out.append(" ".join(toks[i:i + n]))
    return out


def _block(texts, vec):
    """한 TfidfVectorizer 의 변환 결과를 (n, d) 밀집 행렬로."""
    vocab = vec.vocabulary_
    idf = np.asarray(vec._tfidf.idf_, dtype=np.float64)
    lo, hi = vec.ngram_range
    gram = _char_ngrams if vec.analyzer == "char" else _word_ngrams
    X = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for r, t in enumerate(texts):
        doc = t.lower() if vec.lowercase else t
        row = X[r]
        for g in gram(doc, lo, hi):
            j = vocab.get(g)
            if j is not None:
                row[j] += 1.0
    if vec._tfidf.sublinear_tf:
        nz = X > 0
        X[nz] = np.log(X[nz]) + 1.0
    X *= idf
    if vec._tfidf.norm == "l2":
        nrm = np.sqrt((X * X).sum(1, keepdims=True))
        np.divide(X, nrm, out=X, where=nrm > 0)
    return X


def predict(bundle, texts):
    """tfidf_model.predict 와 같은 (n, 9) 확률."""
    texts = [normalize(t, **VARIANTS[bundle["variant"]]) for t in texts]
    X = np.hstack([_block(texts, bundle["char"]), _block(texts, bundle["word"])])
    P = np.zeros((len(texts), len(bundle["models"])), dtype=np.float64)
    for i, clf in enumerate(bundle["models"]):
        z = X @ np.asarray(clf.coef_, dtype=np.float64).ravel() \
            + float(np.asarray(clf.intercept_).ravel()[0])
        # liblinear 이진 분류: classes_ = [0, 1] 이면 sigmoid(z) 가 양성 확률
        assert list(np.asarray(clf.classes_)) == [0, 1], clf.classes_
        P[:, i] = 1.0 / (1.0 + np.exp(-z))
    return P


def verify():
    """sklearn 이 만든 홀드아웃 확률을 재현하는지 확인한다."""
    from common_local import TRAIN_DIR, load as load_data, split
    R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
    ref = np.load(R / "tfidf_holdout_probs.npy")
    _, texts, Y = load_data(TRAIN_DIR, "train")
    _, te = split(texts, Y, seed=2026)
    # 확률 파일은 **홀드아웃 순서**(0..5839)로 저장된다. holdout_Y.npy 가
    # Y[te] 인 것과 같은 정렬이므로, 전역 인덱스로 참조하면 엉뚱한 행을 본다.
    k = 300
    got = predict(load(R / "submission_final16" / "tfidf.pkl"),
                  [texts[i] for i in te[:k]])
    err = np.abs(got - ref[:k])
    print(f"검증: 홀드아웃 {k}통화 × 9증상")
    print(f"  최대 오차 {err.max():.3e} · 평균 오차 {err.mean():.3e}")
    flips = int(((got >= .5) != (ref[:k] >= .5)).sum())
    print(f"  0.5 판정 불일치 {flips} / {got.size:,}")
    ok = err.max() < 1e-6 and flips == 0
    print("  →", "재현 성공" if ok else "재현 실패 — 측정에 쓸 수 없습니다")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
