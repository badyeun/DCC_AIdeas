"""Text-only NB log-count-ratio + logistic regression for nine symptoms.

This is NB-SVM-inspired feature reweighting with logistic regression, not an
SVM. Vocabulary, IDF, count ratios and classifiers are fitted only from the
explicit training texts and targets. Prediction never updates those objects.
"""
from collections.abc import Mapping
import numbers
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


N_LABELS = 9
SCHEMA_VERSION = 1
DEFAULT_CONFIG = {
    "char_max_features": 80000,
    "word_max_features": 20000,
    "min_df": 2,
    "C": 1.0,
    "max_iter": 500,
    "alpha": 1.0,
    "class_weight": None,
    "seed": 42,
}


def _texts(values):
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("texts must be an iterable of text strings, never annotations")
    try:
        values = list(values)
    except TypeError as error:
        raise TypeError("texts must be an iterable of text strings") from error
    if any(not isinstance(value, str) for value in values):
        raise TypeError("texts must contain text strings only, never annotations")
    # The same label-independent normalization applies at fit and inference.
    return [" ".join(unicodedata.normalize("NFC", value).split()) for value in values]


def _targets(values, n_samples):
    values = np.asarray(values)
    if values.shape != (n_samples, N_LABELS):
        raise ValueError(f"targets must have shape ({n_samples}, {N_LABELS})")
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
        raise ValueError("targets must be numeric binary indicators")
    if np.iscomplexobj(values) or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("targets must contain finite binary indicators 0 or 1")
    return values.astype(np.int8, copy=False)


def _config(config):
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown sparse config keys: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **supplied}
    for key in ("char_max_features", "word_max_features", "min_df", "max_iter"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(value)
    if result["max_iter"] > 500:
        raise ValueError("max_iter must not exceed 500")
    for key in ("C", "alpha"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if result["class_weight"] not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")
    if isinstance(result["seed"], bool) or not isinstance(result["seed"], numbers.Integral):
        raise ValueError("seed must be an integer")
    result["seed"] = int(result["seed"])
    if not 0 <= result["seed"] <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return result


def _fit_features(texts, config):
    matrices, vectorizers = [], []
    for analyzer, ngram_range, maximum in (
        ("char", (2, 5), config["char_max_features"]),
        ("word", (1, 2), config["word_max_features"]),
    ):
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range, min_df=config["min_df"],
            max_features=maximum, lowercase=False, sublinear_tf=True,
            dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as error:
            # Tiny/all-blank inputs can have no qualifying n-grams in one or
            # both channels. Keep the sample and use the remaining features.
            expected = ("empty vocabulary", "After pruning, no terms remain", "max_df corresponds to < documents than min_df")
            if not any(message in str(error) for message in expected):
                raise
            vectorizer = None
        else:
            matrices.append(matrix)
        vectorizers.append(vectorizer)
    matrix = (sparse.hstack(matrices, format="csr", dtype=np.float32) if matrices
              else sparse.csr_matrix((len(texts), 0), dtype=np.float32))
    return vectorizers, matrix


def fit_model(texts, targets, config=None):
    """Fit a joblib-serializable artifact using only the supplied fit split."""
    texts = _texts(texts)
    if not texts:
        raise ValueError("At least one training text is required")
    targets = _targets(targets, len(texts))
    config = _config(config)
    vectorizers, features = _fit_features(texts, config)
    estimators = []
    for index in range(N_LABELS):
        labels = targets[:, index]
        positives = int(labels.sum())
        if positives in (0, len(texts)) or features.shape[1] == 0:
            estimators.append({"constant": float(positives / len(texts))})
            continue
        positive_mass = np.asarray(features[labels == 1].sum(axis=0), dtype=np.float64).ravel() + config["alpha"]
        negative_mass = np.asarray(features[labels == 0].sum(axis=0), dtype=np.float64).ravel() + config["alpha"]
        # Sum-normalized smoothed feature masses, as in NB log-count ratios.
        ratio = (np.log(positive_mass / positive_mass.sum())
                 - np.log(negative_mass / negative_mass.sum())).astype(np.float32)
        if not np.isfinite(ratio).all():
            raise ValueError("Non-finite NB feature ratios")
        classifier = LogisticRegression(
            C=config["C"], max_iter=config["max_iter"], class_weight=config["class_weight"],
            solver="liblinear", n_jobs=1, random_state=config["seed"],
        )
        classifier.fit(features.multiply(ratio).tocsr(), labels)
        estimators.append({"ratio": ratio, "classifier": classifier})
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": "nb_log_count_ratio_logistic_regression",
        "n_labels": N_LABELS,
        "n_features": int(features.shape[1]),
        "fit_samples": len(texts),
        "config": config,
        "vectorizers": vectorizers,
        "estimators": estimators,
    }


def _validate_model(model):
    if not isinstance(model, Mapping) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported sparse model schema")
    if model.get("model_type") != "nb_log_count_ratio_logistic_regression" or model.get("n_labels") != N_LABELS:
        raise ValueError("Sparse model must be a nine-label NB logistic regression artifact")
    count = model.get("n_features")
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
        raise ValueError("Invalid sparse feature count")
    vectorizers, estimators = model.get("vectorizers"), model.get("estimators")
    if not isinstance(vectorizers, (tuple, list)) or len(vectorizers) != 2:
        raise ValueError("Sparse model requires two vectorizer slots")
    vocabulary_size = 0
    for vectorizer in vectorizers:
        if vectorizer is None:
            continue
        if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
            raise ValueError("Invalid or unfitted sparse vectorizer")
        if not np.isfinite(vectorizer.idf_).all():
            raise ValueError("Non-finite vectorizer IDF")
        vocabulary_size += len(vectorizer.vocabulary_)
    if vocabulary_size != count:
        raise ValueError("Vectorizer vocabulary and sparse feature count differ")
    if not isinstance(estimators, (tuple, list)) or len(estimators) != N_LABELS:
        raise ValueError("Sparse model requires nine label estimators")
    for estimator in estimators:
        if not isinstance(estimator, Mapping):
            raise ValueError("Invalid sparse label estimator")
        if "constant" in estimator:
            value = estimator["constant"]
            if not isinstance(value, numbers.Real) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Constant probability must be finite and within [0, 1]")
            continue
        ratio = np.asarray(estimator.get("ratio"))
        if ratio.shape != (count,) or not np.issubdtype(ratio.dtype, np.floating) or not np.isfinite(ratio).all():
            raise ValueError("Invalid NB feature ratio shape or values")
        classifier = estimator.get("classifier")
        if (not isinstance(classifier, LogisticRegression) or not hasattr(classifier, "classes_")
                or not np.array_equal(classifier.classes_, [0, 1])
                or getattr(classifier, "n_features_in_", None) != count
                or not np.isfinite(classifier.coef_).all()
                or not np.isfinite(classifier.intercept_).all()):
            raise ValueError("Invalid fitted binary logistic regression estimator")


def predict_model(model, texts):
    """Return finite float32 probabilities of shape [n, 9] without fitting."""
    _validate_model(model)
    texts = _texts(texts)
    if not texts:
        return np.empty((0, N_LABELS), dtype=np.float32)
    matrices = [vectorizer.transform(texts) for vectorizer in model["vectorizers"] if vectorizer is not None]
    features = (sparse.hstack(matrices, format="csr", dtype=np.float32) if matrices
                else sparse.csr_matrix((len(texts), 0), dtype=np.float32))
    if features.shape[1] != model["n_features"] or not np.isfinite(features.data).all():
        raise ValueError("Invalid inference feature shape or values")
    output = np.empty((len(texts), N_LABELS), dtype=np.float32)
    for index, estimator in enumerate(model["estimators"]):
        if "constant" in estimator:
            output[:, index] = estimator["constant"]
        else:
            output[:, index] = estimator["classifier"].predict_proba(
                features.multiply(estimator["ratio"]).tocsr())[:, 1]
    if not np.isfinite(output).all() or (output < 0).any() or (output > 1).any():
        raise ValueError("Invalid sparse output probabilities")
    return output
