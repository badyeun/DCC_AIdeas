"""Behavior checks for text-only NB feature reweighting and prediction."""
import copy

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from mission3_ver9.sparse import fit_model, predict_model


@pytest.fixture
def fitted():
    texts = ["고열 체온 뜨겁다", "고열 체온 뜨겁다", "고열 체온 높다", "고열 체온 높다",
             "복통 배 아프다", "복통 배 아프다", "복통 배 아프다", "복통 배 아프다"]
    targets = np.zeros((len(texts), 9), dtype=np.int8)
    targets[:4, 0] = 1
    targets[4:, 3] = 1
    targets[:, 8] = 1
    return fit_model(texts, targets), texts, targets


def test_probabilities_learn_evidence_and_keep_all_nine_labels(fitted):
    model, texts, _ = fitted
    probabilities = predict_model(model, texts)
    assert probabilities.shape == (8, 9)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    assert probabilities[:4, 0].mean() > probabilities[4:, 0].mean()
    assert probabilities[4:, 3].mean() > probabilities[:4, 3].mean()
    np.testing.assert_array_equal(probabilities[:, 1], np.zeros(8))
    np.testing.assert_array_equal(probabilities[:, 8], np.ones(8))


def test_unseen_inference_never_fits_or_changes_vocabulary(fitted, monkeypatch):
    model, _, _ = fitted
    before = [(dict(vectorizer.vocabulary_), vectorizer.idf_.copy()) for vectorizer in model["vectorizers"]]

    def forbidden(*args, **kwargs):
        raise AssertionError("inference attempted to fit")

    monkeypatch.setattr(TfidfVectorizer, "fit", forbidden)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbidden)
    probabilities = predict_model(model, ["UNSEEN_NEVER_LEARN_THIS_TOKEN", ""])
    assert probabilities.shape == (2, 9)
    for vectorizer, (vocabulary, idf) in zip(model["vectorizers"], before):
        assert vectorizer.vocabulary_ == vocabulary
        np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert predict_model(model, []).shape == (0, 9)


def test_joblib_roundtrip_preserves_predictions(fitted, tmp_path):
    model, texts, _ = fitted
    checkpoint = tmp_path / "sparse.joblib"
    joblib.dump(model, checkpoint)
    restored = joblib.load(checkpoint)
    np.testing.assert_array_equal(predict_model(model, texts), predict_model(restored, texts))


def test_no_vocabulary_preserves_samples_and_returns_fit_prior():
    targets = np.zeros((2, 9), dtype=np.int8)
    targets[1, 0] = 1
    targets[:, 8] = 1
    model = fit_model(["", "   "], targets)
    probabilities = predict_model(model, ["new text", ""])
    assert model["n_features"] == 0
    np.testing.assert_array_equal(probabilities[:, 0], [0.5, 0.5])
    np.testing.assert_array_equal(probabilities[:, 1], [0.0, 0.0])
    np.testing.assert_array_equal(probabilities[:, 8], [1.0, 1.0])
    single = fit_model(["x"], targets[:1])
    np.testing.assert_array_equal(predict_model(single, ["x"]), targets[:1])


@pytest.mark.parametrize("bad_targets", [np.zeros((2, 8)), np.full((2, 9), np.nan),
                                         np.full((2, 9), 2), np.full((2, 9), "1")])
def test_target_schema_rejected(bad_targets):
    with pytest.raises(ValueError):
        fit_model(["one", "two"], bad_targets)


def test_annotations_invalid_config_and_corrupt_artifact_rejected(fitted):
    model, _, _ = fitted
    with pytest.raises(TypeError, match="never annotations"):
        predict_model(model, [{"text": "고열", "symptom": ["고열"]}])
    with pytest.raises(TypeError):
        fit_model("bare string", np.zeros((1, 9)))
    for config in ({"max_iter": 501}, {"C": float("nan")}, {"alpha": 0},
                   {"min_df": 0}, {"unknown": 1}):
        with pytest.raises(ValueError):
            fit_model(["valid", "text"], np.zeros((2, 9)), config)
    corrupt = copy.deepcopy(model)
    corrupt["estimators"][0]["ratio"][0] = np.nan
    with pytest.raises(ValueError, match="ratio"):
        predict_model(corrupt, ["고열"])
    corrupt = copy.deepcopy(model)
    corrupt["estimators"][1]["constant"] = 1.1
    with pytest.raises(ValueError, match="Constant probability"):
        predict_model(corrupt, ["고열"])
