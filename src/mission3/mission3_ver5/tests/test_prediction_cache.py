"""Cache reuse must preserve scores and reject stale/corrupt raw predictions."""
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from mission3_ver5.prediction_cache import PredictionCache


@pytest.mark.parametrize("change", ["none", "location", "text", "order", "weights", "tokenizer",
                                   "code", "batch", "execution", "label_order", "corruption", "checksum"])
def test_cache_identity_and_recovery(tmp_path, change):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "vocab.txt").write_text("vocabulary")
    weights, source = tmp_path / "model.ckpt", tmp_path / "inference.py"
    weights.write_bytes(b"first model")
    source.write_text("original inference")
    paths, texts = [Path("a.json"), Path("b.json")], ["call one", "call two"]
    config = {"device": "cpu", "max_length": 512, "stride": 64,
              "window_batch_size": 2, "inference_batch_size": 8}
    execution, labels = {"version": "one"}, ["symptom_a", "symptom_b"]
    directory = tmp_path / "cache"
    values = np.array([[0.1, 0.4], [0.9, 0.3]], dtype=np.float32)
    compute = Mock(return_value=values)

    def predict():
        cache = PredictionCache(directory, paths, texts, tokenizer, labels, execution)
        return cache.get("neural", weights, config, [source], compute)

    np.testing.assert_array_equal(predict(), values)
    assert compute.call_count == 1
    if change == "location":
        paths = [tmp_path / "moved" / p for p in paths]
        copied = tmp_path / "copied.ckpt"
        copied.write_bytes(weights.read_bytes())
        weights = copied
    elif change == "text":
        texts[0] += " changed"
    elif change == "order":
        paths, texts = paths[::-1], texts[::-1]
    elif change == "weights":
        weights.write_bytes(b"other model")
    elif change == "tokenizer":
        (tokenizer / "vocab.txt").write_text("other vocabulary")
    elif change == "code":
        source.write_text("new inference")
    elif change == "batch":
        config["window_batch_size"] = 1
    elif change == "execution":
        execution["version"] = "two"
    elif change == "label_order":
        labels.reverse()
    elif change == "corruption":
        next(directory.glob("*.npz")).write_bytes(b"interrupted download")
    elif change == "checksum":
        file = next(directory.glob("*.npz"))
        with np.load(file, allow_pickle=False) as stored:
            data = {key: stored[key].copy() for key in stored.files}
        data["probabilities"][0, 0] = 0.8
        np.savez_compressed(file, **data)
    np.testing.assert_array_equal(predict(), values)
    assert compute.call_count == (1 if change in ("none", "location") else 2)
    assert not list(directory.glob("*.tmp"))


def test_invalid_probabilities_are_not_saved(tmp_path):
    weights = tmp_path / "model"
    weights.write_bytes(b"weights")
    cache = PredictionCache(tmp_path / "cache", [Path("a.json")], ["text"], tmp_path / "tokenizer",
                            ["symptom"], {})
    config = {"device": "cpu", "max_length": 512, "stride": 64,
              "window_batch_size": 2, "inference_batch_size": 8}
    with pytest.raises(ValueError, match="Invalid raw"):
        cache.get("neural", weights, config, [], lambda: np.array([[float("nan")]]))
    assert not list((tmp_path / "cache").glob("*.npz"))
