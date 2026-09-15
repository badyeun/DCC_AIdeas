"""Exercise a real, portable training/inference bundle without network or a GPU."""

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mission3_ver4.data import TARGET_SYMPTOMS
from mission3_ver4 import pipeline
from mission3_ver4.pipeline import evaluate_csv, load_manifest, predict_pipeline, train_pipeline


def _write_annotation(path, text, symptoms):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "utterances": [{"text": text, "speaker": "caller", "startAt": 1}],
        "symptom": symptoms,
    }, ensure_ascii=False), encoding="utf-8")


def _read_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["label file name", "symptom"]
        rows = list(reader)
    values = {}
    for row in rows:
        labels = ast.literal_eval(row["symptom"])
        assert isinstance(labels, list)
        assert len(set(labels)) == len(labels)
        assert all(label in TARGET_SYMPTOMS for label in labels)
        assert row["label file name"] not in values
        values[row["label file name"]] = [int(label in labels) for label in TARGET_SYMPTOMS]
    return rows, values


def test_actual_pipeline_resume_and_portable_offline_submission(tmp_path):
    torch.set_num_threads(1)
    pretrained = tmp_path / "tiny_pretrained"
    pretrained.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "call", "context"]
    vocab += [f"cue{i}" for i in range(9)] + [f"id{i}" for i in range(144)]
    vocabulary_path = pretrained / "vocab.txt"
    vocabulary_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary_path), do_lower_case=False)
    tokenizer.save_pretrained(pretrained)
    torch.manual_seed(3)
    BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    )).save_pretrained(pretrained)
    train_dir = tmp_path / "Training" / "TL_서울_구급"
    truth = np.random.default_rng(24).integers(0, 2, size=(144, 9), dtype=np.int8)
    training_texts = {}
    for index in reversed(range(len(truth))):
        cues = " ".join(f"cue{label}" for label in np.flatnonzero(truth[index]))
        prefix = "context " * (25 if index % 11 == 0 else 1)
        text = f"call id{index} {prefix}{cues}"
        name = f"call_{index:03d}.json"
        training_texts[name] = text
        _write_annotation(train_dir / name, text,
                          [name for name, flag in zip(TARGET_SYMPTOMS, truth[index]) if flag])
    config = {
        "model_name": str(pretrained), "require_cuda": False, "device": "cpu",
        "n_folds": 2, "calibration_size": 0.2, "calibration_folds": 2,
        "max_length": 16, "stride": 4, "max_epochs": 1, "patience": 1,
        "batch_size": 8, "inference_batch_size": 8, "window_batch_size": 4,
        "gradient_accumulation_steps": 2, "dropout": 0,
        "gradient_checkpointing": True, "char_features": 100,
        "word_features": 100, "min_df": 1,
    }
    original_bundle = tmp_path / "original_bundle"
    manifest_path, report = train_pipeline(train_dir, original_bundle, config)
    manifest = load_manifest(manifest_path)
    assert report["training_samples"] == 144
    assert report["validation_used_for_training_or_calibration"] is False
    assert len(manifest["folds"]) == 2
    assert np.isfinite(report["calibration_crossfit_macro_f1"])
    with np.load(original_bundle / "calibration_predictions.npz") as calibration:
        assert calibration["neural"].shape == calibration["sparse"].shape == calibration["targets"].shape
        assert calibration["neural"].shape[1] == 9
        assert set(calibration["filenames"]).issubset(training_texts)
    splits = json.loads((original_bundle / "splits.json").read_text(encoding="utf-8"))
    calibration_names = set(splits["calibration"])
    for fold in splits["folds"]:
        assert set(fold["fit"]).isdisjoint(fold["stop"])
        assert (set(fold["fit"]) | set(fold["stop"])).isdisjoint(calibration_names)
        assert set(fold["fit"]) | set(fold["stop"]) | calibration_names == set(training_texts)
    artifact_names = [*manifest["folds"], "sparse.joblib"]
    hashes = {name: hashlib.sha256((original_bundle / name).read_bytes()).hexdigest()
              for name in artifact_names}
    with patch("mission3_ver4.pipeline.train_model", side_effect=AssertionError("Completed fold retrained")), \
         patch("mission3_ver4.pipeline.fit_sparse", side_effect=AssertionError("Completed sparse model retrained")):
        resumed_path, resumed_report = train_pipeline(train_dir, original_bundle, config)
    assert resumed_path == manifest_path
    assert resumed_report["fold_best_epochs"] == report["fold_best_epochs"]
    assert hashes == {name: hashlib.sha256((original_bundle / name).read_bytes()).hexdigest()
                      for name in artifact_names}
    with pytest.raises(ValueError, match="다른 데이터/설정/코드"):
        train_pipeline(train_dir, original_bundle, {**config, "dropout": 0.1})

    # Paths deliberately sort differently from basenames; metadata is poisoned
    # after obtaining a reference to prove it never influences inference.
    inference_dir = tmp_path / "Test" / "labels"
    input_paths = [inference_dir / "z" / "aaa.json", inference_dir / "bbb.json",
                   inference_dir / "a" / "zzz.json", inference_dir / "middle.json"]
    for index, path in enumerate(input_paths):
        _write_annotation(path, training_texts[f"call_{index:03d}.json"], [])
    paths, probabilities, predictions = predict_pipeline(
        manifest_path, inference_dir, tmp_path / "reference.csv", device="cpu", batch_size=2,
    )
    assert probabilities.shape == (4, 9)
    reference_by_name = {path.name: prediction.tolist() for path, prediction in zip(paths, predictions)}
    for path in input_paths:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        annotation["symptom"] = {"invalid": ["this must not be read"]}
        annotation["utterances"][0]["speaker"] = {"invalid": "ignored metadata"}
        annotation["unrelated"] = "cue8 cue8 cue8"
        path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    portable_bundle = tmp_path / "moved" / "portable_bundle"
    portable_bundle.parent.mkdir()
    shutil.move(str(original_bundle), portable_bundle)
    pretrained.rename(tmp_path / "hidden_pretrained")
    assert not pretrained.exists()
    assert not original_bundle.exists()
    script = portable_bundle / "submission" / "inference.py"
    assert script.exists()
    output = tmp_path / "portable_predictions.csv"
    independent_cwd = tmp_path / "unrelated_cwd"
    independent_cwd.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty_hf_cache"), "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    completed = subprocess.run([
        sys.executable, str(script), "--audio_dir", str(tmp_path / "missing_audio"),
        "--label_dir", str(inference_dir), "--ckpt_path", str(portable_bundle / manifest_path.name),
        "--output", str(output), "--device", "cpu", "--batch_size", "1",
    ], cwd=independent_cwd, env=environment, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    rows, actual = _read_predictions(output)
    assert [row["label file name"] for row in rows] == [path.name for path in sorted(input_paths)]
    assert actual == reference_by_name


def test_csv_evaluation_matches_filenames_instead_of_row_order(tmp_path):
    labels = tmp_path / "ground_truth"
    _write_annotation(labels / "z" / "aaa.json", "alpha", ["고열", "구토"])
    _write_annotation(labels / "a" / "zzz.json", "beta", ["두통"])
    _write_annotation(labels / "middle.json", "gamma", [])
    output = tmp_path / "independent_predictions.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label file name", "symptom"])
        writer.writerows([
            ["aaa.json", str(["고열"])], ["middle.json", str(["복통"])],
            ["zzz.json", str(["두통", "구토"])],
        ])
    result = evaluate_csv(labels, output)
    # 고열=1, 두통=1, 구토=0, 복통=0, all other absent labels=0.
    assert result["macro_f1"] == pytest.approx(2 / 9)
    assert result["samples"] == 3
    assert result["per_symptom"]["고열"]["f1"] == 1
    assert result["per_symptom"]["구토"]["support"] == 1


def test_five_saturated_float32_folds_average_within_probability_bounds(tmp_path, monkeypatch):
    labels = tmp_path / "labels"
    _write_annotation(labels / "call.json", "text only", {"ignored": True})
    manifest = {
        "config": {"max_length": 16, "stride": 4}, "tokenizer": "tokenizer",
        "folds": [f"fold_{index}.ckpt" for index in range(5)], "sparse": "sparse.joblib",
        "calibrator": {"neural_weights": [1.0] * 9, "thresholds": [0.5] * 9},
    }
    monkeypatch.setattr(pipeline, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "tokenize_dialogues", lambda texts, *args: [[] for _ in texts])
    monkeypatch.setattr(pipeline, "predict_checkpoint",
                        lambda path, documents, tokenizer, config: np.ones((len(documents), 9), dtype=np.float32))
    monkeypatch.setattr(pipeline.joblib, "load", lambda path: object())
    monkeypatch.setattr(pipeline, "predict_sparse", lambda model, texts: np.zeros((len(texts), 9), dtype=np.float32))
    output = tmp_path / "saturated.csv"
    _, probabilities, predictions = predict_pipeline(tmp_path / "manifest.ckpt", labels, output, device="cpu")
    np.testing.assert_array_equal(probabilities, np.ones((1, 9)))
    np.testing.assert_array_equal(predictions, np.ones((1, 9)))
    _, rows = _read_predictions(output)
    assert rows["call.json"] == [1] * 9
