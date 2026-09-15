"""Actual tiny CPU training, cache isolation, and portable offline CLI inference."""
import ast
from contextlib import contextmanager, ExitStack
import copy
import csv
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

from mission3_ver5 import pipeline
from mission3_ver5.ver4_runtime import pipeline as legacy
from mission3_ver5.ver4_runtime.data import TARGET_SYMPTOMS, load_dataset


def _annotation(path, text, symptoms):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"utterances": [{"text": text, "speaker": "caller"}],
                                "symptom": symptoms}, ensure_ascii=False), encoding="utf-8")


@contextmanager
def _restore(path):
    original = path.read_bytes()
    try:
        yield
    finally:
        path.write_bytes(original)


def _build_base(root):
    torch.set_num_threads(1)
    pretrained = root / "pretrained"
    pretrained.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "call", "context"]
    vocab += [f"cue{i}" for i in range(9)] + [f"id{i}" for i in range(144)]
    vocab_path = pretrained / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(pretrained)
    torch.manual_seed(3)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
                         num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(pretrained)
    train_dir = root / "Training" / "TL_서울_구급"
    truth = np.random.default_rng(24).integers(0, 2, size=(144, 9), dtype=np.int8)
    texts = {}
    for i in reversed(range(144)):
        prefix = "context " * (25 if i % 11 == 0 else 1)
        text = f"call id{i} {prefix}" + " ".join(f"cue{j}" for j in np.flatnonzero(truth[i]))
        name = f"call_{i:03d}.json"
        texts[name] = text
        _annotation(train_dir / name, text, [s for s, flag in zip(TARGET_SYMPTOMS, truth[i]) if flag])
    config = {
        "model_name": str(pretrained), "device": "cpu", "require_cuda": False,
        "n_folds": 2, "calibration_size": 0.2, "calibration_folds": 2,
        "max_length": 16, "stride": 4, "max_epochs": 1, "patience": 1,
        "batch_size": 8, "inference_batch_size": 8, "window_batch_size": 4,
        "gradient_accumulation_steps": 2, "dropout": 0, "gradient_checkpointing": True,
        "char_features": 100, "word_features": 100, "min_df": 1,
    }
    base_dir = root / "base_original"
    manifest_path, report = legacy.train_pipeline(train_dir, base_dir, config)
    return pretrained, train_dir, texts, base_dir, legacy.load_manifest(manifest_path), report


def _assert_bad_base_rejected(train_dir, base_dir, manifest):
    # A changed dataset must fail before trusting saved probability arrays.
    first = next(train_dir.glob("*.json"))
    with _restore(first):
        annotation = json.loads(first.read_text())
        annotation["utterances"][0]["text"] += " changed"
        first.write_text(json.dumps(annotation), encoding="utf-8")
        with pytest.raises(ValueError, match="Training"):
            pipeline._read_base(train_dir, base_dir, 1)

    cache_path = base_dir / "calibration_predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        original = {key: cache[key].copy() for key in cache.files}
    for corruption, message in [("targets", "targets mismatch"), ("names", "filenames mismatch"),
                                ("nonfinite", "Invalid ver4 raw"), ("shape", "Invalid ver4 raw")]:
        with _restore(cache_path):
            values = {key: value.copy() for key, value in original.items()}
            if corruption == "targets":
                values["targets"][0, 0] = 1 - values["targets"][0, 0]
            elif corruption == "names":
                values["filenames"][0] = values["filenames"][1]
            elif corruption == "nonfinite":
                values["neural"][0, 0] = np.nan
            else:
                values["sparse"] = values["sparse"][:, :8]
            np.savez_compressed(cache_path, **values)
            with pytest.raises(ValueError, match=message):
                pipeline._read_base(train_dir, base_dir, 1)

    split_path = base_dir / "splits.json"
    splits = json.loads(split_path.read_text())
    with _restore(split_path):
        invalid = copy.deepcopy(splits)
        invalid["folds"][0]["fit"][0] = splits["calibration"][0]
        split_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="leakage"):
            pipeline._read_base(train_dir, base_dir, 1)

    # Keep the synthetic fingerprint internally consistent to reach the separate
    # normalized-text group guard rather than merely failing dataset identity.
    fit_file = train_dir / splits["folds"][0]["fit"][0]
    cal_file = train_dir / splits["calibration"][0]
    checkpoint_path = base_dir / "mission3_ver4_model.ckpt"
    with _restore(fit_file), _restore(checkpoint_path):
        annotation = json.loads(fit_file.read_text())
        annotation["utterances"] = json.loads(cal_file.read_text())["utterances"]
        fit_file.write_text(json.dumps(annotation), encoding="utf-8")
        paths, texts, targets = load_dataset(train_dir, include_targets=True)
        changed = copy.deepcopy(manifest)
        changed["signature"] = legacy._fingerprint(paths, texts, targets, changed["config"])
        torch.save(changed, checkpoint_path)
        with pytest.raises(ValueError, match="text-group leakage"):
            pipeline._read_base(train_dir, base_dir, 1)


def _csv_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["label file name", "symptom"]
        rows = list(reader)
    result = {}
    for row in rows:
        labels = ast.literal_eval(row["symptom"])
        assert isinstance(labels, list) and len(set(labels)) == len(labels)
        assert all(label in TARGET_SYMPTOMS for label in labels)
        assert row["label file name"] not in result
        result[row["label file name"]] = [int(s in labels) for s in TARGET_SYMPTOMS]
    return rows, result


def test_actual_tiny_pipeline_reuses_cache_and_exports_portable_offline_bundle(tmp_path):
    pretrained, train_dir, texts, base_dir, base_manifest, base_report = _build_base(tmp_path)
    _assert_bad_base_rejected(train_dir, base_dir, base_manifest)
    output_dir = tmp_path / "ver5_original"
    config = {
        "device": "cpu", "require_cuda": False, "warmup_epochs": 1, "finetune_epochs": 1,
        "patience": 1, "attention_dim": 8, "batch_size": 8, "window_batch_size": 4,
        "inference_batch_size": 8, "gradient_accumulation_steps": 2, "dropout": 0,
        "gradient_checkpointing": True, "log_every": 0,
    }
    with ExitStack() as patches:
        for name in ("train_model", "fit_sparse", "predict_checkpoint"):
            patches.enter_context(patch.object(legacy, name, side_effect=AssertionError(f"ver4 {name} must be reused")))
        patches.enter_context(patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("Use saved encoder weights")))
        checkpoint_path, report = pipeline.train_pipeline(train_dir, base_dir, output_dir, config)
    manifest = pipeline.load_manifest(checkpoint_path)
    assert report["new_models_trained"] == 1 and report["base_models_retrained"] == 0
    assert report["validation_used_for_training_or_selection"] is False
    assert report["training_samples"] == 144
    assert report["baseline_crossfit_macro_f1"] == base_report["calibration_crossfit_macro_f1"]
    assert {row["phase"] for row in report["candidate_training"]["history"]} == {"baseline", "warmup", "finetune"}
    assert (output_dir / "candidate_best.ckpt").is_file()
    for name in [*base_manifest["folds"], "sparse.joblib"]:
        assert (output_dir / "base" / name).read_bytes() == (base_dir / name).read_bytes()
    if report["selected_name"] == "ver4_baseline":
        assert manifest["calibrator"]["candidate_weights"] == [0.0] * 9
        for key in ("neural_weights", "thresholds"):
            assert manifest["calibrator"][key] == base_manifest["calibrator"][key]

    with patch.object(pipeline.neural, "train_model", side_effect=AssertionError("Completed candidate retrained")), \
         patch.object(pipeline.neural, "predict_checkpoint", side_effect=AssertionError("Completed candidate inferred again")), \
         patch.object(pipeline.calibration, "calibrate_crossfit", side_effect=AssertionError("Completed calibration repeated")):
        resumed_path, resumed_report = pipeline.train_pipeline(train_dir, base_dir, output_dir, config)
    assert resumed_path == checkpoint_path and resumed_report == report
    with pytest.raises(ValueError, match="OUTPUT_DIR"):
        pipeline.train_pipeline(train_dir, base_dir, output_dir, {**config, "dropout": 0.1})

    inference_dir = tmp_path / "Test" / "labels"
    input_paths = [inference_dir / "z" / "aaa.json", inference_dir / "bbb.json",
                   inference_dir / "a" / "zzz.json", inference_dir / "middle.json"]
    for i, path in enumerate(input_paths):
        _annotation(path, texts[f"call_{i:03d}.json"], [])
    paths, _, predictions = pipeline.predict_pipeline(checkpoint_path, inference_dir,
                                                       tmp_path / "reference.csv", device="cpu", batch_size=2)
    reference = {p.name: prediction.tolist() for p, prediction in zip(paths, predictions)}
    # Exercise candidate inference even if the actual selection safely fell back.
    forced_path = output_dir / "forced_candidate.ckpt"
    forced = copy.deepcopy(manifest)
    forced["calibrator"] = {"neural_weights": [0.75] * 9, "candidate_weights": [0.5] * 9,
                             "thresholds": [0.5] * 9}
    torch.save(forced, forced_path)
    with patch.object(pipeline.neural, "predict_checkpoint", wraps=pipeline.neural.predict_checkpoint) as spy:
        forced_paths, probabilities, _ = pipeline.predict_pipeline(forced_path, inference_dir,
                                                                    tmp_path / "forced.csv", device="cpu", batch_size=2)
    assert spy.call_count == 1
    # Place test thresholds inside observed ranges to avoid a vacuous all-one CSV.
    thresholds = (probabilities.min(axis=0) + probabilities.max(axis=0)) / 2
    forced["calibrator"]["thresholds"] = thresholds.tolist()
    forced_predictions = probabilities >= thresholds
    assert np.any(forced_predictions) and not np.all(forced_predictions)
    forced_reference = {p.name: prediction.tolist() for p, prediction in zip(forced_paths, forced_predictions)}
    torch.save(forced, forced_path)
    for path in input_paths:
        annotation = json.loads(path.read_text())
        annotation["symptom"] = {"poisoned": "must not be inspected at inference"}
        annotation["utterances"][0]["speaker"] = {"invalid": "ignored"}
        annotation["unrelated"] = "cue8 cue8 cue8"
        path.write_text(json.dumps(annotation), encoding="utf-8")

    portable = tmp_path / "moved" / "portable"
    portable.parent.mkdir()
    shutil.move(str(output_dir), portable)
    base_dir.rename(tmp_path / "hidden_base")
    pretrained.rename(tmp_path / "hidden_pretrained")
    assert not output_dir.exists() and not base_dir.exists() and not pretrained.exists()
    script = portable / "submission" / "inference.py"
    assert script.is_file()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HOME": str(tmp_path / "empty_hf_cache"), "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "PYTHONDONTWRITEBYTECODE": "1"})
    for name, expected in [(checkpoint_path.name, reference), (forced_path.name, forced_reference)]:
        output = tmp_path / (name + ".csv")
        completed = subprocess.run([
            sys.executable, str(script), "--audio_dir", str(tmp_path / "missing_audio"),
            "--label_dir", str(inference_dir), "--ckpt_path", str(portable / name),
            "--output", str(output), "--device", "cpu", "--batch_size", "1",
        ], cwd=unrelated, env=environment, text=True, capture_output=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
        rows, actual = _csv_predictions(output)
        assert [row["label file name"] for row in rows] == [p.name for p in sorted(input_paths)]
        assert actual == expected
