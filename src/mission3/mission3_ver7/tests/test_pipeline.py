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
from transformers import BertConfig, BertModel, BertTokenizerFast, ElectraConfig, ElectraModel

from mission3_ver7 import pipeline, pretrained as public_model
from mission3_ver7.ver4_runtime import pipeline as legacy
from mission3_ver7.ver4_runtime.data import TARGET_SYMPTOMS, load_dataset


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


def _build_public(root, base_pretrained):
    directory = root / "public_electra"
    directory.mkdir()
    # Deliberately different ids and architecture from the original Bert model.
    vocab = (base_pretrained / "vocab.txt").read_text().splitlines()
    vocab = vocab[:5] + ["newword"] + list(reversed(vocab[5:]))
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(directory)
    torch.manual_seed(37)
    ElectraModel(ElectraConfig(vocab_size=len(vocab), embedding_size=12, hidden_size=12,
                              num_hidden_layers=2, num_attention_heads=3, intermediate_size=24,
                              max_position_embeddings=64, hidden_dropout_prob=0,
                              attention_probs_dropout_prob=0)).save_pretrained(directory)
    metadata = {"format": public_model.FORMAT, "complete": True, "model_id": "test/tiny-electra",
                "revision": "a" * 40,
                "files": {p.name: {"size": p.stat().st_size, "sha256": pipeline._sha256(p)}
                          for p in directory.iterdir() if p.is_file()}}
    (directory / public_model.MANIFEST_NAME).write_text(json.dumps(metadata), encoding="utf-8")
    public_model.validate_pretrained(directory)
    return directory


@pytest.mark.parametrize("bundle_base", [True, False], ids=["portable", "external_base"])
def test_fresh_encoder_pipeline_split_isolation_cache_and_offline_export(tmp_path, bundle_base):
    base_pretrained, train_dir, texts, base_dir, base_manifest, base_report = _build_base(tmp_path)
    public_dir = _build_public(tmp_path, base_pretrained)
    _assert_bad_base_rejected(train_dir, base_dir, base_manifest)
    output_dir = tmp_path / "ver7_original"
    config = {
        "model_name": "test/tiny-electra", "device": "cpu", "require_cuda": False,
        "max_epochs": 1, "patience": 1, "stop_crossfit_splits": 2,
        "bundle_base": bundle_base, "attention_dim": 8, "batch_size": 8, "window_batch_size": 4,
        "inference_batch_size": 8, "gradient_accumulation_steps": 2, "dropout": 0,
        "gradient_checkpointing": True, "log_every": 0, "max_length": 20, "stride": 4,
    }
    with patch.object(legacy, "train_model", side_effect=AssertionError("Must not retrain old neural")), \
         patch.object(legacy, "fit_sparse", side_effect=AssertionError("Must not refit old sparse")), \
         patch.object(legacy, "predict_checkpoint", wraps=legacy.predict_checkpoint) as old_predict:
        checkpoint, report = pipeline.train_pipeline(train_dir, base_dir, output_dir, config, pretrained_dir=public_dir)
    # Exactly one original fold, its own stop only: never a leaking ensemble benchmark.
    assert old_predict.call_count == 1
    assert Path(old_predict.call_args.args[0]).name == base_manifest["folds"][0]
    assert len(old_predict.call_args.args[1]) == report["candidate_stop_samples"]
    manifest = pipeline.load_manifest(checkpoint)
    assert manifest["portable"] is bundle_base
    assert report["new_models_trained"] == 1 and report["base_models_retrained"] == 0
    assert not report["validation_used_for_training_or_selection"]
    assert report["training_samples"] == 144
    assert report["baseline_crossfit_macro_f1"] == base_report["calibration_crossfit_macro_f1"]
    assert 0 <= report["candidate_standalone_crossfit_macro_f1"] <= 1
    assert report["candidate_training"]["completed_epochs"] == 1
    saved = torch.load(output_dir / manifest["candidate"], weights_only=True, map_location="cpu")
    assert saved["encoder_config"]["model_type"] == "electra"
    assert saved["initialization"] == "local_public_pretrained_encoder_and_new_label_attention_head"
    exported_tok = BertTokenizerFast.from_pretrained(output_dir / "tokenizer", local_files_only=True)
    base_tok = BertTokenizerFast.from_pretrained(base_dir / "tokenizer", local_files_only=True)
    assert exported_tok.convert_tokens_to_ids("call") != base_tok.convert_tokens_to_ids("call")
    assert "feature_cache" not in report and "head_trials" not in report["candidate_training"]
    for name in pipeline.RUNTIME_FILES:
        assert (output_dir / "submission" / name).read_bytes() == (Path(pipeline.__file__).parent / name).read_bytes()
    for file in (Path(pipeline.__file__).parent / "ver4_runtime").iterdir():
        if file.is_file():
            assert (output_dir / "submission" / "ver4_runtime" / file.name).read_bytes() == file.read_bytes()
    assert (output_dir / "base").is_dir() is bundle_base
    if report["selected_name"] == "ver4_baseline":
        assert manifest["calibrator"]["candidate_weights"] == [0.] * 9
        for key in ("neural_weights", "thresholds"):
            assert manifest["calibrator"][key] == base_manifest["calibrator"][key]

    # Completed pipeline and a crash just before final publication reuse all model work.
    with patch.object(pipeline.neural, "train_model", side_effect=AssertionError("Completed run retrained")), \
         patch.object(pipeline.neural, "predict_checkpoint", side_effect=AssertionError("Completed cal repeated")), \
         patch.object(legacy, "predict_checkpoint", side_effect=AssertionError("Completed baseline repeated")):
        repeated_path, repeated = pipeline.train_pipeline(train_dir, base_dir, output_dir, config, pretrained_dir=public_dir)
        assert repeated == report and repeated_path == checkpoint
        checkpoint.unlink()
        (output_dir / "mission3_ver7_metrics.json").unlink()
        checkpoint, report = pipeline.train_pipeline(train_dir, base_dir, output_dir, config, pretrained_dir=public_dir)
    with pytest.raises(ValueError, match="OUTPUT_DIR"):
        pipeline.train_pipeline(train_dir, base_dir, output_dir, {**config, "dropout": .2}, pretrained_dir=public_dir)
    with pytest.raises(ValueError, match="Unknown config"):
        pipeline.train_pipeline(train_dir, base_dir, output_dir, {**config, "head_epochs": 1}, pretrained_dir=public_dir)
    with _restore(public_dir / "model.safetensors"):
        with (public_dir / "model.safetensors").open("ab") as stream:
            stream.write(b"bad")
        with pytest.raises(ValueError, match="무결성"):
            pipeline.train_pipeline(train_dir, base_dir, output_dir, config, pretrained_dir=public_dir)

    inference_dir = tmp_path / "Validation" / "labels"
    for i in range(4):
        path = inference_dir / f"{i}.json"
        _annotation(path, texts[f"call_{i:03d}.json"], [TARGET_SYMPTOMS[i]])
        annotation = json.loads(path.read_text())
        annotation["utterances"].append({"text": "newword context cue8", "speaker": "dispatcher"})
        path.write_text(json.dumps(annotation), encoding="utf-8")
    paths, source_texts, _ = load_dataset(inference_dir, include_targets=False)
    candidate_texts = pipeline.data.candidate_texts(source_texts, exported_tok)
    assert all(" [SEP] " in text for text in candidate_texts)
    base_documents = legacy.tokenize_dialogues(source_texts, base_tok, 16, 4)
    candidate_documents = pipeline.neural.tokenize_dialogues(candidate_texts, exported_tok, 20, 4)
    # Force a known fallback and still require candidate-only diagnostics to run.
    forced_path = output_dir / "diagnostic.ckpt"
    forced = copy.deepcopy(manifest)
    forced["calibrator"] = {"neural_weights": [.75] * 9, "candidate_weights": [0.] * 9,
                             "thresholds": [.5] * 9, "blend_space": "raw"}
    torch.save(forced, forced_path)
    cache_dir = tmp_path / "shared_predictions"
    with patch.object(pipeline.neural, "predict_checkpoint", wraps=pipeline.neural.predict_checkpoint) as new_predict, \
         patch.object(legacy, "predict_checkpoint", wraps=legacy.predict_checkpoint) as old_predict:
        _, probabilities, predictions = pipeline.predict_pipeline(
            forced_path, inference_dir, tmp_path / "final.csv", device="cpu", batch_size=2,
            cache_dir=cache_dir, candidate_output=tmp_path / "candidate.csv")
    assert new_predict.call_count == 1 and old_predict.call_count == 2
    assert new_predict.call_args.args[1] == candidate_documents
    assert all(call.args[1] == base_documents for call in old_predict.call_args_list)
    reference_final = (tmp_path / "final.csv").read_bytes()
    reference_candidate = (tmp_path / "candidate.csv").read_bytes()
    assert len(_csv_predictions(tmp_path / "candidate.csv")[0]) == 4
    for path in paths:
        annotation = json.loads(path.read_text())
        annotation["symptom"] = {"poison": "do not read"}
        annotation["utterances"][0]["speaker"] = {"poison": "ignored"}
        annotation["other"] = "cue8"
        path.write_text(json.dumps(annotation), encoding="utf-8")
    with patch.object(pipeline.neural, "predict_checkpoint", side_effect=AssertionError("New model cache missed")), \
         patch.object(legacy, "predict_checkpoint", side_effect=AssertionError("Old model cache missed")), \
         patch.object(legacy, "predict_sparse", side_effect=AssertionError("Sparse cache missed")):
        pipeline.predict_pipeline(forced_path, inference_dir, tmp_path / "reused.csv", device="cpu", batch_size=2,
                                  cache_dir=cache_dir, candidate_output=tmp_path / "reused_candidate.csv")
        # A calibration-only change reuses every raw model output.
        changed_blend = {**forced, "calibrator": {**forced["calibrator"], "blend_space": "logit",
                                                  "candidate_weights": [.5] * 9}}
        changed_path = output_dir / "changed_blend.ckpt"
        torch.save(changed_blend, changed_path)
        _, mixed, _ = pipeline.predict_pipeline(changed_path, inference_dir, tmp_path / "mixed.csv", device="cpu",
                                               batch_size=2, cache_dir=cache_dir)
        assert not np.array_equal(mixed, probabilities)
    assert reference_final == (tmp_path / "reused.csv").read_bytes()
    assert reference_candidate == (tmp_path / "reused_candidate.csv").read_bytes()

    # Distinct changed candidate weights invalidate only their own cache and must match manifest hash.
    candidate_path = output_dir / manifest["candidate"]
    with _restore(candidate_path):
        updated = torch.load(candidate_path, map_location="cpu", weights_only=True)
        updated["model_state"]["head.label_bias"] += .2
        torch.save(updated, candidate_path)
        with pytest.raises(ValueError, match="weights hash"):
            pipeline.load_manifest(forced_path)
        candidate_changed = {**changed_blend, "candidate_sha256": pipeline._sha256(candidate_path)}
        candidate_changed_path = output_dir / "changed_candidate.ckpt"
        torch.save(candidate_changed, candidate_changed_path)
        with patch.object(legacy, "predict_checkpoint", side_effect=AssertionError("Unchanged base reran")), \
             patch.object(legacy, "predict_sparse", side_effect=AssertionError("Unchanged sparse reran")), \
             patch.object(pipeline.neural, "predict_checkpoint", wraps=pipeline.neural.predict_checkpoint) as spy:
            _, changed, _ = pipeline.predict_pipeline(candidate_changed_path, inference_dir, tmp_path / "new_candidate.csv",
                                                      device="cpu", batch_size=2, cache_dir=cache_dir)
        assert spy.call_count == 1 and not np.array_equal(changed, mixed)

    # Offline exported CLI works with only saved results + old base; no pretrained/source import path.
    portable = tmp_path / "moved_ver7"
    output_dir.rename(portable)
    moved_base = tmp_path / "moved_base"
    base_dir.rename(moved_base)
    moved_public = tmp_path / "moved_public"
    public_dir.rename(moved_public)
    base_pretrained.rename(tmp_path / "hidden_old_pretrained")
    with patch.object(pipeline.neural, "train_model", side_effect=AssertionError("Relocation retrained")):
        moved_checkpoint, moved_report = pipeline.train_pipeline(train_dir, moved_base, portable, config,
                                                                 pretrained_dir=moved_public)
    assert moved_checkpoint.name == checkpoint.name and moved_report == report
    base_args = [] if bundle_base else ["--base_dir", str(moved_base)]
    if not bundle_base:
        with pytest.raises(FileNotFoundError, match="--base_dir"):
            pipeline.load_manifest(portable / checkpoint.name)
    for i, path in enumerate(paths):
        annotation = json.loads(path.read_text())
        annotation["symptom"] = [TARGET_SYMPTOMS[i]]
        path.write_text(json.dumps(annotation), encoding="utf-8")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HOME": str(tmp_path / "empty_hf"), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    environment.pop("PYTHONPATH", None)
    final_dir = tmp_path / "cli_predictions"
    completed = subprocess.run([
        sys.executable, str(portable / "submission" / "inference.py"), "--audio_dir", "",
        "--label_dir", str(inference_dir), "--ckpt_path", str(portable / forced_path.name),
        "--output", str(final_dir / "mission3.csv"), "--candidate_output", str(final_dir / "candidate_only.csv"),
        "--evaluate", "--device", "cpu", "--batch_size", "2", "--cache_dir", str(cache_dir), *base_args,
    ], cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (final_dir / "mission3.csv").read_bytes() == reference_final
    assert (final_dir / "candidate_only.csv").read_bytes() == reference_candidate
    for filename in ("validation_metrics.json", "candidate_validation_metrics.json"):
        metrics = json.loads((final_dir / filename).read_text())
        assert metrics["samples"] == 4 and 0 <= metrics["macro_f1"] <= 1
