"""Portable artifacts and tiny, mocked model runs; no pretrained downloads/GPU."""
import ast
import copy
import csv
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mission3_ver9 import pipeline
from mission3_ver9.budget import BudgetExceeded


class TinyTokenizer:
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text('{"tiny": true}', encoding="utf-8")


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """Keep artifact copying, hashes, caching and publication real; mock ML cost."""
    base_dir, output_dir = tmp_path / "original_ver4", tmp_path / "ver9"
    base_dir.mkdir()
    (base_dir / "tokenizer").mkdir()
    (base_dir / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    for name in ("fold_1.ckpt", "fold_2.ckpt", "sparse.joblib"):
        (base_dir / name).write_bytes(name.encode())
    base = {
        "format": "mission3_ver4", "version": 1,
        "target_symptoms": list(pipeline.TARGET_SYMPTOMS),
        "signature": "original-training-signature",
        "folds": ["fold_1.ckpt", "fold_2.ckpt"], "sparse": "sparse.joblib", "tokenizer": "tokenizer",
        "config": {"max_length": 512, "stride": 64, "inference_batch_size": 8},
        "calibrator": {"neural_weights": [0.25] * 9, "thresholds": [0.42] * 9},
        "versions": {},
    }
    torch.save(base, base_dir / "mission3_ver4_model.ckpt")
    paths = [tmp_path / "Training" / f"sample_{i:02d}.json" for i in range(12)]
    texts = [f"training dialogue {i}" for i in range(12)]
    targets = np.asarray([[(i + j) % 2 for j in range(9)] for i in range(12)], dtype=np.int8)
    fit, stop, cal = np.arange(4), np.arange(4, 8), np.arange(8, 12)
    groups = np.asarray([f"group_{i}" for i in range(12)])
    splits = {"calibration": [paths[i].name for i in cal], "folds": [
        {"fit": [paths[i].name for i in fit], "stop": [paths[i].name for i in stop]},
        {"fit": [paths[i].name for i in stop], "stop": [paths[i].name for i in fit]},
    ]}
    pipeline.write_json(base_dir / "splits.json", splits)
    hashes = {p.relative_to(base_dir).as_posix(): pipeline.assets._sha256(p)
              for p in base_dir.rglob("*") if p.is_file() and p.name != "splits.json"}
    base_neural = np.full((4, 9), 0.7)
    base_sparse = np.full((4, 9), 0.3)
    read_base = Mock(return_value=(base, paths, texts, targets, fit, stop, cal, groups, base_neural, base_sparse))
    monkeypatch.setattr(pipeline.assets, "_read_base", read_base)
    monkeypatch.setattr(pipeline.assets, "_versions", lambda: {})
    monkeypatch.setattr(pipeline.assets, "_release", lambda: None)
    monkeypatch.setattr(pipeline, "_identity", lambda *args: (
        "tiny-run-signature", {"base_artifact_sha256": hashes, "config": args[0]}))
    monkeypatch.setattr(pipeline, "export_code", lambda output: None)
    monkeypatch.setattr(pipeline.pretrained, "validate_pretrained", lambda *args: {
        "model_id": pipeline.pretrained.MODEL_ID, "revision": pipeline.pretrained.MODEL_REVISION,
        "files": {},
    })
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", Mock(return_value=TinyTokenizer()))
    monkeypatch.setattr(pipeline.data, "candidate_texts", lambda texts, tokenizer: list(texts))
    monkeypatch.setattr(pipeline.neural, "tokenize_dialogues", lambda texts, *args: list(texts))
    candidate_neural = np.full((4, 9), 0.8)
    candidate_sparse = np.full((4, 9), 0.2)

    def train(*args, **kwargs):
        checkpoint = args[6]
        checkpoint.write_bytes(b"completed candidate checkpoint")
        checkpoint.with_name(checkpoint.stem + "_last.ckpt").write_bytes(b"completed optimizer state")
        return {"best_epoch": 1, "stopped_for_budget": False, "completed": True}

    train_mock = Mock(side_effect=train)
    monkeypatch.setattr(pipeline.neural, "train_model", train_mock)
    monkeypatch.setattr(pipeline.neural, "predict_checkpoint", Mock(return_value=candidate_neural))
    sparse_fit = Mock(return_value={"mock_training_only_model": True})
    monkeypatch.setattr(pipeline.sparse, "fit_model", sparse_fit)
    monkeypatch.setattr(pipeline.sparse, "predict_model", Mock(return_value=candidate_sparse))
    proposed = {"neural_weights": [0.5] * 9, "candidate_weights": [0.5] * 9,
                "thresholds": [0.61] * 9, "blend_space": "logit"}
    comparison = {"selected_name": "candidate_mixture", "selection_gain": 0.05}
    compare = Mock(side_effect=lambda *a, **k: (copy.deepcopy(proposed), copy.deepcopy(comparison)))
    monkeypatch.setattr(pipeline.calibration, "compare_candidates", compare)
    diagnostic = {"candidate_standalone_crossfit_macro_f1": 0.7}
    standalone = Mock(return_value=({"thresholds": [0.55] * 9}, diagnostic))
    monkeypatch.setattr(pipeline.calibration, "fit_candidate_only_crossfit", standalone)

    def run(**kwargs):
        options = {"require_cuda": False, "device": "cpu", "max_models": 1}
        options.update(kwargs.pop("config", {}))
        return pipeline.train_pipeline(tmp_path / "Training", base_dir, output_dir,
                                       config=options, pretrained_dir=tmp_path / "pretrained", **kwargs)

    return SimpleNamespace(base_dir=base_dir, output_dir=output_dir, base=base, run=run,
                           paths=paths, texts=texts, targets=targets, fit=fit, stop=stop, cal=cal,
                           read_base=read_base, train=train_mock, sparse_fit=sparse_fit,
                           proposed=proposed, comparison=comparison, compare=compare, standalone=standalone,
                           expected_candidate=0.75 * candidate_neural + 0.25 * candidate_sparse)


def test_training_api_has_no_validation_input():
    assert not any("validation" in name.lower() for name in inspect.signature(pipeline.train_pipeline).parameters)
    with pytest.raises(TypeError):
        pipeline.train_pipeline("train", "base", "out", validation_dir="must not read")


@pytest.mark.parametrize("config", [
    {"unknown": 1}, {"max_models": 0}, {"max_models": 3}, {"max_models": True},
    {"final_reserve_seconds": 1499}, {"calibration_reserve_seconds": 599},
    {"min_calibration_gain": 0.001}, {"min_calibration_improved_folds": 2},
    {"model_name": "unapproved/model"}, {"max_epochs": 4},
])
def test_unknown_options_and_relaxed_guards_are_rejected(config):
    with pytest.raises(ValueError):
        pipeline._config(config)


@pytest.mark.parametrize("config", [
    {"final_reserve_seconds": float("nan")}, {"calibration_reserve_seconds": float("inf")},
    {"min_calibration_gain": float("nan")}, {"min_calibration_improved_folds": 3.5},
])
def test_nonfinite_and_noninteger_guards_rejected(config):
    with pytest.raises(ValueError):
        pipeline._config(config)


def test_tiny_training_adopts_candidate_from_training_only_partitions(tiny_run):
    run = tiny_run
    report = run.run()
    manifest = pipeline.load_manifest(run.output_dir / pipeline.MANIFEST)
    assert report["complete"] and not report["official_validation_used_for_selection"]
    assert manifest["selected_name"] == "candidate_mixture"
    assert manifest["selection"] == run.proposed
    assert manifest["candidate_calibrator"] == {"thresholds": [0.55] * 9}
    assert manifest["candidate_models"] == ["candidate_1.ckpt"]
    train_args = run.train.call_args.args
    assert train_args[0] == run.texts[:4] and train_args[2] == run.texts[4:8]
    assert run.sparse_fit.call_args.args[0] == run.texts[:8]
    np.testing.assert_array_equal(run.compare.call_args.args[0], run.targets[run.cal])
    np.testing.assert_allclose(run.compare.call_args.args[3], run.expected_candidate)
    assert run.compare.call_args.kwargs["min_gain"] == 0.002
    assert run.compare.call_args.kwargs["min_improved_folds"] == 3
    assert not (run.output_dir / "candidate_1_last.ckpt").exists()
    assert (run.output_dir / "base/mission3_ver4_model.ckpt").exists()


def test_failed_adoption_keeps_exact_original_calibrator(tiny_run):
    run = tiny_run
    run.comparison["selected_name"] = "ver4_baseline"
    run.proposed["thresholds"] = [0.01] * 9  # Cross-fit refit must not replace original fallback.
    report = run.run()
    manifest = pipeline.load_manifest(run.output_dir / pipeline.MANIFEST)
    assert report["selected_name"] == "ver4_baseline"
    assert manifest["selection"] == {**run.base["calibrator"], "candidate_weights": [0.0] * 9, "blend_space": "raw"}


def test_no_completed_epoch_keeps_baseline_and_skips_sparse_selection(tiny_run):
    run = tiny_run
    run.train.side_effect = None
    run.train.return_value = {"best_epoch": 0, "stopped_for_budget": True, "completed": False}
    report = run.run()
    manifest = pipeline.load_manifest(run.output_dir / pipeline.MANIFEST)
    assert report["status"] == "no_trained_candidate"
    assert manifest["candidate_models"] == []
    assert manifest["selected_name"] == "ver4_baseline"
    run.sparse_fit.assert_not_called()
    run.compare.assert_not_called()


def test_timeout_leaves_a_valid_portable_fallback(tiny_run):
    run = tiny_run
    run.train.side_effect = BudgetExceeded("simulated deadline")
    with pytest.raises(BudgetExceeded):
        run.run()
    manifest = pipeline.load_manifest(run.output_dir / pipeline.MANIFEST)
    assert manifest["status"] == "baseline_ready"
    assert manifest["selected_name"] == "ver4_baseline"
    assert not (run.output_dir / "training_metrics.json").exists()
    assert (run.base_dir / "fold_1.ckpt").read_bytes() == b"fold_1.ckpt"


def test_completed_run_reuses_frozen_selection_without_retraining(tiny_run):
    first = tiny_run.run()
    second = tiny_run.run()
    assert first == second
    assert tiny_run.train.call_count == tiny_run.compare.call_count == 1


def test_attempt_receipt_is_published_only_after_valid_portable_fallback(tiny_run, tmp_path):
    run = tiny_run
    marker = tmp_path / "local_attempt/portable_model_ready.json"

    def interrupt_after_receipt(*args, **kwargs):
        receipt = json.loads(marker.read_text())
        assert receipt == {"signature": "tiny-run-signature", "output_dir": str(run.output_dir.resolve())}
        manifest = pipeline.load_manifest(run.output_dir / pipeline.MANIFEST)
        assert manifest["status"] == "baseline_ready"
        raise BudgetExceeded("candidate time ended after valid fallback publication")

    run.train.side_effect = interrupt_after_receipt
    with pytest.raises(BudgetExceeded):
        run.run(attempt_marker=marker)
    assert marker.is_file()


def test_completed_reuse_writes_receipt_for_this_new_attempt(tiny_run, tmp_path):
    run = tiny_run
    first = run.run()
    marker = tmp_path / "new_attempt/portable_model_ready.json"
    second = run.run(attempt_marker=marker)
    assert first == second
    assert run.train.call_count == 1
    assert json.loads(marker.read_text()) == {"signature": "tiny-run-signature", "output_dir": str(run.output_dir.resolve())}


def test_rejected_existing_experiment_never_publishes_attempt_receipt(tiny_run, tmp_path):
    run = tiny_run
    run.output_dir.mkdir()
    pipeline.write_json(run.output_dir / "run_identity.json", {"signature": "different experiment"})
    marker = tmp_path / "failed_attempt/portable_model_ready.json"
    with pytest.raises(ValueError, match="different experiment"):
        run.run(attempt_marker=marker)
    assert not marker.exists()
    run.train.assert_not_called()


def test_bundle_can_move_without_original_inputs(tiny_run, tmp_path):
    tiny_run.run()
    relocated = tmp_path / "downloaded_elsewhere" / "ver9"
    relocated.parent.mkdir()
    shutil.move(tiny_run.output_dir, relocated)
    shutil.rmtree(tiny_run.base_dir)
    manifest = pipeline.load_manifest(relocated / pipeline.MANIFEST)
    base = pipeline.base_pipeline.load_manifest(relocated / manifest["base_manifest"])
    assert base["signature"] == tiny_run.base["signature"]
    assert all(not Path(name).is_absolute() for name in manifest["artifact_sha256"])


@pytest.mark.parametrize("mutation", ["missing", "corrupted"])
def test_missing_or_corrupted_artifact_rejected(tiny_run, mutation):
    tiny_run.run()
    artifact = tiny_run.output_dir / "base/fold_1.ckpt"
    if mutation == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"not the original trained weights")
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.load_manifest(tiny_run.output_dir / pipeline.MANIFEST)


@pytest.mark.parametrize("mutation", ["missing_hash_map", "missing_candidate_hash", "missing_base_hash"])
def test_manifest_requires_digest_coverage_for_referenced_artifacts(tiny_run, mutation):
    tiny_run.run()
    path = tiny_run.output_dir / pipeline.MANIFEST
    manifest = json.loads(path.read_text())
    if mutation == "missing_hash_map":
        manifest.pop("artifact_sha256")
    elif mutation == "missing_candidate_hash":
        manifest["artifact_sha256"].pop("candidate_1.ckpt")
    else:
        manifest["artifact_sha256"].pop("base/fold_1.ckpt")
    pipeline.write_json(path, manifest)
    with pytest.raises(ValueError):
        pipeline.load_manifest(path)


@pytest.mark.parametrize("name", ["../outside.ckpt", "/tmp/outside.ckpt", "..\\outside.ckpt"])
def test_artifact_paths_must_stay_inside_portable_bundle(tiny_run, name):
    tiny_run.run()
    path = tiny_run.output_dir / pipeline.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["base_manifest"] = name
    pipeline.write_json(path, manifest)
    with pytest.raises(ValueError):
        pipeline.load_manifest(path)


def test_inference_is_label_blind_and_writes_standard_csv(tiny_run, tmp_path, monkeypatch):
    run = tiny_run
    run.comparison["selected_name"] = "ver4_baseline"
    run.run()
    labels = tmp_path / "official_validation"
    labels.mkdir()
    original = {
        "utterances": [{"text": "체온이 높아요", "speaker": "ignored metadata"}, {"text": "언제부터인가요?"}],
        "symptom": {"intentionally_invalid_target": True}, "patient_age": "MUST_NOT_ENTER_MODEL",
    }
    (labels / "b.json").write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    (labels / "a.json").write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    seen = []

    def raw_base(root, base, paths, texts, config, cache_dir):
        seen.append(texts)
        return np.full((2, 9), 0.8), np.full((2, 9), 0.2)

    monkeypatch.setattr(pipeline, "_raw_base", raw_base)
    output = tmp_path / "prediction/result.csv"
    paths, probabilities, predictions = pipeline.predict_pipeline(
        run.output_dir / pipeline.MANIFEST, labels, output, device="cpu")
    assert [p.name for p in paths] == ["a.json", "b.json"]
    assert seen == [["체온이 높아요\n언제부터인가요?"] * 2]
    np.testing.assert_allclose(probabilities, 0.25 * 0.8 + 0.75 * 0.2)
    assert not predictions.any()
    with output.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["label file name", "symptom"]
        assert [ast.literal_eval(row["symptom"]) for row in reader] == [[], []]
    before = output.read_bytes()
    original["symptom"] = list(pipeline.TARGET_SYMPTOMS)
    original["patient_age"] = "completely different metadata"
    for path in paths:
        path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    pipeline.predict_pipeline(run.output_dir / pipeline.MANIFEST, labels, output, device="cpu")
    assert output.read_bytes() == before


def test_failed_csv_write_does_not_replace_previous_final(tiny_run, tmp_path, monkeypatch):
    run = tiny_run
    run.comparison["selected_name"] = "ver4_baseline"
    run.run()
    monkeypatch.setattr(pipeline, "load_dataset", lambda *a, **k: ([Path("sample.json")], ["text only"], None))
    monkeypatch.setattr(pipeline, "_raw_base", lambda *a, **k: (np.full((1, 9), 0.8), np.full((1, 9), 0.2)))
    output = tmp_path / "already_finished.csv"
    output.write_text("previous complete predictions", encoding="utf-8")

    def fail(paths, probabilities, thresholds, temporary):
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline, "write_predictions", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        pipeline.predict_pipeline(run.output_dir / pipeline.MANIFEST, tmp_path, output, device="cpu")
    assert output.read_text(encoding="utf-8") == "previous complete predictions"


def test_candidate_inference_and_standalone_use_their_separate_frozen_thresholds(tiny_run, tmp_path, monkeypatch):
    run = tiny_run
    run.run()
    paths = [Path("first.json"), Path("second.json")]
    monkeypatch.setattr(pipeline, "load_dataset", lambda *a, **k: (paths, ["first text", "second text"], None))
    base_neural, base_sparse = np.full((2, 9), 0.7), np.full((2, 9), 0.3)
    monkeypatch.setattr(pipeline, "_raw_base", lambda *a, **k: (base_neural, base_sparse))
    pipeline.neural.predict_checkpoint.return_value = np.full((2, 9), 0.8)
    pipeline.sparse.predict_model.return_value = np.full((2, 9), 0.2)
    output, standalone = tmp_path / "final.csv", tmp_path / "diagnostics/standalone.csv"
    _, probabilities, final_predictions = pipeline.predict_pipeline(
        run.output_dir / pipeline.MANIFEST, tmp_path, output, device="cpu", candidate_output=standalone)
    candidate = np.full((2, 9), 0.75 * 0.8 + 0.25 * 0.2)
    expected = pipeline.calibration.apply_selected(base_neural, base_sparse, candidate, run.proposed)
    np.testing.assert_array_equal(probabilities, expected)
    np.testing.assert_array_equal(final_predictions, expected >= 0.61)
    with standalone.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [ast.literal_eval(row["symptom"]) for row in rows] == [list(pipeline.TARGET_SYMPTOMS)] * 2
    assert not (tmp_path / "final.csv.tmp").exists()
    assert not (tmp_path / "diagnostics/standalone.csv.tmp").exists()
