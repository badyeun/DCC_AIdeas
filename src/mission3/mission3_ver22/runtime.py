"""Cached utterance-evidence representation experiments with Training-only selection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from . import data, head, evidence, source_bundle
from .bootstrap import prepare_data
from .budget import Budget, BudgetExceeded

OTHER_LABELS = [0, 1, 2, 3, 4, 5, 7, 8]


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dataset_digest(texts, targets):
    value = hashlib.sha256()
    for text, target in zip(texts, targets):
        value.update(text.encode("utf-8"))
        value.update(b"\0")
        value.update(np.asarray(target, dtype=np.int8).tobytes())
    return value.hexdigest()


def _hash(path, budget=None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            if budget:
                budget.check("hash artifact")
            value.update(block)
    return value.hexdigest()


def _safe(root, relative):
    root = Path(root).resolve()
    candidate = root / relative
    result = candidate.resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("Artifact path escapes its run directory")
    for part in (candidate, *candidate.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError("Symlink artifacts are not supported")
    return result


def _verify_hashes(root, hashes, budget=None):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing completed-artifact hashes")
    for relative, expected in hashes.items():
        path = _safe(root, relative)
        if not path.is_file() or _hash(path, budget) != expected:
            raise ValueError("Saved artifact changed: " + relative)


def _copy_file(source, target, budget):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    with Path(source).open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(4 * 1024 * 1024):
            budget.check("publish verified artifact")
            writer.write(block)
    if _hash(source, budget) != _hash(temporary, budget):
        raise OSError("Published copy content mismatch")
    temporary.replace(target)


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _save_feature_cache(directory, cache, identity, budget):
    """Keep large FP32 arrays on local scratch disk, never in Drive results."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata, hashes = {}, {}
    for key, value in cache.items():
        if not isinstance(value, np.ndarray):
            continue
        if Path(key).name != key or "/" in key or "\\" in key:
            raise ValueError("Invalid feature-cache array name")
        destination = directory / (key + ".npy")
        temporary = destination.with_suffix(".npy.tmp")
        budget.check("save local feature cache", training=True)
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(destination)
        hashes[destination.name] = _hash(destination, budget)
        metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": int(value.nbytes)}
    saved = {"identity": identity, "timing": _serial(cache.get("_timing", {})),
             "arrays": metadata, "artifact_hashes": hashes, "storage": "local scratch only"}
    data.write_json(directory / "complete.json", saved)
    return saved


def validate_settings(settings):
    result = dict(settings)
    if result.get("version") != "ver22":
        raise ValueError("This runner supports ver22 only")
    if result.get("seed") != 42 or result.get("holdout_size") != .2:
        raise ValueError("The verified ver19 split requires seed 42 and holdout_size 0.2")
    if result.get("refit_full_training") is not True:
        raise ValueError("Accepted recipes must be refit on the full-Training source model")
    if result.get("batch_size") != 4 or result.get("encoder_window_batch") != 4:
        raise ValueError("Verified source inference requires the original batch and window batch sizes of 4")
    for key in ("batch_size", "encoder_window_batch", "max_evidence_epochs", "evidence_batch_size",
                "early_stopping_patience", "ce_warmup_epochs", "log_every"):
        if type(result.get(key)) is not int or result[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if result["max_evidence_epochs"] > 20 or result["evidence_batch_size"] > 256:
        raise ValueError("Evidence training permits at most 20 epochs and batch size 256")
    if result["ce_warmup_epochs"] >= result["max_evidence_epochs"]:
        raise ValueError("CE warmup must leave at least one mixed-loss epoch")
    for key in ("evidence_learning_rate", "positive_weight", "gce_q", "gce_mix", "time_budget_minutes",
                "finalization_reserve_seconds", "budget_safety_factor"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    weight_decay = result.get("evidence_weight_decay")
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError("evidence_weight_decay must be finite and nonnegative")
    if result["gce_q"] > 1 or result["gce_mix"] > .5:
        raise ValueError("GCE requires q <= 1 and mixture <= 0.5")
    if result["time_budget_minutes"] > 180 or result["budget_safety_factor"] < 1:
        raise ValueError("Budget must be at most 180 minutes with safety factor >= 1")
    if result["finalization_reserve_seconds"] >= result["time_budget_minutes"] * 60:
        raise ValueError("Finalization reserve must fit within the shared budget")
    for key in ("run_validation", "allow_cpu"):
        if type(result.get(key)) is not bool:
            raise ValueError(key + " must be boolean")
    for key in ("source_run", "resume_run", "work_dir", "output_dir", "input_zip", "train_dir", "validation_dir"):
        if not isinstance(result.get(key, ""), str):
            raise ValueError(key + " must be a path string")
    return result


def _stable_settings(settings):
    volatile = {"deadline_monotonic", "input_zip", "train_dir", "validation_dir", "source_run", "work_dir", "output_dir",
                "resume_run", "log_every", "time_budget_minutes", "finalization_reserve_seconds", "budget_safety_factor"}
    return {key: value for key, value in settings.items() if key not in volatile}


def _device(settings):
    import torch
    if not torch.cuda.is_available() and not settings["allow_cpu"]:
        raise RuntimeError("Colab에서 GPU 런타임을 선택하세요. 전체 데이터 CPU 실행은 자동으로 시작하지 않습니다.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serial(item) for item in value]
    return value


def _cached_features(checkpoint, texts, targets, settings, root, role, budget):
    """Scope-specific cache; immutable identity and fresh hashes gate reuse."""
    checkpoint = Path(checkpoint)
    manifest = _read(checkpoint)
    identity = {"role": role, "training_digest": dataset_digest(texts, targets),
                "checkpoint_digest": _hash(checkpoint, budget),
                "weights_digest": _hash(_safe(checkpoint.parent, manifest["weights"]), budget),
                "training_scope": manifest["training_scope"], "fitted_samples": manifest["fitted_samples"],
                "cache_format": "ragged_turns_global256_fp32_v1", "batch_size": settings["batch_size"]}
    scope = "fit_only" if role == "selection" else "all_training"
    if manifest["training_scope"] != scope:
        raise ValueError("Wrong encoder training scope for feature-cache role")
    groups = [np.arange(len(texts))]
    if role == "selection":
        split_path = root / "training_split.npz"
        identity["split_digest"] = _hash(split_path, budget)
        with np.load(split_path, allow_pickle=False) as split:
            groups = [split["fit"].copy(), split["holdout"].copy()]
        if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(texts))):
            raise ValueError("Feature-cache split must partition all original Training rows")
    cache_dir = Path(settings["work_dir"]) / "feature_caches" / root.name / role
    marker = cache_dir / "complete.json"
    summary = root / "caches" / role / "cache_summary.json"
    if marker.is_file():
        saved = _read(marker)
        if saved.get("identity") != identity:
            raise ValueError("Feature-cache identity changed; start a fresh run")
        _verify_hashes(cache_dir, saved["artifact_hashes"], budget)
        cache = {key: np.load(_safe(cache_dir, key + ".npy"), allow_pickle=False, mmap_mode="r")
                 for key in saved["arrays"]}
        evidence._validate_cache(cache)
        if len(cache["predictions"]) != len(texts):
            raise ValueError("Feature-cache row count mismatch")
        cache["_timing"] = saved.get("timing", {})
        data.write_json(summary, saved)
        return cache
    model, tokenizer, loaded = head.load_model(checkpoint, _device(settings))
    try:
        config = {**loaded, **settings, "encoder_window_batch": loaded["encoder_window_batch"]}
        # Historical holdout predictions used a holdout-only ordered loader.
        # Preserve its batchmates/padding to avoid avoidable FP32 decision drift.
        blocks, timings = [], []
        for indices in groups:
            block = evidence.extract_cache(model, tokenizer, [texts[int(i)] for i in indices], config, _device(settings), budget)
            blocks.append(block)
            timings.append(block.get("_timing", {}))
        cache = evidence.merge_caches(blocks, groups, len(texts))
        del block, blocks
        cache["_timing"] = {"batching": "fit_then_historical_holdout" if role == "selection" else "all_training",
                            "groups": timings}
    finally:
        del model, tokenizer
        _release()
    budget.check("save completed feature cache", training=True)
    saved = _save_feature_cache(cache_dir, cache, identity, budget)
    data.write_json(summary, saved)
    return cache


def _comparison(targets, before, after):
    baseline, current = data.score_predictions(targets, before), data.score_predictions(targets, after)
    if not np.array_equal(before[:, OTHER_LABELS], after[:, OTHER_LABELS]):
        raise ValueError("A supposedly frozen symptom decision changed")
    per_symptom = {}
    for j, label in enumerate(data.TARGET_SYMPTOMS):
        truth, old, new = targets[:, j], before[:, j], after[:, j]
        per_symptom[label] = {"f1_delta": current["per_symptom"][label]["f1"] - baseline["per_symptom"][label]["f1"],
            "paired": {"fn_to_tp": int(((truth == 1) & (old == 0) & (new == 1)).sum()),
                       "tp_to_fn": int(((truth == 1) & (old == 1) & (new == 0)).sum()),
                       "fp_to_tn": int(((truth == 0) & (old == 1) & (new == 0)).sum()),
                       "tn_to_fp": int(((truth == 0) & (old == 0) & (new == 1)).sum())}}
    macro = current["macro_f1"] - baseline["macro_f1"]
    nausea = current["per_symptom"]["오심"]["f1"] - baseline["per_symptom"]["오심"]["f1"]
    return {"available": True, "reference_version": "ver19", "reference_metrics": baseline,
            "new_metrics": current, "macro_f1_delta": macro, "nausea_f1_delta": nausea,
            "both_macro_and_nausea_improved": macro > 0 and nausea > 0,
            "status": "improved" if macro > 0 and nausea > 0 else "not_improved",
            "other_eight_predictions_unchanged": True, "per_symptom": per_symptom}


def _choose(records):
    baseline = records[0]
    improved = [record for record in records[1:]
                if record["metrics"]["macro_f1"] > baseline["metrics"]["macro_f1"]
                and record["metrics"]["per_symptom"]["오심"]["f1"] > baseline["metrics"]["per_symptom"]["오심"]["f1"]]
    return max(improved, key=lambda record: record["metrics"]["macro_f1"]) if improved else baseline


def _parameter_count(state):
    return sum(int(np.asarray(value).size) for value in state.values())


def _save_records(root, records, holdout, targets):
    rows = []
    for record in records:
        destination = root / "experiments" / record["candidate_id"]
        destination.mkdir(parents=True, exist_ok=True)
        data.write_json(destination / "history.json", _serial(record["history"]))
        data.write_json(destination / "training_metrics.json", record["metrics"])
        _save_npz(destination / "evidence.npz", **record["state"])
        _save_npz(destination / "holdout_predictions.npz", indices=holdout, targets=targets[holdout], predictions=record["predictions"])
        rows.append({"candidate": record["candidate_id"], "macro_f1": record["metrics"]["macro_f1"],
                     "nausea_f1": record["metrics"]["per_symptom"]["오심"]["f1"],
                     "selected_epochs": record["epoch"], "stop_reason": record["stop_reason"]})
    with (root / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _predict_loaded(model, tokenizer, texts, manifest, device, budget=None, *, training=False):
    import torch
    from torch.utils.data import DataLoader
    from .neural import EncodeDataset
    dataset = EncodeDataset(texts, None, tokenizer, manifest["max_length"], manifest["overlap"], utterance_evidence=True)
    loader = DataLoader(dataset, batch_size=manifest["batch_size"], shuffle=False, num_workers=0, collate_fn=dataset.collate)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if budget:
                budget.check("actual model inference", training=training)
            logits, _ = model({key: value.to(device) for key, value in batch.items()})
            predictions.append(logits.argmax(-1).cpu().numpy().astype(np.int8))
    if not predictions:
        raise ValueError("Inference requires nonempty text input")
    return np.concatenate(predictions)


def _confirm_selected(source, records, texts, targets, holdout, settings, budget):
    """Confirm the chosen cached candidate through the actual fit-only model."""
    selected = _choose(records)
    audit = {"scope": "Training holdout only", "validation_accessed": False, "candidates": []}
    if selected["candidate_id"] == "baseline":
        audit["status"] = "no_improved_cached_candidate"
        return selected, audit
    budget.check("confirm actual holdout candidate", training=True)
    device = _device(settings)
    model, tokenizer, manifest = head.load_model(source["fit_checkpoint"], device)
    confirmed = set()
    try:
        while selected["candidate_id"] != "baseline" and selected["candidate_id"] not in confirmed:
            budget.check("confirm actual holdout candidate", training=True)
            parameter_audit = evidence.set_evidence_state(model, selected["state"])
            predicted = _predict_loaded(model, tokenizer, [texts[int(i)] for i in holdout],
                                        manifest, device, budget, training=True)
            comparison = _comparison(targets[holdout], records[0]["predictions"], predicted)
            audit["candidates"].append({"candidate_id": selected["candidate_id"],
                "cached_vs_actual_nausea_disagreements": int((selected["predictions"][:, 6] != predicted[:, 6]).sum()),
                "actual_metrics": comparison["new_metrics"], "parameter_audit": parameter_audit})
            selected["cached_metrics"] = selected["metrics"]
            selected["metrics"] = comparison["new_metrics"]
            selected["predictions"] = predicted
            selected["prediction_scope"] = "actual fit-only source model forward"
            confirmed.add(selected["candidate_id"])
            selected = _choose(records)
        audit["status"] = "confirmed" if selected["candidate_id"] != "baseline" else "no_actual_improvement"
        return selected, audit
    finally:
        del model, tokenizer
        _release()


def predict_checkpoint(checkpoint, texts, budget=None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, manifest = head.load_model(checkpoint, device)
    try:
        return _predict_loaded(model, tokenizer, texts, manifest, device, budget)
    finally:
        del model, tokenizer
        _release()


def inference(checkpoint, label_dir, output, budget=None):
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predicted = predict_checkpoint(checkpoint, texts, budget)
    data.write_predictions(paths, predicted, output)
    return predicted


def _export_and_evaluate(source, selected, trained_evidence, settings, root, texts, validation_dir, budget, fallback_reason):
    """Keep original full encoder and global heads; export a complete evidence update."""
    import torch
    device = _device(settings)
    checkpoint = source["full_checkpoint"]
    model, tokenizer, original = head.load_model(checkpoint, device)
    actual_candidate = selected["candidate_id"] if trained_evidence is not None else "baseline"
    try:
        if settings["run_validation"]:
            val_paths, val_texts, truth = data.load_dataset(validation_dir, include_targets=True)
        else:
            val_paths, val_texts, truth = None, texts[:12], None
        config = {**original, **settings, "batch_size": original["batch_size"], "encoder_window_batch": original["encoder_window_batch"]}
        cache = evidence.extract_cache(model, tokenizer, val_texts, config, device, budget, training=False)
        baseline = cache["predictions"].copy()
        if trained_evidence is not None:
            _, cache_predicted = evidence.apply_evidence(cache, trained_evidence, settings, device, budget)
            audit = evidence.set_evidence_state(model, trained_evidence)
            # Export fidelity uses the actual inference device and kernels.
            # Cached evidence optimization and full forward may differ near an argmax tie.
            expected = _predict_loaded(model, tokenizer, val_texts, original, device, budget)
            audit["cache_vs_model_nausea_disagreements"] = int((cache_predicted[:, 6] != expected[:, 6]).sum())
        else:
            expected = baseline.copy()
            audit = {"changed_parameters": [], "all_source_parameters_unchanged": True}
        if not np.array_equal(expected[:, OTHER_LABELS], baseline[:, OTHER_LABELS]):
            raise ValueError("Other-eight symptom preservation failed before export")
        destination = root / "final_model"
        destination.mkdir(parents=True, exist_ok=True)
        source_assets = _safe(checkpoint.parent, original["assets"])
        for path in sorted(source_assets.rglob("*")):
            if path.is_file():
                _copy_file(path, destination / "assets" / path.relative_to(source_assets), budget)
        weight_path = destination / "weights.pt"
        if trained_evidence is None:
            _copy_file(_safe(checkpoint.parent, original["weights"]), weight_path, budget)
        else:
            temporary = weight_path.with_name("weights.pt.tmp")
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
            budget.check("publish completed final evidence")
            temporary.replace(weight_path)
        manifest = {**original, "version": "ver22", "source_version": "ver19",
            "weights": "final_model/weights.pt", "assets": "final_model/assets",
            "selected_candidate": actual_candidate, "evidence_epochs": selected["epoch"] if trained_evidence is not None else 0,
            "evidence_fitted_samples": len(texts) if trained_evidence is not None else 0,
            "evidence_training": {"candidate_id": actual_candidate,
                "learning_rate": settings["evidence_learning_rate"], "batch_size": settings["evidence_batch_size"],
                "weight_decay": settings["evidence_weight_decay"], "positive_weight": settings["positive_weight"],
                "gce_q": settings["gce_q"], "gce_mix": settings["gce_mix"] if actual_candidate == "evidence_ce_gce" else 0,
                "ce_warmup_epochs": settings["ce_warmup_epochs"],
                "encoder_frozen": True, "global_projection_frozen": True, "all_nine_global_heads_frozen": True,
                "evidence_frozen": trained_evidence is None, "other_eight_heads_frozen": True,
                "trainable_parameters": _parameter_count(trained_evidence) if trained_evidence is not None else 0},
            "refit_skipped_reason": fallback_reason, "baseline_reused": trained_evidence is None,
            "source_full_weights_sha256": _hash(_safe(checkpoint.parent, original["weights"]), budget)}
        final_path = root / "mission3_ver22_model.ckpt"
        data.write_json(final_path, manifest)
    finally:
        del model, tokenizer
        _release()
    actual = predict_checkpoint(final_path, val_texts, budget)
    if not np.array_equal(actual, expected):
        raise ValueError("Portable reloaded predictions differ from the in-memory evidence result")
    if not np.array_equal(actual[:, OTHER_LABELS], baseline[:, OTHER_LABELS]):
        raise ValueError("Frozen eight-symptom decisions changed after reload")
    verification = {"samples": len(actual), "decisions": int(actual.size), "portable_reloaded_decisions_match": True,
                    "other_eight_decisions_match_original_full_model": True, "parameter_audit": _serial(audit),
                    "scope": "all Validation rows" if settings["run_validation"] else "Training probe only"}
    data.write_json(root / "preservation_verification.json", verification)
    if settings["run_validation"]:
        data.write_predictions(val_paths, actual, root / "mission3.csv")
        data.write_json(root / "validation_metrics.json", data.score_predictions(truth, actual))
        report = _comparison(truth, baseline, actual)
        report["comparison_scope"] = "Descriptive final Validation only; never used for recipe selection or reverting a trained evidence"
        data.write_json(root / "validation_comparison.json", report)
        _save_npz(root / "validation_predictions.npz", targets=truth, predictions=actual, baseline_predictions=baseline)
    return manifest


def _environment():
    import torch
    from importlib.metadata import version
    return {"packages": {name: version(name) for name in ("torch", "transformers", "numpy", "scikit-learn")},
            "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def run(settings, runtime_sources=None):
    started = time.monotonic()
    settings = validate_settings(settings)
    deadline = settings.get("deadline_monotonic", started + settings["time_budget_minutes"] * 60)
    if deadline > started + settings["time_budget_minutes"] * 60 + 1:
        raise ValueError("Shared deadline exceeds the allowed budget")
    budget = Budget(deadline, settings["finalization_reserve_seconds"], settings["budget_safety_factor"])
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8-sig") for p in Path(__file__).parent.iterdir()
                                if p.is_file() and p.suffix in (".py", ".json", ".md", ".txt")}
    sources = {k: v for k, v in sources.items() if not k.startswith("test_") and k != "build_notebook.py"}
    output = Path(settings["output_dir"]).expanduser().resolve()
    resume = settings.get("resume_run", "")
    root = Path(resume).expanduser().resolve() if resume else output / _stamp()
    if root == output or not root.is_relative_to(output):
        raise ValueError("resume_run must be inside output_dir")
    if resume and not (root / "run_identity.json").is_file():
        raise ValueError("resume_run is not initialized")
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    writable = not bool(resume)
    try:
        budget.check("prepare inputs")
        train_dir, valid_dir = prepare_data(settings["input_zip"], settings["work_dir"], settings.get("train_dir", ""), settings.get("validation_dir", ""))
        _, texts, targets = data.load_dataset(train_dir, include_targets=True)
        fit, holdout = data.make_split(texts, targets, .2, 42)
        identity = {"version": "ver22", "training_digest": dataset_digest(texts, targets),
                    "settings_digest": _digest(_stable_settings(settings)),
                    "source_digest": _digest({k: v for k, v in sources.items() if not k.endswith(".md")})}
        if resume:
            if _read(root / "run_identity.json") != identity:
                raise ValueError("Resume data, settings or source changed")
            with np.load(root / "training_split.npz", allow_pickle=False) as saved:
                if not np.array_equal(fit, saved["fit"]) or not np.array_equal(holdout, saved["holdout"]):
                    raise ValueError("Resume split changed")
        else:
            data.write_json(root / "run_identity.json", identity)
            data.write_json(root / "settings.json", {k: v for k, v in settings.items() if k != "deadline_monotonic"})
            _save_npz(root / "training_split.npz", fit=fit, holdout=holdout)
            for name, content in sources.items():
                if Path(name).name != name:
                    raise ValueError("Embedded source names must be flat")
                destination = root / "submission" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content.encode("utf-8"))
        if (root / "run_complete.json").is_file():
            _verify_hashes(root, _read(root / "run_complete.json")["artifact_hashes"], budget)
            print("Completed ver22 artifacts verified; no additional training.", flush=True)
            return root
        writable = True
        _device(settings)
        data.write_json(root / "runtime.json", {"status": "running", "message": "Only run_complete.json confirms completion"})
        data.write_json(root / "environment.json", _environment())
        source = source_bundle.prepare_source(settings.get("source_run", ""), Path(settings["work_dir"]) / "verified_ver19", budget)
        verified = source_bundle.verify_training(texts, targets, fit, holdout, source)
        data.write_json(root / "source_verified.json", _serial(verified))
        print("Verified original ver19 fit-only and full-Training models. Extracting fixed features.", flush=True)
        cache = _cached_features(source["fit_checkpoint"], texts, targets, settings, root, "selection", budget)
        with np.load(source["root"] / "holdout_predictions.npz", allow_pickle=False) as saved:
            if (not np.array_equal(saved["indices"], holdout) or not np.array_equal(saved["targets"], targets[holdout])
                    or not np.array_equal(saved["predictions"], cache["predictions"][holdout])):
                raise ValueError("Current fit-only encoder does not reproduce the pinned ver19 holdout predictions; no candidate was trained")
        data.write_json(root / "baseline_verification.json", {"samples": len(holdout), "all_nine_decisions_match_pinned_ver19": True})
        data.write_json(root / "pilot_report.json", _serial(cache.get("_timing", {})))
        records = evidence.evidence_candidates(cache, targets, fit, holdout, settings, budget, device=_device(settings))
        search_budget_limited = (records[0].get("search_stop_reason") == "budget"
                                 or any(record["stop_reason"] == "budget" for record in records))
        del cache
        _release()
        try:
            selected, confirmation = _confirm_selected(source, records, texts, targets, holdout, settings, budget)
        except BudgetExceeded:
            search_budget_limited = True
            selected = records[0]
            confirmation = {"status": "budget_before_confirmation", "scope": "Training holdout only",
                            "validation_accessed": False}
        data.write_json(root / "candidate_confirmation.json", _serial(confirmation))
        _save_records(root, records, holdout, targets)
        report = _comparison(targets[holdout], records[0]["predictions"], selected["predictions"])
        data.write_json(root / "training_reference_comparison.json", report)
        data.write_json(root / "training_metrics.json", selected["metrics"])
        _save_npz(root / "holdout_predictions.npz", indices=holdout, targets=targets[holdout], predictions=selected["predictions"])
        improved = report["both_macro_and_nausea_improved"]
        data.write_json(root / "final_selection.json", {"candidate_id": selected["candidate_id"], "selected_epochs": selected["epoch"],
            "metrics": selected["metrics"], "selection_policy": "Training only: jointly improve Macro and nausea F1, then highest Macro F1; exact ties retain earlier candidate",
            "evidence_candidates": [r["candidate_id"] for r in records], "evidence_search_budget_limited": search_budget_limited,
            "both_macro_and_nausea_improved": improved})
        trained_evidence = None
        reason = "budget_before_complete_evidence_search" if search_budget_limited else "training_reference_not_jointly_improved"
        if improved:
            full_cache = None
            try:
                budget.check("begin full-Training evidence refit", training=True)
                full_cache = _cached_features(source["full_checkpoint"], texts, targets, settings, root, "full_refit", budget)
                state, history = evidence.refit_evidence(full_cache, targets, selected["candidate_id"], selected["epoch"], settings, budget,
                                                        device=_device(settings))
                # Publish only a refit that completed all selected epochs on full Training.
                trained_evidence = state
                data.write_json(root / "final_history.json", _serial(history))
                _save_npz(root / "final_evidence.npz", **state)
                reason = None
            except BudgetExceeded:
                reason = "budget_before_complete_evidence_refit"
                trained_evidence = None
                print("Full evidence refit incomplete; preserving the original full ver19 model.", flush=True)
            finally:
                del full_cache
                _release()
        manifest = _export_and_evaluate(source, selected, trained_evidence, settings, root, texts, valid_dir, budget, reason)
        budget.check("complete verified run")
        status = "complete" if trained_evidence is not None else ("complete_budget_limited" if improved or search_budget_limited else "complete_not_improved")
        evidence_sidecars = {"final_evidence.npz", "final_history.json"}
        data.write_json(root / "evidence_artifacts.json", {"active": trained_evidence is not None,
            "active_paths": sorted(evidence_sidecars) if trained_evidence is not None else [],
            "note": "Only active_paths belong to this export; other evidence sidecars may remain from an interrupted earlier attempt."})
        data.write_json(root / "runtime.json", {"status": status, "seconds": time.monotonic() - started,
            "seconds_scope": "worker invocation; shared notebook deadline includes installation", "selected_candidate": selected["candidate_id"],
            "exported_candidate": manifest["selected_candidate"], "baseline_reused": manifest["baseline_reused"],
            "evidence_search_budget_limited": search_budget_limited,
            "both_macro_and_nausea_improved_on_training": improved, "training_scope": "all_training", "fitted_samples": len(texts),
            "refit_skipped_reason": reason, "evidence_trainable_parameters": _parameter_count(records[0]["state"]), "other_eight_preserved": True,
            "budget_minutes_remaining": budget.remaining() / 60})
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "run_complete.json"
                       and not p.name.endswith((".tmp", ".copying"))
                       and (trained_evidence is not None or p.relative_to(root).as_posix() not in evidence_sidecars))
        hashes = {p.relative_to(root).as_posix(): _hash(p, budget) for p in files}
        data.write_json(root / "run_complete.json", {"status": status, "artifact_hashes": hashes})
        print("ver22 artifacts complete:", status, flush=True)
        return root
    except BaseException as error:
        if writable and not (root / "run_complete.json").exists():
            data.write_json(root / "runtime.json", {"status": "budget_exhausted" if isinstance(error, BudgetExceeded) else "failed",
                            "seconds": time.monotonic() - started, "error_type": type(error).__name__,
                            "message": "No completion marker: artifacts are incomplete; keep the previous submission."})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    inference(args.checkpoint, args.label_dir, args.output)


if __name__ == "__main__":
    main()
