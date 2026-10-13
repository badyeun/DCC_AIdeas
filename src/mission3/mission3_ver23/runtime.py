"""Bounded all-label encoder continuation with fixed categorical ensembles."""
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

from . import data, head, adaptation, source_bundle, portability
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


def validate_settings(settings):
    result = dict(settings)
    fixed = {"version": "ver23", "seed": 42, "holdout_size": .2,
             "batch_size": 4, "encoder_window_batch": 4, "max_length": 512, "overlap": 64,
             "trainable_layers": 4, "scheduler_horizon_epochs": 2,
             "refit_full_training": True}
    for key, expected in fixed.items():
        if type(result.get(key)) is not type(expected) or result[key] != expected:
            raise ValueError("Pinned experiment setting changed: " + key)
    for key in ("max_epochs", "gradient_accumulation", "pilot_optimizer_steps", "log_every"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if result["max_epochs"] > 2:
        raise ValueError("This continuation permits at most two added epochs")
    for key in ("learning_rate", "head_learning_rate", "rdrop_alpha", "time_budget_minutes",
                "finalization_reserve_seconds", "budget_safety_factor"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    weight_decay = result.get("weight_decay")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if result.get("positive_class_weights") != list(adaptation.POSITIVE_WEIGHTS):
        raise ValueError("Use the original nine positive class weights")
    if result["time_budget_minutes"] > 180 or result["budget_safety_factor"] < 1:
        raise ValueError("Use at most 180 minutes and safety factor at least one")
    if result["finalization_reserve_seconds"] >= result["time_budget_minutes"] * 60:
        raise ValueError("Finalization reserve must fit inside the shared budget")
    for key in ("allow_cpu", "run_validation"):
        if type(result.get(key)) is not bool:
            raise ValueError(key + " must be boolean")
    for key in ("input_zip", "output_dir", "work_dir", "train_dir", "validation_dir", "source_run", "resume_run"):
        if not isinstance(result.get(key, ""), str):
            raise ValueError(key + " must be a path string")
    forbidden = ("threshold", "calibration", "ensemble_weight", "logit_shift")
    if any(any(token in key.lower() for token in forbidden) for key in result):
        raise ValueError("Decision thresholds, calibration and fitted ensemble weights are not supported")
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


def _comparison(targets, before, after):
    baseline, current = data.score_predictions(targets, before), data.score_predictions(targets, after)
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
    accepted = macro > 0 and nausea >= 0
    return {"available": True, "reference_version": "ver19", "reference_metrics": baseline,
            "new_metrics": current, "macro_f1_delta": macro, "nausea_f1_delta": nausea,
            "acceptance_gate_passed": accepted, "both_macro_and_nausea_improved": macro > 0 and nausea > 0,
            "status": "improved" if accepted else "not_improved", "per_symptom": per_symptom,
            "other_eight_predictions_unchanged": bool(np.array_equal(before[:, OTHER_LABELS], after[:, OTHER_LABELS]))}


def _choose(records):
    baseline = records[0]
    improved = [record for record in records[1:]
                if record["metrics"]["macro_f1"] > baseline["metrics"]["macro_f1"]
                and record["metrics"]["per_symptom"]["오심"]["f1"] >= baseline["metrics"]["per_symptom"]["오심"]["f1"]]
    return max(improved, key=lambda record: record["metrics"]["macro_f1"]) if improved else baseline


def _baseline_record(logits, targets):
    predicted = logits.argmax(-1).astype(np.int8)
    return {"candidate_id": "baseline", "recipe": "baseline", "mode": "baseline", "epoch": 0,
            "metrics": data.score_predictions(targets, predicted), "predictions": predicted,
            "logits": logits, "history": [], "refit_seconds": 0., "stop_reason": "original_ver19"}


def _save_records(root, records, holdout, targets):
    rows = []
    baseline_probabilities = portability.probabilities(records[0]["logits"])
    for record in records:
        destination = root / "experiments" / record["candidate_id"]
        data.write_json(destination / "history.json", _serial(record["history"]))
        data.write_json(destination / "training_metrics.json", record["metrics"])
        probabilities = portability.probabilities(record["logits"])
        if record["mode"] == "mean":
            probabilities = (baseline_probabilities + probabilities) / 2
        if not np.array_equal(probabilities.argmax(-1), record["predictions"]):
            raise ValueError("Saved candidate probabilities do not reproduce its selected predictions")
        _save_npz(destination / "holdout_predictions.npz", indices=holdout, targets=targets[holdout],
                  predictions=record["predictions"], component_logits=record["logits"],
                  probabilities=probabilities,
                  decision_rule=np.asarray("equal_probability_mean" if record["mode"] == "mean" else "single_model_argmax"))
        rows.append({"candidate": record["candidate_id"], "macro_f1": record["metrics"]["macro_f1"],
                     "nausea_f1": record["metrics"]["per_symptom"]["오심"]["f1"],
                     "selected_epochs": record["epoch"], "stop_reason": record["stop_reason"]})
    with (root / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_search(source, texts, targets, fit, holdout, settings, root, budget):
    """Only fit rows update weights; only Training holdout selects candidates."""
    import torch
    from .neural import seed_everything
    held_texts = [texts[int(i)] for i in holdout]
    fit_texts = [texts[int(i)] for i in fit]
    started = time.monotonic()
    baseline_logits = portability.predict_logits(source["fit_checkpoint"], held_texts, budget, training=True)
    evaluation_seconds = time.monotonic() - started
    records = [_baseline_record(baseline_logits, targets[holdout])]
    with np.load(source["root"] / "holdout_predictions.npz", allow_pickle=False) as saved:
        if (not np.array_equal(saved["indices"], holdout) or not np.array_equal(saved["targets"], targets[holdout])
                or not np.array_equal(saved["predictions"], records[0]["predictions"])):
            raise ValueError("Current fit-only encoder does not reproduce the pinned ver19 holdout predictions; no candidate was trained")
    data.write_json(root / "baseline_verification.json", {"samples": len(holdout), "all_nine_decisions_match_pinned_ver19": True,
                                                        "inference_seconds": evaluation_seconds})
    baseline_probabilities = portability.probabilities(baseline_logits)
    limited = False
    device = _device(settings)
    startup_estimate, pilot_estimate = 180., 120.
    for recipe in adaptation.RECIPES:
        model = tokenizer = optimizer_state = loader = None
        history, recipe_records = [], []
        stopped = "max_epochs"
        try:
            retained_refit = _choose(records).get("refit_seconds", 0.)
            # Loading, full-fit tokenization and the pilot happen before its
            # throughput report. Reserve them before spending an existing win.
            if retained_refit and not budget.can_start(
                    retained_refit + startup_estimate * 1.5 + pilot_estimate * 2, training=True):
                raise BudgetExceeded("candidate startup preserves the selected complete refit")
            budget.check("begin continuation candidate", training=True)
            setup_started = time.monotonic()
            seed_everything(settings["seed"])
            model, tokenizer, manifest = head.load_model(source["fit_checkpoint"], device)
            if manifest["training_scope"] != "fit_only":
                raise ValueError("Candidate training must use the fit-only source")
            audit = adaptation.configure_adaptation(model, settings["trainable_layers"])
            frozen_before = adaptation.frozen_digest(model)
            data.write_json(root / "experiments" / recipe / "parameter_audit.json", audit)
            loader = adaptation.make_loader(model, tokenizer, fit_texts, targets[fit], settings, shuffle=True)
            optimizer_state = adaptation.make_optimizer(model, settings, len(loader), device)
            startup_estimate = max(180., time.monotonic() - setup_started)
            for epoch in range(1, settings["max_epochs"] + 1):
                retained_refit = _choose(records).get("refit_seconds", 0.)
                if history:
                    estimate = history[-1]["training_seconds"] + evaluation_seconds + max(
                        retained_refit, history[-1]["training_seconds"] * len(texts) / len(fit) * epoch + 120.)
                    if not budget.can_start(estimate, training=True):
                        raise BudgetExceeded("candidate epoch reserves complete full-data refit")
                row = adaptation.train_epoch(model, loader, settings, device, optimizer_state, budget, epoch, recipe,
                    pilot_context={"holdout_batches": math.ceil(len(holdout) / settings["batch_size"]),
                                   "full_ratio": len(texts) / len(fit), "refit_epochs": epoch,
                                   "minimum_refit_seconds": retained_refit,
                                   "report_paths": [root / "experiments" / recipe / f"pilot_epoch_{epoch}.json",
                                                    root / "pilot_report.json"]})
                if row.get("pilot_report"):
                    pilot_estimate = max(120., row["pilot_report"]["measured_seconds"] * 1.2)
                if row["samples"] != len(fit):
                    raise ValueError("Candidate epoch did not consume every original fit row")
                evaluated = time.monotonic()
                logits = portability.predict_loaded_logits(model, tokenizer, held_texts, manifest, device, budget, training=True)
                evaluation_seconds = time.monotonic() - evaluated
                mixed = (baseline_probabilities + portability.probabilities(logits)) / 2
                row["evaluation_seconds"] = evaluation_seconds
                row["single_metrics"] = data.score_predictions(targets[holdout], logits.argmax(-1))
                row["mean_metrics"] = data.score_predictions(targets[holdout], mixed.argmax(-1))
                history.append(row)
                data.write_json(root / "experiments" / recipe / "history.json", _serial(history))
                for mode, scores in (("single", logits), ("mean", mixed)):
                    candidate_id = recipe + ("_mean" if mode == "mean" else "")
                    predicted = scores.argmax(-1).astype(np.int8)
                    record = {"candidate_id": candidate_id, "recipe": recipe, "mode": mode, "epoch": epoch,
                        "metrics": row[mode + "_metrics"], "predictions": predicted,
                        # Store original logits; mean probabilities are saved explicitly below.
                        "logits": logits, "history": history, "stop_reason": "max_epochs",
                        "refit_seconds": row["training_seconds"] * len(texts) / len(fit) * epoch + 120.}
                    # Keep every completed epoch eligible: an ineligible later nausea regression
                    # must never hide an earlier candidate that passed both selection conditions.
                    records.append(record)
                    recipe_records.append(record)
                    _save_npz(root / "experiments" / candidate_id / f"epoch_{epoch}_holdout.npz",
                        indices=holdout, targets=targets[holdout], predictions=predicted,
                        component_logits=logits, probabilities=mixed if mode == "mean" else portability.probabilities(logits),
                        decision_rule=np.asarray("equal_probability_mean" if mode == "mean" else "single_model_argmax"))
                print(f"{recipe} epoch {epoch}: Macro F1 {row['single_metrics']['macro_f1']:.6f}; "
                      f"fixed mean {row['mean_metrics']['macro_f1']:.6f}", flush=True)
            if adaptation.frozen_digest(model) != frozen_before:
                raise ValueError("Frozen encoder prefix changed during candidate training")
        except BudgetExceeded:
            limited, stopped = True, "budget"
            print(f"{recipe}: stopped before an incomplete epoch could become a candidate.", flush=True)
        finally:
            for record in recipe_records:
                record["stop_reason"] = stopped
            data.write_json(root / "experiments" / recipe / "search_status.json",
                            {"stop_reason": stopped, "completed_epochs": len(history), "fit_rows": len(fit),
                             "partial_epoch_published": False})
            del model, tokenizer, optimizer_state, loader
            _release()
        if limited:
            break
    return records, limited


def _refit(source, selected, texts, targets, settings, root, budget):
    from .neural import seed_everything
    device = _device(settings)
    if not budget.can_start(selected["refit_seconds"], training=True):
        raise BudgetExceeded("admit complete full-Training continuation")
    seed_everything(settings["seed"])
    model, tokenizer, manifest = head.load_model(source["full_checkpoint"], device)
    optimizer_state = loader = None
    try:
        if manifest["training_scope"] != "all_training" or manifest["fitted_samples"] != len(texts):
            raise ValueError("Full refit requires the separate all-Training ver19 source")
        audit = adaptation.configure_adaptation(model, settings["trainable_layers"])
        frozen_before = adaptation.frozen_digest(model)
        loader = adaptation.make_loader(model, tokenizer, texts, targets, settings, shuffle=True)
        optimizer_state = adaptation.make_optimizer(model, settings, len(loader), device)
        history = []
        for epoch in range(1, selected["epoch"] + 1):
            row = adaptation.train_epoch(model, loader, settings, device, optimizer_state, budget, epoch, selected["recipe"],
                pilot_context={"refit": True, "refit_epochs": selected["epoch"],
                               "remaining_epochs": selected["epoch"] - epoch + 1,
                               "report_paths": [root / f"refit_pilot_epoch_{epoch}.json"]})
            if row["samples"] != len(texts):
                raise ValueError("Full refit epoch did not consume every original Training row")
            history.append(row)
            data.write_json(root / "final_history.json", _serial(history))
        if adaptation.frozen_digest(model) != frozen_before:
            raise ValueError("Frozen encoder prefix changed during full refit")
        audit.update({"frozen_state_unchanged": True, "frozen_sha256": frozen_before,
                      "training_scope": "all_training", "fitted_samples": len(texts), "added_epochs": selected["epoch"],
                      "source_checkpoint": "separate original full-Training ver19; no fit checkpoint transplant"})
        data.write_json(root / "final_parameter_audit.json", audit)
        return adaptation.trainable_state(model)
    finally:
        del model, tokenizer, optimizer_state, loader
        _release()


def _export_and_evaluate(source, selected, trained_state, settings, root, texts, validation_dir, budget, fallback_reason):
    """Validation is descriptive, after selection; it cannot trigger model replacement."""
    device = _device(settings)
    if settings["run_validation"]:
        val_paths, val_texts, truth = data.load_dataset(validation_dir, include_targets=True)
    else:
        val_paths, val_texts, truth = None, texts[:12], None
    baseline_logits = portability.predict_logits(source["full_checkpoint"], val_texts, budget)
    baseline = baseline_logits.argmax(-1).astype(np.int8)
    expected = baseline.copy()
    if trained_state is not None:
        model, tokenizer, original = head.load_model(source["full_checkpoint"], device)
        try:
            adaptation.configure_adaptation(model, settings["trainable_layers"])
            adaptation.restore_trainable_state(model, trained_state)
            logits = portability.predict_loaded_logits(model, tokenizer, val_texts, original, device, budget)
            expected = ((portability.probabilities(logits) + portability.probabilities(baseline_logits)) / 2
                        if selected["mode"] == "mean" else logits).argmax(-1).astype(np.int8)
        finally:
            del model, tokenizer
            _release()
    final_path, manifest = portability.export(source, selected, trained_state, settings, root, budget, fallback_reason)
    actual = portability.predict_checkpoint(final_path, val_texts, budget)
    if not np.array_equal(actual, expected):
        raise ValueError("Portable reloaded predictions differ from the in-memory final result")
    data.write_json(root / "export_verification.json", {"passed": True, "samples": len(val_texts),
        "all_nine_predictions_match": True, "model_reloaded_from_saved_files": True,
        "prediction_rule": "equal probability mean then argmax" if trained_state is not None and selected["mode"] == "mean" else "argmax"})
    if truth is not None:
        data.write_predictions(val_paths, actual, root / "validation_predictions.csv")
        data.write_predictions(val_paths, actual, root / "mission3.csv")
        data.write_json(root / "validation_metrics.json", data.score_predictions(truth, actual))
        report = _comparison(truth, baseline, actual)
        report["comparison_scope"] = "Descriptive final Validation only; never used for recipe selection or reverting a trained model"
        data.write_json(root / "validation_comparison.json", report)
        _save_npz(root / "validation_predictions.npz", targets=truth, predictions=actual, baseline_predictions=baseline)
    return manifest


def predict_checkpoint(checkpoint, texts, budget=None):
    return portability.predict_checkpoint(checkpoint, texts, budget)


def inference(checkpoint, label_dir, output, budget=None):
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predicted = predict_checkpoint(checkpoint, texts, budget)
    data.write_predictions(paths, predicted, output)
    return predicted


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
    if resume and not (root / "run_complete.json").is_file():
        raise ValueError("Only a completed run can be resumed; clear resume_run to start a fresh experiment")
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    writable = not bool(resume)
    try:
        budget.check("prepare inputs")
        train_dir, valid_dir = prepare_data(settings["input_zip"], settings["work_dir"], settings.get("train_dir", ""), settings.get("validation_dir", ""))
        _, texts, targets = data.load_dataset(train_dir, include_targets=True)
        fit, holdout = data.make_split(texts, targets, .2, 42)
        identity = {"version": "ver23", "training_digest": dataset_digest(texts, targets),
                    "settings_digest": _digest(_stable_settings(settings)),
                    "source_digest": _digest({k: v for k, v in sources.items() if not k.endswith(".md")})}
        if resume:
            if _read(root / "run_identity.json") != identity:
                raise ValueError("Resume data, settings or source changed")
            _verify_hashes(root, _read(root / "run_complete.json")["artifact_hashes"], budget)
            print("Completed ver23 artifacts verified; no additional training.", flush=True)
            return root
        data.write_json(root / "run_identity.json", identity)
        data.write_json(root / "settings.json", {k: v for k, v in settings.items() if k != "deadline_monotonic"})
        _save_npz(root / "training_split.npz", fit=fit, holdout=holdout)
        for name, content in sources.items():
            if Path(name).name != name:
                raise ValueError("Embedded source names must be flat")
            destination = root / "submission" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content.encode("utf-8"))
        _device(settings)
        data.write_json(root / "runtime.json", {"status": "running", "message": "Only run_complete.json confirms completion"})
        data.write_json(root / "environment.json", _environment())
        source = source_bundle.prepare_source(settings.get("source_run", ""), Path(settings["work_dir"]) / "verified_ver19", budget)
        verified = source_bundle.verify_training(texts, targets, fit, holdout, source)
        data.write_json(root / "source_verified.json", _serial(verified))
        print("Verified original fit-only and full-Training ver19 models. Starting bounded continuation.", flush=True)
        records, limited = _candidate_search(source, texts, targets, fit, holdout, settings, root, budget)
        selected = _choose(records)
        # Save one representative per mode while retaining every epoch's separate NPZ/history.
        representatives = [records[0]]
        for candidate_id in dict.fromkeys(r["candidate_id"] for r in records[1:]):
            candidates = [r for r in records[1:] if r["candidate_id"] == candidate_id]
            eligible = [r for r in candidates if _choose([records[0], r]) is r]
            representatives.append(max(eligible or candidates, key=lambda r: r["metrics"]["macro_f1"]))
        _save_records(root, representatives, holdout, targets)
        report = _comparison(targets[holdout], records[0]["predictions"], selected["predictions"])
        data.write_json(root / "training_reference_comparison.json", report)
        data.write_json(root / "training_metrics.json", selected["metrics"])
        _save_npz(root / "holdout_predictions.npz", indices=holdout, targets=targets[holdout], predictions=selected["predictions"])
        improved = selected["candidate_id"] != "baseline"
        data.write_json(root / "final_selection.json", {"candidate_id": selected["candidate_id"], "recipe": selected["recipe"],
            "mode": selected["mode"], "selected_epochs": selected["epoch"], "added_epochs": selected["epoch"],
            "metrics": selected["metrics"], "selection_policy": "Training only: Macro improves and nausea does not decrease, then highest Macro; exact ties retain earlier candidate",
            "search_budget_limited": limited, "acceptance_gate_passed": improved, "validation_accessed": False,
            "both_macro_and_nausea_improved": report["both_macro_and_nausea_improved"],
            "holdout_is_reused_exploratory_data": True})
        trained_state = None
        reason = "budget_before_complete_search" if limited else "training_reference_not_improved"
        if improved:
            try:
                trained_state = _refit(source, selected, texts, targets, settings, root, budget)
                reason = None
            except BudgetExceeded:
                reason = "budget_before_complete_refit"
                print("Full continuation refit incomplete; preserving the original full ver19 model.", flush=True)
        manifest = _export_and_evaluate(source, selected, trained_state, settings, root, texts, valid_dir, budget, reason)
        status = "complete" if trained_state is not None else ("complete_budget_limited" if improved or limited else "complete_not_improved")
        data.write_json(root / "runtime.json", {"status": status, "seconds": time.monotonic() - started,
            "seconds_scope": "worker invocation; shared notebook deadline includes installation",
            "selected_candidate": selected["candidate_id"], "exported_candidate": manifest["selected_candidate"],
            "baseline_reused": manifest["baseline_reused"], "search_budget_limited": limited,
            "training_acceptance_gate_passed": improved, "training_scope": "all_training", "fitted_samples": len(texts),
            "added_epochs": selected["epoch"] if trained_state is not None else 0,
            "refit_skipped_reason": reason, "budget_minutes_remaining": budget.remaining() / 60})
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "run_complete.json"
                       and not p.name.endswith((".tmp", ".copying")))
        hashes = {p.relative_to(root).as_posix(): _hash(p, budget) for p in files}
        budget.check("complete verified run")
        data.write_json(root / "run_complete.json", {"status": status, "artifact_hashes": hashes})
        print("ver23 artifacts complete:", status, flush=True)
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
