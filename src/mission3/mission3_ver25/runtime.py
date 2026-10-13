"""Fresh QLoRA/control comparison with a frozen confirmation protocol."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np

from . import analysis, batching, data, trainer
from .bootstrap import prepare_data
from .budget import Budget, BudgetExceeded

ORDER = ("qwen", "control")
FORMAT = "mission3_ver25_pointer_v1"
EXPECTED_TRAINING_SAMPLES = 29200
PILOT_FORMAT = "mission3_ver25_pilot_v1"


class _TrainingBudget:
    def __init__(self, budget):
        self.budget = budget

    def check(self, stage, training=False):
        return self.budget.check(stage, training=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_settings(settings):
    from .build_notebook import default_settings
    defaults = default_settings()
    result = dict(settings)
    flexible = {"input_zip", "output_dir", "work_dir", "train_dir", "validation_dir",
                "run_validation", "max_epochs", "time_budget_minutes", "run_mode", "source_run",
                "finalization_reserve_seconds", "budget_safety_factor", "log_every", "allow_cpu"}
    unknown = set(result) - set(defaults) - {"deadline_monotonic", "invocation_started_monotonic"}
    if unknown:
        raise ValueError("Unknown experiment settings: " + ", ".join(sorted(unknown)))
    for key, value in defaults.items():
        unlimited_budget = key == "time_budget_minutes" and key in result and result[key] is None
        if key not in result or (not unlimited_budget and type(result[key]) is not type(value)):
            raise ValueError("Missing setting or invalid type: " + key)
        if key not in flexible and result[key] != value:
            raise ValueError("Pinned experiment setting changed: " + key)
    if not 1 <= result["max_epochs"] <= 3:
        raise ValueError("Use 1-3 pilot epochs")
    if result["run_mode"] not in ("pilot", "full"):
        raise ValueError("run_mode must be pilot or full")
    if result["run_mode"] == "full" and not result["source_run"].strip():
        raise ValueError("Full training requires a completed pilot source_run")
    if result["run_mode"] == "pilot" and result["source_run"]:
        raise ValueError("Set source_run only for a separate full run")
    minutes = result["time_budget_minutes"]
    if ((minutes is not None and (minutes <= 0
                                 or result["finalization_reserve_seconds"] >= 60 * minutes))
            or not 1200 <= result["finalization_reserve_seconds"] <= 1800
            or not math.isfinite(result["budget_safety_factor"])
            or result["budget_safety_factor"] < 1.25 or result["log_every"] < 1):
        raise ValueError("Invalid time budget or reserve")
    for key in ("deadline_monotonic", "invocation_started_monotonic"):
        if key not in result:
            continue
        timestamp = result[key]
        if key == "deadline_monotonic" and minutes is None and timestamp is None:
            continue
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Invalid invocation timestamp: " + key)
    return result


def dataset_digest(texts, targets):
    return data.dataset_digest(texts, targets)


def _hash(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if budget:
                budget.check("artifact hash")
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device(settings):
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if not settings["allow_cpu"]:
        raise RuntimeError("CUDA GPU required for the QLoRA experiment")
    return "cpu"


def resolve_specs(cache_dir, budget, source_specs=None):
    """Resolve mutable references once, then use immutable snapshots everywhere."""
    from huggingface_hub import HfApi, snapshot_download
    specs = (read_json(Path(__file__).with_name("model_specs.json")) if source_specs is None
             else {kind: dict(source_specs[kind]) for kind in ORDER})
    for name in ORDER:
        budget.check("resolve pretrained revision", training=True)
        spec = specs[name]
        revision = spec["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            revision = HfApi().model_info(spec["model_id"], revision=revision).sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pretrained model revision is not immutable")
        snapshot = snapshot_download(spec["model_id"], revision=revision, cache_dir=str(cache_dir),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
            ignore_patterns=["original/*", "onnx/*", "openvino/*"])
        spec.update(kind=name, revision=revision, snapshot_path=str(snapshot))
        budget.check("pretrained download", training=True)
    return specs


def _loader(kind, tokenizer, texts, targets, indices, settings, *, shuffle=False, device="cpu"):
    return trainer.make_loader(tokenizer, texts, targets=targets, indices=indices, kind=kind,
        batch_size=settings[kind + "_batch_size"], max_length=settings[kind + "_max_length"],
        shuffle=shuffle, seed=settings["seed"], pin_memory=device == "cuda")


def _optimizer(kind, model, loader, horizon, settings, device):
    optimizer = trainer.build_optimizer(model, kind, settings)
    updates = math.ceil(len(loader) / settings[kind + "_gradient_accumulation"]) * horizon
    scheduler = trainer.build_scheduler(optimizer, updates, settings["warmup_ratio"])
    return optimizer, scheduler, trainer.make_scaler(device)


def _progress(name, every):
    def report(progress):
        if progress["successful_updates"] % every == 0:
            print(f"{name}: {progress['examples']} rows; {progress['elapsed_seconds'] / 60:.1f} min", flush=True)
    return report


def timing_estimates(probes, fit_count, full_count, development_count, confirmation_count, settings):
    estimates = {}
    for kind in ORDER:
        probe = probes[kind]
        step, predict = probe["seconds_per_example"], probe["seconds_per_prediction"]
        if not all(math.isfinite(x) and x > 0 for x in (step, predict)):
            raise ValueError("Pilot timings must be finite and positive")
        batch = settings[kind + "_batch_size"]
        estimates[kind] = {"fit_epoch_seconds": step * math.ceil(fit_count / batch) * batch,
            "development_seconds": predict * development_count,
            "confirmation_seconds": predict * confirmation_count,
            "full_epoch_seconds": step * math.ceil(full_count / batch) * batch,
            "setup_seconds": max(30., probe.get("setup_seconds", 30.))}
    return estimates


def plan_epochs(probes, fit_count, full_count, development_count, confirmation_count, settings, budget):
    """Admit only the requested small comparison, never a full-data refit."""
    estimates = timing_estimates(probes, fit_count, full_count, development_count, confirmation_count, settings)
    for epochs in range(settings["max_epochs"], 0, -1):
        comparison = sum((p["fit_epoch_seconds"] + p["development_seconds"]) * epochs
            + 2 * p["setup_seconds"] + 60 * epochs for p in estimates.values())
        if budget.can_start(comparison, training=True):
            return {"common_epochs": epochs, "estimates": estimates,
                "full_refit_planned": False, "confirmation_planned": False,
                "estimated_work_seconds": comparison, "comparison_seconds": comparison,
                "fitted_samples": fit_count, "remaining_seconds": budget.remaining(),
                "safety_factor": budget.safety_factor,
                "finalization_reserve_seconds": budget.finalization_reserve_s}
    raise BudgetExceeded("one complete subset epoch and development evaluation do not fit")


def estimate_full_run(estimates, selection, settings, *, records=None, fit_count=None, full_count=None,
                      preparation_seconds=0.):
    selected = selection["components"]
    refit = 0.
    for kind in selected:
        epoch_seconds = estimates[kind]["full_epoch_seconds"]
        if records is not None:
            epoch_seconds = max(epoch_seconds, records[kind]["actual_epoch_seconds"] * full_count / fit_count)
        refit += epoch_seconds * selection["selected_epochs"][kind] + estimates[kind]["setup_seconds"]
    confirmation = sum(p["confirmation_seconds"] + p["setup_seconds"] for p in estimates.values())
    # Use development-size inference as a conservative proxy without opening official Validation.
    validation = sum(estimates[kind]["development_seconds"] for kind in selected)
    raw = refit + confirmation + validation + 120 * len(selected)
    if not math.isfinite(preparation_seconds) or preparation_seconds < 0:
        raise ValueError("Preparation estimate must be finite and nonnegative")
    return {"selected_candidate": selection["selected_candidate"], "selected_epochs": selection["selected_epochs"],
        "refit_seconds": refit, "confirmation_seconds": confirmation,
        "validation_proxy_seconds": validation, "estimated_work_seconds": raw,
        "estimated_preparation_seconds": preparation_seconds,
        "recommended_budget_minutes": math.ceil(((raw + preparation_seconds) * settings["budget_safety_factor"]
            + settings["finalization_reserve_seconds"]) / 60),
        "safety_factor": settings["budget_safety_factor"],
        "notes": "Estimate only; includes measured preparation/batch-search time, safety margin and export reserve. "
                 "Future source verification, installation/download and GPU speeds can differ. "
                 "Batch search repeats on the future GPU. No full run starts automatically."}


def choose_candidate(scores, records, min_macro_gain=.01):
    # Stable tie order prefers the simpler single model, with control first.
    selected = max(("control", "qwen", "ensemble"), key=lambda name: scores[name]["macro_f1"])
    gain = scores[selected]["macro_f1"] - scores["control"]["macro_f1"]
    components = ["control", "qwen"] if selected == "ensemble" else [selected]
    return {"selected_candidate": selected, "components": components,
        "selected_epochs": {name: records[name]["selected_epochs"] for name in components},
        "selection_data": "Training development only", "confirmation_can_change_selection": False,
        "macro_f1_delta": gain, "improved": gain > 0,
        "expansion_recommended": gain >= min_macro_gain,
        "expansion_macro_gain": min_macro_gain,
        "rule": "highest development Macro F1; ties control then qwen then fixed mean; no nausea veto",
        "historical_best_comparison": "not evaluated; this control is a fresh single KLUE model"}


def _fit_candidate(kind, spec, texts, targets, fit, development, settings, plan, device, budget, root):
    budget.check("candidate load", training=True)
    model, tokenizer = trainer.create_model(spec, settings, device)
    loader = _loader(kind, tokenizer, texts, targets, fit, settings, shuffle=True, device=device)
    validation = _loader(kind, tokenizer, texts, None, development, settings, device=device)
    optimizer, scheduler, scaler = _optimizer(kind, model, loader, plan["common_epochs"], settings, device)
    weights = trainer.positive_weights(targets[fit])
    best, best_logits, history = None, None, []
    for epoch in range(1, plan["common_epochs"] + 1):
        print(f"{kind}: fit epoch {epoch}/{plan['common_epochs']}", flush=True)
        stats = trainer.train_epoch(model, loader, optimizer, device=device, pos_weight=weights,
            accumulation_steps=settings[kind + "_gradient_accumulation"], scheduler=scheduler,
            scaler=scaler, budget=budget, on_update=_progress(kind, settings["log_every"]))
        if not stats["completed_epoch"]:
            raise RuntimeError("Candidate epoch did not finish")
        evaluation_started = time.monotonic()
        logits = trainer.predict_logits(model, validation, device=device, budget=_TrainingBudget(budget))
        evaluation_seconds = time.monotonic() - evaluation_started
        score = analysis.metrics(targets[development], logits)
        record = {"candidate": kind, "macro_f1": score["macro_f1"], "selected_epochs": epoch,
            "train_seconds": stats["elapsed_seconds"], "peak_vram_bytes": stats.get("peak_vram_bytes"),
            "evaluation_seconds": evaluation_seconds, "overflow_retries": stats.get("overflow_retries", 0),
            "fitted_samples": len(fit), "positive_weights": weights.tolist()}
        history.append({**record, "metrics": score})
        if best is None or score["macro_f1"] > best["macro_f1"]:
            best, best_logits = record, logits.copy()
            budget.check("save fit checkpoint")
            trainer.save_model(root / ("fit_" + kind) / ("epoch_" + str(epoch)),
                               model, tokenizer, spec, settings, "fit_only", epoch)
            data.write_json(root / (kind + "_development_metrics.json"), score)
            _save_npz(root / (kind + "_development_predictions.npz"), indices=development,
                targets=targets[development], logits=best_logits, predictions=trainer.fixed_decisions(best_logits))
        data.write_json(root / (kind + "_history.json"), history)
        print(f"{kind}: development Macro F1={score['macro_f1']:.6f}", flush=True)
    best["actual_epoch_seconds"] = max(row["train_seconds"] for row in history)
    del model, tokenizer, loader, validation, optimizer, scheduler, scaler
    _release()
    return best, best_logits


def _compare(root, stage, targets, indices, logits, budget):
    logits = dict(logits)
    logits["ensemble"] = analysis.ensemble_logits(logits["control"], logits["qwen"])
    scores = {name: analysis.metrics(targets[indices], values) for name, values in logits.items()}
    changes = {name: analysis.paired_error_changes(targets[indices], logits["control"], logits[name])
               for name in ("qwen", "ensemble")}
    intervals = {}
    for name in ("qwen", "ensemble"):
        budget.check("paired call bootstrap", training=True)
        intervals[name] = analysis.paired_bootstrap_ci(targets[indices], logits["control"], logits[name])
    data.write_json(root / (stage + "_comparison.json"), {"metrics": scores,
        "paired_error_changes": changes, "paired_bootstrap": intervals,
        "bootstrap_unit": "call", "decision": "fixed sigmoid >= 0.5",
        "ensemble": "fixed 1:1 probability mean, mathematical tie handled by logit sum sign"})
    _save_npz(root / (stage + "_predictions.npz"), indices=indices, targets=targets[indices],
              **{name + "_logits": values for name, values in logits.items()})
    return scores


def _confirmation(root, texts, targets, indices, records, settings, device, budget, *, source_root=None):
    logits = {}
    source_root = root if source_root is None else source_root
    for kind in ORDER:
        budget.check("confirmation reload", training=True)
        model, tokenizer, _ = trainer.load_model(source_root / ("fit_" + kind)
            / ("epoch_" + str(records[kind]["selected_epochs"])), device)
        loader = _loader(kind, tokenizer, texts, None, indices, settings, device=device)
        logits[kind] = trainer.predict_logits(model, loader, device=device, budget=_TrainingBudget(budget))
        del model, tokenizer, loader
        _release()
    return _compare(root, "confirmation", targets, indices, logits, budget)


def _refit(kind, spec, epochs, texts, targets, settings, plan, device, budget, root):
    budget.check("full Training refit load", training=True)
    model, tokenizer = trainer.create_model(spec, settings, device)
    loader = _loader(kind, tokenizer, texts, targets, None, settings, shuffle=True, device=device)
    optimizer, scheduler, scaler = _optimizer(kind, model, loader, plan["common_epochs"], settings, device)
    weights = trainer.positive_weights(targets)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"{kind}: full Training epoch {epoch}/{epochs}", flush=True)
        stats = trainer.train_epoch(model, loader, optimizer, device=device, pos_weight=weights,
            accumulation_steps=settings[kind + "_gradient_accumulation"], scheduler=scheduler,
            scaler=scaler, budget=budget, on_update=_progress("refit " + kind, settings["log_every"]))
        if not stats["completed_epoch"] or stats["examples"] != len(texts):
            raise RuntimeError("Full Training refit did not cover all Training rows")
        history.append({"epoch": epoch, "examples": stats["examples"],
            "seconds": stats["elapsed_seconds"], "peak_vram_bytes": stats.get("peak_vram_bytes"),
            "overflow_retries": stats.get("overflow_retries", 0)})
        data.write_json(root / (kind + "_final_history.json"), history)
    directory = root / ("final_" + kind)
    budget.check("export final component")
    trainer.save_model(directory, model, tokenizer, spec, settings, "all_training", epochs)
    sample = [texts[i] for i in np.linspace(0, len(texts) - 1, min(16, len(texts)), dtype=int)]
    expected = trainer.predict_logits(model, _loader(kind, tokenizer, sample, None, None, settings),
                                      device=device, budget=budget)
    del model, tokenizer, loader, optimizer, scheduler, scaler
    _release()
    budget.check("reload final component")
    model, tokenizer, metadata = trainer.load_model(directory, device)
    if metadata["training_scope"] != "all_training":
        raise ValueError("Final component is not a full Training model")
    actual = trainer.predict_logits(model, _loader(kind, tokenizer, sample, None, None, settings),
                                    device=device, budget=budget)
    verified = (np.allclose(expected, actual, rtol=1e-4, atol=1e-4)
                and np.array_equal(trainer.fixed_decisions(expected), trainer.fixed_decisions(actual)))
    if not verified:
        raise ValueError("Locally reloaded checkpoint predictions differ")
    data.write_json(root / (kind + "_portable_verification.json"), {"samples": len(sample),
        "decisions_equal": True, "local_files_only": True,
        "max_abs_logit_difference": float(np.max(abs(expected - actual))),
        "positive_weights": weights.tolist(), "fitted_samples": len(texts),
        "scheduler_horizon_epochs": plan["common_epochs"]})
    del model, tokenizer
    _release()
    return {"kind": kind, "model_directory": directory.name, "selected_epochs": epochs}


def _sources(root, runtime_sources):
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.iterdir()
        if p.suffix in {".py", ".txt", ".json", ".md"} and not p.name.startswith("test_")}
    submission = root / "submission"
    submission.mkdir(exist_ok=False)
    for name, source in sources.items():
        if Path(name).name != name:
            raise ValueError("Unsafe source bundle name")
        (submission / name).write_text(source, encoding="utf-8")
    packages = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "scikit-learn")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    data.write_json(root / "environment.json", versions)


def _diagnostic(root, status, stage, started, records=None):
    saved_fit = sorted(p.parent.relative_to(root).as_posix()
                       for p in root.glob("fit_*/epoch_*/manifest.json"))
    if saved_fit and status == "budget_not_started":
        status = "fit_only_budget"
    result = {"version": "ver25", "status": status, "stage": stage,
        "training_scope": "fit_only" if records or saved_fit else "none", "submission_ready": False,
        "elapsed_worker_seconds": time.monotonic() - started,
        "completed_fit_candidates": list(records or {}), "saved_fit_checkpoints": saved_fit,
        "validation_evaluated": (root / "validation_metrics.json").is_file()}
    data.write_json(root / "runtime.json", result)
    data.write_json(root / "diagnostic_complete.json", result)
    return root


def _verify_pilot_exports(root, texts, development, records, logits, settings, device, budget):
    verification = {}
    indices = development[:16]
    for kind in ORDER:
        budget.check("reload pilot checkpoint")
        model, tokenizer, metadata = trainer.load_model(root / ("fit_" + kind)
            / ("epoch_" + str(records[kind]["selected_epochs"])), device)
        if metadata["training_scope"] != "fit_only":
            raise ValueError("Pilot component has an unexpected training scope")
        actual = trainer.predict_logits(model,
            _loader(kind, tokenizer, texts, None, indices, settings, device=device),
            device=device, budget=budget)
        expected = logits[kind][:len(indices)]
        if (not np.allclose(expected, actual, rtol=1e-4, atol=1e-4)
                or not np.array_equal(trainer.fixed_decisions(expected), trainer.fixed_decisions(actual))):
            raise ValueError("Locally reloaded pilot predictions differ")
        verification[kind] = {"samples": len(indices), "decisions_equal": True,
            "local_files_only": True, "max_abs_logit_difference": float(np.max(abs(expected - actual)))}
        del model, tokenizer
        _release()
    data.write_json(root / "pilot_portable_verification.json", verification)


def _load_pilot_source(source_run, texts, targets, fit, development, confirmation, subset,
                       split_report, subset_report, settings, budget):
    source = Path(source_run).expanduser().resolve()
    marker_path = source / "pilot_complete.json"
    if not marker_path.is_file():
        raise ValueError("source_run must contain a completed pilot_complete.json; incomplete runs cannot start full training")
    marker = read_json(marker_path)
    if (marker.get("format") != PILOT_FORMAT or marker.get("status") != "pilot_complete"
            or marker.get("training_scope") != "training_subset" or marker.get("submission_ready") is not False
            or marker.get("confirmation_evaluated") is not False or marker.get("validation_evaluated") is not False):
        raise ValueError("Invalid completed pilot source")
    hashes = marker.get("hashes", {})
    required = {"pilot_recipe.json", "final_selection.json", "model_specs.json", "settings.json",
        "training_data.json", "split.npz", "split_manifest.json", "subset.npz", "subset_manifest.json",
        "pilot_portable_verification.json", "development_comparison.json"}
    if not required.issubset(hashes):
        raise ValueError("Pilot completion marker is missing required artifact hashes")
    for name, expected in hashes.items():
        relative = Path(name)
        path = source / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(source) or not path.is_file()
                or _hash(path, budget) != expected):
            raise ValueError("Pilot source artifact is missing or changed: " + name)
    previous = validate_settings(read_json(source / "settings.json"))
    if previous["run_mode"] != "pilot":
        raise ValueError("source_run must be a pilot run")
    recipe = read_json(source / "pilot_recipe.json")
    selection = read_json(source / "final_selection.json")
    specs = read_json(source / "model_specs.json")
    if (recipe.get("format") != PILOT_FORMAT or recipe.get("target_symptoms") != list(data.TARGET_SYMPTOMS)
            or recipe.get("training_dataset_digest") != dataset_digest(texts, targets)
            or recipe.get("selection") != selection
            or recipe.get("common_epochs") != selection.get("common_epochs")
            or not 1 <= recipe.get("common_epochs", 0) <= previous["max_epochs"]):
        raise ValueError("Pilot recipe or Training data does not match this full run")
    for name, current in (("split_manifest.json", split_report), ("subset_manifest.json", subset_report)):
        if read_json(source / name) != current:
            raise ValueError("Pilot split or subset does not match current Training data")
    with np.load(source / "split.npz", allow_pickle=False) as saved:
        if any(not np.array_equal(saved[key], values)
               for key, values in (("fit", fit), ("development", development), ("confirmation", confirmation))):
            raise ValueError("Pilot split indices changed")
    with np.load(source / "subset.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["indices"], subset):
            raise ValueError("Pilot subset indices changed")
    records = recipe.get("records", {})
    scores = read_json(source / "development_comparison.json")["metrics"]
    expected_selection = choose_candidate(scores, records, previous["min_macro_gain"])
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("Pilot selection differs from the frozen development decision")
    if selection.get("training_dataset_digest") != recipe["training_dataset_digest"]:
        raise ValueError("Pilot selection data digest differs")
    for kind in ORDER:
        epoch = records[kind]["selected_epochs"]
        if type(epoch) is not int or not 1 <= epoch <= recipe["common_epochs"]:
            raise ValueError("Invalid selected pilot epoch")
        if (f"fit_{kind}/epoch_{epoch}/manifest.json" not in hashes
                or not re.fullmatch(r"[0-9a-f]{40}", specs[kind]["revision"])):
            raise ValueError("Pilot source lacks a selected checkpoint or pinned pretrained revision")
    # validate_settings pins every model/data hyperparameter in both runs. Only
    # the new run's paths, timing budget, hardware, and measured batches change.
    return source, recipe, specs


def run(settings, runtime_sources=None):
    settings = validate_settings(settings)
    started = time.monotonic()
    minutes = settings["time_budget_minutes"]
    deadline = (None if minutes is None else
                min(float(settings.get("deadline_monotonic", started + minutes * 60)), started + minutes * 60))
    invocation_started = settings.get("invocation_started_monotonic",
                                      started if deadline is None else deadline - minutes * 60)
    budget = Budget(deadline, settings["finalization_reserve_seconds"], settings["budget_safety_factor"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    root = Path(settings["output_dir"]).expanduser().resolve() / stamp
    root.mkdir(parents=True, exist_ok=False)
    data.write_json(root / "settings.json", settings)
    settings = {**settings, "base_store_dir": str(root / "shared_base")}
    stage, records = "prepare", {}
    data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
    try:
        _sources(root, runtime_sources)
        device = _device(settings)
        import torch
        hardware = {"device": device, "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda}
        if device == "cuda":
            properties = torch.cuda.get_device_properties(0)
            hardware.update(gpu_name=properties.name, vram_bytes=properties.total_memory,
                            bf16_supported=torch.cuda.is_bf16_supported())
        data.write_json(root / "hardware.json", hardware)
        budget.check(stage, training=True)
        train_dir, validation_dir = prepare_data(settings["input_zip"], settings["work_dir"],
            train_dir=settings["train_dir"], validation_dir=settings["validation_dir"])
        _, texts, targets = data.load_dataset(train_dir, include_targets=True)
        if len(texts) != EXPECTED_TRAINING_SAMPLES:
            raise ValueError("Training must contain all 29,200 provided calls before model downloads")
        fit, development, confirmation, split_report = data.three_way_split(texts, targets, settings["seed"])
        subset, subset_report = data.sample_fit_subset(texts, targets, fit,
            sample_count=settings["pilot_samples"], seed=settings["seed"])
        _save_npz(root / "split.npz", fit=fit, development=development, confirmation=confirmation)
        data.write_json(root / "split_manifest.json", split_report)
        _save_npz(root / "subset.npz", indices=subset)
        data.write_json(root / "subset_manifest.json", subset_report)
        digest = dataset_digest(texts, targets)
        data.write_json(root / "training_data.json", {"samples": len(texts), "fit_samples": len(fit),
            "pilot_samples": len(subset), "requested_pilot_samples": settings["pilot_samples"],
            "development_samples": len(development), "confirmation_samples": len(confirmation),
            "digest": digest, "labels_changed": 0, "samples_removed": 0,
            "training_count_matches_expected_29200": len(texts) == EXPECTED_TRAINING_SAMPLES})
        data.write_json(root / "training_diagnostics.json", data.training_diagnostics(texts, targets))
        source, source_recipe, source_specs = None, None, None
        if settings["run_mode"] == "full":
            stage = "validate_pilot_source"
            source, source_recipe, source_specs = _load_pilot_source(settings["source_run"], texts, targets,
                fit, development, confirmation, subset, split_report, subset_report, settings, budget)
            data.write_json(root / "source_pilot.json", {"source_run": str(source),
                "completion_sha256": _hash(source / "pilot_complete.json", budget),
                "training_dataset_digest": digest, "selection_reused_without_search": True})
        stage = "pretrained_download"
        specs = resolve_specs(Path(settings["work_dir"]) / "hf_cache", budget, source_specs=source_specs)
        # Do not put machine-specific HF cache paths into the portable recipe.
        data.write_json(root / "model_specs.json", {k: {a: b for a, b in v.items() if a != "snapshot_path"}
                                                  for k, v in specs.items()})
        for kind, spec in specs.items():
            notices = root / "licenses" / kind
            notices.mkdir(parents=True)
            copied = []
            snapshot = Path(spec["snapshot_path"])
            for notice_path in sorted(snapshot.iterdir()):
                if notice_path.is_file() and (notice_path.name.startswith("LICENSE") or notice_path.name == "README.md"):
                    shutil.copyfile(notice_path, notices / notice_path.name)
                    copied.append(notice_path.name)
            data.write_json(notices / "attribution.json", {"model_id": spec["model_id"],
                "revision": spec["revision"], "source": "https://huggingface.co/" + spec["model_id"],
                "license_note": spec.get("license_note"), "upstream_files": copied,
                "modification": "9-output classifier trained on provided Training; Qwen NF4 and LoRA adaptation"})
        probes = {}
        for kind in ORDER:
            stage = kind + "_batch_benchmark"
            data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
            print(f"Measuring {kind} batch throughput; benchmark weights are discarded.", flush=True)
            settings, probes[kind] = batching.autotune(kind, specs[kind], texts, targets,
                subset, development, settings, device, budget, root)
            data.write_json(root / "batch_selection.json", probes)
        data.write_json(root / "effective_settings.json", settings)
        preparation_seconds = max(0., time.monotonic() - invocation_started)
        stage = "budget_admission"
        if settings["run_mode"] == "pilot":
            plan = plan_epochs(probes, len(subset), len(texts), len(development), len(confirmation), settings, budget)
        else:
            records, selection = source_recipe["records"], source_recipe["selection"]
            estimates = timing_estimates(probes, len(subset), len(texts), len(development), len(confirmation), settings)
            full_estimate = estimate_full_run(estimates, selection, settings,
                preparation_seconds=preparation_seconds)
            data.write_json(root / "full_run_estimate.json", full_estimate)
            plan = {"common_epochs": source_recipe["common_epochs"], "estimates": estimates,
                "full_refit_planned": True, "confirmation_planned": True,
                "estimated_work_seconds": full_estimate["estimated_work_seconds"],
                "remaining_seconds": budget.remaining(), "fitted_samples": len(texts)}
            if not budget.can_start(full_estimate["estimated_work_seconds"], training=True):
                data.write_json(root / "budget_plan.json", {**plan, "admitted": False})
                return _diagnostic(root, "budget_not_started", "full_training_admission", started)
        data.write_json(root / "budget_plan.json", plan)
        if settings["run_mode"] == "pilot":
            logits = {}
            for kind in ORDER:
                stage = kind + "_fit"
                data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
                records[kind], logits[kind] = _fit_candidate(kind, specs[kind], texts, targets, subset, development,
                    settings, plan, device, budget, root)
            stage = "development_selection"
            scores = _compare(root, "development", targets, development, logits, budget)
            with (root / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["candidate", "macro_f1", "nausea_f1", "selected_epochs"])
                for name in ("control", "qwen", "ensemble"):
                    epochs = records[name]["selected_epochs"] if name in records else json.dumps(
                        {kind: records[kind]["selected_epochs"] for kind in ORDER})
                    writer.writerow([name, scores[name]["macro_f1"], scores[name]["nausea_f1"], epochs])
            selection = choose_candidate(scores, records, settings["min_macro_gain"])
            selection.update(common_epochs=plan["common_epochs"], training_dataset_digest=digest)
            data.write_json(root / "final_selection.json", selection)
            data.write_json(root / "training_metrics.json", scores[selection["selected_candidate"]])
            data.write_json(root / "pilot_recipe.json", {"format": PILOT_FORMAT,
                "target_symptoms": list(data.TARGET_SYMPTOMS), "training_dataset_digest": digest,
                "common_epochs": plan["common_epochs"], "records": records, "selection": selection})
            data.write_json(root / "full_run_estimate.json", estimate_full_run(plan["estimates"], selection, settings,
                records=records, fit_count=len(subset), full_count=len(texts), preparation_seconds=preparation_seconds))
            stage = "verify_pilot_export"
            _verify_pilot_exports(root, texts, development, records, logits, settings, device, budget)
            report = {"version": "ver25", "format": PILOT_FORMAT, "run_mode": "pilot",
                "status": "pilot_complete", "training_scope": "training_subset", "submission_ready": False,
                "fitted_samples": len(subset), "training_samples": len(texts), "device": device,
                "confirmation_evaluated": False, "validation_evaluated": False,
                "elapsed_worker_seconds": time.monotonic() - started,
                "elapsed_invocation_seconds": max(0., time.monotonic() - invocation_started)}
            data.write_json(root / "runtime.json", report)
            hashes = {p.relative_to(root).as_posix(): _hash(p, budget)
                      for p in sorted(root.rglob("*")) if p.is_file()}
            budget.check("publish pilot completion marker")
            data.write_json(root / "pilot_complete.json", {**report, "hashes": hashes})
            print("Pilot complete: results saved. Full training requires a separate explicit full run.", flush=True)
            return root

        data.write_json(root / "final_selection.json", selection)
        data.write_json(root / "pilot_recipe.json", source_recipe)
        # Selection is written BEFORE the reserved confirmation labels are evaluated.
        stage = "frozen_confirmation"
        confirmed = _confirmation(root, texts, targets, confirmation, records, settings, device, budget, source_root=source)
        data.write_json(root / "confirmation_protocol.json", {"selection_frozen_before_evaluation": True,
            "selection_changed": False, "comparison_count": 1,
            "selected_candidate": selection["selected_candidate"],
            "selected_macro_delta": confirmed[selection["selected_candidate"]]["macro_f1"] - confirmed["control"]["macro_f1"],
            "limitation": "Training has historical experiment exposure; this is not a new external test set."})
        stage = "full_training_admission"
        expected = sum(plan["estimates"][kind]["full_epoch_seconds"] * selection["selected_epochs"][kind]
            + plan["estimates"][kind]["setup_seconds"] for kind in selection["components"])
        if not budget.can_start(expected, training=True):
            return _diagnostic(root, "budget_not_started", stage, started)
        components = []
        for kind in selection["components"]:
            stage = kind + "_full_training_refit"
            data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
            components.append(_refit(kind, specs[kind], selection["selected_epochs"][kind], texts, targets,
                                      settings, plan, device, budget, root))
        manifest = {"format": FORMAT, "version": "ver25", "training_scope": "all_training",
            "fitted_samples": len(texts), "training_dataset_digest": digest,
            "target_symptoms": list(data.TARGET_SYMPTOMS), "components": components,
            "combination": "fixed_probability_mean" if len(components) == 2 else "single",
            "decision": "independent sigmoid >= 0.5", "selection": selection}
        checkpoint = root / "mission3_ver25_model.ckpt"
        data.write_json(checkpoint, manifest)
        stage = "final_validation"
        if settings["run_validation"]:
            budget.check(stage)
            paths, validation_texts, validation_targets = data.load_dataset(validation_dir, include_targets=True)
            final_logits = _predict_components(root, manifest, validation_texts, device, budget)
            predictions = trainer.fixed_decisions(final_logits)
            data.write_predictions(paths, predictions, root / "mission3.csv")
            data.write_json(root / "validation_metrics.json", analysis.metrics(validation_targets, final_logits))
            _save_npz(root / "validation_predictions.npz", logits=final_logits,
                      targets=validation_targets, predictions=predictions)
        stage = "finalize"
        status = "complete" if selection["improved"] else "complete_not_improved"
        report = {"version": "ver25", "run_mode": "full", "status": status, "training_scope": "all_training",
            "fitted_samples": len(texts), "validation_evaluated": settings["run_validation"],
            "device": device, "elapsed_worker_seconds": time.monotonic() - started,
            "elapsed_invocation_seconds": max(0., time.monotonic() - invocation_started),
            "submission_ready": settings["run_validation"]}
        data.write_json(root / "runtime.json", report)
        hashes = {p.relative_to(root).as_posix(): _hash(p, budget) for p in sorted(root.rglob("*")) if p.is_file()}
        budget.check("publish completion marker")
        data.write_json(root / "run_complete.json", {**report, "hashes": hashes})
        return root
    except BudgetExceeded:
        # Completed best-epoch fit checkpoints remain valid diagnostics. A
        # partial all-data refit never changes their scope or becomes complete.
        local_records = records if settings["run_mode"] == "pilot" else {}
        status = "fit_only_budget" if local_records else "budget_not_started"
        return _diagnostic(root, status, stage, started, local_records)
    except Exception as error:
        data.write_json(root / "runtime.json", {"status": "failed", "stage": stage,
            "error_type": type(error).__name__, "elapsed_worker_seconds": time.monotonic() - started,
            "submission_ready": False})
        raise


def _validated_manifest(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    manifest = read_json(checkpoint)
    if (manifest.get("format") != FORMAT or manifest.get("training_scope") != "all_training"
            or manifest.get("target_symptoms") != list(data.TARGET_SYMPTOMS)):
        raise ValueError("Unrecognized or fit-only ver25 checkpoint")
    components = manifest.get("components", [])
    names = [item.get("kind") for item in components]
    if names not in (["control"], ["qwen"], ["control", "qwen"]):
        raise ValueError("Invalid model components")
    expected = "single" if len(names) == 1 else "fixed_probability_mean"
    if manifest.get("combination") != expected:
        raise ValueError("Invalid fixed model combination")
    for item in components:
        if item.get("model_directory") != "final_" + item["kind"]:
            raise ValueError("Invalid component path")
        path = checkpoint.parent / item["model_directory"]
        if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.parent):
            raise ValueError("Model component escapes run directory")
    return checkpoint.parent, manifest


def _predict_components(root, manifest, texts, device, budget=None):
    logits = []
    for item in manifest["components"]:
        if budget:
            budget.check("offline component load")
        model, tokenizer, metadata = trainer.load_model(root / item["model_directory"], device)
        if metadata["training_scope"] != "all_training":
            raise ValueError("Inference component was not refit on full Training")
        loader = trainer.make_loader(tokenizer, texts, kind=item["kind"],
            batch_size=metadata.get("batch_size", 1), max_length=metadata["max_length"])
        logits.append(trainer.predict_logits(model, loader, device=device, budget=budget))
        del model, tokenizer, loader
        _release()
    return logits[0] if len(logits) == 1 else analysis.ensemble_logits(*logits)


def inference(checkpoint, label_dir, output):
    import torch
    root, manifest = _validated_manifest(checkpoint)
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    logits = _predict_components(root, manifest, texts, "cuda" if torch.cuda.is_available() else "cpu")
    data.write_predictions(paths, trainer.fixed_decisions(logits), output)
    return Path(output)


def inference_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="", help="Interface compatibility only; audio is unused")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    inference(args.checkpoint, args.label_dir, args.output)


if __name__ == "__main__":
    inference_main()
