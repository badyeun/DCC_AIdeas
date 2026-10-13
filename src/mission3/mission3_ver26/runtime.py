"""Diagnostic-only three-epoch QLoRA/control learning curves on a fixed subset."""
from __future__ import annotations

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
EXPECTED_TRAINING_SAMPLES = 29200
SCREENING_FORMAT = "mission3_ver26_learning_curve_v1"


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
                "time_budget_minutes", "run_mode", "source_run",
                "finalization_reserve_seconds", "budget_safety_factor", "log_every", "allow_cpu"}
    unknown = set(result) - set(defaults) - {"deadline_monotonic", "invocation_started_monotonic"}
    if unknown:
        raise ValueError("Unknown experiment settings: " + ", ".join(sorted(unknown)))
    for key, value in defaults.items():
        valid_budget_type = (key == "time_budget_minutes" and key in result
                             and type(result[key]) in (int, float))
        if key not in result or (not valid_budget_type and type(result[key]) is not type(value)):
            raise ValueError("Missing setting or invalid type: " + key)
        if key not in flexible and result[key] != value:
            raise ValueError("Pinned experiment setting changed: " + key)
    if result["run_mode"] != "pilot":
        raise ValueError("ver26 is screening-only; full training is disabled. Confirm epochs on full fit data first.")
    if result["source_run"]:
        raise ValueError("ver26 starts a fresh screening; source_run must be empty")
    minutes = result["time_budget_minutes"]
    if ((not math.isfinite(minutes) or not 0 < minutes <= 480
         or result["finalization_reserve_seconds"] >= 60 * minutes)
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
    epochs = settings["max_epochs"]
    comparison = sum((p["fit_epoch_seconds"] + p["development_seconds"]) * epochs
        + 2 * p["setup_seconds"] + 60 * epochs for p in estimates.values())
    if not budget.can_start(comparison, training=True):
        raise BudgetExceeded("all three screening epochs and development evaluations do not fit")
    return {"common_epochs": epochs, "scheduler_horizon_epochs": epochs, "estimates": estimates,
        "full_refit_planned": False, "confirmation_planned": False,
        "estimated_work_seconds": comparison, "comparison_seconds": comparison,
        "fitted_samples": fit_count, "remaining_seconds": budget.remaining(),
        "safety_factor": budget.safety_factor,
        "finalization_reserve_seconds": budget.finalization_reserve_s,
        "partial_epoch_plan_allowed": False}


def choose_candidate(scores, records, min_macro_gain=.01):
    # Stable tie order prefers the simpler single model, with control first.
    selected = max(("control", "qwen", "ensemble"), key=lambda name: scores[name]["macro_f1"])
    gain = scores[selected]["macro_f1"] - scores["control"]["macro_f1"]
    components = ["control", "qwen"] if selected == "ensemble" else [selected]
    return {"selected_candidate": selected, "components": components,
        "selected_epochs": {name: records[name]["selected_epochs"] for name in components},
        "selection_data": "Training development only", "confirmation_can_change_selection": False,
        "macro_f1_delta": gain, "improved": gain > 0,
        "control_relative_gain_reaches_reference": gain >= min_macro_gain,
        "reference_macro_gain": min_macro_gain,
        "control_gain_does_not_prove_more_epochs_help": True,
        "rule": "highest development Macro F1; ties control then qwen then fixed mean; no nausea veto",
        "historical_best_comparison": "not evaluated; this control is a fresh single KLUE model"}


def summarize_learning_curve(histories):
    """Describe within-run epoch evidence without authorizing a full-data run."""
    candidates = {}
    for kind in ORDER:
        rows = histories[kind]
        if [row["selected_epochs"] for row in rows] != [1, 2, 3]:
            raise ValueError("Learning-curve summary requires all three ordered epochs")
        best = max(rows, key=lambda row: row["macro_f1"])
        later_gain = max(row["macro_f1"] for row in rows[1:]) - rows[0]["macro_f1"]
        candidates[kind] = {
            "per_epoch": [{"epoch": row["selected_epochs"], "macro_f1": row["macro_f1"],
                "macro_average_precision": row["metrics"]["macro_average_precision"],
                "nausea_f1": row["metrics"]["nausea_f1"],
                "train_mean_loss": row["train_mean_loss"],
                "development_weighted_bce": row["development_weighted_bce"]} for row in rows],
            "best_epoch": best["selected_epochs"], "best_macro_f1": best["macro_f1"],
            "best_later_epoch_minus_epoch1_macro_f1": later_gain,
            "best_epoch_is_one": best["selected_epochs"] == 1,
            "later_epoch_improved": later_gain > 0,
            "plateau_or_decline_after_epoch1": later_gain <= 0,
            "interpretation": ("Later epochs improved development Macro F1 within this three-epoch schedule."
                if later_gain > 0 else "No later epoch improved development Macro F1 over epoch one.")}
    return {"scheduler_horizon_epochs": 3, "candidates": candidates,
        "full_fit_confirmation_required": True, "automatic_full_training_recommended": False,
        "epoch_one_is_not_ver25_one_epoch_schedule": True,
        "limitation": "Within-run development evidence only; small differences may be noise. "
                      "The subset best epoch is not an established full-data optimum."}


def _fit_candidate(kind, spec, texts, targets, fit, development, settings, plan, device, budget, root):
    budget.check("candidate load", training=True)
    model, tokenizer = trainer.create_model(spec, settings, device)
    loader = _loader(kind, tokenizer, texts, targets, fit, settings, shuffle=True, device=device)
    validation = _loader(kind, tokenizer, texts, None, development, settings, device=device)
    optimizer, scheduler, scaler = _optimizer(kind, model, loader, plan["common_epochs"], settings, device)
    weights = trainer.positive_weights(targets[fit])
    best, best_logits, history = None, None, []
    total_updates = 0
    for epoch in range(1, plan["common_epochs"] + 1):
        print(f"{kind}: fit epoch {epoch}/{plan['common_epochs']}", flush=True)
        learning_rates_start = [float(group["lr"]) for group in optimizer.param_groups]
        stats = trainer.train_epoch(model, loader, optimizer, device=device, pos_weight=weights,
            accumulation_steps=settings[kind + "_gradient_accumulation"], scheduler=scheduler,
            scaler=scaler, budget=budget, on_update=_progress(kind, settings["log_every"]))
        if not stats["completed_epoch"]:
            raise RuntimeError("Candidate epoch did not finish")
        evaluation_started = time.monotonic()
        logits = trainer.predict_logits(model, validation, device=device, budget=_TrainingBudget(budget))
        evaluation_seconds = time.monotonic() - evaluation_started
        score = analysis.metrics(targets[development], logits)
        # Match the training weighted BCE reduction without a second model pass.
        truth = targets[development].astype(np.float64)
        values = logits.astype(np.float64)
        positive_weights = weights.cpu().numpy().astype(np.float64)
        development_loss = float(np.mean((1 - truth) * np.logaddexp(0, values)
            + truth * positive_weights * np.logaddexp(0, -values)))
        total_updates += stats["successful_updates"]
        record = {"candidate": kind, "macro_f1": score["macro_f1"], "selected_epochs": epoch,
            "train_seconds": stats["elapsed_seconds"], "peak_vram_bytes": stats.get("peak_vram_bytes"),
            "evaluation_seconds": evaluation_seconds, "overflow_retries": stats.get("overflow_retries", 0),
            "fitted_samples": len(fit), "positive_weights": weights.tolist(),
            "train_mean_loss": stats["mean_loss"], "development_weighted_bce": development_loss,
            "successful_optimizer_updates": stats["successful_updates"],
            "cumulative_optimizer_updates": total_updates,
            "learning_rates_start": learning_rates_start,
            "learning_rates_end": [float(group["lr"]) for group in optimizer.param_groups],
            "scheduler_horizon_epochs": plan["common_epochs"]}
        history.append({**record, "metrics": score})
        epoch_folder = root / "learning_curve" / kind
        epoch_folder.mkdir(parents=True, exist_ok=True)
        data.write_json(epoch_folder / ("epoch_" + str(epoch) + "_metrics.json"), history[-1])
        _save_npz(epoch_folder / ("epoch_" + str(epoch) + "_predictions.npz"), indices=development,
            targets=targets[development], logits=logits, predictions=trainer.fixed_decisions(logits))
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


def _sources(root, runtime_sources):
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.iterdir()
        if p.suffix in {".py", ".txt", ".json", ".md"} and not p.name.startswith("test_")}
    submission = root / "source_snapshot"
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
    result = {"version": "ver26", "status": status, "stage": stage,
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
    data.write_json(root / "screening_portable_verification.json", verification)


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
        stage = "pretrained_download"
        specs = resolve_specs(Path(settings["work_dir"]) / "hf_cache", budget)
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
        stage = "budget_admission"
        plan = plan_epochs(probes, len(subset), len(texts), len(development), len(confirmation), settings, budget)
        data.write_json(root / "budget_plan.json", plan)
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
        selection.update(common_epochs=plan["common_epochs"], training_dataset_digest=digest,
            diagnostic_only=True, full_fit_confirmation_required=True,
            selected_epoch_is_not_full_data_optimum=True)
        data.write_json(root / "screening_selection.json", selection)
        data.write_json(root / "learning_curve_summary.json", summarize_learning_curve({
            kind: read_json(root / (kind + "_history.json")) for kind in ORDER}))
        data.write_json(root / "training_metrics.json", scores[selection["selected_candidate"]])
        data.write_json(root / "screening_recipe.json", {"format": SCREENING_FORMAT,
            "target_symptoms": list(data.TARGET_SYMPTOMS), "training_dataset_digest": digest,
            "common_epochs": plan["common_epochs"], "records": records, "selection": selection})
        data.write_json(root / "screening_protocol.json", {
            "purpose": "training_subset_learning_curve", "scheduler_horizon_epochs": plan["common_epochs"],
            "epoch_one_comparable_to_ver25_one_epoch_scheduler": False,
            "full_training_enabled": False, "submission_ready": False,
            "next_gate": "Confirm the epoch and recipe on all 20,440 fit samples before freezing a full refit.",
            "limitations": ["Development has prior experiment exposure; scores are not a fresh external test.",
                           "The best epoch on 4,000 rows is not the optimum full-data epoch.",
                           "No score improvement or Macro F1 above 0.7 is guaranteed."]})
        stage = "verify_pilot_export"
        _verify_pilot_exports(root, texts, development, records, logits, settings, device, budget)
        report = {"version": "ver26", "format": SCREENING_FORMAT, "run_mode": "pilot",
            "status": "screening_complete", "training_scope": "training_subset", "submission_ready": False,
            "full_training_enabled": False, "scheduler_horizon_epochs": plan["common_epochs"],
            "fitted_samples": len(subset), "training_samples": len(texts), "device": device,
            "confirmation_evaluated": False, "validation_evaluated": False,
            "elapsed_worker_seconds": time.monotonic() - started,
            "elapsed_invocation_seconds": max(0., time.monotonic() - invocation_started)}
        data.write_json(root / "runtime.json", report)
        hashes = {p.relative_to(root).as_posix(): _hash(p, budget)
                  for p in sorted(root.rglob("*")) if p.is_file()}
        budget.check("publish pilot completion marker")
        data.write_json(root / "screening_complete.json", {**report, "hashes": hashes})
        print("Screening complete: results saved. Confirm epochs on full fit data before any full refit.", flush=True)
        return root
    except BudgetExceeded:
        # Keep partial curves diagnostic; only three completed epochs can publish completion.
        local_records = records
        status = "fit_only_budget" if local_records else "budget_not_started"
        return _diagnostic(root, status, stage, started, local_records)
    except Exception as error:
        data.write_json(root / "runtime.json", {"status": "failed", "stage": stage,
            "error_type": type(error).__name__, "elapsed_worker_seconds": time.monotonic() - started,
            "submission_ready": False})
        raise


