"""Training-only, single-view symptom experiments within one shared time budget."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from . import data
from .bootstrap import prepare_data
from .budget import Budget, BudgetExceeded


CANDIDATES = (
    ("large_partial_ce", [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]),
)
SELECTION_POLICY = ("Training only: highest Macro F1 among epochs improving both Macro and nausea F1 "
                    "over verified ver19; otherwise highest Macro F1 for diagnosis without full refit")


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def dataset_digest(texts, targets):
    digest = hashlib.sha256()
    for text, target in zip(texts, targets):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(target, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(root, relative):
    candidate = root / relative
    path = candidate.resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Artifact path must stay inside its run directory")
    for part in (candidate, *candidate.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError("Symlink artifacts are not supported")
    return path


def _verify_hashes(root, hashes, budget=None):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Completed artifacts require nonempty hashes")
    for relative, expected in hashes.items():
        if budget:
            budget.check("verify saved artifacts")
        path = _safe(root, relative)
        if not path.is_file() or _hash(path) != expected:
            raise ValueError("Completed artifact is missing or changed: " + relative)


def _model_hashes(root, manifest, extras=()):
    assets = _safe(root, manifest["assets"])
    weights = _safe(root, manifest["weights"])
    if not weights.is_file() or not assets.is_dir():
        raise FileNotFoundError("Model weights or assets are missing")
    files = [weights, *sorted(p for p in assets.rglob("*") if p.is_file()),
             *(_safe(root, name) for name in extras)]
    if not any(p.parent.name == "tokenizer" for p in files):
        raise FileNotFoundError("Tokenizer assets are missing")
    return {p.relative_to(root.resolve()).as_posix(): _hash(p) for p in files}


def _publish_tree(source, destination, budget):
    """Copy immutable local artifacts; caller writes completion marker last."""
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        budget.check("publish completed model")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".copying")
        with path.open("rb") as reader, temporary.open("wb") as writer:
            while block := reader.read(4 * 1024 * 1024):
                budget.check("copy model to output storage")
                writer.write(block)
        if _hash(path) != _hash(temporary):
            raise OSError("Published model copy failed its content hash check")
        temporary.replace(target)


def validate_settings(settings):
    result = dict(settings)
    if result.get("version") != "ver20" or result.get("backend") != "neural":
        raise ValueError("This runner supports ver20 neural experiments only")
    if result.get("label_attention", False) or result.get("contrastive_weight", 0) != 0:
        raise ValueError("ver20 uses a single CE view without attention or contrastive loss")
    if result.get("model_id") != "klue/roberta-large" or result.get("revision") != "28d911204e9022eda172571ca8cc61eaffd942f7":
        raise ValueError("ver20 requires the pinned KLUE RoBERTa large checkpoint")
    if (type(result.get("expected_encoder_layers")) is not int or result["expected_encoder_layers"] != 24
            or type(result.get("trainable_encoder_layers")) is not int or result["trainable_encoder_layers"] != 8):
        raise ValueError("ver20 freezes embeddings and the first 16 of 24 layers, training only the last 8")
    if result.get("loss_mode", "ce") != "ce" or result.get("utterance_evidence", False):
        raise ValueError("ver20 requires ordinary weighted CE and the global dialogue classifier")
    if type(result.get("pilot_optimizer_steps")) is not int or result["pilot_optimizer_steps"] < 1:
        raise ValueError("pilot_optimizer_steps must be a positive integer")
    if result.get("refit_full_training") is not True:
        raise ValueError("Full refit is requested and skipped only when the time budget cannot fit it")
    for key in ("batch_size", "encoder_window_batch", "gradient_accumulation", "max_epochs", "patience", "max_length", "log_every"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not 1 <= result["max_epochs"] <= 3:
        raise ValueError("ver20 supports at most three epochs per training phase")
    if type(result.get("scheduler_horizon_epochs")) is not int or result["scheduler_horizon_epochs"] != 3:
        raise ValueError("ver20 retains a fixed three-epoch learning-rate schedule")
    if type(result.get("seed")) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(result.get("overlap")) is not int or not 0 <= result["overlap"] < result["max_length"] - 2:
        raise ValueError("overlap must fit within a token window")
    if not 0 < result.get("holdout_size", 0) < .5:
        raise ValueError("holdout_size must lie in (0, .5)")
    for key in ("learning_rate", "head_learning_rate", "time_budget_minutes", "budget_safety_factor", "finalization_reserve_seconds"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be positive and finite")
    if result["time_budget_minutes"] > 180:
        raise ValueError("The execution budget cannot exceed 180 minutes")
    if result["finalization_reserve_seconds"] >= result["time_budget_minutes"] * 60:
        raise ValueError("Finalization reserve must be smaller than the total budget")
    if result["budget_safety_factor"] < 1:
        raise ValueError("budget_safety_factor must be at least one")
    if not isinstance(result.get("resume_run", ""), str):
        raise ValueError("resume_run must be a run-directory path or empty string")
    result.update(label_attention=False, contrastive_weight=0.0, ce_views=1)
    return result


def _stable_settings(settings):
    excluded = {"deadline_monotonic", "input_zip", "train_dir", "validation_dir", "work_dir", "output_dir",
                "resume_run", "log_every", "time_budget_minutes", "finalization_reserve_seconds", "budget_safety_factor"}
    return {key: value for key, value in settings.items() if key not in excluded}


def _loader(texts, targets, tokenizer, settings, shuffle):
    from torch.utils.data import DataLoader
    from .neural import EncodeDataset
    dataset = EncodeDataset(texts, targets, tokenizer, settings["max_length"], settings["overlap"])
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=dataset.collate)


def _save_weights(model, path):
    import torch
    temporary = path.with_name(path.name + ".tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def _load_weights(model, path):
    import torch
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)


def _predict(model, loader, device, budget=None, training=False):
    import torch
    from .neural import predictions
    result = []
    model.eval()
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            if budget:
                budget.check("evaluate model", training=training)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits, _ = model({key: value.to(device) for key, value in batch.items()})
            result.append(predictions(logits).cpu().numpy())
    if not result:
        raise ValueError("Cannot predict an empty dataset")
    return np.concatenate(result).astype(np.int8)


def _optimizer(model, settings, batches, device):
    import torch
    identifiers = {id(p) for p in model.encoder.parameters()}
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for p in model.parameters() if id(p) not in identifiers and p.requires_grad]
    if not encoder or not heads:
        raise ValueError("Both trainable encoder layers and classification heads are required")
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": settings["learning_rate"]},
        {"params": heads, "lr": settings["head_learning_rate"]}], weight_decay=.01)
    updates = math.ceil(batches / settings["gradient_accumulation"]) * settings["scheduler_horizon_epochs"]
    warmup = max(1, int(updates * .1))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else max(0., (updates - step) / max(1, updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)
    return optimizer, scheduler, scaler, dtype


def _pilot_projection(settings, *, microbatches, measured_batches, measured_seconds,
                      total_batches, holdout_batches, full_ratio, budget, device, peak_memory_gb=0.):
    """Estimate work from real first-epoch updates; never create extra updates."""
    if measured_batches < 1 or measured_seconds <= 0:
        raise ValueError("Throughput pilot must contain measured training batches")
    per_batch = measured_seconds / measured_batches
    epoch_seconds = per_batch * total_batches
    # Using training-batch cost for inference deliberately overestimates it.
    evaluate_seconds = per_batch * holdout_batches
    unfinished_epoch = per_batch * max(0, total_batches - microbatches)
    first_completion = unfinished_epoch + evaluate_seconds + 120
    feasible = [epochs for epochs in range(1, settings["max_epochs"] + 1)
                if budget.can_start(first_completion + (epochs - 1) * (epoch_seconds + evaluate_seconds)
                                    + epochs * epoch_seconds * full_ratio + 120, training=True)]
    decision = ("continue_with_refit_projection" if feasible else
                "fit_only_risk" if budget.can_start(first_completion, training=True) else "stop_no_complete_epoch")
    return {"observed_optimizer_steps": math.ceil(microbatches / settings["gradient_accumulation"]),
            "observed_microbatches": microbatches, "measured_microbatches": measured_batches,
            "measured_seconds": measured_seconds, "estimated_epoch_seconds": epoch_seconds,
            "estimated_holdout_seconds": evaluate_seconds,
            "remaining_seconds": budget.remaining(), "projected_feasible_epochs": max(feasible, default=0),
            "decision": decision, "device": str(device), "peak_cuda_memory_gb": peak_memory_gb,
            "includes_training_updates": True, "resets_or_repeats_updates": False,
            "limitations": "Early shuffled batches estimate throughput only; later sequence lengths, I/O and GPU contention may differ."}


def _train_epoch(model, loader, settings, device, state, budget, epoch, pilot_context=None):
    import torch
    from .neural import classification_loss
    optimizer, scheduler, scaler, dtype = state
    accumulation = settings["gradient_accumulation"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total, counts = 0., np.zeros(9, dtype=np.int64)
    started = time.monotonic()
    measure_started, measure_start_batch, pilot_written = started, 0, False
    pilot_batch = min(len(loader), (settings["pilot_optimizer_steps"] + 1) * accumulation)
    if pilot_context and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(loader):
        budget.check("train microbatch", training=True)
        counts += batch["targets"].sum(0).numpy()
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, _ = model(batch)
            loss = classification_loss(logits, batch["targets"], positive_class_weights=settings["positive_class_weights"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        group_start = step // accumulation * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        loss_total += float(loss.detach())
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if pilot_context and step + 1 == accumulation and len(loader) > accumulation:
            if device.type == "cuda":
                torch.cuda.synchronize()
            measure_started, measure_start_batch = time.monotonic(), step + 1
        if pilot_context and not pilot_written and step + 1 == pilot_batch:
            if device.type == "cuda":
                torch.cuda.synchronize()
            report = _pilot_projection(
                settings, microbatches=step + 1, measured_batches=step + 1 - measure_start_batch,
                measured_seconds=max(time.monotonic() - measure_started, 1e-9), total_batches=len(loader),
                holdout_batches=pilot_context["holdout_batches"], full_ratio=pilot_context["full_ratio"],
                budget=budget, device=device,
                peak_memory_gb=torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.)
            for path in pilot_context["report_paths"]:
                data.write_json(path, report)
            pilot_written = True
            print(f"Large throughput pilot: estimated epoch {report['estimated_epoch_seconds'] / 60:.1f} min; "
                  f"projected fit+refit epochs {report['projected_feasible_epochs']}; {report['decision']}", flush=True)
            if report["decision"] == "stop_no_complete_epoch":
                raise BudgetExceeded("pilot predicts no completed epoch within the training budget")
        if (step + 1) % settings["log_every"] == 0:
            print(f"{settings['candidate_id']} epoch {epoch}: {step + 1}/{len(loader)} batches; "
                  f"{budget.remaining() / 60:.1f} minutes left", flush=True)
    return {"epoch": epoch, "loss": loss_total / len(loader), "classification_loss": loss_total / len(loader),
            "ce_views": 1, "scheduler_horizon_epochs": settings["scheduler_horizon_epochs"],
            "positive_target_counts": counts.tolist(), "training_seconds": time.monotonic() - started}


def _selection_key(metrics, reference_metrics=None):
    qualifies = bool(reference_metrics is not None
                     and metrics["macro_f1"] > reference_metrics["macro_f1"]
                     and metrics["per_symptom"][data.TARGET_SYMPTOMS[6]]["f1"]
                         > reference_metrics["per_symptom"][data.TARGET_SYMPTOMS[6]]["f1"])
    return qualifies, metrics["macro_f1"]


def _fit_selection(model, fit_loader, hold_loader, truth, indices, settings, device, local, budget, full_ratio,
                   pilot_report_path=None, reference_metrics=None):
    state = _optimizer(model, settings, len(fit_loader), device)
    best, stale, history, stop = None, 0, [], "max_epochs"
    for epoch in range(1, settings["max_epochs"] + 1):
        if history:
            next_cost = max(row["epoch_seconds"] for row in history)
            refit_cost = max(row["training_seconds"] for row in history) * full_ratio * epoch + 120
            if not budget.can_start(next_cost + refit_cost, training=True):
                stop = "budget_reserving_refit"
                break
        try:
            started = time.monotonic()
            pilot_context = None
            if epoch == 1:
                paths = [local / "pilot_report.json"]
                if pilot_report_path is not None:
                    paths.append(pilot_report_path)
                pilot_context = {"report_paths": paths, "holdout_batches": len(hold_loader), "full_ratio": full_ratio}
            row = _train_epoch(model, fit_loader, settings, device, state, budget, epoch,
                               pilot_context=pilot_context)
            evaluate_started = time.monotonic()
            predicted = _predict(model, hold_loader, device, budget, training=True)
            row["evaluation_seconds"] = time.monotonic() - evaluate_started
            metrics = data.score_predictions(truth, predicted)
            row["holdout_macro_f1"], row["holdout_metrics"] = metrics["macro_f1"], metrics
            row["both_macro_and_nausea_improved"] = _selection_key(metrics, reference_metrics)[0]
            save_started = time.monotonic()
            if best is None or _selection_key(metrics, reference_metrics) > _selection_key(best["metrics"], reference_metrics):
                budget.check("save selected epoch")
                _save_weights(model, local / "selection_weights.pt")
                np.savez_compressed(local / "holdout_predictions.npz", indices=indices, targets=truth, predictions=predicted)
                best, stale = {"selected_epochs": epoch, "metrics": metrics}, 0
            else:
                stale += 1
            row["save_seconds"] = time.monotonic() - save_started
            row["epoch_seconds"] = time.monotonic() - started
            history.append(row)
            data.write_json(local / "selection_history.json", history)
            print(f"{settings['candidate_id']} Training epoch {epoch}: Macro F1={metrics['macro_f1']:.6f}; "
                  f"{row['epoch_seconds'] / 60:.1f} minutes", flush=True)
            if stale >= settings["patience"]:
                stop = "early_stopping"
                break
        except BudgetExceeded:
            stop = "budget_during_epoch"
            break
    if best is None:
        raise BudgetExceeded("No complete Training epoch and holdout evaluation fit the budget")
    return {**best, "history": history, "stop_reason": stop}


def _manifest(settings, revision, epochs, samples, scope, partial_audit=None):
    return {"format": "mission3_categorical_v1", "kind": "neural", "version": "ver20",
            "labels": list(data.TARGET_SYMPTOMS), "decision_rule": "categorical predict/argmax",
            "model_id": settings["model_id"], "resolved_revision": revision,
            "selected_epochs": epochs, "fitted_samples": samples, "training_scope": scope,
            "positive_class_weights": settings["positive_class_weights"], "label_attention": False, "contrastive": False,
            "trainable_encoder_layers": settings["trainable_encoder_layers"],
            "expected_encoder_layers": settings["expected_encoder_layers"], "partial_finetuning": partial_audit,
            "scheduler_horizon_epochs": settings["scheduler_horizon_epochs"],
            "inference_precision": "CUDA BF16 when supported, otherwise CUDA FP16; CPU FP32",
            "max_length": settings["max_length"], "overlap": settings["overlap"],
            "batch_size": settings["batch_size"], "encoder_window_batch": settings["encoder_window_batch"],
            "assets": "assets", "weights": "selection_weights.pt" if scope == "fit_only" else "final_weights.pt"}


def _device(settings):
    import torch
    if not torch.cuda.is_available() and not settings.get("allow_cpu", False):
        raise RuntimeError("Select a Colab GPU runtime before starting ver20")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_candidate(texts, targets, fit, holdout, settings, local, budget, pilot_report_path=None,
                    reference_metrics=None):
    from .neural import DialogueClassifier, prepare_encoder, seed_everything, configure_partial_finetuning
    device = _device(settings)
    seed_everything(settings["seed"])
    budget.check("prepare candidate encoder", training=True)
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], local / "assets")
    audit = configure_partial_finetuning(encoder, settings["trainable_encoder_layers"], settings["expected_encoder_layers"])
    data.write_json(local / "partial_finetuning.json", audit)
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False)
    model.encoder_window_batch = settings["encoder_window_batch"]
    model.to(device)
    try:
        budget.check("tokenize candidate", training=True)
        fit_loader = _loader([texts[i] for i in fit], targets[fit], tokenizer, settings, True)
        hold_loader = _loader([texts[i] for i in holdout], targets[holdout], tokenizer, settings, False)
        result = _fit_selection(model, fit_loader, hold_loader, targets[holdout], holdout, settings, device,
                                local, budget, len(targets) / len(fit), pilot_report_path=pilot_report_path,
                                reference_metrics=reference_metrics)
        manifest = _manifest(settings, revision, result["selected_epochs"], len(fit), "fit_only", audit)
        data.write_json(local / "candidate_model.ckpt", manifest)
        data.write_json(local / "training_metrics.json", result["metrics"])
        return manifest, result
    finally:
        del model, encoder
        _release()


def refit_selected(texts, targets, settings, epochs, local, budget):
    """Only a fully completed fresh refit is eligible for publication."""
    from .neural import DialogueClassifier, prepare_encoder, seed_everything, configure_partial_finetuning
    device = _device(settings)
    seed_everything(settings["seed"])
    budget.check("prepare fresh refit", training=True)
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], local / "assets")
    audit = configure_partial_finetuning(encoder, settings["trainable_encoder_layers"], settings["expected_encoder_layers"])
    data.write_json(local / "partial_finetuning.json", audit)
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False)
    model.encoder_window_batch = settings["encoder_window_batch"]
    model.to(device)
    try:
        loader = _loader(texts, targets, tokenizer, settings, True)
        state = _optimizer(model, settings, len(loader), device)
        history = []
        for epoch in range(1, epochs + 1):
            row = _train_epoch(model, loader, settings, device, state, budget, epoch)
            history.append(row)
            data.write_json(local / "final_history.json", history)
            print(f"Fresh all-Training refit epoch {epoch}/{epochs}: loss={row['loss']:.5f}", flush=True)
        budget.check("save completed full refit")
        _save_weights(model, local / "final_weights.pt")
        probe = _predict(model, _loader(texts[:12], None, tokenizer, settings, False), device, budget)
        manifest = _manifest(settings, revision, epochs, len(targets), "all_training", audit)
        data.write_json(local / "final_model.ckpt", manifest)
        return manifest, probe
    finally:
        del model, encoder
        _release()


def predict_checkpoint(checkpoint, texts, budget=None):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .neural import DialogueClassifier, configure_partial_finetuning
    checkpoint = Path(checkpoint)
    manifest, root = _read(checkpoint), checkpoint.parent
    if manifest.get("format") != "mission3_categorical_v1" or tuple(manifest["labels"]) != data.TARGET_SYMPTOMS:
        raise ValueError("Unsupported checkpoint or symptom ordering")
    if (manifest.get("version") != "ver20" or manifest["kind"] != "neural"
            or manifest["label_attention"] or manifest["contrastive"]):
        raise ValueError("ver20 inference requires its single-view classifier")
    if (manifest.get("model_id") != "klue/roberta-large"
            or manifest.get("resolved_revision") != "28d911204e9022eda172571ca8cc61eaffd942f7"
            or manifest.get("expected_encoder_layers") != 24 or manifest.get("trainable_encoder_layers") != 8
            or manifest.get("scheduler_horizon_epochs") != 3
            or not isinstance(manifest.get("partial_finetuning"), dict)
            or manifest.get("decision_rule") != "categorical predict/argmax"):
        raise ValueError("Checkpoint partial-finetuning provenance is missing or incompatible")
    if budget:
        budget.check("load final checkpoint")
    assets = _safe(root, manifest["assets"])
    config = AutoConfig.from_pretrained(assets / "encoder", local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    audit = configure_partial_finetuning(encoder, manifest["trainable_encoder_layers"], manifest["expected_encoder_layers"])
    if audit != manifest["partial_finetuning"]:
        raise ValueError("Saved partial-finetuning audit disagrees with the encoder architecture")
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True, trust_remote_code=False)
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False)
    model.encoder_window_batch = manifest["encoder_window_batch"]
    _load_weights(model, _safe(root, manifest["weights"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        return _predict(model, _loader(texts, None, tokenizer, manifest, False), device, budget)
    finally:
        del model, encoder
        _release()


def inference(checkpoint, label_dir, output, budget=None):
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predicted = predict_checkpoint(checkpoint, texts, budget)
    if budget:
        budget.check("save final predictions")
    data.write_predictions(paths, predicted, output)
    return paths, predicted


def _candidate_record(settings, texts, targets, fit, holdout, root, budget, reference_metrics=None):
    name = settings["candidate_id"]
    directory = root / "experiments" / name
    completed = directory / "completed.json"
    stable = _stable_settings(settings)
    if completed.is_file():
        record = _read(completed)
        if record["settings"] != stable or record["settings_digest"] != _digest(stable) or record["candidate_id"] != name:
            raise ValueError("Saved candidate configuration changed")
        if Path(record["directory"]).parts[:2] != ("experiments", name):
            raise ValueError("Saved candidate directory changed")
        _verify_hashes(root, record["artifact_hashes"], budget)
        original = _read(_safe(root, record["directory"] + "/candidate_model.ckpt"))
        expected = {**original, "weights": record["directory"] + "/" + original["weights"],
                    "assets": record["directory"] + "/" + original["assets"]}
        if expected != record["manifest"]:
            raise ValueError("Saved candidate manifest changed")
        print("Reusing completed Training candidate:", name, flush=True)
    else:
        started, attempt = time.monotonic(), _stamp()
        local = Path(settings["work_dir"]) / "training" / root.name / name / attempt
        local.mkdir(parents=True, exist_ok=False)
        data.write_json(local / "settings.json", stable)
        manifest, result = train_candidate(texts, targets, fit, holdout, settings, local, budget,
                                          pilot_report_path=root / "pilot_report.json", reference_metrics=reference_metrics)
        destination = directory / attempt
        _publish_tree(local, destination, budget)
        relative = destination.relative_to(root).as_posix()
        root_manifest = {**manifest, "weights": relative + "/" + manifest["weights"],
                         "assets": relative + "/" + manifest["assets"]}
        record = {"candidate_id": name, "directory": relative, "settings": stable,
                  "settings_digest": _digest(stable), "manifest": root_manifest,
                  "metrics": result["metrics"], "history": result["history"],
                  "stop_reason": result["stop_reason"], "seconds": time.monotonic() - started}
        record["artifact_hashes"] = _model_hashes(root, root_manifest, (
            relative + "/holdout_predictions.npz", relative + "/selection_history.json",
            relative + "/candidate_model.ckpt", relative + "/training_metrics.json", relative + "/settings.json",
            relative + "/partial_finetuning.json", relative + "/pilot_report.json"))
        budget.check("commit completed candidate")
        data.write_json(completed, record)
    with np.load(_safe(root, record["directory"] + "/holdout_predictions.npz"), allow_pickle=False) as saved:
        if not np.array_equal(saved["indices"], holdout) or not np.array_equal(saved["targets"], targets[holdout]):
            raise ValueError("Saved candidate Training targets or split changed")
        if data.score_predictions(targets[holdout], saved["predictions"]) != record["metrics"]:
            raise ValueError("Saved candidate scores disagree with discrete predictions")
    if record["history"] != _read(_safe(root, record["directory"] + "/selection_history.json")):
        raise ValueError("Saved timing or epoch history changed")
    return record


def _refit_estimate(record, full_ratio):
    return (max(row["training_seconds"] for row in record["history"]) * full_ratio
            * record["manifest"]["selected_epochs"] + 120)


def _write_table(root, records):
    columns = ["candidate", "selected_epochs", "macro_f1", "nausea_f1", "nausea_precision", "nausea_recall",
               "seconds", "stop_reason", "positive_class_weights"]
    temporary = root / "candidate_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            nausea = record["metrics"]["per_symptom"][data.TARGET_SYMPTOMS[6]]
            writer.writerow({"candidate": record["candidate_id"], "selected_epochs": record["manifest"]["selected_epochs"],
                             "macro_f1": record["metrics"]["macro_f1"], "nausea_f1": nausea["f1"],
                             "nausea_precision": nausea["precision"], "nausea_recall": nausea["recall"],
                             "seconds": record["seconds"], "stop_reason": record["stop_reason"],
                             "positive_class_weights": json.dumps(record["settings"]["positive_class_weights"])})
    temporary.replace(root / "candidate_summary.csv")


def _final_model(selected, texts, targets, holdout, settings, root, budget, refit_allowed=True):
    checkpoint, ready = root / "mission3_ver20_model.ckpt", root / "final_model_ready.json"
    if ready.is_file():
        saved = _read(ready)
        if saved["selected_candidate"] != selected["candidate_id"]:
            raise ValueError("Final model selection changed during resume")
        _verify_hashes(root, saved["artifact_hashes"], budget)
        return checkpoint, _read(checkpoint)
    manifest = dict(selected["manifest"])
    reason = "budget_before_refit" if refit_allowed else "training_reference_not_jointly_improved"
    with np.load(_safe(root, selected["directory"] + "/holdout_predictions.npz"), allow_pickle=False) as saved:
        expected_probe = saved["predictions"][:12].copy()
    probe_texts = [texts[i] for i in holdout[:len(expected_probe)]]
    estimate = _refit_estimate(selected, len(targets) / manifest["fitted_samples"])
    if refit_allowed and budget.can_start(estimate, training=True):
        attempt = _stamp()
        local = Path(settings["work_dir"]) / "training" / root.name / "full_refit" / attempt
        local.mkdir(parents=True, exist_ok=False)
        final_settings = {**settings, **selected["settings"]}
        data.write_json(local / "settings.json", _stable_settings(final_settings))
        try:
            new_manifest, new_probe = refit_selected(texts, targets, final_settings,
                                                     manifest["selected_epochs"], local, budget)
        except BudgetExceeded:
            reason = "budget_during_refit"
            print("Refit did not finish within its training budget; retaining the completed fit-only candidate.", flush=True)
        else:
            destination = root / "final_model" / attempt
            _publish_tree(local, destination, budget)
            relative = destination.relative_to(root).as_posix()
            manifest = {**new_manifest, "weights": relative + "/" + new_manifest["weights"],
                        "assets": relative + "/" + new_manifest["assets"]}
            expected_probe, probe_texts = new_probe, texts[:len(new_probe)]
            data.write_json(root / "final_history.json", _read(local / "final_history.json"))
            reason = None
    elif refit_allowed:
        print("Full refit does not fit the remaining budget; using the completed fit-only model.", flush=True)
    else:
        print("Skipping full refit: Training Macro F1 and nausea F1 did not both improve. "
              "The saved fit-only candidate is experimental; retain the prior submission.", flush=True)
    manifest.update(selected_candidate=selected["candidate_id"], refit_skipped_reason=reason)
    data.write_json(checkpoint, manifest)
    actual = predict_checkpoint(checkpoint, probe_texts, budget)
    if not np.array_equal(actual, expected_probe):
        raise ValueError("Reloaded final model differs from its saved selection or in-memory predictions")
    hashes = _model_hashes(root, manifest, (checkpoint.name,))
    budget.check("commit verified final model")
    data.write_json(ready, {"selected_candidate": selected["candidate_id"], "reloaded_probe_samples": len(expected_probe),
                           "artifact_hashes": hashes})
    return checkpoint, manifest


def _environment():
    import torch
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ("torch", "transformers", "numpy", "scikit-learn", "iterative-stratification", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"packages": packages, "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def run(settings, runtime_sources=None):
    from .reference import verify_training_reference, compare_reference, reference_summary
    started = time.monotonic()
    settings = validate_settings(settings)
    deadline = settings.get("deadline_monotonic", started + settings["time_budget_minutes"] * 60)
    if deadline > started + settings["time_budget_minutes"] * 60 + 1:
        raise ValueError("Supplied deadline exceeds the allowed execution budget")
    budget = Budget(deadline, finalization_reserve_s=settings["finalization_reserve_seconds"],
                    safety_factor=settings["budget_safety_factor"])
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.iterdir()
                                 if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json")}
    sources = {key: value for key, value in sources.items() if not key.startswith("test_") and key != "build_notebook.py"}
    output = Path(settings["output_dir"]).expanduser().resolve()
    resume = settings.get("resume_run", "")
    root = Path(resume).expanduser().resolve() if resume else output / _stamp()
    if root == output or not root.is_relative_to(output):
        raise ValueError("resume_run must be a child run directory inside output_dir")
    if resume and not (root / "run_identity.json").is_file():
        raise ValueError("resume_run is not initialized")
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    print("Result run:", root, flush=True)
    writable_status = not bool(resume)
    records, skipped = [], []
    try:
        budget.check("prepare inputs")
        _device(settings)
        train_dir, validation_dir = prepare_data(settings["input_zip"], settings["work_dir"],
                                                settings.get("train_dir", ""), settings.get("validation_dir", ""))
        paths, texts, targets = data.load_dataset(train_dir, include_targets=True)
        fit, holdout = data.make_split(texts, targets, settings["holdout_size"], settings["seed"])
        training_digest = dataset_digest(texts, targets)
        reference = verify_training_reference(training_digest, targets, fit, holdout,
                                              split_seed=settings["seed"], holdout_size=settings["holdout_size"])
        identity = {"version": "ver20", "training_digest": training_digest,
                    "source_digest": _digest({k: v for k, v in sources.items() if not k.endswith(".md")}),
                    "settings_digest": _digest(_stable_settings(settings)), "candidates": CANDIDATES,
                    "model_selection": SELECTION_POLICY}
        # JSON normalizes tuples, so identity is stored and compared in that canonical representation.
        identity = json.loads(json.dumps(identity))
        if resume:
            if _read(root / "run_identity.json") != identity:
                raise ValueError("Resume source, settings, or Training data changed")
            with np.load(root / "training_split.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["fit"], fit) or not np.array_equal(saved["holdout"], holdout):
                    raise ValueError("Resume Training split changed")
        else:
            data.write_json(root / "run_identity.json", identity)
            data.write_json(root / "settings.json", {k: v for k, v in settings.items() if k != "deadline_monotonic"})
            np.savez_compressed(root / "training_split.npz", fit=fit, holdout=holdout)
            submission = root / "submission"
            submission.mkdir()
            for name, source in sources.items():
                if Path(name).name != name:
                    raise ValueError("Runtime sources must have flat filenames")
                (submission / name).write_bytes(source.encode("utf-8"))
        if (root / "run_complete.json").is_file():
            _verify_hashes(root, _read(root / "run_complete.json")["artifact_hashes"], budget)
            print("Run already complete; its saved final artifacts were verified.", flush=True)
            return root
        writable_status = True
        data.write_json(root / "training_reference_verified.json", reference_summary(reference))
        data.write_json(root / "runtime.json", {"status": "running", "seconds": time.monotonic() - started,
                        "message": "Only run_complete.json confirms completed final artifacts."})
        data.write_json(root / ("environment_" + _stamp() + ".json"), _environment())
        data.write_json(root / "experiment_plan.json", {"candidates": CANDIDATES, "ce_views": 1,
                        "selection": SELECTION_POLICY, "max_epochs": settings["max_epochs"],
                        "training_reference": "ver19 saved fit-only predictions; no baseline retraining",
                        "encoder_layers": {"total": settings["expected_encoder_layers"],
                                           "trainable": settings["trainable_encoder_layers"]},
                        "time_budget_minutes": settings["time_budget_minutes"],
                        "finalization_reserve_seconds": settings["finalization_reserve_seconds"],
                        "budget_fallback": "completed fit-only candidate; never incomplete refit"})
        final_already_ready = (root / "final_model_ready.json").is_file()
        for name, weights in CANDIDATES:
            completed = root / "experiments" / name / "completed.json"
            if not completed.is_file() and final_already_ready:
                skipped.append({"candidate": name, "reason": "final_model_already_selected"})
                continue
            if not completed.is_file() and records:
                previous = records[0]
                minimum_trial = max(row["epoch_seconds"] for row in previous["history"]) * min(3, settings["max_epochs"]) + 120
                current_best = max(records, key=lambda row: row["metrics"]["macro_f1"])
                estimate = minimum_trial + _refit_estimate(current_best, len(targets) / len(fit))
                if not budget.can_start(estimate, training=True):
                    skipped.append({"candidate": name, "reason": "budget_reserving_refit"})
                    print("Skipping additional candidate to preserve finalization/refit time:", name, flush=True)
                    continue
            config = {**settings, "candidate_id": name, "positive_class_weights": weights}
            try:
                records.append(_candidate_record(config, texts, targets, fit, holdout, root, budget,
                                                  reference_metrics=reference["metrics"]))
            except BudgetExceeded:
                skipped.append({"candidate": name, "reason": "budget_no_completed_epoch"})
                if not records:
                    raise
                break
            _write_table(root, records)
        if not records:
            raise BudgetExceeded("No completed candidate is available")
        selected = max(records, key=lambda row: row["metrics"]["macro_f1"])
        data.write_json(root / "final_selection.json", {"candidate_id": selected["candidate_id"],
                        "metrics": selected["metrics"], "selection_policy": SELECTION_POLICY,
                        "selected_epochs": selected["manifest"]["selected_epochs"], "skipped_candidates": skipped})
        data.write_json(root / "training_metrics.json", selected["metrics"])
        saved = _safe(root, selected["directory"] + "/holdout_predictions.npz")
        shutil.copy2(saved, root / "holdout_predictions.npz")
        with np.load(saved, allow_pickle=False) as values:
            comparison = compare_reference(reference, values["predictions"], targets, holdout)
        data.write_json(root / "training_reference_comparison.json", comparison)
        qualified = comparison["both_macro_and_nausea_improved"]
        print("Training comparison with ver19: both Macro F1 and nausea F1 improved =", qualified, flush=True)
        if not qualified:
            print("This is an experimental candidate, not a verified improvement. Existing results remain unchanged.", flush=True)
        checkpoint, manifest = _final_model(selected, texts, targets, holdout, settings, root, budget,
                                             refit_allowed=qualified)
        hashes = _read(root / "final_model_ready.json")["artifact_hashes"]
        for name in ("training_reference_comparison.json", "training_reference_verified.json",
                     "holdout_predictions.npz", "training_metrics.json", "final_selection.json", "pilot_report.json"):
            hashes[name] = _hash(root / name)
        if settings.get("run_validation", True):
            predicted_paths, predicted = inference(checkpoint, validation_dir, root / "mission3.csv", budget)
            budget.check("score final Validation CSV")
            scored_paths, _, truth = data.load_dataset(validation_dir, include_targets=True)
            if [p.name for p in scored_paths] != [p.name for p in predicted_paths]:
                raise ValueError("Validation row order changed")
            metrics = data.score_predictions(truth, predicted)
            data.write_json(root / "validation_metrics.json", metrics)
            print(f"Final Validation Macro F1={metrics['macro_f1']:.6f}", flush=True)
            for name in ("mission3.csv", "validation_metrics.json"):
                hashes[name] = _hash(root / name)
        limited = bool(skipped) or manifest["training_scope"] != "all_training" or any(r["stop_reason"].startswith("budget") for r in records)
        budget.check("commit final run")
        status = "complete_not_improved" if not qualified else "complete_budget_limited" if limited else "complete"
        data.write_json(root / "runtime.json", {"status": status,
                        "seconds": time.monotonic() - started, "seconds_scope": "worker invocation; notebook deadline includes installation",
                        "candidate_count": len(records), "selected_candidate": selected["candidate_id"],
                        "both_macro_and_nausea_improved_on_training": qualified,
                        "training_reference_version": "ver19", "baseline_retrained": False,
                        "trainable_encoder_layers": manifest["trainable_encoder_layers"],
                        "training_scope": manifest["training_scope"], "fitted_samples": manifest["fitted_samples"],
                        "refit_skipped_reason": manifest["refit_skipped_reason"], "skipped_candidates": skipped,
                        "budget_minutes_remaining": max(0., budget.remaining() / 60)})
        data.write_json(root / "run_complete.json", {"artifact_hashes": hashes})
    except BudgetExceeded:
        if writable_status:
            data.write_json(root / "runtime.json", {"status": "budget_exhausted", "seconds": time.monotonic() - started,
                            "candidate_count": len(records), "skipped_candidates": skipped,
                            "message": "No newly completed run; existing completed candidate artifacts are preserved."})
        print("Time budget exhausted. Completed artifacts remain in the result directory.", flush=True)
    except BaseException as error:
        if writable_status:
            data.write_json(root / "runtime.json", {"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                            "seconds": time.monotonic() - started, "error_type": type(error).__name__})
        raise
    return root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    inference(args.ckpt_path, args.label_dir, args.output)


if __name__ == "__main__":
    main()
