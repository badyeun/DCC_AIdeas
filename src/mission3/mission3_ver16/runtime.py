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


POSITIVE_CLASS_WEIGHTS = [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]
CANDIDATES = (
    {"candidate_id": "baseline_ce", "label_correlation": False},
    {"candidate_id": "cornet", "label_correlation": True},
)


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
    from .neural import validate_correlation_config
    result = dict(settings)
    if result.get("version") != "ver16" or result.get("backend") != "neural":
        raise ValueError("This runner supports ver16 neural experiments only")
    if result.get("label_attention", False) or result.get("contrastive_weight", 0) != 0:
        raise ValueError("ver16 uses a single CE view without attention or contrastive loss")
    if result.get("refit_full_training") is not True:
        raise ValueError("Full refit is requested and skipped only when the time budget cannot fit it")
    for key in ("batch_size", "encoder_window_batch", "gradient_accumulation", "max_epochs", "patience", "max_length", "log_every"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not 1 <= result["max_epochs"] <= 3:
        raise ValueError("ver16 supports at most three selection epochs")
    horizon = result.get("scheduler_horizon_epochs", 4)
    if type(horizon) is not int or horizon != 4:
        raise ValueError("ver16 keeps its matched four-epoch scheduler horizon")
    result["scheduler_horizon_epochs"] = horizon
    result.update(validate_correlation_config(result.get("correlation_hidden", 32),
                                             result.get("correlation_depth", 1),
                                             result.get("correlation_dropout", .1)))
    result.update(_architecture(result))
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
    with torch.inference_mode():
        for batch in loader:
            if budget:
                budget.check("evaluate model", training=training)
            logits, _ = model({key: value.to(device) for key, value in batch.items()})
            result.append(predictions(logits).cpu().numpy())
    if not result:
        raise ValueError("Cannot predict an empty dataset")
    return np.concatenate(result).astype(np.int8)


def _optimizer(model, settings, batches, device):
    import torch
    encoder = list(model.encoder.parameters())
    identifiers = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if id(p) not in identifiers]
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": settings["learning_rate"]},
        {"params": heads, "lr": settings["head_learning_rate"]}], weight_decay=.01)
    updates = math.ceil(batches / settings["gradient_accumulation"]) * settings.get("scheduler_horizon_epochs", 4)
    warmup = max(1, int(updates * .1))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else max(0., (updates - step) / max(1, updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)
    return optimizer, scheduler, scaler, dtype


def _train_epoch(model, loader, settings, device, state, budget, epoch):
    import torch
    from .neural import classification_loss
    optimizer, scheduler, scaler, dtype = state
    accumulation = settings["gradient_accumulation"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total, counts = 0., np.zeros(9, dtype=np.int64)
    started = time.monotonic()
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
        if (step + 1) % settings["log_every"] == 0:
            print(f"{settings['candidate_id']} epoch {epoch}: {step + 1}/{len(loader)} batches; "
                  f"{budget.remaining() / 60:.1f} minutes left", flush=True)
    return {"epoch": epoch, "loss": loss_total / len(loader), "classification_loss": loss_total / len(loader),
            "ce_views": 1, "scheduler_horizon_epochs": settings.get("scheduler_horizon_epochs", 4),
            "positive_target_counts": counts.tolist(), "training_seconds": time.monotonic() - started}


def _fit_selection(model, fit_loader, hold_loader, truth, indices, settings, device, local, budget, full_ratio):
    state = _optimizer(model, settings, len(fit_loader), device)
    best, stale, history, stop = None, 0, [], "max_epochs"
    for epoch in range(1, settings["max_epochs"] + 1):
        if history:
            next_cost = max(row["epoch_seconds"] for row in history)
            refit_cost = max(row["training_seconds"] for row in history) * full_ratio * epoch + 120
            if not budget.can_start(next_cost + refit_cost, training=True):
                # Prior runs improved chiefly at epoch three. Prefer reaching that
                # point on the fit partition when a fresh full refit cannot fit.
                minimum_useful_epochs = min(3, settings["max_epochs"])
                if epoch > minimum_useful_epochs or not budget.can_start(next_cost, training=True):
                    stop = "budget_reserving_refit"
                    break
        try:
            started = time.monotonic()
            row = _train_epoch(model, fit_loader, settings, device, state, budget, epoch)
            evaluate_started = time.monotonic()
            predicted = _predict(model, hold_loader, device, budget, training=True)
            row["evaluation_seconds"] = time.monotonic() - evaluate_started
            metrics = data.score_predictions(truth, predicted)
            row["holdout_macro_f1"], row["holdout_metrics"] = metrics["macro_f1"], metrics
            save_started = time.monotonic()
            if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
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


def _manifest(settings, revision, epochs, samples, scope):
    return {"format": "mission3_categorical_v1", "kind": "neural", "version": "ver16",
            "labels": list(data.TARGET_SYMPTOMS), "decision_rule": "categorical predict/argmax",
            "model_id": settings["model_id"], "resolved_revision": revision,
            "selected_epochs": epochs, "fitted_samples": samples, "training_scope": scope,
            "positive_class_weights": settings["positive_class_weights"], "label_attention": False, "contrastive": False,
            **_architecture(settings), "scheduler_horizon_epochs": settings.get("scheduler_horizon_epochs", 4),
            "max_length": settings["max_length"], "overlap": settings["overlap"],
            "batch_size": settings["batch_size"], "encoder_window_batch": settings["encoder_window_batch"],
            "assets": "assets", "weights": "selection_weights.pt" if scope == "fit_only" else "final_weights.pt"}


def _architecture(settings):
    from .neural import validate_correlation_config
    enabled = settings.get("label_correlation", False)
    if type(enabled) is not bool:
        raise ValueError("label_correlation must be boolean")
    return {"label_correlation": enabled,
            **validate_correlation_config(settings.get("correlation_hidden", 32),
                                          settings.get("correlation_depth", 1),
                                          settings.get("correlation_dropout", .1))}


def _device(settings):
    import torch
    if not torch.cuda.is_available() and not settings.get("allow_cpu", False):
        raise RuntimeError("Select a Colab GPU runtime before starting ver16")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_candidate(texts, targets, fit, holdout, settings, local, budget):
    from .neural import DialogueClassifier, prepare_encoder, seed_everything
    device = _device(settings)
    seed_everything(settings["seed"])
    budget.check("prepare candidate encoder", training=True)
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], local / "assets")
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False, **_architecture(settings))
    model.encoder_window_batch = settings["encoder_window_batch"]
    model.to(device)
    try:
        budget.check("tokenize candidate", training=True)
        fit_loader = _loader([texts[i] for i in fit], targets[fit], tokenizer, settings, True)
        hold_loader = _loader([texts[i] for i in holdout], targets[holdout], tokenizer, settings, False)
        result = _fit_selection(model, fit_loader, hold_loader, targets[holdout], holdout, settings, device,
                                local, budget, len(targets) / len(fit))
        manifest = _manifest(settings, revision, result["selected_epochs"], len(fit), "fit_only")
        data.write_json(local / "candidate_model.ckpt", manifest)
        data.write_json(local / "training_metrics.json", result["metrics"])
        return manifest, result
    finally:
        del model, encoder
        _release()


def refit_selected(texts, targets, settings, epochs, local, budget):
    """Only a fully completed fresh refit is eligible for publication."""
    from .neural import DialogueClassifier, prepare_encoder, seed_everything
    if type(epochs) is not int or not 1 <= epochs <= settings["max_epochs"] <= 3:
        raise ValueError("Refit epochs must match a completed selection epoch between one and three")
    device = _device(settings)
    seed_everything(settings["seed"])
    budget.check("prepare fresh refit", training=True)
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], local / "assets")
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False, **_architecture(settings))
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
        manifest = _manifest(settings, revision, epochs, len(targets), "all_training")
        data.write_json(local / "final_model.ckpt", manifest)
        return manifest, probe
    finally:
        del model, encoder
        _release()


def predict_checkpoint(checkpoint, texts, budget=None):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .neural import DialogueClassifier
    checkpoint = Path(checkpoint)
    manifest, root = _read(checkpoint), checkpoint.parent
    if manifest.get("format") != "mission3_categorical_v1" or tuple(manifest["labels"]) != data.TARGET_SYMPTOMS:
        raise ValueError("Unsupported checkpoint or symptom ordering")
    if manifest["kind"] != "neural" or manifest["label_attention"] or manifest["contrastive"] or manifest.get("version") != "ver16":
        raise ValueError("ver16 inference requires its single-view classifier")
    if not {"label_correlation", "correlation_hidden", "correlation_depth", "correlation_dropout"} <= manifest.keys():
        raise ValueError("ver16 checkpoint must declare its correlation architecture")
    architecture = _architecture(manifest)
    if budget:
        budget.check("load final checkpoint")
    assets = _safe(root, manifest["assets"])
    config = AutoConfig.from_pretrained(assets / "encoder", local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True, trust_remote_code=False)
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False, **architecture)
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


def _candidate_record(settings, texts, targets, fit, holdout, root, budget):
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
        manifest, result = train_candidate(texts, targets, fit, holdout, settings, local, budget)
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
            relative + "/candidate_model.ckpt", relative + "/training_metrics.json", relative + "/settings.json"))
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
               "seconds", "stop_reason", "positive_class_weights", "label_correlation"]
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
                             "positive_class_weights": json.dumps(record["settings"]["positive_class_weights"]),
                             "label_correlation": record["manifest"]["label_correlation"]})
    temporary.replace(root / "candidate_summary.csv")


def _final_model(selected, texts, targets, holdout, settings, root, budget):
    checkpoint, ready = root / "mission3_ver16_model.ckpt", root / "final_model_ready.json"
    if ready.is_file():
        saved = _read(ready)
        if saved["selected_candidate"] != selected["candidate_id"]:
            raise ValueError("Final model selection changed during resume")
        _verify_hashes(root, saved["artifact_hashes"], budget)
        return checkpoint, _read(checkpoint)
    manifest = dict(selected["manifest"])
    reason = "budget_before_refit"
    with np.load(_safe(root, selected["directory"] + "/holdout_predictions.npz"), allow_pickle=False) as saved:
        expected_probe = saved["predictions"][:12].copy()
    probe_texts = [texts[i] for i in holdout[:len(expected_probe)]]
    estimate = _refit_estimate(selected, len(targets) / manifest["fitted_samples"])
    if budget.can_start(estimate, training=True):
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
    else:
        print("Full refit does not fit the remaining budget; using the completed fit-only model.", flush=True)
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
    started = time.monotonic()
    settings = validate_settings(settings)
    deadline = settings.get("deadline_monotonic", started + settings["time_budget_minutes"] * 60)
    if deadline > started + settings["time_budget_minutes"] * 60 + 1:
        raise ValueError("Supplied deadline exceeds the allowed execution budget")
    budget = Budget(deadline, finalization_reserve_s=settings["finalization_reserve_seconds"],
                    safety_factor=settings["budget_safety_factor"])
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.iterdir()
                                 if p.is_file() and p.suffix in (".py", ".md", ".txt")}
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
        identity = {"version": "ver16", "training_digest": dataset_digest(texts, targets),
                    "source_digest": _digest({k: v for k, v in sources.items() if not k.endswith(".md")}),
                    "settings_digest": _digest(_stable_settings(settings)), "candidates": CANDIDATES,
                    "model_selection": "Training internal holdout Macro F1 only"}
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
                (submission / name).write_text(source, encoding="utf-8")
        if (root / "run_complete.json").is_file():
            _verify_hashes(root, _read(root / "run_complete.json")["artifact_hashes"], budget)
            print("Run already complete; its saved final artifacts were verified.", flush=True)
            return root
        writable_status = True
        data.write_json(root / "runtime.json", {"status": "running", "seconds": time.monotonic() - started,
                        "message": "Only run_complete.json confirms completed final artifacts."})
        data.write_json(root / ("environment_" + _stamp() + ".json"), _environment())
        data.write_json(root / "experiment_plan.json", {"candidates": CANDIDATES,
                        "positive_class_weights": POSITIVE_CLASS_WEIGHTS, "ce_views": 1,
                        "selection": "Training Macro F1 only", "max_epochs": settings["max_epochs"],
                        "scheduler_horizon_epochs": settings["scheduler_horizon_epochs"],
                        "time_budget_minutes": settings["time_budget_minutes"],
                        "finalization_reserve_seconds": settings["finalization_reserve_seconds"],
                        "budget_fallback": "completed fit-only candidate; never incomplete refit"})
        final_already_ready = (root / "final_model_ready.json").is_file()
        for candidate in CANDIDATES:
            name = candidate["candidate_id"]
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
            config = {**settings, **candidate, "positive_class_weights": list(POSITIVE_CLASS_WEIGHTS)}
            try:
                records.append(_candidate_record(config, texts, targets, fit, holdout, root, budget))
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
                        "metrics": selected["metrics"], "selection_policy": "Training Macro F1 only",
                        "selected_epochs": selected["manifest"]["selected_epochs"], "skipped_candidates": skipped})
        data.write_json(root / "training_metrics.json", selected["metrics"])
        saved = _safe(root, selected["directory"] + "/holdout_predictions.npz")
        shutil.copy2(saved, root / "holdout_predictions.npz")
        with np.load(saved, allow_pickle=False) as values:
            data.export_errors([paths[i] for i in holdout], [texts[i] for i in holdout], targets[holdout],
                               values["predictions"], root / "training_errors")
        checkpoint, manifest = _final_model(selected, texts, targets, holdout, settings, root, budget)
        hashes = _read(root / "final_model_ready.json")["artifact_hashes"]
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
        data.write_json(root / "runtime.json", {"status": "complete_budget_limited" if limited else "complete",
                        "seconds": time.monotonic() - started, "seconds_scope": "worker invocation; notebook deadline includes installation",
                        "candidate_count": len(records), "selected_candidate": selected["candidate_id"],
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
