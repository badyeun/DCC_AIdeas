"""Strict offline inference and publication of complete ver23 models."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch

from . import adaptation, data, head


ENSEMBLE_FORMAT = "mission3_categorical_ensemble_v1"
NEURAL_FORMAT = "mission3_categorical_v1"


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Model manifests must be JSON objects")
    return value


def _safe(root, relative):
    if (not isinstance(relative, str) or not relative or "\x00" in relative
            or PureWindowsPath(relative).drive or PureWindowsPath(relative).root
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative.replace("\\", "/")).parts):
        raise ValueError("Portable model paths must be relative and cannot contain parent traversal")
    return head._safe(root, relative)


def _check(budget, stage, training=False):
    if budget is not None:
        budget.check(stage, training=training)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probabilities(logits):
    values = np.asarray(logits)
    if (values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (9, 2)
            or not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating)
            or not np.isfinite(values).all()):
        raise ValueError("Probabilities require finite real logits with shape (samples, 9, 2)")
    values = values.astype(np.float64)
    exponent = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def predict_loaded_logits(model, tokenizer, texts, manifest, device, budget=None, training=False):
    """The original ordered FP32 inference path, without stochastic views."""
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    texts = list(texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Inference requires nonempty text-only dialogues")
    if manifest.get("batch_size") != 4 or manifest.get("encoder_window_batch") != 4:
        raise ValueError("Portable ver23 inference preserves the source batch and window sizes")
    _check(budget, "portable tokenization", training)
    loader = adaptation.make_loader(model, tokenizer, texts, None, manifest, shuffle=False)
    _check(budget, "portable tokenization complete", training)
    device = torch.device(device)
    model.float().to(device).eval()
    model.encoder_window_batch = 4
    blocks = []
    with torch.inference_mode():
        for batch in loader:
            _check(budget, "portable FP32 inference", training)
            logits, _ = model({key: value.to(device) for key, value in batch.items()})
            values = logits.detach().float().cpu().numpy()
            if values.ndim != 3 or values.shape[1:] != (9, 2) or not np.isfinite(values).all():
                raise FloatingPointError("Portable model produced invalid logits")
            blocks.append(values)
    if not blocks:
        raise ValueError("Portable inference produced no batches")
    result = np.concatenate(blocks).astype(np.float32, copy=False)
    if len(result) != len(texts):
        raise ValueError("Portable inference row count differs from input")
    _check(budget, "portable FP32 inference complete", training)
    return result


def predict_logits(checkpoint, texts, budget=None, training=False):
    checkpoint = Path(checkpoint)
    manifest = _read(checkpoint)
    if manifest.get("format") != NEURAL_FORMAT or manifest.get("kind") != "neural":
        raise ValueError("Logit inference requires one neural component; nested ensembles are forbidden")
    _safe(checkpoint.parent, manifest.get("weights"))
    _safe(checkpoint.parent, manifest.get("assets"))
    _check(budget, "load portable neural model", training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, loaded = head.load_model(checkpoint, device)
    try:
        return predict_loaded_logits(model, tokenizer, texts, loaded, device, budget, training)
    finally:
        del model, tokenizer
        _release()


def predict_checkpoint(checkpoint, texts, budget=None, training=False):
    checkpoint = Path(checkpoint)
    manifest = _read(checkpoint)
    texts = list(texts)
    if manifest.get("format") == NEURAL_FORMAT and manifest.get("kind") == "neural":
        return predict_logits(checkpoint, texts, budget, training).argmax(-1).astype(np.int8)
    components = manifest.get("components")
    if (manifest.get("format") != ENSEMBLE_FORMAT or manifest.get("kind") != "ensemble"
            or manifest.get("version") != "ver23" or tuple(manifest.get("labels", ())) != data.TARGET_SYMPTOMS
            or manifest.get("combination") != "equal_probability_mean"
            or not isinstance(components, list) or len(components) != 2
            or "weights" in manifest or "component_weights" in manifest):
        raise ValueError("Unsupported ensemble; exactly two equally weighted neural components are required")
    paths = [_safe(checkpoint.parent, relative) for relative in components]
    if paths[0] == paths[1]:
        raise ValueError("Ensemble components must be distinct checkpoints")
    # Validate both components before loading either expensive model.
    for path in paths:
        component = _read(path)
        if (component.get("format") != NEURAL_FORMAT or component.get("kind") != "neural"
                or tuple(component.get("labels", ())) != data.TARGET_SYMPTOMS):
            raise ValueError("Nested or incompatible ensemble components are forbidden")
    first = probabilities(predict_logits(paths[0], texts, budget, training))
    second = probabilities(predict_logits(paths[1], texts, budget, training))
    if first.shape != second.shape:
        raise ValueError("Ensemble component predictions have different shapes")
    return ((first + second) * .5).argmax(-1).astype(np.int8)


def _hash(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            _check(budget, "hash export source")
            digest.update(block)
    return digest.hexdigest()


def _copy(source, target, budget):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("Export must not overwrite its source")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    _check(budget, "copy export artifact")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(4 * 1024 * 1024):
            _check(budget, "copy export artifact")
            writer.write(block)
    _check(budget, "publish copied artifact")
    temporary.replace(target)


def export(source, selected, trained_state, settings, root, budget, fallback_reason):
    """Publish a fully refitted model, fixed mean, or byte-preserved baseline.

    ``trained_state`` must come from complete all-Training continuation. Runtime
    orchestration owns that provenance and the subsequent round-trip inference
    check; this helper never treats a partial state as a completed fit.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode = selected.get("mode")
    if mode not in ("single", "mean", "baseline"):
        raise ValueError("Export mode must be single, mean, or baseline")
    if trained_state is None:
        if mode != "baseline" and not fallback_reason:
            raise ValueError("Missing completed adaptation state without an explicit fallback reason")
        actual_mode, actual_candidate, added_epochs = "baseline", "baseline", 0
    else:
        if (mode == "baseline" or selected.get("recipe") not in adaptation.RECIPES
                or type(selected.get("epoch")) is not int or selected["epoch"] not in (1, 2)):
            raise ValueError("Completed adaptation requires a predefined recipe and one or two epochs")
        actual_mode, actual_candidate, added_epochs = mode, selected["candidate_id"], selected["epoch"]
    checkpoint = Path(source["full_checkpoint"])
    original = _read(checkpoint)
    if (original.get("training_scope") != "all_training" or original.get("fitted_samples") != 29200
            or original.get("version") != "ver19"):
        raise ValueError("Export must start from the verified full-Training ver19 source")
    source_weights = _safe(checkpoint.parent, original["weights"])
    source_assets = _safe(checkpoint.parent, original["assets"])
    if not source_assets.is_dir():
        raise ValueError("Source model assets must be a directory")
    files = sorted(path for path in source_assets.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Source model assets are empty")
    for path in files:
        if not path.resolve().is_relative_to(source_assets.resolve()):
            raise ValueError("Source model asset escapes its asset directory")
        _copy(path, root / "final_model" / "assets" / path.relative_to(source_assets), budget)
    source_hash = _hash(source_weights, budget)
    common = {**original, "version": "ver23", "source_version": "ver19",
              "assets": "final_model/assets", "training_scope": "all_training", "fitted_samples": 29200,
              "source_full_weights_sha256": source_hash, "source_selected_epochs": original.get("selected_epochs"),
              "source_scheduler_horizon_epochs": original.get("scheduler_horizon_epochs"),
              "source_trained_gce_epochs": original.get("trained_gce_epochs", 0),
              "requested_candidate": selected.get("candidate_id"), "refit_skipped_reason": fallback_reason}
    baseline_manifest = {**common, "weights": "final_model/baseline_weights.pt", "added_epochs": 0,
                         "baseline_reused": True, "selected_candidate": "baseline", "mode": "baseline"}
    if actual_mode in ("baseline", "mean"):
        _copy(source_weights, root / baseline_manifest["weights"], budget)
        if _hash(root / baseline_manifest["weights"], budget) != source_hash:
            raise ValueError("Copied baseline weights do not match the verified source")
    adapted_manifest = None
    if trained_state is not None:
        _check(budget, "load final source for adaptation export")
        model, tokenizer, loaded = head.load_model(checkpoint, "cpu")
        try:
            config_audit = adaptation.configure_adaptation(model, settings.get("trainable_layers", 4))
            restoration = adaptation.restore_trainable_state(model, trained_state)
            weights = root / "final_model" / "adapted_weights.pt"
            temporary = weights.with_name(weights.name + ".tmp")
            _check(budget, "save completed adaptation weights")
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
            _check(budget, "publish completed adaptation weights")
            temporary.replace(weights)
            adapted_manifest = {**common, "weights": "final_model/adapted_weights.pt", "mode": "single",
                "added_epochs": added_epochs,
                "selected_epochs": int(original.get("selected_epochs", 0)) + added_epochs,
                "trained_gce_epochs": int(original.get("trained_gce_epochs", 0)) + added_epochs,
                "scheduler_horizon_epochs": 2,
                "loss_schedule": list(original.get("loss_schedule", [])) + [
                    {"epoch": int(original.get("selected_epochs", 0)) + added,
                     "active_loss_mode": "nausea_ce_gce", "recipe": selected["recipe"]}
                    for added in range(1, added_epochs + 1)],
                "baseline_reused": False, "selected_candidate": actual_candidate, "recipe": selected["recipe"],
                "adaptation_training": {"scope": "all_training", "fitted_samples": 29200,
                    "epochs": added_epochs, "learning_rate": settings.get("learning_rate", 5e-6),
                    "head_learning_rate": settings.get("head_learning_rate", 2e-5),
                    "weight_decay": settings.get("weight_decay", .01),
                    "gradient_accumulation": settings.get("gradient_accumulation", 4),
                    "scheduler_horizon_epochs": 2, "gce_q": .7, "gce_mix": .5,
                    "rdrop_alpha": settings.get("rdrop_alpha", 1.) if selected["recipe"] == "continue_rdrop" else 0.,
                    **config_audit}, "parameter_audit": restoration}
        finally:
            del model, tokenizer
            _release()
    final_path = root / "mission3_ver23_model.ckpt"
    if actual_mode == "baseline":
        manifest = baseline_manifest
    elif actual_mode == "single":
        manifest = adapted_manifest
    else:
        data.write_json(root / "baseline_component.ckpt", baseline_manifest)
        data.write_json(root / "adapted_component.ckpt", adapted_manifest)
        manifest = {key: value for key, value in adapted_manifest.items() if key not in ("weights", "assets")}
        manifest.update({"format": ENSEMBLE_FORMAT, "kind": "ensemble", "mode": "mean",
                         "components": ["baseline_component.ckpt", "adapted_component.ckpt"],
                         "combination": "equal_probability_mean", "baseline_reused": False,
                         "decision_rule": "argmax of the equal mean of two class-probability vectors"})
    _check(budget, "publish final model manifest")
    data.write_json(final_path, manifest)
    return final_path, manifest
