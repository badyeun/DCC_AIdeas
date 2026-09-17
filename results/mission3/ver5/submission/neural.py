"""Warm-started token-evidence residual and completed-epoch training resume.

The saved ver4 encoder/global head is retained. A zero-initialized residual
adds label-specific token evidence without changing initial inference logits.
Resume restores the last *completed* epoch; an interrupted epoch is replayed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

if __package__:
    from .ver4_runtime.neural import (
        DialogueClassifier as LegacyDialogueClassifier,
        DialogueCollator, DialogueDataset, NUM_LABELS, _amp, _device, _loader,
        _optimizer, _predict, _targets, tokenize_dialogues,
    )
else:
    from ver4_runtime.neural import (
        DialogueClassifier as LegacyDialogueClassifier,
        DialogueCollator, DialogueDataset, NUM_LABELS, _amp, _device, _loader,
        _optimizer, _predict, _targets, tokenize_dialogues,
    )


MODEL_FORMAT = "mission3_ver5_neural"
LAST_FORMAT = "mission3_ver5_last"
DEFAULTS = {
    "warmup_epochs": 1, "finetune_epochs": 1, "patience": 2,
    "finetune_top_layers": 4,
    "learning_rate": 5e-6, "head_learning_rate": 1e-4,
    "warmup_head_learning_rate": 3e-4, "attention_dim": 256, "dropout": 0.2,
    "batch_size": 2, "window_batch_size": 2,
    "gradient_accumulation_steps": 16, "inference_batch_size": 8,
    "gradient_checkpointing": True, "device": "cuda", "log_every": 100,
    "layerwise_lr_decay": 0.95, "weight_decay": 0.01,
    "pos_weight_power": 0.5, "warmup_ratio": 0.1,
}


class DialogueClassifier(LegacyDialogueClassifier):
    """One encoder forward supplies both global and label-wise token evidence."""

    def __init__(self, encoder, num_labels=NUM_LABELS, dropout=0.2,
                 window_batch_size=8, attention_dim=256, special_token_ids=()):
        super().__init__(encoder, num_labels, dropout, window_batch_size)
        if int(attention_dim) < 1:
            raise ValueError("attention_dim must be positive")
        self.attention_dim = int(attention_dim)
        self.special_token_ids = tuple(sorted(set(int(x) for x in special_token_ids)))
        self.token_projection = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, self.attention_dim), nn.Tanh(),
        )
        self.token_queries = nn.Parameter(torch.empty(num_labels, self.attention_dim))
        nn.init.normal_(self.token_queries, std=0.02)
        # Initial residual is exactly zero, including for empty dialogues.
        self.token_evidence_weights = nn.Parameter(torch.zeros(num_labels, self.attention_dim))
        self.token_evidence_bias = nn.Parameter(torch.zeros(num_labels))

    def _pool_tokens(self, hidden, input_ids, attention_mask):
        projected = self.token_projection(hidden)
        valid = attention_mask.bool()
        for token_id in self.special_token_ids:
            valid = valid & input_ids.ne(token_id)
        # Autocast can lower a matmul even when both operands were explicitly
        # cast to float32; normalize the resulting scores in float32 as well.
        scores = (projected.float() @ self.token_queries.float().T / math.sqrt(self.attention_dim)).float()
        scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid.unsqueeze(-1)
        # All-special/empty windows produce zero evidence, never NaN.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("wsl,wsd->wld", weights.to(projected.dtype), projected)

    def forward(self, doc_indices, **inputs):
        representations, token_representations = [], []
        for start in range(0, len(doc_indices), self.window_batch_size):
            window_inputs = {key: value[start:start + self.window_batch_size]
                             for key, value in inputs.items()}
            hidden = self.encoder(**window_inputs).last_hidden_state
            mask = window_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            representations.append(self.projection(torch.cat([hidden[:, 0], mean], -1)))
            token_representations.append(self._pool_tokens(
                hidden, window_inputs["input_ids"], window_inputs["attention_mask"],
            ))
        windows = self.dropout(torch.cat(representations))
        token_windows = torch.cat(token_representations)
        logits = []
        for document_index in range(int(doc_indices.max().item()) + 1):
            selected_indices = doc_indices == document_index
            selected = windows[selected_indices]
            scores = selected @ self.attention_queries.T / math.sqrt(selected.shape[-1])
            attention = torch.softmax(scores.float(), dim=0).to(selected.dtype)
            pooled = attention.T @ selected
            global_logits = (self.dropout(pooled) * self.label_weights).sum(-1) + self.label_bias
            token_pooled = (token_windows[selected_indices] * attention.unsqueeze(-1)).sum(dim=0)
            evidence = (self.dropout(token_pooled) * self.token_evidence_weights).sum(-1)
            logits.append(global_logits + evidence + self.token_evidence_bias)
        return torch.stack(logits)


def _read_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)


def _encoder(configuration):
    values = dict(configuration)
    model_type = values.pop("model_type")
    return AutoModel.from_config(AutoConfig.for_model(model_type, **values))


def _special_ids(tokenizer):
    return sorted({int(value) for name in ("pad_token_id", "cls_token_id", "sep_token_id",
                                         "bos_token_id", "eos_token_id")
                   if (value := getattr(tokenizer, name, None)) is not None})


def warm_start(init_checkpoint_path, tokenizer, config):
    """Create a residual model from a ver4 checkpoint without a model download."""
    checkpoint = _read_checkpoint(init_checkpoint_path)
    if checkpoint.get("format_version") != 1 or "model_state" not in checkpoint:
        raise ValueError("Expected a ver4 fold checkpoint for warm start")
    if checkpoint.get("format") in (MODEL_FORMAT, LAST_FORMAT):
        raise ValueError("Warm start requires the original ver4 fold checkpoint")
    architecture = dict(checkpoint["architecture"])
    architecture.update(
        attention_dim=int(config.get("attention_dim", DEFAULTS["attention_dim"])),
        dropout=float(config.get("dropout", architecture["dropout"])),
        window_batch_size=int(config.get("window_batch_size", architecture["window_batch_size"])),
        special_token_ids=_special_ids(tokenizer),
    )
    model = DialogueClassifier(_encoder(checkpoint["encoder_config"]), **architecture)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    expected = {key for key in model.state_dict() if key.startswith("token_")}
    if set(missing) != expected or unexpected:
        raise ValueError(f"Warm-start parameter mismatch: missing={missing}, unexpected={unexpected}")
    metadata = {
        "encoder_config": checkpoint["encoder_config"], "architecture": architecture,
        "model_name": checkpoint.get("model_name"),
        "init_epoch": checkpoint.get("epoch"), "init_seed": checkpoint.get("seed"),
    }
    return model, metadata


def _from_saved(checkpoint, config):
    if checkpoint.get("format") not in (MODEL_FORMAT, LAST_FORMAT) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported ver5 neural checkpoint")
    architecture = dict(checkpoint["architecture"])
    if "window_batch_size" in config:
        architecture["window_batch_size"] = int(config["window_batch_size"])
    model = DialogueClassifier(_encoder(checkpoint["encoder_config"]), **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def predict_checkpoint(checkpoint_path, documents, tokenizer, config):
    device = _device(config)
    checkpoint = _read_checkpoint(checkpoint_path)
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("Inference requires the best ver5 checkpoint, not a resume checkpoint")
    model = _from_saved(checkpoint, config).to(device)
    del checkpoint
    loader = _loader(documents, None, tokenizer, config.get("inference_batch_size", 8))
    probabilities = _predict(model, loader, device).astype(np.float32, copy=False)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Inference produced non-finite probabilities")
    return probabilities


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _settings(config):
    settings = {name: config.get(name, value) for name, value in DEFAULTS.items()}
    for name in ("warmup_epochs", "finetune_epochs", "patience", "batch_size", "window_batch_size",
                 "gradient_accumulation_steps", "inference_batch_size", "attention_dim"):
        value = settings[name]
        minimum = 0 if name == "warmup_epochs" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("learning_rate", "head_learning_rate", "warmup_head_learning_rate"):
        if not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    top_layers = settings["finetune_top_layers"]
    if isinstance(top_layers, bool) or not isinstance(top_layers, int) or top_layers < -1:
        raise ValueError("finetune_top_layers must be an integer >= -1 (-1 means the full encoder)")
    if not 0 <= float(settings["dropout"]) < 1 or not 0 <= float(settings["warmup_ratio"]) < 1:
        raise ValueError("dropout/warmup_ratio must be in [0, 1)")
    if float(settings["weight_decay"]) < 0 or not math.isfinite(float(settings["pos_weight_power"])):
        raise ValueError("Invalid weight decay or positive-class weight power")
    return settings


def _fingerprint(settings, seed, init_checkpoint_path):
    path = Path(init_checkpoint_path).resolve()
    data = {"settings": settings, "seed": int(seed), "init_checkpoint": str(path)}
    # The caller's run_signature binds data/code/content; this guards accidental
    # config changes even when a caller incorrectly reuses the same signature.
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values["name"], np.asarray(values["keys"], dtype=np.uint32),
                         values["position"], values["has_gauss"], values["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed; exact epoch resume is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _phase(epoch, settings):
    return "warmup" if epoch <= settings["warmup_epochs"] else "finetune"


def _finetune_blocks(model, settings):
    """Locate the top BERT/RoBERTa blocks without changing checkpoint structure."""
    count = settings["finetune_top_layers"]
    if count <= 0:
        return []
    blocks = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("Partial fine-tuning requires a BERT/RoBERTa encoder.layer block list")
    # Tiny local test models can have fewer blocks than the production model.
    return list(blocks[-min(count, len(blocks)):])


def _configure_phase(model, phase, settings):
    model.encoder.requires_grad_(False)
    if phase == "finetune":
        if settings["finetune_top_layers"] == -1:
            model.encoder.requires_grad_(True)
        else:
            for block in _finetune_blocks(model, settings):
                block.requires_grad_(True)
    train_encoder = any(parameter.requires_grad for parameter in model.encoder.parameters())
    if train_encoder and settings["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.encoder.gradient_checkpointing_disable()
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False


def _train_mode(model, phase, settings):
    """Keep frozen feature extraction deterministic while training the top blocks."""
    model.train()
    model.encoder.eval()
    if phase == "finetune":
        if settings["finetune_top_layers"] == -1:
            model.encoder.train()
        else:
            for block in _finetune_blocks(model, settings):
                block.train()


def _parameter_counts(model):
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()
                                            if parameter.requires_grad),
    }


def _phase_optimizer(model, phase, settings, batches):
    options = dict(settings)
    if phase == "warmup":
        options["head_learning_rate"] = settings["warmup_head_learning_rate"]
    optimizer = _optimizer(model, options)
    epoch_count = settings["warmup_epochs"] if phase == "warmup" else settings["finetune_epochs"]
    steps = math.ceil(batches / settings["gradient_accumulation_steps"]) * epoch_count
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * settings["warmup_ratio"]), num_training_steps=steps,
    )
    return optimizer, scheduler


def _score(model, loader, device, targets):
    probabilities = _predict(model, loader, device)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Stopping predictions contain non-finite values")
    valid = targets.sum(axis=0) > 0
    if not valid.any():
        raise ValueError("Stopping split has no positive labels")
    return float(average_precision_score(targets[:, valid], probabilities[:, valid], average="macro")), int(valid.sum())


def _check_identity(checkpoint, run_signature, configuration_fingerprint):
    if checkpoint.get("run_signature") != run_signature:
        raise ValueError("Checkpoint run signature mismatch; use a new output directory")
    if checkpoint.get("configuration_fingerprint") != configuration_fingerprint:
        raise ValueError("Checkpoint config fingerprint mismatch; use a new output directory")


def _best_payload(last):
    keep = ("format_version", "encoder_config", "architecture", "model_name", "init_epoch", "init_seed",
            "model_state", "seed", "run_signature", "configuration_fingerprint", "settings")
    return {**{key: last[key] for key in keep}, "format": MODEL_FORMAT,
            "epoch": last["best_epoch"], "stop_ap": last["best_stop_ap"]}


def train_model(train_documents, train_targets, stop_documents, stop_targets,
                tokenizer, config, checkpoint_path, seed, init_checkpoint_path, run_signature):
    """Adapt one clean fold and resume only from a fully saved epoch boundary.

    The supplied stopping documents must be the original warm-start fold's stop
    split. Base ensemble predictions and calibration/Validation targets are not
    accepted here. Phase transitions deliberately start a fresh optimizer.
    """
    train_targets = _targets(train_targets, train_documents, "train")
    stop_targets = _targets(stop_targets, stop_documents, "stop")
    if not isinstance(run_signature, str) or not run_signature:
        raise ValueError("run_signature must be a nonempty string")
    settings = _settings(config)
    fingerprint = _fingerprint(settings, seed, init_checkpoint_path)
    device = _device(settings)
    checkpoint_path = Path(checkpoint_path)
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.ckpt")
    best_checkpoint = _read_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    if best_checkpoint is not None:
        if best_checkpoint.get("format") != MODEL_FORMAT:
            raise ValueError("Existing best checkpoint is not a ver5 model")
        _check_identity(best_checkpoint, run_signature, fingerprint)
    last = _read_checkpoint(last_path) if last_path.exists() else None
    if last is not None:
        if last.get("format") != LAST_FORMAT or last.get("format_version") != 1:
            raise ValueError("Unsupported epoch-resume checkpoint")
        _check_identity(last, run_signature, fingerprint)
        # Last is committed first. A crash before writing best is recoverable
        # because the newly selected best weights are exactly last.model_state.
        best_matches = (best_checkpoint is not None and best_checkpoint.get("epoch") == last["best_epoch"]
                        and best_checkpoint.get("stop_ap") == last["best_stop_ap"])
        if not best_matches:
            if last["best_epoch"] != last["completed_epochs"]:
                raise ValueError("Best checkpoint is missing/inconsistent with the resume record")
            _atomic_save(_best_payload(last), checkpoint_path)
        if last["completed"]:
            return {"best_epoch": last["best_epoch"], "best_stop_ap": last["best_stop_ap"],
                    "history": last["history"], "completed_epochs": last["completed_epochs"],
                    "resumed": True, "last_checkpoint": str(last_path)}
    elif best_checkpoint is not None:
        raise ValueError("Best checkpoint exists without epoch-resume state; use a new output directory")
    del best_checkpoint

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if last is None:
        model, metadata = warm_start(init_checkpoint_path, tokenizer, settings)
    else:
        model = _from_saved(last, settings)
        metadata = {key: last[key] for key in ("encoder_config", "architecture", "model_name", "init_epoch", "init_seed")}
    model.to(device)
    train_loader = _loader(train_documents, train_targets, tokenizer, settings["batch_size"], shuffle=True, seed=seed)
    stop_loader = _loader(stop_documents, None, tokenizer, settings["inference_batch_size"])
    positives = train_targets.sum(axis=0)
    if (positives == 0).any():
        raise ValueError("Every training label needs positive examples")
    pos_weight = np.clip(((len(train_targets) - positives) / positives).clip(min=1)
                         ** settings["pos_weight_power"], 1, 5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    completed_epochs = last["completed_epochs"] if last else 0
    phase = _phase(completed_epochs + 1, settings)
    _configure_phase(model, phase, settings)
    optimizer, scheduler = _phase_optimizer(model, phase, settings, len(train_loader))
    if last is not None:
        if phase == last["phase"]:
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        _restore_rng(last["rng"], train_loader.generator)
        history = list(last["history"])
        best_epoch, best_ap, stale = last["best_epoch"], last["best_stop_ap"], last["stale_finetune_epochs"]
        print(f"완료된 epoch {completed_epochs}부터 재개합니다. 중단된 epoch는 처음부터 다시 실행합니다.", flush=True)
    else:
        baseline_started = time.monotonic()
        best_ap, valid_count = _score(model, stop_loader, device, stop_targets)
        best_epoch, stale = 0, 0
        history = [{"epoch": 0, "phase": "baseline", "phase_epoch": 0, "train_loss": None,
                    "stop_macro_ap": best_ap, "stop_labels_with_positives": valid_count,
                    "minutes": (time.monotonic() - baseline_started) / 60}]
        print(f"epoch=0 baseline stop_AP={best_ap:.6f} (warm-start model, no training)", flush=True)
    resumed = last is not None
    del last

    def save_state(epoch, done, improved):
        print(f"epoch={epoch} 재개 상태 저장 중: {last_path.name} (모델·optimizer 포함)", flush=True)
        state = {"format": LAST_FORMAT, "format_version": 1, **metadata,
                 "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                 "seed": int(seed), "run_signature": run_signature,
                 "configuration_fingerprint": fingerprint, "settings": settings,
                 "completed_epochs": epoch, "phase": phase, "completed": done,
                 "best_epoch": best_epoch, "best_stop_ap": best_ap,
                 "stale_finetune_epochs": stale, "history": history,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": _rng_state(train_loader.generator)}
        _atomic_save(state, last_path)
        if improved:
            print(f"epoch={epoch} 최적 가중치 저장 중: {checkpoint_path.name}", flush=True)
            _atomic_save(_best_payload(state), checkpoint_path)

    if not resumed:
        save_state(0, False, True)
    total_epochs = settings["warmup_epochs"] + settings["finetune_epochs"]
    accumulation = settings["gradient_accumulation_steps"]
    for epoch in range(completed_epochs + 1, total_epochs + 1):
        next_phase = _phase(epoch, settings)
        if next_phase != phase:
            phase = next_phase
            _configure_phase(model, phase, settings)
            optimizer, scheduler = _phase_optimizer(model, phase, settings, len(train_loader))
            stale = 0
        epoch_started = time.monotonic()
        _train_mode(model, phase, settings)
        parameter_counts = _parameter_counts(model)
        print(f"phase={phase} trainable_parameters={parameter_counts['trainable_parameters']:,}/"
              f"{parameter_counts['total_parameters']:,} "
              f"encoder_trainable={parameter_counts['trainable_encoder_parameters']:,}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, document_count = 0.0, 0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = (step // accumulation) * accumulation
            group_documents = min(accumulation * train_loader.batch_size,
                                  len(train_documents) - group_start * train_loader.batch_size)
            with _amp(device):
                logits = model(**batch)
            # Float32 BCE and actual-document weighting include a short final group.
            loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={step}")
            scaler.scale(loss * len(labels) / group_documents).backward()
            loss_sum += float(loss.detach()) * len(labels)
            document_count += len(labels)
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if settings["log_every"] > 0 and (step + 1) % settings["log_every"] == 0:
                training_elapsed = time.monotonic() - epoch_started
                remaining_training = training_elapsed / (step + 1) * (len(train_loader) - step - 1)
                print(f"seed={seed} phase={phase} epoch={epoch}/{total_epochs} "
                      f"batch={step + 1}/{len(train_loader)} loss={loss_sum / document_count:.5f} "
                      f"minutes={training_elapsed / 60:.1f} "
                      f"train_ETA_minutes={remaining_training / 60:.1f} (평가·저장 제외)", flush=True)
        score, valid_count = _score(model, stop_loader, device, stop_targets)
        row = {"epoch": epoch, "phase": phase,
               "phase_epoch": epoch if phase == "warmup" else epoch - settings["warmup_epochs"],
               "train_loss": loss_sum / document_count, "stop_macro_ap": score,
               "stop_labels_with_positives": valid_count, "minutes": (time.monotonic() - epoch_started) / 60,
               **parameter_counts}
        history.append(row)
        improved = score > best_ap + 1e-6
        if improved:
            best_epoch, best_ap, stale = epoch, score, 0
        elif phase == "finetune":
            stale += 1
        completed_epochs = epoch
        done = epoch == total_epochs or (phase == "finetune" and stale >= settings["patience"])
        save_state(epoch, done, improved)
        print(f"phase={phase} epoch={epoch}/{total_epochs} loss={row['train_loss']:.5f} "
              f"stop_AP={score:.6f} best_epoch={best_epoch} minutes={row['minutes']:.1f}", flush=True)
        if done:
            break
    return {"best_epoch": best_epoch, "best_stop_ap": best_ap, "history": history,
            "completed_epochs": completed_epochs, "resumed": resumed, "last_checkpoint": str(last_path)}
