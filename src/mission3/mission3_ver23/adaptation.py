"""Continue the verified text model with trainable upper encoder layers.

The two predefined recipes keep the recorded labels and raw argmax inference.
R-Drop adds symmetric KL between two stochastic training forwards only. Frozen
prefix dropout is disabled; the upper encoder, global heads and evidence learn.
"""
from __future__ import annotations

import hashlib
from itertools import islice
import math
from numbers import Real
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .budget import BudgetExceeded
from . import data
from .neural import EncodeDataset, training_loss


RECIPES = ("continue_gce", "continue_rdrop")
POSITIVE_WEIGHTS = [1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def configure_adaptation(model, trainable_layers=4):
    """Validate and configure the original twelve-layer RoBERTa architecture."""
    _positive_int(trainable_layers, "trainable_layers")
    encoder = model.encoder
    config = getattr(encoder, "config", None)
    stack = getattr(encoder, "encoder", None)
    layers = getattr(stack, "layer", None)
    embeddings = getattr(encoder, "embeddings", None)
    pooler = getattr(encoder, "pooler", None)
    if (getattr(config, "model_type", None) != "roberta"
            or getattr(config, "num_hidden_layers", None) != 12
            or not isinstance(layers, nn.ModuleList) or len(layers) != 12
            or not isinstance(embeddings, nn.Module) or trainable_layers > 12
            or any(not isinstance(layer, nn.Module) or not list(layer.parameters()) for layer in layers)
            or (pooler is not None and not isinstance(pooler, nn.Module))):
        raise ValueError("Adaptation requires twelve RoBERTa layers and valid embeddings")
    if (model.label_attention or model.contrastive or not isinstance(model.evidence, nn.Module)
            or model.class_weights.shape[0] != 9 or model.class_weights.shape[-1] != 2):
        raise ValueError("Adaptation requires the original nine-head evidence architecture")
    boundary = 12 - trainable_layers
    trainable_modules = [*layers[boundary:], model.projection, model.evidence]
    allowed = {id(parameter) for module in trainable_modules for parameter in module.parameters()}
    allowed.update((id(model.class_weights), id(model.class_bias)))
    frozen_modules = [embeddings, *layers[:boundary]] + ([pooler] if pooler is not None else [])
    frozen = {id(parameter) for module in frozen_modules for parameter in module.parameters()}
    frozen.update((id(model.feature_weights), id(model.feature_bias)))
    if allowed & frozen:
        raise ValueError("Frozen and trainable components must not share parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)
        parameter.grad = None
    checkpointing = bool(getattr(encoder, "supports_gradient_checkpointing", False))
    if checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(config, "use_cache"):
        config.use_cache = False
    audit = {"total_encoder_layers": 12, "trainable_layer_indices": list(range(boundary, 12)),
             "frozen_layer_indices": list(range(boundary)), "frozen_prefix_eval": True,
             "global_projection_trainable": True, "all_nine_heads_trainable": True,
             "evidence_trainable": True, "unused_feature_projection_frozen": True,
             "gradient_checkpointing": checkpointing,
             "gradient_checkpointing_use_reentrant": False,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    model._mission3_adaptation = audit
    enforce_frozen_eval(model)
    return dict(audit)


def enforce_frozen_eval(model):
    audit = getattr(model, "_mission3_adaptation", None)
    if audit is None:
        raise ValueError("Configure adaptation before training")
    model.encoder.embeddings.eval()
    for index in audit["frozen_layer_indices"]:
        model.encoder.encoder.layer[index].eval()
    if getattr(model.encoder, "pooler", None) is not None:
        model.encoder.pooler.eval()


def trainable_state(model):
    if not getattr(model, "_mission3_adaptation", None):
        raise ValueError("Configure adaptation before capturing trainable weights")
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def frozen_digest(model):
    digest = hashlib.sha256()
    frozen = {name: parameter for name, parameter in model.named_parameters() if not parameter.requires_grad}
    frozen.update(dict(model.named_buffers()))
    for name, value in sorted(frozen.items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def restore_trainable_state(model, state):
    """Validate before writes and prove the frozen state remains unchanged."""
    if not getattr(model, "_mission3_adaptation", None):
        raise ValueError("Configure adaptation before restoring trainable weights")
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Trainable state keys do not match this configured model")
    for name, parameter in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("Trainable state has an invalid tensor")
    before = frozen_digest(model)
    changed_tensors, changed_elements = 0, 0
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state[name].to(parameter.device)
            differences = int((parameter != value).sum())
            changed_tensors += int(differences > 0)
            changed_elements += differences
            parameter.copy_(value)
    after = frozen_digest(model)
    if before != after:
        raise RuntimeError("Restoring adaptation weights changed a frozen tensor")
    return {"frozen_state_unchanged": True, "frozen_sha256": after,
            "changed_tensors": changed_tensors, "changed_elements": changed_elements}


def adaptation_loss(logits1, targets, logits2=None, *, recipe="continue_gce", alpha=1.,
                    positive_class_weights=None):
    if recipe not in RECIPES:
        raise ValueError("Unknown adaptation recipe")
    alpha = _number(alpha, "rdrop_alpha")
    weights = POSITIVE_WEIGHTS if positive_class_weights is None else positive_class_weights
    objective = {"positive_class_weights": weights, "loss_mode": "nausea_ce_gce",
                 "gce_q": .7, "gce_mix": .5, "ce_warmup_epochs": 1, "epoch": 2}
    first = training_loss(logits1, targets, **objective)
    if recipe == "continue_gce":
        if logits2 is not None:
            raise ValueError("Single-view continuation does not accept a second view")
        supervised, consistency = first, first.new_zeros(())
    else:
        if logits2 is None or logits2.shape != logits1.shape:
            raise ValueError("R-Drop requires two matching logits tensors")
        second = training_loss(logits2, targets, **objective)
        supervised = (first + second) / 2
        log_first, log_second = F.log_softmax(logits1.float(), -1), F.log_softmax(logits2.float(), -1)
        # Both probability distributions retain gradients, as in symmetric KL.
        consistency = .5 * ((log_first.exp() - log_second.exp()) * (log_first - log_second)).sum(-1).mean()
    total = supervised + alpha * consistency
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Adaptation objective is nonfinite")
    return total, {"supervised_loss": float(supervised.detach()), "consistency_loss": float(consistency.detach())}


def make_loader(model, tokenizer, texts, targets, settings, shuffle=False):
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be boolean")
    dataset = EncodeDataset(texts, targets, tokenizer, max_length=settings.get("max_length", 512),
                            overlap=settings.get("overlap", 64), utterance_evidence=model.evidence is not None)
    generator = torch.Generator().manual_seed(settings.get("seed", 42))
    return DataLoader(dataset, batch_size=_positive_int(settings.get("batch_size", 4), "batch_size"),
                      shuffle=shuffle, collate_fn=dataset.collate, generator=generator, num_workers=0)


def make_optimizer(model, settings, batches, device):
    if not getattr(model, "_mission3_adaptation", None):
        raise ValueError("Configure adaptation before creating its optimizer")
    _positive_int(batches, "batches")
    device = torch.device(device)
    accumulation = _positive_int(settings.get("gradient_accumulation", 4), "gradient_accumulation")
    horizon = _positive_int(settings.get("scheduler_horizon_epochs", 2), "scheduler_horizon_epochs")
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    if not encoder or not heads:
        raise ValueError("Both upper encoder layers and classification paths must train")
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": _number(settings.get("learning_rate", 5e-6), "learning_rate", positive=True)},
        {"params": heads, "lr": _number(settings.get("head_learning_rate", 2e-5), "head_learning_rate", positive=True)}],
        weight_decay=_number(settings.get("weight_decay", .01), "weight_decay"))
    updates = math.ceil(batches / accumulation) * horizon
    warmup = max(1, int(updates * .1))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else max(0., (updates - step) / max(1, updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)
    return optimizer, scheduler, scaler, dtype


def _pilot_report(settings, *, microbatches, measured_batches, measured_seconds, total_batches,
                  budget, device, context):
    if measured_batches < 1 or measured_seconds <= 0:
        raise ValueError("Throughput pilot requires actual measured batches")
    per_batch = measured_seconds / measured_batches
    epoch_seconds = per_batch * total_batches
    evaluate_seconds = per_batch * context.get("holdout_batches", 0)
    refit = bool(context.get("refit", False))
    ratio = _number(context.get("full_ratio", 1.25), "full_ratio", positive=True)
    refit_epochs = _positive_int(context.get("refit_epochs", 1), "refit_epochs")
    remaining_epochs = _positive_int(context.get("remaining_epochs", refit_epochs), "remaining_epochs")
    unfinished = per_batch * max(0, total_batches - microbatches)
    minimum_refit = _number(context.get("minimum_refit_seconds", 0.), "minimum_refit_seconds")
    refit_seconds = 0. if refit else max(epoch_seconds * ratio * refit_epochs + 120., minimum_refit)
    completion_seconds = unfinished + (epoch_seconds * (remaining_epochs - 1) if refit else 0.)
    required_seconds = completion_seconds + evaluate_seconds + refit_seconds + 120.
    feasible = budget.can_start(required_seconds, training=True)
    return {"observed_microbatches": microbatches, "measured_microbatches": measured_batches,
            "observed_optimizer_steps": math.ceil(microbatches / settings.get("gradient_accumulation", 4)),
            "measured_seconds": measured_seconds, "estimated_epoch_seconds": epoch_seconds,
            "estimated_holdout_seconds": evaluate_seconds, "estimated_refit_seconds": refit_seconds,
            "estimated_required_seconds": required_seconds, "remaining_seconds": budget.remaining(),
            "decision": "continue_with_refit_reserve" if feasible else "stop_reserving_complete_refit",
            "device": str(device), "includes_training_updates": True, "resets_or_repeats_updates": False,
            "phase": "full_refit" if refit else "candidate_fit",
            "limitations": "Early shuffled batches estimate throughput; later lengths, I/O and GPU contention may differ."}


def _optimizer_step(model, optimizer, scheduler, scaler):
    """Let FP16 scaling recover from overflow without advancing the schedule."""
    enabled = bool(scaler.is_enabled())
    scaler.unscale_(optimizer)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=not enabled)
    if enabled and not bool(torch.isfinite(norm)):
        # Unscale has already recorded any nonfinite gradient for GradScaler.
        # A nonfinite norm from otherwise finite values is not such an overflow
        # signal and must not silently execute a zeroed optimizer update.
        if all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise FloatingPointError("Nonfinite clipping norm without a GradScaler overflow signal")
    previous_scale = scaler.get_scale() if enabled else None
    scaler.step(optimizer)
    scaler.update()
    successful = not enabled or scaler.get_scale() >= previous_scale
    if successful:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return successful


def train_epoch(model, loader, settings, device, state, budget, epoch, recipe, pilot_context=None):
    """Complete an epoch or raise; partial epoch weights are never publishable.

    Accumulation weights each microbatch by its sample count in the group. This
    remains correct for both a short last microbatch and a short final group.
    The pilot consumes real updates and reserves a complete full-data refit.
    """
    _positive_int(epoch, "epoch")
    if recipe not in RECIPES or len(loader) < 1:
        raise ValueError("An explicit adaptation recipe and nonempty loader are required")
    device = torch.device(device)
    optimizer, scheduler, scaler, dtype = state
    accumulation = _positive_int(settings.get("gradient_accumulation", 4), "gradient_accumulation")
    pilot_steps = _positive_int(settings.get("pilot_optimizer_steps", 10), "pilot_optimizer_steps")
    log_every = _positive_int(settings.get("log_every", 250), "log_every")
    model.train()
    enforce_frozen_eval(model)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0., "supervised_loss": 0., "consistency_loss": 0.}
    counts = torch.zeros(9, dtype=torch.long)
    samples = microbatches = updates = attempts = skipped = 0
    started = time.monotonic()
    measure_started, measure_start_batch = started, 0
    warmup_recorded = False
    report = None
    if pilot_context is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    try:
        for group_start in range(0, len(loader), accumulation):
            budget.check("adaptation accumulation group", training=True)
            group = list(islice(iterator, accumulation))
            if len(group) != min(accumulation, len(loader) - group_start):
                raise ValueError("Loader length changed during adaptation")
            group_samples = sum(int(batch["targets"].shape[0]) for batch in group)
            if group_samples < 1:
                raise ValueError("Adaptation received an empty accumulation group")
            for batch in group:
                budget.check("adaptation microbatch", training=True)
                batch_samples = int(batch["targets"].shape[0])
                if batch_samples < 1:
                    raise ValueError("Adaptation received an empty microbatch")
                counts += batch["targets"].detach().cpu().sum(0).long()
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logits1, _ = model(batch)
                    logits2 = model(batch)[0] if recipe == "continue_rdrop" else None
                    loss, details = adaptation_loss(logits1, batch["targets"], logits2, recipe=recipe,
                        alpha=settings.get("rdrop_alpha", 1.),
                        positive_class_weights=settings.get("positive_class_weights", POSITIVE_WEIGHTS))
                scaler.scale(loss * (batch_samples / group_samples)).backward()
                for key, value in {"loss": float(loss.detach()), **details}.items():
                    totals[key] += value * batch_samples
                samples += batch_samples
                microbatches += 1
            budget.check("adaptation optimizer update", training=True)
            successful = _optimizer_step(model, optimizer, scheduler, scaler)
            attempts += 1
            updates += int(successful)
            skipped += int(not successful)
            if (pilot_context is not None and successful and not warmup_recorded
                    and updates == 1 and microbatches < len(loader)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                measure_started, measure_start_batch = time.monotonic(), microbatches
                warmup_recorded = True
            if (pilot_context is not None and report is None
                    and (updates >= pilot_steps + 1 or microbatches == len(loader))):
                if updates == 0:
                    raise FloatingPointError("No successful optimizer update completed this epoch")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                report = _pilot_report(settings, microbatches=microbatches,
                    measured_batches=microbatches - measure_start_batch,
                    measured_seconds=max(time.monotonic() - measure_started, 1e-9), total_batches=len(loader),
                    budget=budget, device=device, context=pilot_context)
                report.update({"observed_optimizer_steps": updates, "observed_optimizer_attempts": attempts,
                               "skipped_optimizer_steps": skipped, "warmup_update_excluded": warmup_recorded})
                report["peak_cuda_memory_gb"] = (torch.cuda.max_memory_allocated(device) / 1024**3
                                                  if device.type == "cuda" else 0.)
                for path in pilot_context.get("report_paths", []):
                    data.write_json(path, report)
                print(f"{recipe} pilot: estimated epoch {report['estimated_epoch_seconds'] / 60:.1f} min; "
                      f"{report['decision']}", flush=True)
                if report["decision"] != "continue_with_refit_reserve":
                    raise BudgetExceeded("adaptation pilot reserves a complete refit")
            if microbatches % log_every == 0:
                print(f"{recipe} epoch {epoch}: {microbatches}/{len(loader)} batches; "
                      f"{budget.remaining() / 60:.1f} minutes left", flush=True)
        if updates == 0:
            raise FloatingPointError("No successful optimizer update completed this epoch")
        budget.check("completed adaptation epoch", training=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {"epoch": epoch, **{key: value / samples for key, value in totals.items()},
            "classification_loss": totals["supervised_loss"] / samples,
            "configured_loss_mode": "nausea_ce_gce", "active_loss_mode": "nausea_ce_gce",
            "gce_q": .7, "gce_mix": .5, "source_warmup_already_completed": True,
            "ce_views": 2 if recipe == "continue_rdrop" else 1,
            "rdrop_alpha": settings.get("rdrop_alpha", 1.) if recipe == "continue_rdrop" else 0.,
            "scheduler_horizon_epochs": settings.get("scheduler_horizon_epochs", 2),
            "samples": samples, "microbatches": microbatches, "optimizer_steps": updates,
            "optimizer_attempts": attempts, "skipped_optimizer_steps": skipped,
            "positive_target_counts": counts.tolist(), "training_seconds": time.monotonic() - started,
            "pilot_report": report}
