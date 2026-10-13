"""Learn contextual nausea evidence while preserving the frozen ver19 model.

All encoder windows are evaluated once per source checkpoint. Exact FP32
utterance representations are stored as ragged arrays; no turn is truncated.
Only the existing nausea evidence module is optimized on original fit labels.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import data
from .budget import BudgetExceeded
from .head import (FEATURE_DIM, NAUSEA, OTHER_LABELS, _finite, _indices, _key,
                   _positive_int, _state_hashes, _sync, load_model)
from .neural import EncodeDataset, UtteranceEvidence, seed_everything


RECIPES = ("evidence_ce", "evidence_ce_gce")
STATE_PREFIX = "initial_evidence__"
STATE_KEYS = ("projection.0.weight", "projection.0.bias", "attention.weight",
              "attention.bias", "output.weight", "output.bias")
ROW_KEYS = ("global_features", "global_logits", "evidence", "logits", "predictions")


def _settings(settings):
    settings = settings or {}
    seed = settings.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    result = {
        "seed": seed,
        "max_evidence_epochs": _positive_int(settings.get("max_evidence_epochs", 8), "max_evidence_epochs"),
        "evidence_batch_size": _positive_int(settings.get("evidence_batch_size", 128), "evidence_batch_size"),
        "evidence_learning_rate": _finite(settings.get("evidence_learning_rate", 1e-4), "evidence_learning_rate", strict=True),
        "evidence_weight_decay": _finite(settings.get("evidence_weight_decay", .01), "evidence_weight_decay"),
        "early_stopping_patience": _positive_int(settings.get("early_stopping_patience", 3), "early_stopping_patience"),
        "positive_weight": _finite(settings.get("positive_weight", 3.), "positive_weight", strict=True),
        "gce_q": _finite(settings.get("gce_q", .7), "gce_q", strict=True),
        "gce_mix": _finite(settings.get("gce_mix", .5), "gce_mix"),
        "ce_warmup_epochs": _positive_int(settings.get("ce_warmup_epochs", 1), "ce_warmup_epochs"),
    }
    if result["gce_q"] > 1 or result["gce_mix"] > .5:
        raise ValueError("GCE requires q <= 1 and mix <= 0.5")
    return result


def initial_state(cache):
    return {name: np.asarray(cache[STATE_PREFIX + name]).copy() for name in STATE_KEYS}


def _validate_state(state, encoder_hidden, evidence_hidden=None):
    if not isinstance(state, dict) or set(state) != set(STATE_KEYS):
        raise ValueError("Evidence state must contain exactly the original module tensors")
    weight = np.asarray(state["projection.0.weight"])
    if weight.ndim != 2 or not 1 <= weight.shape[0] <= 512:
        raise ValueError("Invalid evidence projection shape")
    hidden = weight.shape[0]
    if evidence_hidden is not None and hidden != evidence_hidden:
        raise ValueError("Evidence state has a different hidden dimension")
    shapes = {"projection.0.weight": (hidden, encoder_hidden * 3 + FEATURE_DIM),
              "projection.0.bias": (hidden,), "attention.weight": (1, hidden),
              "attention.bias": (1,), "output.weight": (2, hidden), "output.bias": (2,)}
    for name, shape in shapes.items():
        value = np.asarray(state[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid finite FP32 evidence parameter: " + name)
    return hidden


def _validate_cache(cache):
    features = np.asarray(cache["global_features"])
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) < 1:
        raise ValueError("Cache requires nonempty 256-dimensional global features")
    count = len(features)
    shapes = {"global_features": (count, FEATURE_DIM), "global_logits": (count, 2),
              "evidence": (count, 2), "logits": (count, 9, 2)}
    for name, shape in shapes.items():
        value = np.asarray(cache[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid finite FP32 cache array: " + name)
    turns, offsets = np.asarray(cache["turn_features"]), np.asarray(cache["turn_offsets"])
    if (turns.ndim != 2 or turns.shape[1] < 1 or turns.dtype != np.float32
            or offsets.shape != (count + 1,)
            or offsets.dtype != np.int64 or offsets[0] != 0
            or offsets[-1] != len(turns) or np.any(np.diff(offsets) < 0)):
        raise ValueError("Invalid ragged utterance cache or row offsets")
    # The full Training cache can exceed 2.5 GiB. Keep validation scratch space
    # bounded, including when the caller supplies memory-mapped arrays.
    for start in range(0, len(turns), 8192):
        if not np.isfinite(turns[start:start + 8192]).all():
            raise ValueError("Nonfinite cached utterance representation")
    dropout = np.asarray(cache["evidence_dropout"])
    if dropout.shape != () or dropout.dtype != np.float32 or not 0 <= float(dropout) < 1:
        raise ValueError("Invalid evidence dropout configuration")
    _validate_state(initial_state(cache), turns.shape[1])
    predicted = data._binary(cache["predictions"], count, "cache predictions")
    if not np.array_equal(predicted, cache["logits"].argmax(-1)):
        raise ValueError("Stored predictions differ from raw logit argmax")
    rebuilt = cache["global_logits"] + cache["evidence"]
    if (not np.allclose(rebuilt, cache["logits"][:, NAUSEA], rtol=1e-4, atol=1e-5)
            or not np.array_equal(rebuilt.argmax(-1), predicted[:, NAUSEA])):
        raise ValueError("Frozen global and evidence logits do not reconstruct nausea")
    return count


def extract_cache(model, tokenizer, texts, settings, device, budget=None, *, training=True):
    """Capture original evidence inputs after complete overlap reconstruction."""
    texts = list(texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Cache extraction requires nonempty text-only dialogues")
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    if (model.label_attention or model.contrastive or model.evidence is None
            or tuple(model.class_weights.shape) != (9, FEATURE_DIM, 2)):
        raise ValueError("Cache extraction requires the original ver19 evidence architecture")
    batch_size = _positive_int(settings.get("batch_size", 4), "batch_size")
    model.requires_grad_(False)
    model.float().to(torch.device(device)).eval()
    rows = {name: [] for name in ROW_KEYS if name != "predictions"}
    turn_rows, turn_counts = [], []
    captured = {}

    def capture_inputs(module, inputs):
        if captured or len(inputs) != 3:
            raise RuntimeError("Unexpected evidence input hook invocation")
        turns, mask, global_features = inputs
        if (turns.ndim != 3 or mask.shape != turns.shape[:2] or mask.dtype != torch.bool
                or global_features.shape != (len(turns), FEATURE_DIM)):
            raise ValueError("Unexpected evidence input shapes")
        captured.update(turns=turns.detach(), mask=mask.detach(), global_features=global_features.detach())

    def capture_output(module, inputs, output):
        if output.shape != (len(captured["turns"]), 2) or "evidence" in captured:
            raise ValueError("Unexpected evidence output shape")
        captured["evidence"] = output.detach()

    hooks = [model.evidence.register_forward_pre_hook(capture_inputs),
             model.evidence.register_forward_hook(capture_output)]
    started, pilot, maximum_error, batches = time.monotonic(), None, 0., 0
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                if budget:
                    budget.check("tokenize contextual evidence cache", training=training)
                dataset = EncodeDataset(texts[start:start + 64], None, tokenizer,
                                        max_length=settings.get("max_length", 512),
                                        overlap=settings.get("overlap", 64), utterance_evidence=True)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=dataset.collate, num_workers=0)
                for batch in loader:
                    if budget:
                        budget.check("extract contextual evidence cache", training=training)
                    captured.clear()
                    batch = {name: value.to(device) for name, value in batch.items()}
                    logits, _ = model(batch)
                    if set(captured) != {"turns", "mask", "global_features", "evidence"}:
                        raise RuntimeError("Evidence capture did not complete")
                    global_features, residual = captured["global_features"], captured["evidence"]
                    # Repeat the original full-label kernel, avoiding subtraction
                    # cancellation when separating the fixed and learned branches.
                    global_logits = (torch.einsum("bld,ldc->blc",
                                                  global_features[:, None].expand(-1, 9, -1),
                                                  model.class_weights) + model.class_bias)[:, NAUSEA]
                    reconstructed = global_logits + residual
                    maximum_error = max(maximum_error, float((reconstructed - logits[:, NAUSEA]).abs().max()))
                    if (not torch.allclose(reconstructed, logits[:, NAUSEA], rtol=1e-4, atol=1e-5)
                            or not torch.equal(reconstructed.argmax(-1), logits[:, NAUSEA].argmax(-1))):
                        raise RuntimeError("Cached evidence changed original nausea decisions")
                    for turns, mask in zip(captured["turns"], captured["mask"]):
                        valid = turns[mask].float().cpu().numpy().copy()
                        turn_rows.append(valid)
                        turn_counts.append(len(valid))
                    for name, value in (("global_features", global_features), ("global_logits", global_logits),
                                        ("evidence", residual), ("logits", logits)):
                        rows[name].append(value.float().cpu().numpy().copy())
                    batches += 1
                    if batches % 250 == 0:
                        print(f"Evidence cache: {len(turn_counts)}/{len(texts)} rows", flush=True)
                if pilot is None:
                    _sync(device)
                    completed = min(64, len(texts))
                    elapsed = time.monotonic() - started
                    estimate = elapsed / completed * (len(texts) - completed)
                    pilot = {"rows": completed, "elapsed_seconds": elapsed,
                             "estimated_remaining_seconds": estimate, "includes_tokenization": True,
                             "device": str(device)}
                    print(f"Evidence cache pilot: {completed} rows, {elapsed:.1f}s; remaining estimate {estimate:.1f}s", flush=True)
                    if budget and not budget.can_start(estimate, training=training):
                        raise BudgetExceeded("contextual cache pilot admission",
                                             deadline_monotonic=budget.deadline_monotonic)
        result = {name: np.concatenate(values) for name, values in rows.items()}
        result.update(turn_features=np.concatenate(turn_rows),
                      turn_offsets=np.concatenate((np.zeros(1, dtype=np.int64),
                                                   np.cumsum(turn_counts, dtype=np.int64))),
                      evidence_dropout=np.asarray(model.evidence.projection[2].p, dtype=np.float32))
        result.update({STATE_PREFIX + name: value.detach().cpu().numpy().copy()
                       for name, value in model.evidence.state_dict().items()})
        result["predictions"] = result["logits"].argmax(-1).astype(np.int8)
        result["_timing"] = {"pilot": pilot, "rows": len(texts), "turns": len(result["turn_features"]),
                             "elapsed_seconds": time.monotonic() - started,
                             "maximum_reconstruction_error": maximum_error,
                             "evaluation_dtype": "float32", "trainable_model_parameters": 0}
        _validate_cache(result)
        return result
    finally:
        for handle in hooks:
            handle.remove()


def _subset(cache, indices):
    indices = _indices(indices, len(cache["global_features"]), "subset")
    result = {name: value.copy() if isinstance(value, np.ndarray) else value
              for name, value in cache.items() if name not in ROW_KEYS
              and name not in ("turn_features", "turn_offsets", "_timing")}
    result.update({name: cache[name][indices] for name in ROW_KEYS})
    offsets = cache["turn_offsets"]
    pieces = [cache["turn_features"][offsets[index]:offsets[index + 1]] for index in indices]
    result["turn_features"] = np.concatenate(pieces)
    result["turn_offsets"] = np.concatenate((np.zeros(1, dtype=np.int64),
                                             np.cumsum([len(piece) for piece in pieces], dtype=np.int64)))
    return result


def merge_caches(blocks, groups, count):
    """Scatter separately extracted fit/holdout passes into original row order."""
    count = _positive_int(count, "count")
    if not blocks or len(blocks) != len(groups):
        raise ValueError("Every cache block needs its original row indices")
    groups = [_indices(group, count, "merge") for group in groups]
    joined = np.concatenate(groups)
    if len(joined) != count or len(np.unique(joined)) != count:
        raise ValueError("Cache groups must partition all original rows exactly once")
    for block, group in zip(blocks, groups):
        if _validate_cache(block) != len(group):
            raise ValueError("Cache rows do not match their original indices")
        if (block["turn_features"].shape[1] != blocks[0]["turn_features"].shape[1]
                or not np.array_equal(block["evidence_dropout"], blocks[0]["evidence_dropout"])
                or any(not np.array_equal(block[STATE_PREFIX + name], blocks[0][STATE_PREFIX + name])
                       for name in STATE_KEYS)):
            raise ValueError("Cache blocks came from different evidence modules")
    result = {STATE_PREFIX + name: blocks[0][STATE_PREFIX + name].copy() for name in STATE_KEYS}
    result["evidence_dropout"] = blocks[0]["evidence_dropout"].copy()
    for name in ROW_KEYS:
        result[name] = np.empty((count, *blocks[0][name].shape[1:]), dtype=blocks[0][name].dtype)
        for block, group in zip(blocks, groups):
            result[name][group] = block[name]
    counts = np.empty(count, dtype=np.int64)
    for block, group in zip(blocks, groups):
        counts[group] = np.diff(block["turn_offsets"])
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    result["turn_offsets"] = offsets
    result["turn_features"] = np.empty((offsets[-1], blocks[0]["turn_features"].shape[1]), dtype=np.float32)
    for block, group in zip(blocks, groups):
        for row, destination in enumerate(group):
            result["turn_features"][offsets[destination]:offsets[destination + 1]] = block["turn_features"][
                block["turn_offsets"][row]:block["turn_offsets"][row + 1]]
    _validate_cache(result)
    return result


def _new_module(cache, state, device):
    hidden = _validate_state(state, cache["turn_features"].shape[1])
    module = UtteranceEvidence(cache["turn_features"].shape[1], FEATURE_DIM, hidden,
                               float(cache["evidence_dropout"]))
    module.load_state_dict({name: torch.from_numpy(np.asarray(value).copy()) for name, value in state.items()}, strict=True)
    return module.float().to(torch.device(device))


def _batch(cache, indices, device):
    offsets = cache["turn_offsets"]
    counts = offsets[indices + 1] - offsets[indices]
    width = max(1, int(counts.max()))
    turns = np.zeros((len(indices), width, cache["turn_features"].shape[1]), dtype=np.float32)
    mask = np.arange(width)[None] < counts[:, None]
    for row, index in enumerate(indices):
        turns[row, :counts[row]] = cache["turn_features"][offsets[index]:offsets[index + 1]]
    return (torch.from_numpy(turns).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(cache["global_features"][indices]).to(device),
            torch.from_numpy(cache["global_logits"][indices]).to(device))


def _predict(module, cache, options, device, budget=None, *, training=False):
    module.eval()
    result = cache["logits"].copy()
    with torch.inference_mode():
        for start in range(0, len(result), options["evidence_batch_size"]):
            if budget:
                budget.check("evaluate contextual evidence", training=training)
            indices = np.arange(start, min(len(result), start + options["evidence_batch_size"]))
            turns, mask, features, global_logits = _batch(cache, indices, device)
            logits = global_logits + module(turns, mask, features)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Evidence module produced nonfinite logits")
            result[indices, NAUSEA] = logits.cpu().numpy()
    predicted = result.argmax(-1).astype(np.int8)
    if (not np.array_equal(result[:, OTHER_LABELS], cache["logits"][:, OTHER_LABELS])
            or not np.array_equal(predicted[:, OTHER_LABELS], cache["predictions"][:, OTHER_LABELS])):
        raise RuntimeError("A frozen symptom output changed")
    return result, predicted


def apply_evidence(cache, state, settings=None, device="cpu", budget=None, *, training=False):
    _validate_cache(cache)
    module = _new_module(cache, state, device)
    return _predict(module, cache, _settings(settings), device, budget, training=training)


def evidence_loss(logits, targets, candidate_id, *, epoch=1, positive_weight=3., gce_q=.7,
                  gce_mix=.5, ce_warmup_epochs=1):
    """Original labels supervise ordinary CE or the predefined CE/GCE mixture."""
    if candidate_id not in RECIPES:
        raise ValueError("Unknown evidence objective")
    options = _settings({"positive_weight": positive_weight, "gce_q": gce_q, "gce_mix": gce_mix,
                         "ce_warmup_epochs": ce_warmup_epochs})
    epoch = _positive_int(epoch, "epoch")
    if (logits.ndim != 2 or logits.shape[1] != 2 or len(logits) < 1
            or targets.shape != (len(logits),) or targets.dtype != torch.long
            or not bool(((targets == 0) | (targets == 1)).all())
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("Evidence loss requires finite Nx2 logits and binary long targets")
    weights = torch.where(targets.bool(), logits.new_tensor(options["positive_weight"]), logits.new_tensor(1.))
    ce = F.cross_entropy(logits, targets, reduction="none")
    log_probability = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    # The log-domain form retains finite gradients even when a wrong-class
    # probability would underflow to zero before a fractional power.
    gce = -torch.expm1(options["gce_q"] * log_probability) / options["gce_q"]
    mix = options["gce_mix"] if candidate_id == "evidence_ce_gce" and epoch > options["ce_warmup_epochs"] else 0.
    loss = (weights * ((1 - mix) * ce + mix * gce)).mean()
    return loss, {"weighted_ce": float((weights * ce).mean().detach()),
                  "weighted_gce": float((weights * gce).mean().detach()), "gce_mix_applied": mix}


def _snapshot(module):
    return {name: value.detach().cpu().numpy().copy() for name, value in module.state_dict().items()}


def _new_training(cache, options, device):
    seed_everything(options["seed"])
    module = _new_module(cache, initial_state(cache), device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=options["evidence_learning_rate"],
                                  weight_decay=options["evidence_weight_decay"])
    return module, optimizer, np.random.default_rng(options["seed"])


def _train_epoch(module, optimizer, rng, cache, targets, indices, candidate_id, epoch, options, device, budget):
    module.train()
    order = rng.permutation(indices)
    totals = {"loss": 0., "weighted_ce": 0., "weighted_gce": 0.}
    steps, mix = 0, 0.
    started = time.monotonic()
    for start in range(0, len(order), options["evidence_batch_size"]):
        if budget:
            budget.check(f"{candidate_id} epoch {epoch}", training=True)
        chosen = order[start:start + options["evidence_batch_size"]]
        turns, mask, features, global_logits = _batch(cache, chosen, device)
        truth = torch.from_numpy(targets[chosen, NAUSEA].astype(np.int64)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, details = evidence_loss(global_logits + module(turns, mask, features), truth, candidate_id,
                                      epoch=epoch, **{key: options[key] for key in
                                                       ("positive_weight", "gce_q", "gce_mix", "ce_warmup_epochs")})
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite contextual evidence loss")
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters()):
            raise FloatingPointError("Missing or nonfinite contextual evidence gradient")
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1., error_if_nonfinite=True)
        optimizer.step()
        for name, value in {"loss": float(loss.detach()), "weighted_ce": details["weighted_ce"],
                            "weighted_gce": details["weighted_gce"]}.items():
            totals[name] += value * len(chosen)
        steps += 1
        mix = details["gce_mix_applied"]
    _sync(device)
    return {name: value / len(order) for name, value in totals.items()} | {
        "gce_mix_applied": mix, "optimizer_steps": steps, "fitted_samples": len(order),
        "elapsed_seconds": time.monotonic() - started}


def evidence_candidates(cache, targets, fit, holdout, settings, budget=None, device="cpu"):
    count = _validate_cache(cache)
    targets = data._binary(targets, count, "targets")
    fit, holdout = _indices(fit, count, "fit"), _indices(holdout, count, "holdout")
    if len(np.intersect1d(fit, holdout)) or len(fit) + len(holdout) != count:
        raise ValueError("Fit and holdout must form a disjoint complete partition")
    options = _settings(settings)
    held_cache = _subset(cache, holdout)
    baseline_predictions = held_cache["predictions"].copy()
    baseline = data.score_predictions(targets[holdout], baseline_predictions)
    records = [{"candidate_id": "baseline", "epoch": 0, "metrics": baseline,
                "state": initial_state(cache), "predictions": baseline_predictions, "history": [],
                "improved": False, "stop_reason": "unchanged_original_evidence", "trainable_parameters": 0}]
    epoch_estimate = None
    for candidate_id in RECIPES:
        module, optimizer, rng = _new_training(cache, options, device)
        history, best, completed, stale = [], None, 0, 0
        stop_reason = "max_evidence_epochs"
        for epoch in range(1, options["max_evidence_epochs"] + 1):
            try:
                if budget and epoch_estimate is not None and not budget.can_start(epoch_estimate, training=True):
                    raise BudgetExceeded("contextual evidence epoch admission",
                                         deadline_monotonic=budget.deadline_monotonic)
                epoch_started = time.monotonic()
                details = _train_epoch(module, optimizer, rng, cache, targets, fit, candidate_id,
                                       epoch, options, device, budget)
                _, predicted = _predict(module, held_cache, options, device, budget, training=True)
            except BudgetExceeded:
                stop_reason = "budget"
                break
            completed = epoch
            _sync(device)
            epoch_estimate = time.monotonic() - epoch_started
            metrics = data.score_predictions(targets[holdout], predicted)
            history.append({"epoch": epoch, **details, "metrics": metrics,
                            "training_and_evaluation_seconds": epoch_estimate})
            if best is None or _key(metrics, baseline) > _key(best["metrics"], baseline):
                best = {"epoch": epoch, "metrics": metrics, "state": _snapshot(module),
                        "predictions": predicted, "improved": _key(metrics, baseline)[0]}
                stale = 0
            else:
                stale += 1
            print(f"{candidate_id} epoch {epoch}: holdout Macro F1={metrics['macro_f1']:.6f}; "
                  f"nausea F1={metrics['per_symptom'][data.TARGET_SYMPTOMS[NAUSEA]]['f1']:.6f}", flush=True)
            if stale >= options["early_stopping_patience"]:
                stop_reason = "early_stopping"
                break
        if best is not None:
            records.append({"candidate_id": candidate_id, **best, "history": history,
                            "completed_epochs": completed, "stop_reason": stop_reason,
                            "trainable_parameters": sum(parameter.numel() for parameter in module.parameters())})
        if stop_reason == "budget":
            records[0]["search_stop_reason"] = "budget"
            break
    return records


def refit_evidence(cache, targets, candidate_id, epochs, settings, budget=None, device="cpu"):
    """Refit the fixed recipe from the separate full-Training source branch."""
    count = _validate_cache(cache)
    targets = data._binary(targets, count, "targets")
    if candidate_id not in RECIPES:
        raise ValueError("Only a selected evidence recipe can be refitted")
    options = _settings(settings)
    epochs = _positive_int(epochs, "epochs")
    if epochs > options["max_evidence_epochs"]:
        raise ValueError("Refit epochs exceed the declared training budget")
    module, optimizer, rng = _new_training(cache, options, device)
    history = []
    for epoch in range(1, epochs + 1):
        details = _train_epoch(module, optimizer, rng, cache, targets, np.arange(count), candidate_id,
                               epoch, options, device, budget)
        history.append({"epoch": epoch, **details})
        if budget and epoch < epochs and not budget.can_start(details["elapsed_seconds"] * (epochs - epoch), training=True):
            raise BudgetExceeded("remaining full-Training evidence refit admission",
                                 deadline_monotonic=budget.deadline_monotonic)
    return _snapshot(module), history


def set_evidence_state(model, state):
    """Change only the pre-existing nausea evidence tensors, with hash proof."""
    if model.evidence is None or tuple(model.class_weights.shape) != (9, FEATURE_DIM, 2):
        raise ValueError("Cannot patch an incompatible model")
    _validate_state(state, int(model.encoder.config.hidden_size), model.evidence.projection[0].out_features)
    before = _state_hashes(model)
    model.evidence.load_state_dict({name: torch.from_numpy(np.asarray(value).copy()) for name, value in state.items()}, strict=True)
    after = _state_hashes(model)
    changed = [name for name in before if before[name] != after[name]]
    if any(not name.startswith("evidence.") for name in changed):
        raise RuntimeError("Updating evidence changed a frozen model tensor")
    model.requires_grad_(False).eval()
    return {"updated_label_index": NAUSEA,
            "learned_parameter_count": sum(parameter.numel() for parameter in model.evidence.parameters()),
            "other_eight_head_slices_identical": True, "all_global_heads_identical": True,
            "all_non_evidence_tensors_identical": True, "changed_state_entries": changed,
            "checked_state_entries": len(before)}
