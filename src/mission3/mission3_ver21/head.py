"""Frozen ver19 representations and supervised, independent nausea heads.

The encoder and every existing module stay in FP32 evaluation mode. Only a
separate 256-by-2 weight and two biases are optimized. All decisions remain raw
two-class argmax; no thresholds, fitted output offsets, or calibration exist.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import data
from .budget import BudgetExceeded
from .neural import DialogueClassifier, EncodeDataset, seed_everything


NAUSEA = 6
FEATURE_DIM = 256
MODEL_ID = "klue/roberta-base"
REVISION = "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"
OTHER_LABELS = [index for index in range(9) if index != NAUSEA]
RECIPES = ("head_ce", "head_ce_soft_f1")


def _safe(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Checkpoint asset paths must be nonempty relative paths")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Checkpoint asset path escapes its directory")
    return result


def load_model(checkpoint, device):
    """Strictly restore local ver19/ver21 assets without fetching any model."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    checkpoint = Path(checkpoint)
    manifest = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
    if (manifest.get("format") != "mission3_categorical_v1"
            or manifest.get("kind") != "neural"
            or manifest.get("version") not in ("ver19", "ver21")
            or tuple(manifest.get("labels", ())) != data.TARGET_SYMPTOMS
            or manifest.get("label_attention") is not False
            or manifest.get("contrastive") is not False
            or manifest.get("utterance_evidence") is not True
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("resolved_revision") != REVISION):
        raise ValueError("Unsupported checkpoint, architecture, revision, or label ordering")
    architecture = {key: manifest[key] for key in
                    ("utterance_evidence", "evidence_hidden", "evidence_dropout")}
    assets = _safe(checkpoint.parent, manifest["assets"])
    weights_path = _safe(checkpoint.parent, manifest["weights"])
    config = AutoConfig.from_pretrained(assets / "encoder", local_files_only=True,
                                       trust_remote_code=False)
    if (config.model_type != "roberta" or config.num_hidden_layers != 12
            or config.hidden_size != 768):
        raise ValueError("The checkpoint must contain the pinned base encoder architecture")
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Complete overflow windows require a fast tokenizer")
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False,
                               **architecture)
    chunk_size = manifest["encoder_window_batch"]
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("encoder_window_batch must be a positive integer")
    model.encoder_window_batch = chunk_size
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint weights must be a tensor-only state dictionary")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.float().to(torch.device(device)).eval()
    return model, tokenizer, manifest


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value, name, *, minimum=0, strict=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum or (strict and value == minimum)):
        raise ValueError(f"{name} is outside its permitted finite range")
    return float(value)


def _settings(settings):
    seed = settings.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "seed": seed,
        "max_head_epochs": _positive_int(settings.get("max_head_epochs", 80), "max_head_epochs"),
        "head_eval_every": _positive_int(settings.get("head_eval_every", 5), "head_eval_every"),
        "head_learning_rate": _finite(settings.get("head_learning_rate", .001), "head_learning_rate", strict=True),
        "head_weight_decay": _finite(settings.get("head_weight_decay", 0), "head_weight_decay"),
        "positive_weight": _finite(settings.get("positive_weight", 3), "positive_weight", strict=True),
        "soft_f1_weight": _finite(settings.get("soft_f1_weight", .5), "soft_f1_weight"),
    }


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def extract_cache(model, tokenizer, texts, settings, device, budget=None, *, training=True):
    """Read unchanged forward-pass features with hooks, retaining all turns.

    The measured first 64 rows are part of this pass, never extra inference.
    Chunked tokenization permits budget checks and keeps very large datasets
    from delaying the first pilot. Neither labels nor sample identifiers enter
    this cache. The caller must attach checkpoint/dataset/order identities.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("Cannot extract an empty feature cache")
    if model.label_attention or model.contrastive or model.evidence is None:
        raise ValueError("Cache extraction requires the ver19 global/evidence architecture")
    if tuple(model.class_weights.shape) != (9, FEATURE_DIM, 2):
        raise ValueError("Cache extraction requires 256-dimensional global features")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Every dialogue must be a text string")
    batch_size = _positive_int(settings.get("batch_size", 4), "batch_size")
    cache_chunk = 64
    training_stage = training
    if type(training_stage) is not bool:
        raise ValueError("training must be boolean")
    model.requires_grad_(False)
    model.float().to(torch.device(device)).eval()
    feature_rows, evidence_rows, logits_rows = [], [], []
    captured = {}

    def capture_global(module, inputs, output):
        if output.ndim == 3:
            if output.shape[1:] != (9, FEATURE_DIM):
                raise ValueError("Unexpected global feature hook shape")
            if not torch.equal(output, output[:, :1].expand_as(output)):
                raise ValueError("Global features differ across symptom labels")
            if "global" in captured:
                raise ValueError("Multiple global feature hook invocations in one forward")
            captured["global"] = output[:, 0].detach()

    def capture_evidence(module, inputs, output):
        if output.ndim != 2 or output.shape[1] != 2 or "evidence" in captured:
            raise ValueError("Unexpected utterance evidence hook invocation")
        captured["evidence"] = output.detach()

    hooks = [model.dropout.register_forward_hook(capture_global),
             model.evidence.register_forward_hook(capture_evidence)]
    started = time.monotonic()
    pilot = None
    maximum_error = 0.0
    batches = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), cache_chunk):
                if budget:
                    budget.check("tokenize frozen feature cache", training=training_stage)
                dataset = EncodeDataset(
                    texts[start:start + cache_chunk], None, tokenizer,
                    max_length=settings.get("max_length", 512),
                    overlap=settings.get("overlap", 64), utterance_evidence=True,
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=dataset.collate, num_workers=0)
                for batch in loader:
                    if budget:
                        budget.check("extract frozen features", training=training_stage)
                    captured.clear()
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits, _ = model(batch)
                    if set(captured) != {"global", "evidence"}:
                        raise RuntimeError("A required frozen representation hook did not run")
                    global_features, residual = captured["global"], captured["evidence"]
                    rebuilt = (torch.einsum("bd,dc->bc", global_features,
                                            model.class_weights[NAUSEA])
                               + model.class_bias[NAUSEA] + residual)
                    error = float((rebuilt - logits[:, NAUSEA]).abs().max())
                    maximum_error = max(maximum_error, error)
                    if (not torch.allclose(rebuilt, logits[:, NAUSEA], rtol=1e-4, atol=1e-5)
                            or not torch.equal(rebuilt.argmax(-1), logits[:, NAUSEA].argmax(-1))):
                        raise RuntimeError("Frozen feature reconstruction changed original nausea decisions")
                    feature_rows.append(global_features.float().cpu().numpy().copy())
                    evidence_rows.append(residual.float().cpu().numpy().copy())
                    logits_rows.append(logits.float().cpu().numpy().copy())
                    batches += 1
                    if batches % 250 == 0:
                        print(f"Frozen cache: {sum(len(rows) for rows in feature_rows)}/{len(texts)} rows", flush=True)
                if pilot is None:
                    _sync(device)
                    rows = min(cache_chunk, len(texts))
                    elapsed = time.monotonic() - started
                    estimate = elapsed / rows * (len(texts) - rows)
                    pilot = {"rows": rows, "elapsed_seconds": elapsed,
                             "estimated_remaining_seconds": estimate,
                             "includes_tokenization": True, "device": str(device)}
                    print(f"Frozen cache pilot: {rows} rows, {elapsed:.1f}s; remaining estimate {estimate:.1f}s", flush=True)
                    if budget and not budget.can_start(estimate, training=training_stage):
                        raise BudgetExceeded("frozen cache pilot admission",
                                             deadline_monotonic=budget.deadline_monotonic)
        result = {
            "global_features": np.concatenate(feature_rows),
            "evidence": np.concatenate(evidence_rows),
            "logits": np.concatenate(logits_rows),
            "initial_weight": model.class_weights[NAUSEA].detach().cpu().numpy().copy(),
            "initial_bias": model.class_bias[NAUSEA].detach().cpu().numpy().copy(),
        }
        result["predictions"] = result["logits"].argmax(-1).astype(np.int8)
        result["_timing"] = {"pilot": pilot, "rows": len(texts),
                             "elapsed_seconds": time.monotonic() - started,
                             "maximum_reconstruction_error": maximum_error,
                             "evaluation_dtype": "float32", "trainable_model_parameters": 0}
        _validate_cache(result)
        return result
    finally:
        for handle in hooks:
            handle.remove()


def _validate_cache(cache):
    count = len(cache["global_features"])
    if count < 1:
        raise ValueError("Feature cache must not be empty")
    shapes = {"global_features": (count, FEATURE_DIM), "evidence": (count, 2),
              "logits": (count, 9, 2), "initial_weight": (FEATURE_DIM, 2),
              "initial_bias": (2,)}
    for name, shape in shapes.items():
        value = np.asarray(cache[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite FP32 cache array: {name}")
    predicted = data._binary(cache["predictions"], count, "cache predictions")
    if not np.array_equal(predicted, np.asarray(cache["logits"]).argmax(-1)):
        raise ValueError("Cache predictions do not match the stored raw logits")
    return count


def _head_arrays(weight, bias):
    weight, bias = np.asarray(weight), np.asarray(bias)
    if (weight.shape != (FEATURE_DIM, 2) or bias.shape != (2,)
            or weight.dtype != np.float32 or bias.dtype != np.float32
            or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError("Head parameters must be finite FP32 arrays with shapes (256, 2) and (2,)")
    return weight, bias


def apply_head(cache, weight, bias):
    """Replace only nausea logits; preserve eight original logits byte-for-byte."""
    _validate_cache(cache)
    weight, bias = _head_arrays(weight, bias)
    # Whole-batch torch operations use the same kernel as head optimization.
    with torch.inference_mode():
        values = (torch.from_numpy(cache["global_features"]) @ torch.from_numpy(weight)
                  + torch.from_numpy(bias) + torch.from_numpy(cache["evidence"]))
    logits = cache["logits"].copy()
    logits[:, NAUSEA] = values.numpy()
    if not np.isfinite(logits).all():
        raise FloatingPointError("Head produced nonfinite logits")
    predicted = logits.argmax(-1).astype(np.int8)
    if not np.array_equal(predicted[:, OTHER_LABELS], cache["predictions"][:, OTHER_LABELS]):
        raise RuntimeError("A frozen symptom decision changed")
    return logits, predicted


def head_loss(logits, targets, candidate_id, *, positive_weight=3., soft_f1_weight=.5):
    """One true whole-fit CE/soft-F1 objective, without gradient accumulation."""
    if candidate_id not in RECIPES:
        raise ValueError("Unknown supervised head recipe")
    positive_weight = _finite(positive_weight, "positive_weight", strict=True)
    soft_f1_weight = _finite(soft_f1_weight, "soft_f1_weight")
    if (logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] < 1
            or targets.shape != (len(logits),) or targets.dtype != torch.long
            or not bool(((targets == 0) | (targets == 1)).all())
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("Head loss requires finite Nx2 logits and N binary long targets")
    costs = torch.where(targets.bool(), logits.new_tensor(positive_weight), logits.new_tensor(1.))
    ce = (F.cross_entropy(logits, targets, reduction="none") * costs).mean()
    probability = logits.softmax(-1)[:, 1]
    truth = targets.to(probability.dtype)
    soft_f1 = (2 * (probability * truth).sum() + 1e-7) / (probability.sum() + truth.sum() + 1e-7)
    loss = ce + (soft_f1_weight * (1 - soft_f1) if candidate_id == "head_ce_soft_f1" else 0)
    return loss, {"weighted_ce": float(ce.detach()), "soft_f1": float(soft_f1.detach())}


class IndependentHead(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        weight, bias = _head_arrays(weight, bias)
        self.weight = nn.Parameter(torch.from_numpy(weight.copy()))
        self.bias = nn.Parameter(torch.from_numpy(bias.copy()))

    def forward(self, features, evidence):
        return features @ self.weight + self.bias + evidence


def _indices(indices, count, name):
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= count)
            or len(np.unique(values)) != len(values)):
        raise ValueError(f"Invalid unique {name} indices")
    return values.astype(np.int64, copy=False)


def _key(metrics, baseline):
    nausea = data.TARGET_SYMPTOMS[NAUSEA]
    joint = (metrics["macro_f1"] > baseline["macro_f1"]
             and metrics["per_symptom"][nausea]["f1"] > baseline["per_symptom"][nausea]["f1"])
    return joint, metrics["macro_f1"]


def _snapshot(head):
    return (head.weight.detach().cpu().numpy().copy(), head.bias.detach().cpu().numpy().copy())


def _subset(cache, indices):
    return {key: (value[indices] if key in ("global_features", "evidence", "logits", "predictions") else value)
            for key, value in cache.items()}


def _new_training(cache, targets, indices, options):
    seed_everything(options["seed"])
    head = IndependentHead(cache["initial_weight"], cache["initial_bias"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=options["head_learning_rate"],
                                 weight_decay=options["head_weight_decay"])
    features = torch.from_numpy(cache["global_features"][indices].copy())
    evidence = torch.from_numpy(cache["evidence"][indices].copy())
    truth = torch.from_numpy(targets[indices, NAUSEA].astype(np.int64))
    return head, optimizer, features, evidence, truth


def _step(head, optimizer, features, evidence, truth, candidate_id, options):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss, details = head_loss(head(features, evidence), truth, candidate_id,
                              positive_weight=options["positive_weight"],
                              soft_f1_weight=options["soft_f1_weight"])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite supervised head loss")
    loss.backward()
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
           for parameter in head.parameters()):
        raise FloatingPointError("Missing or nonfinite supervised head gradient")
    optimizer.step()
    return {"loss": float(loss.detach()), **details}


def head_candidates(cache, targets, fit, holdout, settings, budget=None):
    """Compare two predeclared supervised recipes against unchanged ver19.

    Only fit targets enter an optimizer. Held-out labels are used exclusively
    to select completed epochs. Exact ties retain the earliest evaluated epoch.
    """
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
                "weight": cache["initial_weight"].copy(), "bias": cache["initial_bias"].copy(),
                "predictions": baseline_predictions, "history": [], "improved": False,
                "stop_reason": "unchanged_original_head", "trainable_parameters": 0}]
    for candidate_id in RECIPES:
        head, optimizer, features, evidence, truth = _new_training(cache, targets, fit, options)
        history, best, completed = [], None, 0
        stop_reason = "max_head_epochs"
        for epoch in range(1, options["max_head_epochs"] + 1):
            try:
                if budget:
                    budget.check(f"{candidate_id} epoch {epoch}", training=True)
            except BudgetExceeded:
                stop_reason = "budget"
                break
            details = _step(head, optimizer, features, evidence, truth, candidate_id, options)
            completed = epoch
            if epoch % options["head_eval_every"] == 0 or epoch == options["max_head_epochs"]:
                weight, bias = _snapshot(head)
                _, predicted = apply_head(held_cache, weight, bias)
                metrics = data.score_predictions(targets[holdout], predicted)
                history.append({"epoch": epoch, **details, "metrics": metrics,
                                "whole_fit_batch_size": len(fit)})
                if best is None or _key(metrics, baseline) > _key(best["metrics"], baseline):
                    best = {"epoch": epoch, "metrics": metrics, "weight": weight,
                            "bias": bias, "predictions": predicted,
                            "improved": _key(metrics, baseline)[0]}
        # Never publish an unevaluated or partially completed optimizer step.
        if best is not None:
            records.append({"candidate_id": candidate_id, **best, "history": history,
                            "stop_reason": stop_reason, "completed_epochs": completed,
                            "trainable_parameters": sum(p.numel() for p in head.parameters())})
        if stop_reason == "budget":
            records[0]["search_stop_reason"] = "budget"
            break
    return records


def refit_head(cache, targets, candidate_id, epochs, settings, budget=None):
    """Apply a fixed recipe to an independently cached full-Training encoder.

    Initialization is the full encoder's own original head, never the weights
    selected on the fit-only encoder. No held-out or Validation labels occur.
    """
    count = _validate_cache(cache)
    targets = data._binary(targets, count, "targets")
    if candidate_id not in RECIPES:
        raise ValueError("Only a selected supervised recipe can be refitted")
    options = _settings(settings)
    epochs = _positive_int(epochs, "epochs")
    if epochs > options["max_head_epochs"]:
        raise ValueError("Refit epochs exceed the declared optimization budget")
    head, optimizer, features, evidence, truth = _new_training(cache, targets, np.arange(count), options)
    history = []
    for epoch in range(1, epochs + 1):
        if budget:
            budget.check(f"full-Training {candidate_id} epoch {epoch}", training=True)
        details = _step(head, optimizer, features, evidence, truth, candidate_id, options)
        history.append({"epoch": epoch, **details, "whole_fit_batch_size": count})
    weight, bias = _snapshot(head)
    return weight, bias, history


def _state_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def set_nausea_head(model, weight, bias):
    """Patch only symptom slice 6, proving every other model tensor unchanged."""
    weight, bias = _head_arrays(weight, bias)
    if tuple(model.class_weights.shape) != (9, FEATURE_DIM, 2):
        raise ValueError("Cannot patch a different classifier architecture")
    before = _state_hashes(model)
    other_weight = model.class_weights[OTHER_LABELS].detach().cpu().clone()
    other_bias = model.class_bias[OTHER_LABELS].detach().cpu().clone()
    with torch.no_grad():
        model.class_weights[NAUSEA].copy_(torch.from_numpy(weight).to(model.class_weights.device))
        model.class_bias[NAUSEA].copy_(torch.from_numpy(bias).to(model.class_bias.device))
    after = _state_hashes(model)
    changed = [name for name in before if before[name] != after[name]]
    if (set(changed) - {"class_weights", "class_bias"}
            or not torch.equal(other_weight, model.class_weights[OTHER_LABELS].detach().cpu())
            or not torch.equal(other_bias, model.class_bias[OTHER_LABELS].detach().cpu())):
        raise RuntimeError("Updating nausea unexpectedly changed frozen model state")
    model.requires_grad_(False).eval()
    return {"updated_label_index": NAUSEA, "learned_parameter_count": FEATURE_DIM * 2 + 2,
            "other_eight_head_slices_identical": True, "all_other_tensors_identical": True,
            "changed_state_entries": changed, "checked_state_entries": len(before)}
