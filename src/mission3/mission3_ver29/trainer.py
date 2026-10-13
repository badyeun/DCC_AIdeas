"""Supervised encoder training with measured batches and exact boundary resume.

All nine predictions use logits >= 0. Model assets reload entirely offline.
Only the supplied Training indices contribute to loss or positive weights.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .budget import BudgetExceeded
from .data import TARGET_SYMPTOMS

FORMAT = "mission3_ver29_full_encoder_v1"
KINDS = ("kure", "control", "klue_asl", "qwen_embedding")
DECISION = "independent_logits_greater_than_or_equal_to_zero"
QWEN_INSTRUCTION = "Identify the patient's symptoms from this Korean emergency call transcript."


def model_protocol(kind):
    """Return the immutable objective and text recipe for one candidate."""
    if kind not in KINDS:
        raise ValueError("Unsupported model kind")
    return {
        "loss": {"name": "weighted_asymmetric_focusing" if kind == "klue_asl" else "weighted_bce",
                 "gamma_positive": 0., "gamma_negative": 2. if kind == "klue_asl" else 0.,
                 "probability_clip": 0., "differentiate_focal_factor": True,
                 "positive_weights": "clip(sqrt(fit_negative/max(fit_positive,1)),1,5)",
                 "reduction": "mean_over_examples_and_nine_labels"},
        "preprocessing": {"format": "fixed_instruction_query" if kind == "qwen_embedding" else "plain_text",
                          "instruction": QWEN_INSTRUCTION if kind == "qwen_embedding" else "",
                          "template": "Instruct: {instruction}\nQuery: {text}" if kind == "qwen_embedding" else "{text}",
                          "truncation": "right", "padding_side": "right"},
        "pooling": {"kure": "cls", "control": "masked_mean", "klue_asl": "masked_mean",
                    "qwen_embedding": "last_nonpadding"}[kind],
        "precision_policy": "native_bfloat16_else_float16_on_cuda",
    }


def prepare_texts(kind, texts):
    """Apply the same label-independent input format at training and inference."""
    protocol = model_protocol(kind)["preprocessing"]
    if not len(texts) or any(not isinstance(text, str) for text in texts):
        raise ValueError("Texts must be a nonempty sequence of strings")
    return [protocol["template"].format(instruction=protocol["instruction"], text=text) for text in texts]


def classification_loss(logits, labels, weights, kind):
    """Stable ASL variant with no clipping and a differentiated focusing factor."""
    if kind not in KINDS:
        raise ValueError("Unsupported model kind")
    logits = logits.float()
    if kind != "klue_asl":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
    positive = labels * weights * F.softplus(-logits)
    negative = (1. - labels) * logits.sigmoid().square() * F.softplus(logits)
    return (positive + negative).mean()


def _default_maximum(kind):
    return 512 if kind in ("control", "klue_asl") else 1024


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _targets(values, rows=None):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != len(TARGET_SYMPTOMS)
            or (rows is not None and len(values) != rows)
            or not np.isin(values, (0, 1)).all()):
        raise ValueError("Targets must be a binary (samples, 9) array")
    return values.astype(np.float32, copy=False)


def positive_weights(fit_targets):
    values = _targets(fit_targets)
    if not len(values):
        raise ValueError("Positive weights need at least one Training row")
    positive = values.sum(0, dtype=np.float64)
    return torch.tensor(np.clip(np.sqrt((len(values) - positive) / np.maximum(positive, 1)), 1, 5),
                        dtype=torch.float32)


def _indices(values, n):
    result = np.asarray(values)
    if (result.ndim != 1 or not len(result) or not np.issubdtype(result.dtype, np.integer)
            or (result < 0).any() or (result >= n).any() or len(np.unique(result)) != len(result)):
        raise ValueError("Indices must be nonempty unique valid integers")
    return result.astype(np.int64)


def _amp_dtype(device):
    if torch.device(device).type != "cuda":
        return None
    with torch.cuda.device(device):
        supported = torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.bfloat16 if supported else torch.float16


def _autocast(device):
    dtype = _amp_dtype(device)
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, kind, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError("Unsupported model kind")
        expected = {"kure": ("xlm-roberta", "roberta"), "control": ("bert", "roberta"),
                    "klue_asl": ("bert", "roberta"),
                    "qwen_embedding": ("qwen3",)}
        if encoder.config.model_type not in expected[kind]:
            raise ValueError(f"Unexpected encoder architecture for {kind}")
        self.encoder, self.kind = encoder, kind
        if hasattr(encoder, "pooler"):
            encoder.pooler = None
        if hasattr(encoder.config, "use_cache"):
            encoder.config.use_cache = False
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(TARGET_SYMPTOMS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if (attention_mask.sum(1) == 0).any():
            raise ValueError("Empty token sequence")
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if self.kind in ("control", "klue_asl") and token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        if self.kind == "qwen_embedding":
            positions = attention_mask.long().cumsum(-1) - 1
            inputs.update(position_ids=positions.masked_fill(attention_mask == 0, 0), use_cache=False)
        hidden = self.encoder(**inputs).last_hidden_state
        if self.kind == "kure":
            pooled = hidden[:, 0]
        elif self.kind in ("control", "klue_asl"):
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last = positions.unsqueeze(0).masked_fill(attention_mask == 0, -1).max(1).values
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        return self.classifier(self.dropout(pooled))


def create_model(kind, spec, settings, device):
    if kind not in KINDS or _get(spec, "kind", kind) != kind:
        raise ValueError("Model kind and source specification disagree")
    source = _get(spec, "snapshot_path") or _get(spec, "model_id")
    if not source:
        raise ValueError("Model source is missing")
    local = Path(str(source)).is_dir()
    revision = _get(spec, "revision")
    if not local and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("Remote model requires an immutable revision")
    kwargs = {"trust_remote_code": False, "local_files_only": local}
    if not local:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True, **kwargs)
    if tokenizer.pad_token_id is None:
        if kind != "qwen_embedding" or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a padding token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoder_kwargs = dict(kwargs)
    if kind != "qwen_embedding":
        encoder_kwargs["add_pooling_layer"] = False
    encoder = AutoModel.from_pretrained(str(source), **encoder_kwargs)
    encoder.config.pad_token_id = tokenizer.pad_token_id
    if kind in ("kure", "qwen_embedding"):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return EncoderClassifier(encoder, kind, _get(settings, "dropout", .1)).to(device), tokenizer


def _tokenize(tokenizer, texts, maximum, kind="control"):
    encoded = tokenizer(prepare_texts(kind, texts), truncation=True, max_length=int(maximum), padding=False)
    keys = [key for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoded]
    return [{key: encoded[key][i] for key in keys} for i in range(len(texts))]


def _length_statistics(tokenizer, texts, rows, maximum, kind="control"):
    lengths = []
    for start in range(0, len(rows), 256):
        encoded = tokenizer(prepare_texts(kind, [texts[int(i)] for i in rows[start:start + 256]]),
                            truncation=False, padding=False, return_attention_mask=False,
                            return_token_type_ids=False)
        lengths.extend(len(value) for value in encoded["input_ids"])
    lengths = np.asarray(lengths, dtype=np.int64)
    return {"training_rows": len(rows), "max_length": maximum,
            "raw_token_percentiles": {str(p): float(np.percentile(lengths, p)) for p in (0, 50, 90, 95, 99, 100)},
            "truncated_training_rows": int((lengths > maximum).sum()),
            "truncated_training_fraction": float(np.mean(lengths > maximum)),
            "truncation": "right_truncation_after_fixed_input_format",
            "preprocessing": model_protocol(kind)["preprocessing"]}


def _batch(tokenizer, tokens, rows, device):
    packed = tokenizer.pad([tokens[int(i)] for i in rows], padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in packed.items()
            if key in ("input_ids", "attention_mask", "token_type_ids")}


def _optimizer(model, kind, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = float(_get(settings, "head_lr", 1e-4) if name.startswith("classifier.")
                   else _get(settings, f"{kind}_lr", 2e-5))
        decay = 0. if name.endswith("bias") or "norm" in name.lower() else float(_get(settings, "weight_decay", .01))
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(decay) or decay < 0:
            raise ValueError("Invalid optimizer settings")
        groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                              for (lr, decay), params in groups.items()])


def _rng():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value):
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(value["cuda"])


def _update(model, tokenizer, tokens, targets, rows, batch_size, optimizer, scaler, weights, device):
    """One complete effective batch, replaying fp16 overflow without skipping rows."""
    model.train()
    rng = _rng()
    for retry in range(9):
        optimizer.zero_grad(set_to_none=True)
        if retry:
            _restore_rng(rng)
        total_loss = 0.
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            labels = torch.as_tensor(targets[part], dtype=torch.float32, device=device)
            with _autocast(device):
                logits = model(**_batch(tokenizer, tokens, part, device))
                loss = classification_loss(logits, labels, weights, model.kind)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            scaler.scale(loss * (len(part) / len(rows))).backward()
            total_loss += float(loss.detach()) * len(part)
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not scaler.is_enabled() and not torch.isfinite(norm):
            raise FloatingPointError("Training gradients are not finite")
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= scale:
            if not torch.isfinite(norm):
                raise FloatingPointError("Accepted update has nonfinite gradients")
            return total_loss / len(rows), retry
    raise FloatingPointError("FP16 overflow retry limit exceeded; no rows were silently skipped")


def _autotune(kind, spec, settings, texts, targets, rows, device, budget):
    effective = int(_get(settings, "effective_batch_size", 16))
    candidates = sorted(set(int(value) for value in _get(settings, "batch_candidates", [1, 2, 4, 8, 16])))
    if not candidates or any(value <= 0 or effective % value for value in candidates):
        raise ValueError("Batch candidates must be positive divisors of effective batch size")
    records = []
    maximum = int(_get(settings, f"{kind}_max_length", _default_maximum(kind)))
    for size in candidates:
        if budget is not None:
            budget.check("batch_autotune", training=True)
        seed_everything(_get(settings, "seed", 42))
        model = optimizer = scaler = None
        try:
            model, tokenizer = create_model(kind, spec, settings, device)
            tokens = _tokenize(tokenizer, texts, maximum, kind)
            optimizer = _optimizer(model, kind, settings)
            scaler = torch.amp.GradScaler("cuda", enabled=_amp_dtype(device) == torch.float16)
            weights = positive_weights(targets[rows]).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # The first real step allocates Adam state and checks the longest rows.
            longest = sorted(rows.tolist(), key=lambda i: len(tokens[i]["input_ids"]), reverse=True)
            longest = np.resize(np.asarray(longest, dtype=np.int64), effective)
            _update(model, tokenizer, tokens, targets, longest, size, optimizer, scaler, weights, device)
            rng = np.random.default_rng(int(_get(settings, "seed", 42)))
            probe = np.resize(rng.permutation(rows), max(effective * 2, int(_get(settings, "benchmark_examples", 64))))
            measured = 0
            _sync(device)
            started = time.monotonic()
            for start in range(0, len(probe), effective):
                if budget is not None:
                    budget.check("batch_autotune_measured_update", training=True)
                part = probe[start:start + effective]
                _update(model, tokenizer, tokens, targets, part, size, optimizer, scaler, weights, device)
                measured += len(part)
            _sync(device)
            elapsed = time.monotonic() - started
            records.append({"batch_size": size, "status": "ok", "examples": measured,
                            "seconds": elapsed, "examples_per_second": measured / elapsed,
                            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0})
        except torch.cuda.OutOfMemoryError:
            records.append({"batch_size": size, "status": "out_of_memory"})
        finally:
            del model, optimizer, scaler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    successful = [item for item in records if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("No batch candidate fits this GPU at the requested sequence length")
    best = max(successful, key=lambda item: item["examples_per_second"])
    return {"selected_batch_size": best["batch_size"], "effective_batch_size": effective,
            "gradient_accumulation_steps": effective // best["batch_size"],
            "seconds_per_training_example": 1. / best["examples_per_second"], "candidates": records}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu_state(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _identity(kind, spec, settings, texts, targets, train_rows, dev_rows, epochs):
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    digest.update(targets.tobytes())
    training_keys = {name: _get(settings, name, default) for name, default in {
        "seed": 42, "effective_batch_size": 16, "head_lr": 1e-4, "weight_decay": .01,
        "warmup_ratio": .05, "dropout": .1, f"{kind}_lr": 2e-5,
        f"{kind}_max_length": _default_maximum(kind)}.items()}
    return {"format": FORMAT, "kind": kind, "model_id": _get(spec, "model_id"),
            "revision": _get(spec, "revision"), "epochs": epochs, "settings": training_keys,
            "data_sha256": digest.hexdigest(), "train_indices": train_rows.tolist(),
            "dev_indices": None if dev_rows is None else dev_rows.tolist(),
            "target_symptoms": list(TARGET_SYMPTOMS), "decision": DECISION,
            "model_protocol": model_protocol(kind)}


def _save_assets(path, model, tokenizer, metadata):
    metadata = {**metadata, "model_protocol": model_protocol(model.kind)}
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory / "encoder_config")
    tokenizer.save_pretrained(directory / "tokenizer")
    _atomic_torch_save(_cpu_state(model), directory / "model.pt")
    files = {p.relative_to(directory).as_posix(): _sha(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    _json(directory / "manifest.json", {"format": FORMAT, "target_symptoms": list(TARGET_SYMPTOMS),
                                        "decision": DECISION, "metadata": metadata, "files": files})


def _save_reload_reference(path, model, tokenizer, tokens, rows, batch_size, device):
    rng = _rng()
    try:
        selected = np.asarray(rows[:16], dtype=np.int64)
        logits = _predict(model, tokenizer, tokens, selected, batch_size, device)
        temporary = Path(path).with_name(Path(path).name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, indices=selected, logits=logits, batch_size=np.asarray(batch_size, dtype=np.int64))
        os.replace(temporary, path)
    finally:
        _restore_rng(rng)


def load_checkpoint(checkpoint_dir, device="cpu"):
    directory = Path(checkpoint_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("target_symptoms") != list(TARGET_SYMPTOMS)
            or manifest.get("decision") != DECISION):
        raise ValueError("Checkpoint format, label order, or decision policy mismatch")
    files = manifest["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(files):
        raise ValueError("Unexpected or missing checkpoint files")
    for relative, digest in files.items():
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or _sha(path) != digest:
            raise ValueError("Checkpoint asset integrity failure")
    metadata = manifest["metadata"]
    if metadata.get("model_protocol") != model_protocol(metadata.get("kind")):
        raise ValueError("Checkpoint objective or preprocessing protocol mismatch")
    if metadata.get("pooling") != model_protocol(metadata["kind"])["pooling"]:
        raise ValueError("Checkpoint pooling protocol mismatch")
    config = AutoConfig.from_pretrained(directory / "encoder_config", local_files_only=True, trust_remote_code=False)
    kwargs = {} if metadata["kind"] == "qwen_embedding" else {"add_pooling_layer": False}
    model = EncoderClassifier(AutoModel.from_config(config, trust_remote_code=False, **kwargs),
                              metadata["kind"], metadata["dropout"])
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    return model.to(device), tokenizer, metadata


@torch.no_grad()
def _predict(model, tokenizer, tokens, rows, batch_size, device, budget=None):
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch_size):
        if budget is not None:
            budget.check("classifier_evaluation", training=True)
        with _autocast(device):
            logits = model(**_batch(tokenizer, tokens, rows[start:start + batch_size], device))
        values = logits.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Prediction logits are not finite")
        chunks.append(values)
    return np.concatenate(chunks).astype(np.float32)


def predict_checkpoint(checkpoint_dir, texts, indices=None, *, device="cpu", batch_size=8, budget=None):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Prediction batch size must be a positive integer")
    model, tokenizer, metadata = load_checkpoint(checkpoint_dir, device)
    try:
        rows = _indices(np.arange(len(texts)) if indices is None else indices, len(texts))
        tokens = _tokenize(tokenizer, texts, metadata["max_length"], metadata["kind"])
        return _predict(model, tokenizer, tokens, rows, batch_size, device, budget)
    finally:
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def _macro_f1(targets, logits):
    predicted = logits >= 0
    gold = targets.astype(bool)
    tp = np.sum(predicted & gold, axis=0)
    denom = np.sum(predicted, axis=0) + np.sum(gold, axis=0)
    return float(np.divide(2 * tp, denom, out=np.zeros(9, dtype=float), where=denom != 0).mean())


def train_model(kind, spec, settings, texts, targets, train_indices, dev_indices,
                output_dir, budget, epochs, resume=False):
    """Train one candidate; pause only at optimizer boundaries, never lose rows.

    ``epochs`` is the fixed scheduler horizon and must remain identical on resume.
    A paused evaluation restarts that evaluation, not the completed Training epoch.
    """
    if kind not in KINDS or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Invalid model kind or epoch count")
    targets = _targets(targets, len(texts))
    rows = _indices(train_indices, len(texts))
    dev = None if dev_indices is None else _indices(dev_indices, len(texts))
    if dev is not None and np.intersect1d(rows, dev).size:
        raise ValueError("Training and development rows must be disjoint")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result_path, state_path = directory / "training_result.json", directory / "training_state.pt"
    identity = _identity(kind, spec, settings, texts, targets, rows, dev, epochs)
    identity_path = directory / "training_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume identity differs in data, model, labels, settings, or scheduler horizon")
        if not resume:
            raise FileExistsError("Training output already exists; choose resume or a fresh output directory")
    elif resume:
        raise FileNotFoundError("Resume requires an existing training identity")
    else:
        _json(identity_path, identity)
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] == "complete":
            result["checkpoint_dir"] = str(directory / "best_model")
            if not (directory / "best_model" / "manifest.json").is_file():
                raise FileNotFoundError("Completed training result has no portable model checkpoint")
            _json(result_path, result)
            return result
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(_get(settings, "allow_cpu", False)):
        raise RuntimeError("Training requires a GPU; CPU is only enabled for explicit local tests")
    seed = int(_get(settings, "seed", 42))
    effective = int(_get(settings, "effective_batch_size", 16))
    if effective < 1:
        raise ValueError("Effective batch size must be positive")
    maximum = int(_get(settings, f"{kind}_max_length", _default_maximum(kind)))
    tuning_path = directory / "batch_benchmark.json"
    started = time.monotonic()
    try:
        if tuning_path.exists():
            tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        else:
            tuning = _autotune(kind, spec, settings, texts, targets, rows, device, budget)
            _json(tuning_path, tuning)
    except BudgetExceeded:
        result = {"status": "paused", "best_epoch": None, "best_score": None,
                  "checkpoint_dir": str(directory / "best_model"), "runtime": {"stage": "batch_autotune"}}
        _json(result_path, result)
        return result
    batch_size = int(tuning["selected_batch_size"])
    seed_everything(seed)  # Benchmark updates and RNG never become Training initialization.
    model, tokenizer = create_model(kind, spec, settings, device)
    tokens = _tokenize(tokenizer, texts, maximum, kind)
    if not (directory / "length_statistics.json").exists():
        _json(directory / "length_statistics.json", _length_statistics(tokenizer, texts, rows, maximum, kind))
    optimizer = _optimizer(model, kind, settings)
    total_updates = math.ceil(len(rows) / effective) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * float(_get(settings, "warmup_ratio", .05))), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_dtype(device) == torch.float16)
    weights = positive_weights(targets[rows]).to(device)
    progress = {"epoch": 0, "next_update": 0, "phase": "train", "global_updates": 0,
                "best_epoch": None, "best_score": None, "history": [], "epoch_loss_sum": 0.,
                "epoch_examples": 0, "overflow_retries": 0, "training_seconds": 0.}
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["batch_size"] != batch_size:
            raise ValueError("Training checkpoint identity or batch size mismatch")
        if state["precision"] != str(_amp_dtype(device) or torch.float32):
            raise ValueError("Exact resume requires the same mixed-precision mode")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        progress = state["progress"]
        _restore_rng(state["rng"])
        del state
    remaining_rows = (epochs - progress["epoch"]) * len(rows) - progress["epoch_examples"]
    estimated_minutes = float(tuning["seconds_per_training_example"]) * remaining_rows / 60.
    print(f"{kind}: selected batch {batch_size}, effective batch {effective}; measured estimate: "
          f"{estimated_minutes:.1f} remaining training minutes (excludes download, tokenization, evaluation and checkpoint saves).", flush=True)

    def save_state():
        _atomic_torch_save({"identity": identity, "batch_size": batch_size,
                           "precision": str(_amp_dtype(device) or torch.float32),
                           "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                           "progress": progress, "rng": _rng()}, state_path)

    status = "complete"
    try:
        while progress["epoch"] < epochs:
            epoch = int(progress["epoch"])
            order = np.random.default_rng(seed + epoch).permutation(rows)
            if progress["phase"] == "train":
                for update in range(progress["next_update"], math.ceil(len(rows) / effective)):
                    if budget is not None:
                        budget.check("classifier_training", training=True)
                    group = order[update * effective:(update + 1) * effective]
                    _sync(device)
                    update_started = time.monotonic()
                    loss, retries = _update(model, tokenizer, tokens, targets, group, batch_size,
                                             optimizer, scaler, weights, device)
                    scheduler.step()
                    _sync(device)
                    progress["training_seconds"] += time.monotonic() - update_started
                    progress["next_update"] = update + 1
                    progress["global_updates"] += 1
                    progress["epoch_loss_sum"] += loss * len(group)
                    progress["epoch_examples"] += len(group)
                    progress["overflow_retries"] += retries
                    every = int(_get(settings, "checkpoint_every_updates", 100))
                    if every > 0 and progress["global_updates"] % every == 0:
                        save_state()
                    log_every = int(_get(settings, "log_every", 50))
                    if log_every > 0 and progress["global_updates"] % log_every == 0:
                        print(f"{kind}: epoch {epoch + 1}/{epochs}, update {progress['global_updates']}/{total_updates}, loss {loss:.5f}", flush=True)
                progress["phase"] = "evaluate"
                save_state()
            score = None
            if dev is not None:
                logits = _predict(model, tokenizer, tokens, dev, batch_size, device, budget)
                score = _macro_f1(targets[dev], logits)
                np.save(directory / f"epoch_{epoch + 1}_development_logits.npy", logits)
            improves = dev is None or progress["best_score"] is None or score > progress["best_score"]
            if improves:
                progress["best_epoch"], progress["best_score"] = epoch + 1, score
                metadata = {"kind": kind, "model_id": _get(spec, "model_id"), "revision": _get(spec, "revision"),
                            "max_length": maximum, "dropout": model.dropout_probability,
                            "selected_epochs": epoch + 1, "training_scope": "all_training" if dev is None else "fit_only",
                            "pos_weight": weights.cpu().tolist(), "decision": DECISION,
                            "pooling": model_protocol(kind)["pooling"]}
                _save_assets(directory / "best_model", model, tokenizer, metadata)
                _save_reload_reference(directory / "reload_reference.npz", model, tokenizer, tokens,
                                       rows if dev is None else dev, batch_size, device)
            progress["history"].append({"epoch": epoch + 1, "macro_f1": score,
                                         "train_mean_loss": progress["epoch_loss_sum"] / progress["epoch_examples"],
                                         "train_examples": progress["epoch_examples"]})
            progress.update(epoch=epoch + 1, next_update=0, phase="train", epoch_loss_sum=0., epoch_examples=0)
            save_state()
    except BudgetExceeded:
        status = "paused"
        save_state()
    finally:
        optimizer.zero_grad(set_to_none=True)
    result = {"status": status, "best_epoch": progress["best_epoch"], "best_score": progress["best_score"],
              "checkpoint_dir": str(directory / "best_model"), "history": progress["history"],
              "runtime": {"invocation_seconds": time.monotonic() - started,
                          "training_seconds": progress["training_seconds"], "global_updates": progress["global_updates"],
                          "overflow_retries": progress["overflow_retries"], "batch_size": batch_size,
                          "effective_batch_size": effective, "precision": str(_amp_dtype(device) or torch.float32),
                          "device": str(device), "epoch": progress["epoch"], "phase": progress["phase"]}}
    _json(directory / "history.json", progress["history"])
    _json(result_path, result)
    if status == "complete" and state_path.exists():
        state_path.unlink()  # Only this trainer's now-obsolete resumable optimizer state.
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
