"""Ver27 weighted-BCE KLUE control and last-token Qwen QLoRA classifier.

Only caller-selected Training rows enter the explicit loss. Quantized frozen
bases are saved once before adapter injection; reload never needs the network.
"""
from __future__ import annotations

from contextlib import nullcontext
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from .data import TARGET_SYMPTOMS

FORMAT = "mission3_ver27_qlora_bce_v1"
DECISION = "independent_logits_greater_than_or_equal_to_zero"
JUDGMENT_SUFFIX = "\n위 통화에서 환자에게 나타난 증상을 판단하세요."
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _get(settings, name, default=None):
    return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)


def _kind(spec):
    value = _get(spec, "kind", _get(spec, "name"))
    if value not in ("control", "qwen"):
        raise ValueError("Model kind must be control or qwen")
    return value


def seed_everything(seed=42):
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
        raise ValueError("Positive weights require at least one fit row")
    positives = values.sum(axis=0, dtype=np.float64)
    ratios = (len(values) - positives) / np.maximum(positives, 1)
    return torch.tensor(np.clip(np.sqrt(ratios), 1, 5), dtype=torch.float32)


def fixed_decisions(logits):
    values = np.asarray(logits)
    if (values.ndim != 2 or values.shape[1] != len(TARGET_SYMPTOMS)
            or not np.isfinite(values).all()):
        raise ValueError("Logits must be finite with shape (samples, 9)")
    return (values >= 0).astype(np.int8)


class MeanPoolClassifier(nn.Module):
    def __init__(self, encoder, dropout=.1):
        super().__init__()
        if encoder.config.model_type not in ("bert", "roberta"):
            raise ValueError("The control requires BERT or RoBERTa")
        if not 0 <= float(dropout) < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.encoder = encoder
        if hasattr(encoder, "pooler"):
            encoder.pooler = None
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(TARGET_SYMPTOMS))
        self.kind = "control"

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


class LastTokenClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        if encoder.config.model_type != "qwen3":
            raise ValueError("The Qwen candidate requires a Qwen3 backbone")
        self.encoder = encoder
        self.encoder.config.use_cache = False
        # This independent head remains float32, trainable, and explicitly saved.
        self.classifier = nn.Linear(encoder.config.hidden_size, len(TARGET_SYMPTOMS))
        self.kind = "qwen"

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if attention_mask.ndim != 2 or (attention_mask.sum(1) == 0).any():
            raise ValueError("Each example needs at least one unpadded token")
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=positions, use_cache=False,
                              return_dict=True).last_hidden_state
        indexes = torch.arange(attention_mask.shape[1], device=hidden.device)
        last = indexes.unsqueeze(0).masked_fill(attention_mask == 0, -1).max(1).values
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        return self.classifier(pooled.to(self.classifier.weight.dtype))


def create_model(spec, settings, device, local_files_only=False):
    """Start each comparison/refit from the pinned public pretrained backbone."""
    kind = _kind(spec)
    model_id, revision = _get(spec, "model_id"), _get(spec, "revision")
    source = _get(spec, "snapshot_path") or model_id
    local = isinstance(source, (str, Path)) and Path(source).is_dir()
    if model_id is None or source is None:
        raise ValueError("A model_id and pretrained source are required")
    if (not Path(str(model_id)).is_dir()
            and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision))):
        raise ValueError("Remote pretrained models require an immutable 40-character revision")
    device = torch.device(device)
    test_only = bool(_get(settings, "test_only_unquantized", False))
    if test_only and not (local and local_files_only):
        raise ValueError("Unquantized Qwen is restricted to explicit local offline tests")
    if kind == "qwen" and not test_only and device.type != "cuda":
        raise RuntimeError("The production NF4 QLoRA recipe requires CUDA")
    seed_everything(_get(settings, "seed", 42))
    options = dict(trust_remote_code=False, local_files_only=local_files_only)
    if not local:
        options["revision"] = revision
    if _get(settings, "cache_dir"):
        options["cache_dir"] = str(_get(settings, "cache_dir"))
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True, **options)
    if tokenizer.pad_token_id is None:
        if kind != "qwen" or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or a Qwen EOS padding fallback")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if kind == "control":
        encoder = AutoModel.from_pretrained(str(source), add_pooling_layer=False, **options)
        model = MeanPoolClassifier(encoder, _get(settings, "control_dropout", .1)).to(device)
    else:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        compute_dtype = _amp_dtype(device) or torch.float32
        loading = dict(options)
        if not test_only:
            loading.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype), device_map={"": str(device)},
                torch_dtype=compute_dtype)
        encoder = AutoModel.from_pretrained(str(source), **loading)
        if encoder.config.model_type != "qwen3":
            raise ValueError("The Qwen candidate requires Qwen3 weights")
        encoder.config.pad_token_id = tokenizer.pad_token_id
        encoder.config.use_cache = False
        # Saving here is essential: after injection the raw module state includes adapters.
        base_store = _get(settings, "base_store_dir")
        if base_store is None:
            raise ValueError("Qwen creation requires a shared base_store_dir for portable saves")
        identity = {"model_id": str(model_id), "revision": revision,
                    "quantized": not test_only, "compute_dtype": str(compute_dtype)}
        base_directory = _save_frozen_base(Path(base_store), encoder, identity)
        if test_only:
            encoder.requires_grad_(False)
            encoder.to(device)
        else:
            encoder = prepare_model_for_kbit_training(
                encoder, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        adapter = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             r=int(_get(settings, "lora_r", 16)),
                             lora_alpha=int(_get(settings, "lora_alpha", 32)),
                             lora_dropout=float(_get(settings, "lora_dropout", .05)),
                             target_modules=list(LORA_TARGETS), bias="none")
        encoder = get_peft_model(encoder, adapter)
        model = LastTokenClassifier(encoder)
        model.classifier.to(device)
        model._base_directory = base_directory
        model._base_identity = identity
    if len(tokenizer) > model.encoder.config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds the model embedding vocabulary")
    return model, tokenizer


class _Rows(Dataset):
    def __init__(self, texts, targets=None, indices=None):
        if not len(texts) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Texts must be a nonempty sequence of strings")
        self.texts = texts
        self.targets = None if targets is None else _targets(targets, len(texts))
        self.indices = np.arange(len(texts), dtype=np.int64) if indices is None else np.asarray(indices)
        if (self.indices.ndim != 1 or not len(self.indices)
                or not np.issubdtype(self.indices.dtype, np.integer)
                or np.any(self.indices < 0) or np.any(self.indices >= len(texts))
                or len(np.unique(self.indices)) != len(self.indices)):
            raise ValueError("Row indices must be unique valid integers")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        return self.texts[row], None if self.targets is None else self.targets[row]


def qwen_token_ids(tokenizer, text, max_length):
    suffix = tokenizer.encode(JUDGMENT_SUFFIX, add_special_tokens=False)
    available = max_length - len(suffix)
    if available < 2:
        raise ValueError("Max length must reserve the full fixed judgment suffix plus transcript tokens")
    transcript = tokenizer.encode(text, add_special_tokens=False)
    if len(transcript) > available:
        head = (available + 1) // 2
        tail = available - head
        transcript = transcript[:head] + transcript[-tail:]
    return transcript + suffix


class _Collate:
    def __init__(self, tokenizer, kind, max_length):
        self.tokenizer, self.kind, self.max_length = tokenizer, kind, max_length

    def __call__(self, rows):
        if self.kind == "qwen":
            inputs = [{"input_ids": qwen_token_ids(self.tokenizer, row[0], self.max_length)} for row in rows]
            tokens = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        else:
            tokens = self.tokenizer([row[0] for row in rows], padding=True,
                                    truncation=True, max_length=self.max_length, return_tensors="pt")
        result = {name: value for name, value in tokens.items()
                  if name in ("input_ids", "attention_mask", "token_type_ids")}
        if rows[0][1] is not None:
            result["labels"] = torch.tensor(np.stack([row[1] for row in rows]), dtype=torch.float32)
        return result


def make_loader(tokenizer, texts, targets=None, indices=None, *, kind, batch_size=8,
                max_length=512, shuffle=False, seed=42, pin_memory=False):
    if kind not in ("control", "qwen"):
        raise ValueError("Model kind must be control or qwen")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    maximum = 1024 if kind == "qwen" else 512
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 2 <= max_length <= maximum:
        raise ValueError(f"Max length must be an integer in [2, {maximum}]")
    if kind == "qwen":
        qwen_token_ids(tokenizer, "", max_length)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(_Rows(texts, targets, indices), batch_size=batch_size,
                      shuffle=bool(shuffle), generator=generator,
                      collate_fn=_Collate(tokenizer, kind, max_length), drop_last=False,
                      num_workers=0, pin_memory=bool(pin_memory))


def length_statistics(tokenizer, texts, kind, max_length):
    if not texts or kind not in ("control", "qwen"):
        raise ValueError("Length statistics require nonempty Training texts and a valid model kind")
    suffix_length = len(tokenizer.encode(JUDGMENT_SUFFIX, add_special_tokens=False)) if kind == "qwen" else 0
    lengths = np.asarray([len(tokenizer.encode(text, add_special_tokens=kind == "control")) + suffix_length
                          for text in texts], dtype=np.int64)
    truncated = np.maximum(lengths - max_length, 0)
    return {"rows": len(texts), "max_length": max_length, "suffix_tokens": suffix_length,
            "raw_token_percentiles": {str(q): float(np.percentile(lengths, q)) for q in (0, 25, 50, 75, 90, 95, 99, 100)},
            "overlength_count": int((lengths > max_length).sum()),
            "overlength_rate": float(np.mean(lengths > max_length)),
            "removed_tokens": int(truncated.sum()),
            "mean_effective_tokens": float(np.minimum(lengths, max_length).mean()),
            "truncation": "transcript_head_tail_with_fixed_suffix" if kind == "qwen" else "right_truncation"}


def build_optimizer(model, kind, settings):
    if kind not in ("control", "qwen"):
        raise ValueError("Model kind must be control or qwen")
    decay = float(_get(settings, "weight_decay", .01))
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = float(_get(settings, "head_lr", 3e-4) if kind == "qwen" and name.startswith("classifier.")
                   else _get(settings, "lora_lr", 1e-4) if kind == "qwen"
                   else _get(settings, "control_lr", 2e-5))
        wd = 0.0 if name.endswith("bias") or "norm" in name.lower() else decay
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(wd) or wd < 0:
            raise ValueError("Invalid optimizer settings")
        groups.setdefault((lr, wd), []).append(parameter)
    if not groups:
        raise ValueError("The model has no trainable parameters")
    return torch.optim.AdamW([{"params": parameters, "lr": lr, "weight_decay": wd}
                              for (lr, wd), parameters in groups.items()])


def build_scheduler(optimizer, total_updates, warmup_ratio=.05):
    if (isinstance(total_updates, bool) or not isinstance(total_updates, int)
            or total_updates < 1 or not 0 <= warmup_ratio < 1):
        raise ValueError("Invalid learning-rate schedule")
    return get_linear_schedule_with_warmup(
        optimizer, int(total_updates * warmup_ratio), total_updates)


def _amp_dtype(device):
    if torch.device(device).type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_scaler(device):
    device = torch.device(device)
    return torch.amp.GradScaler("cuda", enabled=_amp_dtype(device) == torch.float16)


def _autocast(device):
    dtype = _amp_dtype(device)
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, *, device, pos_weight, accumulation_steps=2,
                scheduler=None, scaler=None, budget=None, on_update=None, max_updates=None,
                max_microsteps=None, gradient_clip=1.0, max_overflow_retries=8):
    """Train all rows once, including the final partial accumulation group.

    A pilot can stop after ``max_updates`` successful updates. Such a partial
    epoch is explicitly marked incomplete and must never be a final refit.
    A recoverable fp16 scaler overflow replays the full accumulation group;
    no Training rows or scheduler steps are lost. Timing includes replay work.
    """
    if (isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int)
            or accumulation_steps < 1):
        raise ValueError("Accumulation steps must be a positive integer")
    if max_updates is not None and (isinstance(max_updates, bool)
                                    or not isinstance(max_updates, int) or max_updates < 1):
        raise ValueError("Max updates must be a positive integer")
    if max_microsteps is not None and (isinstance(max_microsteps, bool)
            or not isinstance(max_microsteps, int) or max_microsteps < 1):
        raise ValueError("Max microsteps must be a positive integer")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Gradient clipping must be finite and positive")
    if (isinstance(max_overflow_retries, bool) or not isinstance(max_overflow_retries, int)
            or max_overflow_retries < 0):
        raise ValueError("Maximum overflow retries must be a nonnegative integer")
    planned_batches = min(len(loader), max_microsteps or len(loader))
    if not len(loader):
        raise ValueError("Training loader is empty")
    device = torch.device(device)
    weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    if weights.shape != (len(TARGET_SYMPTOMS),) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Positive weights must contain nine finite positive numbers")
    scaler = make_scaler(device) if scaler is None else scaler
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    successful = attempted = examples = seen_batches = group_rows = 0
    loss_sum = 0.0
    update_seconds = []
    microstep_seconds = []
    group_batches = []
    group_rng = group_cuda_rng = None

    def backward_batch(batch):
        labels = batch["labels"].to(device, non_blocking=True)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()
                  if key != "labels"}
        with _autocast(device):
            loss = loss_function(model(**inputs).float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite")
        # Divide by total rows only after accumulation, including short tails.
        scaler.scale(loss * len(labels)).backward()
        return len(labels), float(loss.detach()) * len(labels)

    _synchronize(device)
    started = previous_update = previous_microstep = time.monotonic()
    try:
        for batch_index, batch in enumerate(loader):
            if budget is not None:
                budget.check("classifier_training", training=True)
            if "labels" not in batch:
                raise ValueError("Training batches must contain labels")
            if scaler.is_enabled():
                if not group_batches:
                    group_rng = torch.get_rng_state()
                    group_cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                group_batches.append({key: value.detach().to("cpu", copy=True) for key, value in batch.items()})
            rows, batch_loss_sum = backward_batch(batch)
            group_rows += rows
            examples += rows
            seen_batches += 1
            loss_sum += batch_loss_sum
            if (batch_index + 1) % accumulation_steps and batch_index + 1 < planned_batches:
                _synchronize(device)
                microstep_finished = time.monotonic()
                microstep_seconds.append(microstep_finished - previous_microstep)
                previous_microstep = microstep_finished
                continue
            retries = 0
            while True:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_rows)
                norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if not scaler.is_enabled() and not torch.isfinite(norm):
                    raise FloatingPointError("Training gradients are not finite")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                attempted += 1
                accepted = scaler.get_scale() >= old_scale
                optimizer.zero_grad(set_to_none=True)
                if accepted:
                    break
                if retries >= max_overflow_retries:
                    raise FloatingPointError("Gradient scaler overflow retry limit exceeded; no rows were skipped")
                retries += 1
                torch.set_rng_state(group_rng)
                if group_cuda_rng is not None:
                    torch.cuda.set_rng_state(group_cuda_rng, device)
                for replay_batch in group_batches:
                    if budget is not None:
                        budget.check("classifier_overflow_replay", training=True)
                    backward_batch(replay_batch)
            group_batches.clear()
            group_rows = 0
            _synchronize(device)
            now = time.monotonic()
            microstep_seconds.append(now - previous_microstep)
            previous_microstep = now
            successful += 1
            update_seconds.append(now - previous_update)
            if scheduler is not None:
                scheduler.step()
            progress = {"successful_updates": successful, "attempted_updates": attempted,
                        "overflow_retries": attempted - successful,
                        "examples": examples, "batches": seen_batches, "microsteps": seen_batches,
                        "elapsed_seconds": now - started,
                        "update_seconds": list(update_seconds),
                        "measured_update_seconds": list(update_seconds[1:])}
            if on_update is not None:
                on_update(progress)
            if max_updates is not None and successful >= max_updates:
                break
            previous_update = now
            if seen_batches >= planned_batches:
                break
    finally:
        optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    if successful == 0:
        raise FloatingPointError("Training completed no successful optimizer updates")
    measured = update_seconds[1:]
    return {"successful_updates": successful, "attempted_updates": attempted,
            "overflow_retries": attempted - successful, "final_scaler_scale": float(scaler.get_scale()),
            "examples": examples, "batches": seen_batches, "microsteps": seen_batches,
            "mean_loss": loss_sum / max(examples, 1),
            "elapsed_seconds": time.monotonic() - started,
            "update_seconds": update_seconds, "measured_update_seconds": measured,
            "seconds_per_update": float(np.mean(measured)) if measured else None,
            "microstep_seconds": microstep_seconds,
            "measured_microstep_seconds": microstep_seconds[min(accumulation_steps, seen_batches - 1):],
            "seconds_per_microstep": float(np.mean(
                microstep_seconds[min(accumulation_steps, seen_batches - 1):])),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "completed_epoch": seen_batches == len(loader)}


@torch.no_grad()
def predict_logits(model, loader, *, device, budget=None, max_batches=None, return_stats=False):
    if max_batches is not None and (isinstance(max_batches, bool)
                                    or not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("Max batches must be a positive integer")
    device = torch.device(device)
    model.eval()
    chunks = []
    _synchronize(device)
    started = time.monotonic()
    for index, batch in enumerate(loader):
        if budget is not None:
            budget.check("classifier_inference", training=False)
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()
                  if key != "labels"}
        with _autocast(device):
            logits = model(**inputs)
        values = logits.float().cpu().numpy()
        fixed_decisions(values)
        chunks.append(values)
        if max_batches is not None and index + 1 >= max_batches:
            break
    if not chunks:
        raise ValueError("Prediction loader is empty")
    _synchronize(device)
    values = np.concatenate(chunks, axis=0)
    elapsed = time.monotonic() - started
    if return_stats:
        return {"logits": values, "examples": len(values), "elapsed_seconds": elapsed,
                "seconds_per_example": elapsed / len(values)}
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _file_hashes(directory, excluded):
    return {path.relative_to(directory).as_posix(): _sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != excluded}


def _verify_files(directory, files, excluded):
    directory = directory.resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("Asset manifest must contain verified files")
    for relative, expected in files.items():
        path = directory / relative
        if (not isinstance(relative, str) or path.is_symlink() or Path(relative).is_absolute()
                or not path.resolve().is_relative_to(directory) or not path.is_file()
                or _sha256(path) != expected):
            raise ValueError("Model asset integrity check failed")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name != excluded}
    if actual != set(files):
        raise ValueError("Model directory contains unverified files")


def _save_frozen_base(store, encoder, identity):
    """Persist the clean frozen backbone before PEFT modifies its modules."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = store / key
    manifest_path = directory / "base_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("format") != FORMAT:
            raise ValueError("Shared base identity differs from the requested pinned source")
        # Full hashes are checked at portable reload; repeated candidate starts
        # need not reread gigabytes of already-written frozen base weights.
        for relative in manifest.get("files", {}):
            if not (directory / relative).is_file():
                raise ValueError("Shared base is incomplete")
        return directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Incomplete shared base directory; use a fresh run")
    directory.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(directory, safe_serialization=True, max_shard_size="4GB")
    _write_json(manifest_path, {"format": FORMAT, "identity": identity,
                              "files": _file_hashes(directory, "base_manifest.json")})
    return directory.resolve()


def save_model(path, model, tokenizer, spec, settings, training_scope, selected_epochs):
    """Save a portable classifier directory and return its recipe metadata.

    Qwen directories reference the run's shared frozen base by a relative path.
    Copy the whole run/submission tree to preserve that relationship. No source
    snapshot or Hugging Face cache is needed by ``load_model``.
    """
    directory = Path(path)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Model save destination must be empty")
    kind = _kind(spec)
    if kind != model.kind:
        raise ValueError("Model and metadata kinds differ")
    if training_scope not in ("fit_only", "all_training", "fit", "full_training"):
        raise ValueError("Training scope must explicitly identify fit-only or all-Training weights")
    if isinstance(selected_epochs, bool) or not isinstance(selected_epochs, int) or selected_epochs < 1:
        raise ValueError("Selected epochs must be a positive integer")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": kind, "model_id": str(_get(spec, "model_id")), "base_revision": _get(spec, "revision"),
        "max_length": int(_get(settings, f"{kind}_max_length", 1024 if kind == "qwen" else 512)),
        "batch_size": int(_get(settings, f"{kind}_batch_size", 1 if kind == "qwen" else 8)),
        "training_scope": training_scope, "selected_epochs": selected_epochs,
        "loss": "BCEWithLogitsLoss_with_explicit_training_subset_pos_weight",
        "pos_weight_rule": "clip(sqrt(N_negative/max(N_positive,1)),1,5)",
        "decision": DECISION, "target_symptoms": list(TARGET_SYMPTOMS),
        "judgment_suffix": JUDGMENT_SUFFIX if kind == "qwen" else None,
    }
    tokenizer.save_pretrained(directory / "tokenizer")
    if kind == "qwen":
        base = getattr(model, "_base_directory", None)
        if base is None or not (Path(base) / "base_manifest.json").is_file():
            raise ValueError("A Qwen checkpoint requires its separately saved clean frozen base")
        metadata["base_relative_path"] = Path(os.path.relpath(base, directory)).as_posix()
        metadata["base_identity"] = model._base_identity
        metadata["base_manifest_sha256"] = _sha256(Path(base) / "base_manifest.json")
        model.encoder.save_pretrained(directory / "adapter", safe_serialization=True)
        torch.save({name: tensor.detach().cpu() for name, tensor in model.classifier.state_dict().items()},
                   directory / "head.pt")
    else:
        metadata["dropout"] = model.dropout_probability
        model.encoder.save_pretrained(directory / "encoder", safe_serialization=True)
        torch.save({name: tensor.detach().cpu() for name, tensor in model.classifier.state_dict().items()},
                   directory / "head.pt")
    _write_json(directory / "manifest.json", {
        "format": FORMAT, "target_symptoms": list(TARGET_SYMPTOMS), "decision": DECISION,
        "metadata": metadata, "files": _file_hashes(directory, "manifest.json")})
    return metadata


def load_model(path, device="cpu"):
    """Integrity-check every local asset, then reload without network access."""
    directory = Path(path)
    if directory.is_file():
        if directory.name != "manifest.json":
            raise ValueError("A model checkpoint file must be its manifest.json")
        directory = directory.parent
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("target_symptoms") != list(TARGET_SYMPTOMS)
            or manifest.get("decision") != DECISION or not isinstance(manifest.get("metadata"), dict)):
        raise ValueError("Checkpoint format, labels, or fixed decision policy is invalid")
    _verify_files(directory, manifest.get("files"), "manifest.json")
    metadata = manifest["metadata"]
    kind = _kind(metadata)
    if metadata.get("target_symptoms") != list(TARGET_SYMPTOMS) or metadata.get("decision") != DECISION:
        raise ValueError("Checkpoint metadata conflicts with the fixed label order or decision")
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", use_fast=True,
                                               local_files_only=True, trust_remote_code=False)
    options = dict(local_files_only=True, trust_remote_code=False)
    if kind == "qwen":
        from peft import PeftModel, prepare_model_for_kbit_training
        relative = metadata.get("base_relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Frozen base must use a portable relative path")
        base = (directory / relative).resolve()
        base_manifest_path = base / "base_manifest.json"
        if _sha256(base_manifest_path) != metadata.get("base_manifest_sha256"):
            raise ValueError("Frozen base manifest integrity check failed")
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        identity = metadata.get("base_identity")
        if (base_manifest.get("format") != FORMAT or base_manifest.get("identity") != identity
                or identity.get("revision") != metadata.get("base_revision")):
            raise ValueError("Frozen base identity does not match the adapter")
        _verify_files(base, base_manifest.get("files"), "base_manifest.json")
        if identity.get("quantized"):
            if device.type != "cuda":
                raise RuntimeError("The saved NF4 Qwen model requires a CUDA device")
            options.update(device_map={"": str(device)}, torch_dtype=_amp_dtype(device))
        encoder = AutoModel.from_pretrained(base, **options)
        encoder.config.use_cache = False
        if identity.get("quantized"):
            # Match training's float32 non-quantized norms/embeddings as well
            # as its frozen 4-bit projections; omitting this changes logits.
            encoder = prepare_model_for_kbit_training(encoder, use_gradient_checkpointing=False)
        else:
            encoder.to(device)
        encoder = PeftModel.from_pretrained(encoder, directory / "adapter", is_trainable=False,
                                            local_files_only=True)
        model = LastTokenClassifier(encoder)
        model._base_directory, model._base_identity = base, identity
        model.classifier.to(device)
    else:
        encoder = AutoModel.from_pretrained(directory / "encoder", add_pooling_layer=False, **options)
        model = MeanPoolClassifier(encoder, metadata["dropout"]).to(device)
    model.classifier.load_state_dict(torch.load(directory / "head.pt", weights_only=True, map_location=device),
                                     strict=True)
    if tokenizer.pad_token_id is None or len(tokenizer) > model.encoder.config.vocab_size:
        raise ValueError("Saved tokenizer and model vocabulary are incompatible")
    if kind == "qwen" and metadata.get("judgment_suffix") != JUDGMENT_SUFFIX:
        raise ValueError("Saved Qwen tokenization suffix differs from this model code")
    return model.eval(), tokenizer, metadata


def predict_texts(model, tokenizer, texts, *, device="cpu", batch_size=16,
                  max_length=512, budget=None):
    loader = make_loader(tokenizer, texts, kind=model.kind, batch_size=batch_size,
                         max_length=max_length, pin_memory=torch.device(device).type == "cuda")
    return predict_logits(model, loader, device=device, budget=budget)


