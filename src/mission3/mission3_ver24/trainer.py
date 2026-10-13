"""Matched BERT/RoBERTa fine-tuning with fixed multilabel decisions.

Only caller-provided Training rows enter the loss. The runtime owns the split,
candidate selection, epoch count, admission checks, and full-data refit.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .data import TARGET_SYMPTOMS


FORMAT = "mission3_ver24_mean_bce_v1"
DECISION = "independent_logits_greater_than_or_equal_to_zero"


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
    """Fixed sqrt5 rule, calculated exclusively from the supplied fit rows."""
    values = _targets(fit_targets)
    if not len(values):
        raise ValueError("Positive weights require at least one fit row")
    positives = values.sum(axis=0, dtype=np.float64)
    ratios = np.maximum((len(values) - positives) / np.maximum(positives, 1), 1)
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
            raise ValueError("Only standard BERT and RoBERTa encoders are supported")
        if not 0 <= float(dropout) < 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.encoder = encoder
        # Mean pooling uses last_hidden_state. A pretrained MLM has no pooler
        # weights, and an unused randomly initialized pooler adds no signal.
        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(TARGET_SYMPTOMS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def create_model(model_name, revision=None, *, seed=42, cache_dir=None,
                 local_files_only=False, dropout=.1):
    """Start from a pinned public encoder, or an explicitly local test snapshot."""
    local = Path(model_name).is_dir()
    if not local and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("Remote pretrained models require an immutable 40-character revision")
    seed_everything(seed)
    options = dict(trust_remote_code=False, local_files_only=local_files_only)
    if not local:
        options["revision"] = revision
    if cache_dir is not None:
        options["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_name), use_fast=True, **options)
    encoder = AutoModel.from_pretrained(str(model_name), add_pooling_layer=False, **options)
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must provide a padding token")
    if len(tokenizer) != encoder.config.vocab_size:
        raise ValueError("Tokenizer and pretrained encoder vocabulary sizes differ")
    model = MeanPoolClassifier(encoder, dropout=dropout)
    return model, tokenizer


class _Rows(Dataset):
    def __init__(self, texts, targets=None, indices=None):
        if not len(texts) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Texts must be a nonempty sequence of strings")
        self.texts = texts
        self.targets = None if targets is None else _targets(targets, len(texts))
        self.indices = (np.arange(len(texts), dtype=np.int64) if indices is None
                        else np.asarray(indices))
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


class _Collate:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        tokens = self.tokenizer([row[0] for row in rows], padding=True,
                                truncation=True, max_length=self.max_length,
                                return_tensors="pt")
        result = {name: value for name, value in tokens.items()
                  if name in ("input_ids", "attention_mask", "token_type_ids")}
        if rows[0][1] is not None:
            result["labels"] = torch.tensor(np.stack([row[1] for row in rows]), dtype=torch.float32)
        return result


def make_loader(tokenizer, texts, targets=None, indices=None, *, batch_size=8,
                max_length=512, shuffle=False, seed=42, pin_memory=False):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 2 <= max_length <= 512:
        raise ValueError("Max length must be an integer in [2, 512]")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(_Rows(texts, targets, indices), batch_size=batch_size,
                      shuffle=bool(shuffle), generator=generator,
                      collate_fn=_Collate(tokenizer, max_length), drop_last=False,
                      num_workers=0, pin_memory=bool(pin_memory))


def build_optimizer(model, lr=2e-5, weight_decay=.01):
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=float(lr))


def build_scheduler(optimizer, total_updates, warmup_ratio=.1):
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
                scheduler=None, scaler=None, budget=None, on_update=None, max_updates=None):
    """Train all rows once, including the final partial accumulation group.

    A pilot can stop after ``max_updates`` successful updates. Such a partial
    epoch is explicitly marked incomplete and must never be a final refit.
    Timing covers tokenization, transfer, forward, backward, and optimizer work.
    """
    if (isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int)
            or accumulation_steps < 1):
        raise ValueError("Accumulation steps must be a positive integer")
    if max_updates is not None and (isinstance(max_updates, bool)
                                    or not isinstance(max_updates, int) or max_updates < 1):
        raise ValueError("Max updates must be a positive integer")
    if not len(loader):
        raise ValueError("Training loader is empty")
    device = torch.device(device)
    weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    if weights.shape != (len(TARGET_SYMPTOMS),) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Positive weights must contain nine finite positive numbers")
    scaler = make_scaler(device) if scaler is None else scaler
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    successful = attempted = examples = seen_batches = group_rows = 0
    loss_sum = 0.0
    update_seconds = []
    _synchronize(device)
    started = previous_update = time.monotonic()
    try:
        for batch_index, batch in enumerate(loader):
            if budget is not None:
                budget.check("encoder_training", training=True)
            if "labels" not in batch:
                raise ValueError("Training batches must contain labels")
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()
                      if key != "labels"}
            with _autocast(device):
                logits = model(**inputs)
                loss = loss_function(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            rows = len(labels)
            # Weight microbatches by row count before dividing accumulated
            # gradients; the final short batch must not be overrepresented.
            scaler.scale(loss * rows).backward()
            group_rows += rows
            examples += rows
            seen_batches += 1
            loss_sum += float(loss.detach()) * rows
            if (batch_index + 1) % accumulation_steps and batch_index + 1 < len(loader):
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_rows)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError("Training gradients are not finite")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            attempted += 1
            accepted = scaler.get_scale() >= old_scale
            optimizer.zero_grad(set_to_none=True)
            group_rows = 0
            _synchronize(device)
            now = time.monotonic()
            if accepted:
                successful += 1
                update_seconds.append(now - previous_update)
                if scheduler is not None:
                    scheduler.step()
                progress = {"successful_updates": successful, "attempted_updates": attempted,
                            "examples": examples, "batches": seen_batches,
                            "elapsed_seconds": now - started,
                            "update_seconds": list(update_seconds),
                            "measured_update_seconds": list(update_seconds[1:])}
                if on_update is not None:
                    on_update(progress)
                if max_updates is not None and successful >= max_updates:
                    break
                previous_update = now
            # An overflow's duration remains in the next successful update's
            # timing, so admission estimates do not hide skipped work.
    finally:
        optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    if successful == 0:
        raise FloatingPointError("Training completed no successful optimizer updates")
    measured = update_seconds[1:]
    return {"successful_updates": successful, "attempted_updates": attempted,
            "examples": examples, "batches": seen_batches,
            "mean_loss": loss_sum / max(examples, 1),
            "elapsed_seconds": time.monotonic() - started,
            "update_seconds": update_seconds, "measured_update_seconds": measured,
            "seconds_per_update": float(np.mean(measured)) if measured else None,
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
            budget.check("encoder_inference", training=False)
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


def _json_metadata(metadata):
    copied = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    if not isinstance(copied, dict):
        raise ValueError("Checkpoint metadata must be an object")
    return copied


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "target_symptoms": list(TARGET_SYMPTOMS),
               "decision": DECISION, "encoder_config": model.encoder.config.to_dict(),
               "dropout": model.dropout_probability, "metadata": _json_metadata(metadata),
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("format") != FORMAT or payload.get("target_symptoms") != list(TARGET_SYMPTOMS)
            or payload.get("decision") != DECISION):
        raise ValueError("Checkpoint format, label order, or fixed decision policy is invalid")
    config_dict = dict(payload["encoder_config"])
    kind = config_dict.pop("model_type")
    if kind not in ("bert", "roberta"):
        raise ValueError("Checkpoint encoder type is unsupported")
    config = AutoConfig.for_model(kind, **config_dict)
    model = MeanPoolClassifier(AutoModel.from_config(
        config, trust_remote_code=False, add_pooling_layer=False), payload["dropout"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), _json_metadata(payload["metadata"])


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(directory, model, tokenizer, metadata):
    """Write a standalone checkpoint and tokenizer; no remote code is required."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Export destination must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    meta = _json_metadata(metadata)
    max_length = meta.get("max_length", 512)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 2 <= max_length <= 512:
        raise ValueError("Export max_length must be an integer in [2, 512]")
    meta["max_length"] = max_length
    tokenizer.save_pretrained(directory / "tokenizer")
    save_checkpoint(directory / "model.pt", model, meta)
    hashes = {path.relative_to(directory).as_posix(): _sha256(path)
              for path in sorted(directory.rglob("*")) if path.is_file()}
    manifest = {"format": FORMAT, "target_symptoms": list(TARGET_SYMPTOMS),
                "decision": DECISION, "files": hashes, "metadata": meta}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def load_export(directory, device="cpu"):
    """Verify every exported asset before local-only tokenizer/model loading."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != FORMAT or manifest.get("target_symptoms") != list(TARGET_SYMPTOMS)
            or manifest.get("decision") != DECISION or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Export manifest is invalid")
    if "model.pt" not in manifest["files"]:
        raise ValueError("Export checkpoint is missing")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if (not isinstance(relative, str) or path.is_symlink()
                or not path.resolve().is_relative_to(directory) or not path.is_file()
                or _sha256(path) != expected):
            raise ValueError("Export asset integrity check failed")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Export includes unverified files")
    model, metadata = load_checkpoint(directory / "model.pt", device=device)
    if metadata != manifest.get("metadata"):
        raise ValueError("Export checkpoint and manifest metadata disagree")
    tokenizer = AutoTokenizer.from_pretrained(
        directory / "tokenizer", local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None or len(tokenizer) != model.encoder.config.vocab_size:
        raise ValueError("Export tokenizer and model are incompatible")
    return model, tokenizer, metadata


def predict_texts(model, tokenizer, texts, *, device="cpu", batch_size=16,
                  max_length=512, budget=None):
    loader = make_loader(tokenizer, texts, batch_size=batch_size, max_length=max_length,
                         pin_memory=torch.device(device).type == "cuda")
    return predict_logits(model, loader, device=device, budget=budget)
