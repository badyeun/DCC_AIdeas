"""Document-level multi-label transformer; every dialogue window is retained."""

from __future__ import annotations

import math
import os
import random
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup


NUM_LABELS = 9


def tokenize_dialogues(texts, tokenizer, max_length=512, stride=64):
    """Return documents of token windows, including the last window of long calls.

    Only ordinary tokenizer special tokens are used. The stride is an overlap in
    tokens, not an upper bound on the number of windows or dialogue length.
    """
    usable_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if usable_length < 1 or not 0 <= stride < usable_length:
        raise ValueError("max_length must fit special tokens; 0 <= stride < usable length")
    texts = list(texts)
    documents = [[] for _ in texts]
    for start in range(0, len(texts), 128):
        encoded = tokenizer(
            texts[start:start + 128], truncation=True, max_length=max_length,
            stride=stride, return_overflowing_tokens=True, padding=False,
            return_attention_mask=True,
        )
        if "overflow_to_sample_mapping" not in encoded:
            raise ValueError("Use a fast tokenizer (AutoTokenizer(..., use_fast=True))")
        for window, sample in enumerate(encoded["overflow_to_sample_mapping"]):
            documents[start + int(sample)].append({
                key: encoded[key][window]
                for key in tokenizer.model_input_names if key in encoded
            })
    if any(not windows for windows in documents):
        raise ValueError("Tokenizer returned a document without any windows")
    return documents


class DialogueDataset(Dataset):
    def __init__(self, documents, targets=None):
        self.documents, self.targets = documents, targets

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        item = {"windows": self.documents[index]}
        if self.targets is not None:
            item["labels"] = self.targets[index]
        return item


class DialogueCollator:
    """Flatten windows for the encoder while keeping exactly one target per call."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        windows, doc_indices = [], []
        for index, example in enumerate(examples):
            if not example["windows"]:
                raise ValueError("A document must contain at least one token window")
            windows.extend(example["windows"])
            doc_indices.extend([index] * len(example["windows"]))
        batch = dict(self.tokenizer.pad(windows, padding=True, return_tensors="pt"))
        batch["doc_indices"] = torch.tensor(doc_indices, dtype=torch.long)
        if "labels" in examples[0]:
            batch["labels"] = torch.from_numpy(np.asarray(
                [example["labels"] for example in examples], dtype=np.float32,
            ))
        return batch


class DialogueClassifier(nn.Module):
    """Label-specific attention pools complete dialogues before computing BCE."""

    def __init__(self, encoder, num_labels=NUM_LABELS, dropout=0.2,
                 window_batch_size=8):
        super().__init__()
        self.encoder = encoder
        self.window_batch_size = int(window_batch_size)
        if self.window_batch_size < 1:
            raise ValueError("window_batch_size must be positive")
        hidden = encoder.config.hidden_size
        self.projection = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.attention_queries = nn.Parameter(torch.empty(num_labels, hidden))
        self.label_weights = nn.Parameter(torch.empty(num_labels, hidden))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.normal_(self.attention_queries, std=0.02)
        nn.init.normal_(self.label_weights, std=0.02)

    def forward(self, doc_indices, **inputs):
        representations = []
        for start in range(0, len(doc_indices), self.window_batch_size):
            window_inputs = {key: value[start:start + self.window_batch_size]
                             for key, value in inputs.items()}
            hidden = self.encoder(**window_inputs).last_hidden_state
            mask = window_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            representations.append(self.projection(torch.cat([hidden[:, 0], mean], -1)))
        windows = self.dropout(torch.cat(representations))
        logits = []
        # Each label learns which part of a potentially long call provides evidence.
        for document_index in range(int(doc_indices.max().item()) + 1):
            selected = windows[doc_indices == document_index]
            scores = selected @ self.attention_queries.T / math.sqrt(selected.shape[-1])
            attention = torch.softmax(scores.float(), dim=0).to(selected.dtype)
            pooled = attention.T @ selected
            logits.append((self.dropout(pooled) * self.label_weights).sum(-1) + self.label_bias)
        return torch.stack(logits)


def _device(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select a GPU runtime.")
    return device


def _amp(device):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


def _loader(documents, targets, tokenizer, batch_size, shuffle=False, seed=0):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DialogueDataset(documents, targets), batch_size=int(batch_size),
        shuffle=shuffle, collate_fn=DialogueCollator(tokenizer),
        num_workers=0, generator=generator,
    )


@torch.inference_mode()
def _predict(model, loader, device):
    model.eval()
    probabilities = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        with _amp(device):
            logits = model(**batch)
        probabilities.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, NUM_LABELS), dtype=np.float32)


def _optimizer(model, config):
    depth = int(getattr(model.encoder.config, "num_hidden_layers", 0))
    decay = float(config.get("layerwise_lr_decay", 0.9))
    if not 0 < decay <= 1:
        raise ValueError("layerwise_lr_decay must be in (0, 1]")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            match = re.search(r"\.layer\.(\d+)\.", name)
            level = int(match.group(1)) + 1 if match else (0 if ".embeddings." in name else depth + 1)
            lr = float(config.get("learning_rate", 1.5e-5)) * decay ** (depth + 1 - level)
        else:
            lr = float(config.get("head_learning_rate", 1e-4))
        wd = 0.0 if parameter.ndim < 2 or name.endswith("bias") else float(config.get("weight_decay", 0.01))
        groups.setdefault((lr, wd), []).append(parameter)
    return torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": wd}
                             for (lr, wd), params in groups.items()])


def _atomic_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _targets(targets, documents, name):
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(documents), NUM_LABELS) or not len(documents):
        raise ValueError(f"{name} requires nonempty documents and targets of shape (N, {NUM_LABELS})")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError(f"{name} targets must contain only 0 and 1")
    return targets


def train_model(train_documents, train_targets, stop_documents, stop_targets,
                tokenizer, config: dict, checkpoint_path: Path, seed: int):
    """Train on documents; select best epoch only by stopping-set macro AP.

    This function does not tune thresholds and never uses calibration targets.
    GPU enforcement for production belongs to the orchestration entry point.
    """
    train_targets = _targets(train_targets, train_documents, "train")
    stop_targets = _targets(stop_targets, stop_documents, "stop")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config)
    max_epochs, patience = int(config.get("max_epochs", 8)), int(config.get("patience", 2))
    accumulation = int(config.get("gradient_accumulation_steps", 4))
    if min(max_epochs, patience, accumulation) < 1:
        raise ValueError("max_epochs, patience and gradient_accumulation_steps must be positive")
    model_name = config.get("model_name", "klue/roberta-large")
    encoder = AutoModel.from_pretrained(model_name)
    if config.get("gradient_checkpointing", True):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False
    architecture = {"num_labels": NUM_LABELS, "dropout": float(config.get("dropout", 0.2)),
                    "window_batch_size": int(config.get("window_batch_size", 8))}
    model = DialogueClassifier(encoder, **architecture).to(device)
    train_loader = _loader(train_documents, train_targets, tokenizer,
                           config.get("batch_size", 2), shuffle=True, seed=seed)
    stop_loader = _loader(stop_documents, None, tokenizer, config.get("inference_batch_size", 4))
    positives = train_targets.sum(axis=0)
    if (positives == 0).any():
        raise ValueError(f"Training split has no positive examples for labels {np.flatnonzero(positives == 0).tolist()}")
    pos_weight = np.clip(((len(train_targets) - positives) / positives).clip(min=1)
                         ** float(config.get("pos_weight_power", 0.5)), 1, 5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, device=device))
    optimizer = _optimizer(model, config)
    total_steps = math.ceil(len(train_loader) / accumulation) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * float(config.get("warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
    valid_labels = stop_targets.sum(axis=0) > 0
    if not valid_labels.any():
        raise ValueError("Stopping split has no positive labels; macro AP cannot select an epoch")
    if not valid_labels.all():
        print(f"Warning: stop AP excludes labels without positives: {np.flatnonzero(~valid_labels).tolist()}", flush=True)
    best_ap, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, document_count = 0.0, 0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            # Match a single full document batch, including a shorter last batch.
            group_start = (step // accumulation) * accumulation
            group_documents = min(accumulation * train_loader.batch_size,
                                  len(train_documents) - group_start * train_loader.batch_size)
            with _amp(device):
                loss = criterion(model(**batch).float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={step}")
            scaler.scale(loss * len(labels) / group_documents).backward()
            loss_sum += loss.detach().item() * len(labels)
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
            log_every = int(config.get("log_every", 100))
            if log_every > 0 and (step + 1) % log_every == 0:
                elapsed = (time.monotonic() - epoch_started) / 60
                print(f"seed={seed} epoch={epoch} batch={step + 1}/{len(train_loader)} "
                      f"loss={loss_sum / document_count:.5f} minutes={elapsed:.1f}", flush=True)
        probabilities = _predict(model, stop_loader, device)
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Stopping predictions contain non-finite values")
        score = float(average_precision_score(stop_targets[:, valid_labels],
                                              probabilities[:, valid_labels], average="macro"))
        row = {"epoch": epoch, "train_loss": loss_sum / document_count,
               "stop_macro_ap": score, "stop_labels_with_positives": int(valid_labels.sum()),
               "minutes": (time.monotonic() - epoch_started) / 60}
        history.append(row)
        print(f"seed={seed} epoch={epoch}/{max_epochs} loss={row['train_loss']:.5f} "
              f"stop_AP={score:.5f} minutes={row['minutes']:.1f}", flush=True)
        if score > best_ap + 1e-6:
            best_ap, best_epoch, stale = score, epoch, 0
            _atomic_save({
                "format_version": 1, "encoder_config": encoder.config.to_dict(),
                "architecture": architecture, "model_name": str(model_name),
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch, "stop_ap": score, "seed": int(seed),
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return {"best_epoch": best_epoch, "best_stop_ap": best_ap, "history": history}


def predict_checkpoint(checkpoint_path, documents, tokenizer, config):
    """Rebuild from the saved config and weights; no pretrained network fetch."""
    device = _device(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported neural checkpoint format: {checkpoint_path}")
    encoder_data = dict(checkpoint["encoder_config"])
    model_type = encoder_data.pop("model_type")
    encoder_config = AutoConfig.for_model(model_type, **encoder_data)
    architecture = dict(checkpoint["architecture"])
    architecture["window_batch_size"] = int(config.get("window_batch_size", architecture["window_batch_size"]))
    model = DialogueClassifier(AutoModel.from_config(encoder_config), **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.to(device)
    loader = _loader(documents, None, tokenizer, config.get("inference_batch_size", 4))
    predictions = _predict(model, loader, device)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Inference predictions contain non-finite values")
    return predictions
