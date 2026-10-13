"""Text-only dialogue classifiers with ordinary binary class decisions.

Each symptom has two class logits. Predictions use the winning class directly;
there is no probability cutoff search, calibration, or post-hoc label weighting.
Public pretrained assets are prepared separately from local checkpoint weights.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from numbers import Real
from pathlib import Path
import random
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


NUM_LABELS = 9
CONTRASTIVE_DIM = 64
_MODEL_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


def seed_everything(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_encoder(model_id, revision, asset_dir):
    """Download an encoder once; store only its config and tokenizer assets.

    The caller stores the trained encoder and head together in its checkpoint.
    Transformers imports are lazy so CPU tests do not require that dependency.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be explicit")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    resolved = getattr(config, "_commit_hash", None)
    if resolved is None and re.fullmatch(r"[0-9a-f]{40}", revision):
        resolved = revision
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError("The immutable pretrained model revision was not resolved")
    # Resolve a mutable branch exactly once, then pin every other asset to it.
    options = {"revision": resolved, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **options)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Complete overflow windows require a fast tokenizer")
    encoder = AutoModel.from_pretrained(model_id, config=config, **options)
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False
    if not getattr(encoder, "supports_gradient_checkpointing", False):
        raise ValueError("The selected encoder must support gradient checkpointing")
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    asset_dir = Path(asset_dir)
    encoder_dir, tokenizer_dir = asset_dir / "encoder", asset_dir / "tokenizer"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    encoder.config.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    return encoder, tokenizer, resolved


def _binary_targets(targets, count):
    values = torch.as_tensor(targets)
    if values.shape != (count, NUM_LABELS):
        raise ValueError(f"targets must have shape ({count}, {NUM_LABELS})")
    if not bool(torch.isfinite(values).all()) or not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("targets must contain only finite binary values")
    return values.to(dtype=torch.long)


def _turn_spans(text, separator):
    """Locate nonempty raw turns in the unchanged separator-joined input."""
    pieces = re.split(r"\r\n?|\n", text)
    delimiter = f" {separator} "
    spans, cursor = [], 0
    for piece in pieces:
        if piece.strip():
            spans.append((cursor, cursor + len(piece), len(spans)))
        cursor += len(piece) + len(delimiter)
    return delimiter.join(pieces), spans


class EncodeDataset(Dataset):
    """Retain all fast-tokenizer overflow windows, including the final tokens."""

    def __init__(self, texts, targets, tokenizer, max_length=512, overlap=64, *,
                 utterance_evidence=False):
        texts = list(texts)
        if type(utterance_evidence) is not bool:
            raise ValueError("utterance_evidence must be boolean")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Every dialogue must be a text string")
        for name, value in (("max_length", max_length), ("overlap", overlap)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        usable = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if usable < 1 or not 0 <= overlap < usable:
            raise ValueError("max_length must fit special tokens and 0 <= overlap < usable tokens")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("The tokenizer must define a padding token")
        separator = getattr(tokenizer, "sep_token", None)
        if not isinstance(separator, str) or not separator:
            raise ValueError("The tokenizer must define an utterance separator token")
        self.tokenizer = tokenizer
        self.utterance_evidence = utterance_evidence
        self.targets = None if targets is None else _binary_targets(targets, len(texts)).clone()
        self.documents = [[] for _ in texts]
        # Unknown tokens still represent original content. Structural markers do
        # not become local evidence, but remain in the unchanged global input.
        structural_ids = set(getattr(tokenizer, "all_special_ids", []))
        structural_ids.discard(getattr(tokenizer, "unk_token_id", None))
        for start in range(0, len(texts), 128):
            dialogues = [re.sub(r"\r\n?|\n", f" {separator} ", text)
                         for text in texts[start:start + 128]]
            turn_spans = [_turn_spans(text, separator)[1] for text in texts[start:start + 128]] if utterance_evidence else None
            options = {"return_offsets_mapping": True} if utterance_evidence else {}
            encoded = tokenizer(
                dialogues, truncation=True, max_length=max_length, stride=overlap,
                return_overflowing_tokens=True, padding=False,
                return_attention_mask=True,
                **options,
            )
            if "overflow_to_sample_mapping" not in encoded:
                raise ValueError("A fast tokenizer must return overflow_to_sample_mapping")
            if "input_ids" not in encoded or "attention_mask" not in encoded:
                raise ValueError("The tokenizer must return input_ids and attention_mask")
            if utterance_evidence and "offset_mapping" not in encoded:
                raise ValueError("Utterance evidence requires fast-tokenizer offset_mapping")
            for index, source in enumerate(encoded["overflow_to_sample_mapping"]):
                source = int(source)
                if not 0 <= source < len(dialogues):
                    raise ValueError("Tokenizer overflow mapping is out of range")
                window = {key: list(encoded[key][index])
                          for key in _MODEL_INPUTS if key in encoded}
                length = len(window["input_ids"])
                if not 0 < length <= max_length or any(len(value) != length for value in window.values()):
                    raise ValueError("Tokenizer returned an invalid window length")
                if not all(value in (0, 1) for value in window["attention_mask"]) or not any(window["attention_mask"]):
                    raise ValueError("A real token window must have a binary, nonempty attention mask")
                if utterance_evidence:
                    offsets = encoded["offset_mapping"][index]
                    if len(offsets) != length:
                        raise ValueError("Tokenizer offsets must align with every token")
                    spans = turn_spans[source]
                    starts = [span[0] for span in spans]
                    turn_ids, occurrence_keys = [], []
                    for token, attended, offset in zip(window["input_ids"], window["attention_mask"], offsets):
                        left, right = map(int, offset)
                        if not 0 <= left <= right <= len(dialogues[source]):
                            raise ValueError("Tokenizer offset is outside the prepared dialogue")
                        position = bisect_right(starts, left) - 1
                        valid = (bool(attended) and token not in structural_ids and right > left
                                 and position >= 0 and right <= spans[position][1])
                        turn_ids.append(spans[position][2] if valid else -1)
                        occurrence_keys.append((left, right, int(token)) if valid else None)
                    window["turn_ids"] = turn_ids
                    window["_occurrence_keys"] = occurrence_keys
                self.documents[start + source].append(window)
        if any(not document for document in self.documents):
            raise ValueError("Tokenizer returned a dialogue without any windows")
        if utterance_evidence:
            for document in self.documents:
                counts = Counter(key for window in document for key in window["_occurrence_keys"] if key is not None)
                for window in document:
                    # Each original token has total mass one across overlapping
                    # windows. Contextual occurrences are averaged, then tokens
                    # within a turn are averaged; long/duplicated turns do not
                    # gain extra attention entries. Adjacent turns are joined
                    # only after this complete document-level reconstruction.
                    window["token_weights"] = [1.0 / counts[key] if key is not None else 0.0
                                               for key in window.pop("_occurrence_keys")]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        item = {"windows": self.documents[index]}
        if self.targets is not None:
            item["targets"] = self.targets[index]
        return item

    def collate(self, examples):
        """Pad tokens and windows; keep one target vector for each dialogue."""
        if not examples or any(not item["windows"] for item in examples):
            raise ValueError("A batch must contain nonempty dialogue windows")
        keys = set(examples[0]["windows"][0])
        if any(set(window) != keys for item in examples for window in item["windows"]):
            raise ValueError("Token windows must use the same model input fields")
        if any(("targets" in item) != ("targets" in examples[0]) for item in examples):
            raise ValueError("A batch cannot mix labeled and unlabeled dialogues")
        batch_size = len(examples)
        window_count = max(len(item["windows"]) for item in examples)
        token_count = max(len(window["input_ids"]) for item in examples for window in item["windows"])
        batch = {key: torch.full(
            (batch_size, window_count, token_count),
            self.tokenizer.pad_token_id if key == "input_ids" else 0,
            dtype=torch.long,
        ) for key in _MODEL_INPUTS if key in keys}
        if self.utterance_evidence:
            batch["turn_ids"] = torch.full((batch_size, window_count, token_count), -1, dtype=torch.long)
            batch["token_weights"] = torch.zeros((batch_size, window_count, token_count), dtype=torch.float32)
        batch["window_mask"] = torch.zeros((batch_size, window_count), dtype=torch.bool)
        for row, item in enumerate(examples):
            for column, window in enumerate(item["windows"]):
                length = len(window["input_ids"])
                for key in keys:
                    batch[key][row, column, :length] = torch.tensor(window[key], dtype=batch[key].dtype)
                batch["window_mask"][row, column] = True
        if "targets" in examples[0]:
            batch["targets"] = torch.stack([item["targets"] for item in examples])
        return batch


def validate_evidence_config(hidden=128, dropout=0.1):
    if type(hidden) is not int or not 1 <= hidden <= 512:
        raise ValueError("evidence_hidden must be an integer between 1 and 512")
    if (isinstance(dropout, bool) or not isinstance(dropout, Real)
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("evidence_dropout must be finite in [0, 1)")
    return {"evidence_hidden": hidden, "evidence_dropout": float(dropout)}


class UtteranceEvidence(nn.Module):
    """Learn an ordered neighboring-turn residual for the nausea head only."""

    def __init__(self, encoder_hidden, global_hidden, hidden=128, dropout=0.1):
        super().__init__()
        validate_evidence_config(hidden, dropout)
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Sequential(nn.Linear(encoder_hidden * 3 + global_hidden, hidden),
                                            nn.GELU(), nn.Dropout(dropout))
            self.attention = nn.Linear(hidden, 1)
            self.output = nn.Linear(hidden, 2)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, turns, turn_mask, global_features):
        if turns.ndim != 3 or turn_mask.shape != turns.shape[:2] or turn_mask.dtype != torch.bool:
            raise ValueError("Turn representations require a matching boolean mask")
        # Compact valid turns in their original chronological order. This also
        # handles a rare turn composed entirely of structural special tokens.
        compact = turns.new_zeros(turns.shape)
        compact_mask = torch.zeros_like(turn_mask)
        for row in range(len(turns)):
            count = int(turn_mask[row].sum())
            compact[row, :count] = turns[row, turn_mask[row]]
            compact_mask[row, :count] = True
        previous = torch.cat((torch.zeros_like(compact[:, :1]), compact[:, :-1]), dim=1)
        following = torch.cat((compact[:, 1:], torch.zeros_like(compact[:, :1])), dim=1)
        global_context = global_features.unsqueeze(1).expand(-1, compact.shape[1], -1)
        evidence = self.projection(torch.cat((previous, compact, following, global_context), dim=-1))
        scores = self.attention(evidence).squeeze(-1).float()
        has_turn = compact_mask.any(1)
        safe_mask = compact_mask.clone()
        safe_mask[~has_turn, 0] = True
        scores = scores.masked_fill(~safe_mask, -torch.inf)
        attention = scores.softmax(1).to(evidence.dtype)
        summary = (attention.unsqueeze(-1) * evidence).sum(1)
        residual = self.output(summary)
        return residual * has_turn.unsqueeze(-1).to(residual.dtype)


class DialogueClassifier(nn.Module):
    """Train nine independent binary heads from all windows of each dialogue."""

    def __init__(self, encoder, label_attention=False, contrastive=False, feature_dim=256, *,
                 utterance_evidence=False, evidence_hidden=128, evidence_dropout=0.1):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(label_attention, bool) or not isinstance(contrastive, bool):
            raise ValueError("label_attention and contrastive must be boolean")
        if type(utterance_evidence) is not bool:
            raise ValueError("utterance_evidence must be boolean")
        validate_evidence_config(evidence_hidden, evidence_dropout)
        if utterance_evidence and (label_attention or contrastive):
            raise ValueError("Utterance evidence requires the single global CE pathway")
        self.encoder = encoder
        self.label_attention = label_attention
        self.contrastive = contrastive
        self.encoder_window_batch = 8
        hidden_size = int(encoder.config.hidden_size)
        self.projection = nn.Sequential(nn.Linear(hidden_size * 2, feature_dim), nn.GELU())
        self.dropout = nn.Dropout(0.1)
        if label_attention:
            self.token_queries = nn.Parameter(torch.empty(NUM_LABELS, hidden_size))
            self.window_queries = nn.Parameter(torch.empty(NUM_LABELS, feature_dim))
            nn.init.normal_(self.token_queries, std=0.02)
            nn.init.normal_(self.window_queries, std=0.02)
        else:
            self.register_parameter("token_queries", None)
            self.register_parameter("window_queries", None)
        self.class_weights = nn.Parameter(torch.empty(NUM_LABELS, feature_dim, 2))
        self.class_bias = nn.Parameter(torch.zeros(NUM_LABELS, 2))
        self.feature_weights = nn.Parameter(torch.empty(NUM_LABELS, feature_dim, CONTRASTIVE_DIM))
        self.feature_bias = nn.Parameter(torch.zeros(NUM_LABELS, CONTRASTIVE_DIM))
        nn.init.xavier_uniform_(self.class_weights)
        nn.init.xavier_uniform_(self.feature_weights)
        self.evidence = (UtteranceEvidence(hidden_size, feature_dim, evidence_hidden, evidence_dropout)
                         if utterance_evidence else None)

    def forward(self, batch):
        window_mask = batch.get("window_mask")
        ids, token_mask = batch.get("input_ids"), batch.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 3:
            raise ValueError("input_ids must have shape (batch, windows, tokens)")
        if not isinstance(token_mask, torch.Tensor) or token_mask.shape != ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not isinstance(window_mask, torch.Tensor) or window_mask.shape != ids.shape[:2] or window_mask.dtype != torch.bool:
            raise ValueError("window_mask must be a boolean (batch, windows) tensor")
        if ids.shape[0] < 1 or not bool(window_mask.any(dim=1).all()):
            raise ValueError("Every dialogue must contain at least one real window")
        if not bool(((token_mask == 0) | (token_mask == 1)).all()):
            raise ValueError("attention_mask must be binary")
        if not bool((token_mask.sum(-1)[window_mask] > 0).all()):
            raise ValueError("Real windows must contain attended tokens")
        chunk_size = self.encoder_window_batch
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("encoder_window_batch must be a positive integer")
        flat_mask = window_mask.flatten()
        turn_sum = turn_mass = None
        if self.evidence is not None:
            turn_ids, token_weights = batch.get("turn_ids"), batch.get("token_weights")
            if (not isinstance(turn_ids, torch.Tensor) or turn_ids.shape != ids.shape or turn_ids.dtype != torch.long
                    or not isinstance(token_weights, torch.Tensor) or token_weights.shape != ids.shape
                    or not token_weights.is_floating_point()):
                raise ValueError("Utterance evidence requires aligned turn_ids and token_weights")
            valid_turn_tokens = turn_ids >= 0
            if (bool((turn_ids < -1).any()) or not bool(torch.isfinite(token_weights).all())
                    or bool((token_weights < 0).any()) or bool((token_weights > 1).any())
                    or bool((token_weights[valid_turn_tokens] <= 0).any())
                    or bool((token_weights[~valid_turn_tokens] != 0).any())
                    or bool((valid_turn_tokens & ~token_mask.bool()).any())
                    or bool((valid_turn_tokens & ~window_mask.unsqueeze(-1)).any())):
                raise ValueError("Invalid turn IDs, padding masks, or overlap occurrence weights")
            turn_count = max(1, int(turn_ids.max()) + 1)
            if turn_count > int(token_mask.sum()):
                raise ValueError("Turn IDs exceed the available token positions")
            selected_turn_ids = turn_ids.flatten(0, 1)[flat_mask]
            selected_weights = token_weights.flatten(0, 1)[flat_mask]
            document_ids = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(window_mask).flatten()[flat_mask]
            turn_sum = torch.zeros((ids.shape[0] * turn_count, self.encoder.config.hidden_size),
                                   dtype=torch.float32, device=ids.device)
            turn_mass = torch.zeros(ids.shape[0] * turn_count, dtype=torch.float32, device=ids.device)
        inputs = {}
        for key in _MODEL_INPUTS:
            if key in batch:
                if batch[key].shape != ids.shape:
                    raise ValueError(f"{key} must match input_ids")
                inputs[key] = batch[key].flatten(0, 1)[flat_mask]
        representations = []
        for start in range(0, len(inputs["input_ids"]), chunk_size):
            chunk = {key: value[start:start + chunk_size] for key, value in inputs.items()}
            hidden = self.encoder(**chunk).last_hidden_state
            if not bool(torch.isfinite(hidden).all()):
                raise FloatingPointError("Encoder produced nonfinite hidden states")
            if self.evidence is not None:
                local_ids = selected_turn_ids[start:start + chunk_size]
                local_weights = selected_weights[start:start + chunk_size]
                present = local_ids >= 0
                rows = document_ids[start:start + chunk_size].unsqueeze(1) * turn_count + local_ids.clamp_min(0)
                turn_sum = turn_sum.index_add(0, rows[present], hidden.float()[present] * local_weights[present, None])
                turn_mass = turn_mass.index_add(0, rows[present], local_weights[present])
            if self.label_attention:
                token_scores = torch.einsum("nth,lh->ntl", hidden.float(), self.token_queries.float())
                token_scores = token_scores / math.sqrt(hidden.shape[-1])
                token_scores = token_scores.masked_fill(~chunk["attention_mask"].bool().unsqueeze(-1), -torch.inf)
                token_attention = torch.softmax(token_scores, dim=1).to(hidden.dtype)
                context = torch.einsum("ntl,nth->nlh", token_attention, hidden)
                cls = hidden[:, 0].unsqueeze(1).expand(-1, NUM_LABELS, -1)
                representations.append(self.projection(torch.cat((cls, context), dim=-1)))
            else:
                mask = chunk["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                representations.append(self.projection(torch.cat((hidden[:, 0], mean), dim=-1)))
        valid = self.dropout(torch.cat(representations, dim=0))
        if not bool(torch.isfinite(valid).all()):
            raise FloatingPointError("Window projection produced nonfinite features")
        padded_shape = (*window_mask.shape, *valid.shape[1:])
        windows = valid.new_zeros(padded_shape)
        windows = windows.flatten(0, 1).index_copy(
            0, flat_mask.nonzero(as_tuple=False).flatten(), valid,
        ).reshape(padded_shape)
        if self.label_attention:
            scores = (windows.float() * self.window_queries.float()[None, None]).sum(-1)
            scores = scores / math.sqrt(windows.shape[-1])
            scores = scores.masked_fill(~window_mask.unsqueeze(-1), -torch.inf)
            attention = torch.softmax(scores, dim=1).to(windows.dtype)
            pooled = (attention.unsqueeze(-1) * windows).sum(dim=1)
        else:
            pooled = windows.sum(dim=1) / window_mask.sum(dim=1, keepdim=True).to(windows.dtype)
            global_features = pooled
            pooled = pooled.unsqueeze(1).expand(-1, NUM_LABELS, -1)
        pooled = self.dropout(pooled)
        logits = torch.einsum("bld,ldc->blc", pooled, self.class_weights) + self.class_bias
        features = torch.einsum("bld,ldf->blf", pooled, self.feature_weights) + self.feature_bias
        features = F.normalize(features.float(), dim=-1)
        if self.evidence is not None:
            turns = (turn_sum / turn_mass.clamp_min(1).unsqueeze(-1)).reshape(ids.shape[0], turn_count, -1)
            turn_mask = turn_mass.reshape(ids.shape[0], turn_count) > 0
            # No new stochastic operation runs until the original global logits
            # and features are complete. All other symptom logits are untouched.
            residual = self.evidence(turns.to(global_features.dtype), turn_mask, global_features)
            nausea_mask = logits.new_zeros((1, NUM_LABELS, 1))
            nausea_mask[:, 6] = 1
            logits = logits + nausea_mask * residual.unsqueeze(1)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(features).all()):
            raise FloatingPointError("Classifier produced nonfinite logits or features")
        return logits, features


def predictions(logits):
    """The learned head's winning class; no fitted post-training cutoff."""
    if logits.ndim != 3 or logits.shape[1:] != (NUM_LABELS, 2):
        raise ValueError("logits must have shape (batch, 9, 2)")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Cannot predict from nonfinite logits")
    return logits.argmax(dim=-1)


def classification_loss(logits, targets, nausea_positive_weight=1.0, *,
                        positive_class_weights=None):
    """Weight positive CE items by label, retaining the batch-by-label mean.

    The scalar nausea weight remains supported for existing callers. Weights
    affect supervised training only; prediction still uses the raw argmax.
    """
    if isinstance(nausea_positive_weight, bool) or not isinstance(nausea_positive_weight, Real):
        raise ValueError("nausea_positive_weight must be a finite positive real number")
    try:
        positive_weight = float(nausea_positive_weight)
    except (OverflowError, ValueError) as error:
        raise ValueError("nausea_positive_weight must be a finite positive real number") from error
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("nausea_positive_weight must be a finite positive real number")
    if positive_class_weights is None:
        class_weights = [1.0] * NUM_LABELS
        class_weights[6] = positive_weight
    else:
        if positive_weight != 1.0:
            raise ValueError("Use positive_class_weights or a nondefault nausea_positive_weight, not both")
        if not isinstance(positive_class_weights, (list, tuple)) or len(positive_class_weights) != NUM_LABELS:
            raise ValueError("positive_class_weights must contain exactly nine finite positive real numbers")
        class_weights = []
        for value in positive_class_weights:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("positive_class_weights must contain exactly nine finite positive real numbers")
            try:
                weight = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError("positive_class_weights must contain exactly nine finite positive real numbers") from error
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("positive_class_weights must contain exactly nine finite positive real numbers")
            class_weights.append(weight)
    predictions(logits)
    targets = _binary_targets(targets, len(logits)).to(logits.device)
    if len(logits) < 1:
        raise ValueError("classification_loss requires a nonempty batch")
    if all(weight == 1.0 for weight in class_weights):
        loss = F.cross_entropy(logits.float().reshape(-1, 2), targets.reshape(-1))
    else:
        per_item = F.cross_entropy(
            logits.float().reshape(-1, 2), targets.reshape(-1), reduction="none",
        ).reshape(-1, NUM_LABELS)
        positive_weights = per_item.new_tensor(class_weights).unsqueeze(0)
        weights = torch.where(targets == 1, positive_weights, 1.0)
        loss = (per_item * weights).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Classification loss is nonfinite")
    return loss


def validate_loss_config(loss_mode="ce", gce_q=0.7, gce_mix=0.5, ce_warmup_epochs=1):
    """Validate a predefined loss recipe; it never changes inference decisions."""
    if loss_mode not in ("ce", "nausea_ce_gce"):
        raise ValueError("loss_mode must be ce or nausea_ce_gce")
    for name, value in (("gce_q", gce_q), ("gce_mix", gce_mix)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(name + " must be a finite real number")
    if not 0 < gce_q <= 1:
        raise ValueError("gce_q must be in (0, 1]")
    if not 0 <= gce_mix <= 0.5:
        raise ValueError("gce_mix must be in [0, 0.5] to retain at least half of CE")
    if type(ce_warmup_epochs) is not int or ce_warmup_epochs < 1:
        raise ValueError("ce_warmup_epochs must be a positive integer")
    return {"loss_mode": loss_mode, "gce_q": float(gce_q),
            "gce_mix": float(gce_mix), "ce_warmup_epochs": ce_warmup_epochs}


def training_loss(logits, targets, *, positive_class_weights=None, loss_mode="ce",
                  gce_q=0.7, gce_mix=0.5, ce_warmup_epochs=1, epoch=1):
    """Mix GCE into nausea only, after CE warmup, preserving every labeled row.

    For the recorded target probability p, GCE is (1 - p**q) / q.
    The nausea gradient is the weighted-CE gradient times
    (1 - mix) + mix * p**q, so even very hard targets retain at least
    half their original CE gradient. This also reduces the nausea task's
    relative gradient; an improvement would not prove annotation noise.
    """
    config = validate_loss_config(loss_mode, gce_q, gce_mix, ce_warmup_epochs)
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    base = classification_loss(logits, targets, positive_class_weights=positive_class_weights)
    if config["loss_mode"] == "ce" or epoch <= config["ce_warmup_epochs"] or config["gce_mix"] == 0:
        return base
    target = _binary_targets(targets, len(logits)).to(logits.device)[:, 6]
    log_probability = F.log_softmax(logits[:, 6].float(), dim=-1).gather(1, target[:, None]).squeeze(1)
    ce = -log_probability
    gce = -torch.expm1(config["gce_q"] * log_probability) / config["gce_q"]
    positive_weight = 1.0 if positive_class_weights is None else float(positive_class_weights[6])
    weight = torch.where(target == 1, positive_weight, 1.0)
    # The unchanged CE path includes all nine labels. Only replace part of
    # the nausea term; retain its original sample weights and denominator.
    correction = (weight * (gce - ce)).mean() * (config["gce_mix"] / NUM_LABELS)
    loss = base + correction
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Mixed nausea loss is nonfinite")
    return loss


def binary_supervised_contrastive_loss(features1, features2, targets, temperature=0.1):
    """Binary supervised contrastive loss per symptom, with balanced anchors.

    The two views come from independent dropout forwards of the same dialogues.
    Same-class examples are positives, self matches are excluded, and each
    present binary class receives equal anchor weight within each symptom.
    Thus the many symptom-negative examples cannot dominate the anchor average.
    """
    if features1.ndim != 3 or features1.shape[1] != NUM_LABELS or features1.shape != features2.shape:
        raise ValueError("Both feature views must have matching (batch, 9, feature_dim) shapes")
    if features1.shape[0] < 1 or features1.shape[2] < 1:
        raise ValueError("Contrastive loss requires nonempty feature views")
    if not bool(torch.isfinite(features1).all()) or not bool(torch.isfinite(features2).all()):
        raise FloatingPointError("Contrastive features must be finite")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    labels = _binary_targets(targets, len(features1)).to(features1.device)
    features = F.normalize(torch.cat((features1.float(), features2.float()), dim=0), dim=-1)
    labels = torch.cat((labels, labels), dim=0)
    self_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    losses = []
    for label in range(NUM_LABELS):
        scores = features[:, label] @ features[:, label].T / float(temperature)
        denominator = torch.logsumexp(scores.masked_fill(self_mask, -torch.inf), dim=1, keepdim=True)
        log_probability = scores - denominator
        positive = (labels[:, label, None] == labels[None, :, label]) & ~self_mask
        anchor_loss = -(log_probability * positive).sum(1) / positive.sum(1)
        class_losses = [anchor_loss[labels[:, label] == value].mean()
                        for value in (0, 1) if bool((labels[:, label] == value).any())]
        losses.append(torch.stack(class_losses).mean())
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Supervised contrastive loss is nonfinite")
    return loss
