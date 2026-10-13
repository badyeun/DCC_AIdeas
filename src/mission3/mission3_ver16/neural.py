"""Text-only dialogue classifiers with ordinary binary class decisions.

Each symptom has two class logits. Predictions use the winning class directly;
there is no probability cutoff search, calibration, or post-hoc label weighting.
Public pretrained assets are prepared separately from local checkpoint weights.
"""
from __future__ import annotations

import math
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


class EncodeDataset(Dataset):
    """Retain all fast-tokenizer overflow windows, including the final tokens."""

    def __init__(self, texts, targets, tokenizer, max_length=512, overlap=64):
        texts = list(texts)
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
        self.targets = None if targets is None else _binary_targets(targets, len(texts)).clone()
        self.documents = [[] for _ in texts]
        for start in range(0, len(texts), 128):
            dialogues = [re.sub(r"\r\n?|\n", f" {separator} ", text)
                         for text in texts[start:start + 128]]
            encoded = tokenizer(
                dialogues, truncation=True, max_length=max_length, stride=overlap,
                return_overflowing_tokens=True, padding=False,
                return_attention_mask=True,
            )
            if "overflow_to_sample_mapping" not in encoded:
                raise ValueError("A fast tokenizer must return overflow_to_sample_mapping")
            if "input_ids" not in encoded or "attention_mask" not in encoded:
                raise ValueError("The tokenizer must return input_ids and attention_mask")
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
                self.documents[start + source].append(window)
        if any(not document for document in self.documents):
            raise ValueError("Tokenizer returned a dialogue without any windows")

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
        batch["window_mask"] = torch.zeros((batch_size, window_count), dtype=torch.bool)
        for row, item in enumerate(examples):
            for column, window in enumerate(item["windows"]):
                length = len(window["input_ids"])
                for key in keys:
                    batch[key][row, column, :length] = torch.tensor(window[key], dtype=torch.long)
                batch["window_mask"][row, column] = True
        if "targets" in examples[0]:
            batch["targets"] = torch.stack([item["targets"] for item in examples])
        return batch


def validate_correlation_config(hidden=32, depth=1, dropout=0.1):
    if type(hidden) is not int or not 1 <= hidden <= 256:
        raise ValueError("correlation_hidden must be an integer between 1 and 256")
    if type(depth) is not int or not 1 <= depth <= 3:
        raise ValueError("correlation_depth must be an integer between 1 and 3")
    if (isinstance(dropout, bool) or not isinstance(dropout, Real)
            or not math.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError("correlation_dropout must be finite in [0, 1)")
    return {"correlation_hidden": hidden, "correlation_depth": depth,
            "correlation_dropout": float(dropout)}


class LabelCorrelation(nn.Module):
    """Jointly learned CorNet-inspired residuals from text-derived evidence only.

    Each block maps nine soft evidences through a small ELU hidden layer. Its
    zero-initialized output starts at the independent-head classifier. This is
    an adaptation for nine binary categorical heads, not the original XML model.
    """

    def __init__(self, hidden=32, depth=1, dropout=0.1):
        super().__init__()
        validate_correlation_config(hidden, depth, dropout)
        # Adding the module must not alter common-head initialization or the RNG
        # state seen by the subsequent shuffle/dropout in the matched baseline.
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(NUM_LABELS, hidden), nn.ELU(), nn.Dropout(dropout),
                              nn.Linear(hidden, NUM_LABELS)) for _ in range(depth)
            ])
            for block in self.blocks:
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)

    def forward(self, evidence):
        if evidence.ndim != 2 or evidence.shape[1] != NUM_LABELS:
            raise ValueError("Correlation evidence must have shape (batch, 9)")
        if not bool(torch.isfinite(evidence).all()):
            raise FloatingPointError("Correlation evidence must be finite")
        value = evidence
        for block in self.blocks:
            value = value + block(torch.sigmoid(value))
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Correlation module produced nonfinite evidence")
        return value


class DialogueClassifier(nn.Module):
    """Train nine binary heads with an optional learned correlation residual."""

    def __init__(self, encoder, label_attention=False, contrastive=False, feature_dim=256, *,
                 label_correlation=False, correlation_hidden=32, correlation_depth=1,
                 correlation_dropout=0.1):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(label_attention, bool) or not isinstance(contrastive, bool):
            raise ValueError("label_attention and contrastive must be boolean")
        if type(label_correlation) is not bool:
            raise ValueError("label_correlation must be boolean")
        validate_correlation_config(correlation_hidden, correlation_depth, correlation_dropout)
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
        self.correlation = (LabelCorrelation(correlation_hidden, correlation_depth, correlation_dropout)
                            if label_correlation else None)

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
            pooled = pooled.unsqueeze(1).expand(-1, NUM_LABELS, -1)
        pooled = self.dropout(pooled)
        logits = torch.einsum("bld,ldc->blc", pooled, self.class_weights) + self.class_bias
        if self.correlation is not None:
            evidence = logits[..., 1] - logits[..., 0]
            residual = self.correlation(evidence) - evidence
            logits = logits + torch.stack((-residual / 2, residual / 2), dim=-1)
        features = torch.einsum("bld,ldf->blf", pooled, self.feature_weights) + self.feature_bias
        features = F.normalize(features.float(), dim=-1)
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
