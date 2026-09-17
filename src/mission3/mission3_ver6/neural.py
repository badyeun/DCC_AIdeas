"""Cached head ablations, direct label attention, and one bounded fine-tune.

The original fold's fit/stop identities are supplied by the pipeline. Six small
heads share ONE frozen encoder feature pass; only the stop-selected winner is
allowed an encoder fine-tune. Training calibration and official Validation
labels never enter this module.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

if __package__:
    from .losses import MultilabelLoss, bernoulli_symmetric_kl
    from .selection import StopSelector, better
    from .ver4_runtime.neural import NUM_LABELS, DialogueCollator, _device, _loader, _optimizer, _targets, tokenize_dialogues
else:
    from losses import MultilabelLoss, bernoulli_symmetric_kl
    from selection import StopSelector, better
    from ver4_runtime.neural import NUM_LABELS, DialogueCollator, _device, _loader, _optimizer, _targets, tokenize_dialogues


MODEL_FORMAT = "mission3_ver6_neural"
LAST_FORMAT = "mission3_ver6_last"
HEAD_FORMAT = "mission3_ver6_head_trial"
PROFILES = {
    "A": {"name": "legacy + ASL", "head_type": "legacy", "loss": "asl", "rdrop": 0.0, "dependency": False},
    "B": {"name": "legacy + distribution-balanced loss", "head_type": "legacy", "loss": "db", "rdrop": 0.0, "dependency": False},
    "C": {"name": "legacy + ASL + 0.2 soft macro-F1", "head_type": "legacy", "loss": "asl_soft_f1", "rdrop": 0.0, "dependency": False},
    "D": {"name": "direct LAAT + weighted BCE", "head_type": "laat", "loss": "bce", "rdrop": 0.0, "dependency": False},
    "E": {"name": "direct LAAT + weighted BCE + head R-Drop", "head_type": "laat", "loss": "bce", "rdrop": 0.5, "dependency": False},
    "F": {"name": "direct LAAT + weighted BCE + head R-Drop + dependency", "head_type": "laat", "loss": "bce", "rdrop": 0.5, "dependency": True},
}
DEFAULTS = {
    "head_epochs": 4, "head_batch_size": 32, "head_learning_rate": 3e-4, "head_patience": 2,
    "profiles": ["A", "B", "C", "D", "E", "F"], "stop_crossfit_splits": 3,
    "finetune_epochs": 1, "finetune_top_layers": 2, "learning_rate": 2e-6,
    "finetune_head_learning_rate": 5e-5, "finetune_rdrop": False,
    "attention_dim": 256, "dropout": 0.2, "batch_size": 2, "window_batch_size": 2,
    "gradient_accumulation_steps": 16, "inference_batch_size": 8,
    "gradient_checkpointing": True, "device": "cuda", "log_every": 100,
    "layerwise_lr_decay": 0.95, "weight_decay": 0.01, "warmup_ratio": 0.1,
}


def native_bf16_supported():
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def _amp(device):
    return torch.amp.autocast(device_type=device.type,
                              dtype=torch.bfloat16 if device.type == "cuda" and native_bf16_supported() else torch.float16,
                              enabled=device.type == "cuda")


def execution_identity(device):
    """Bind epoch resumes to one numerical execution environment.

    Reusing a completed pipeline for inference on another accelerator is valid;
    mixing cached best scores or optimizer/RNG state across GPU precisions is
    not an exact resume. The pipeline handles completed-output reuse upstream.
    """
    device = torch.device(device)
    identity = {"device_type": device.type, "torch": str(torch.__version__),
                "transformers": importlib.metadata.version("transformers"),
                "cuda_version": torch.version.cuda,
                "amp_dtype": "float32" if device.type != "cuda" else "bfloat16" if native_bf16_supported() else "float16",
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if device.type == "cuda":
        identity.update(gpu=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)),
                        cuda_device_count=torch.cuda.device_count())
    else:
        identity["cpu_threads"] = torch.get_num_threads()
    return identity


class LegacyHead(nn.Module):
    """Exactly the ver4 CLS+mean projection and label-specific window pooling."""
    def __init__(self, hidden_size, num_labels=NUM_LABELS, dropout=0.2, **unused):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.attention_queries = nn.Parameter(torch.empty(num_labels, hidden_size))
        self.label_weights = nn.Parameter(torch.empty(num_labels, hidden_size))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.normal_(self.attention_queries, std=0.02)
        nn.init.normal_(self.label_weights, std=0.02)

    def forward(self, documents):
        summaries = [torch.stack([window["legacy_summary"] if "legacy_summary" in window else
                                  torch.cat([window["hidden"][0], window["hidden"].mean(0)])
                                  for window in document]) for document in documents]
        representations = self.dropout(self.projection(pad_sequence(summaries, batch_first=True)))
        lengths = torch.tensor([len(x) for x in summaries], device=representations.device)
        mask = torch.arange(representations.shape[1], device=lengths.device)[None, :] < lengths[:, None]
        scores = (representations @ self.attention_queries.T / math.sqrt(representations.shape[-1])).float()
        attention = scores.masked_fill(~mask[:, :, None], torch.finfo(scores.dtype).min).softmax(1)
        pooled = torch.einsum("bwl,bwh->blh", attention.to(representations.dtype), representations)
        return (self.dropout(pooled) * self.label_weights).sum(-1) + self.label_bias


class LabelAttentionHead(nn.Module):
    """PLM-ICD/LAAT attention over semantic tokens from every window.

    A = softmax(U tanh(W H)); R = A H; each final label logit directly
    classifies its own original-hidden-size representation. No zero residual
    bypass exists. Empty semantic input has a finite bias-only prediction.
    """
    def __init__(self, hidden_size, num_labels=NUM_LABELS, dropout=0.2,
                 attention_dim=256, dependency=False, fit_targets=None):
        super().__init__()
        self.token_projection = nn.Linear(hidden_size, attention_dim)
        self.token_queries = nn.Parameter(torch.empty(num_labels, attention_dim))
        self.label_weights = nn.Parameter(torch.empty(num_labels, hidden_size))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.token_queries, std=0.02)
        nn.init.normal_(self.label_weights, std=0.02)
        self.dependency = bool(dependency)
        targets = torch.as_tensor(fit_targets, dtype=torch.float32) if fit_targets is not None else None
        if targets is not None:
            prior = targets.mean(0).clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                self.label_bias.copy_(prior.logit())
        else:
            prior = torch.full((num_labels,), 0.5)
        if self.dependency:
            graph = torch.zeros(num_labels, num_labels)
            if targets is not None:
                # graph[source,target] uses FIT-only excess conditional rate.
                conditional = (targets.T @ targets) / targets.sum(0).clamp_min(1)[:, None]
                graph = conditional - prior[None, :]
                graph.fill_diagonal_(0)
                graph /= graph.abs().sum(0, keepdim=True).clamp_min(1)
            self.register_buffer("dependency_graph", graph)
            self.register_buffer("fit_prior", prior)
            self.dependency_scale = nn.Parameter(torch.full((num_labels,), 0.1))

    def forward(self, documents):
        rows, lengths = [], []
        for windows in documents:
            tokens = torch.cat([window["hidden"][window["content_mask"]] for window in windows], dim=0)
            lengths.append(len(tokens))
            # A dummy zero row avoids softmax over a zero-length axis.
            rows.append(tokens if len(tokens) else windows[0]["hidden"].new_zeros(1, windows[0]["hidden"].shape[-1]))
        hidden = pad_sequence(rows, batch_first=True)
        lengths = torch.tensor(lengths, device=hidden.device)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
        projected = torch.tanh(self.token_projection(self.dropout(hidden)))
        scores = (projected @ self.token_queries.T).float()
        attention = scores.masked_fill(~mask[:, :, None], torch.finfo(scores.dtype).min).softmax(1)
        attention = attention * mask[:, :, None]
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("btl,bth->blh", attention.to(hidden.dtype), hidden)
        logits = (self.dropout(pooled) * self.label_weights).sum(-1) + self.label_bias
        if self.dependency:
            evidence = (logits.sigmoid() - self.fit_prior) @ self.dependency_graph
            logits = logits + self.dependency_scale.tanh() * evidence
        return logits


class DialogueClassifier(nn.Module):
    def __init__(self, encoder, head_type="laat", num_labels=NUM_LABELS, dropout=0.2,
                 window_batch_size=2, attention_dim=256, special_token_ids=(), dependency=False):
        super().__init__()
        self.encoder, self.window_batch_size = encoder, int(window_batch_size)
        self.special_token_ids = tuple(int(x) for x in special_token_ids)
        arguments = dict(hidden_size=encoder.config.hidden_size, num_labels=num_labels, dropout=dropout,
                         attention_dim=attention_dim, dependency=dependency)
        self.head = LegacyHead(**arguments) if head_type == "legacy" else LabelAttentionHead(**arguments)

    def forward(self, doc_indices, **inputs):
        documents = [[] for _ in range(int(doc_indices.max().item()) + 1)]
        for start in range(0, len(doc_indices), self.window_batch_size):
            subset = {key: value[start:start + self.window_batch_size] for key, value in inputs.items()}
            hidden = self.encoder(**subset).last_hidden_state
            for offset in range(len(hidden)):
                valid = subset["attention_mask"][offset].bool()
                states = hidden[offset][valid]
                ids = subset["input_ids"][offset][valid]
                content = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
                for token_id in self.special_token_ids:
                    content &= ids.ne(token_id)
                documents[int(doc_indices[start + offset])].append({"hidden": states, "content_mask": content})
        return self.head(documents)


def _read_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)


def _encoder(configuration):
    values = dict(configuration)
    return AutoModel.from_config(AutoConfig.for_model(values.pop("model_type"), **values))


def _special_ids(tokenizer):
    return sorted({int(value) for name in ("pad_token_id", "cls_token_id", "sep_token_id", "bos_token_id", "eos_token_id")
                   if (value := getattr(tokenizer, name, None)) is not None})


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
    for name in ("head_epochs", "head_batch_size", "head_patience", "stop_crossfit_splits", "finetune_epochs",
                 "batch_size", "window_batch_size", "gradient_accumulation_steps", "inference_batch_size", "attention_dim"):
        minimum = 0 if name == "finetune_epochs" else (2 if name == "stop_crossfit_splits" else 1)
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    count = settings["finetune_top_layers"]
    if isinstance(count, bool) or not isinstance(count, int) or count < -1:
        raise ValueError("finetune_top_layers must be an integer >= -1")
    for name in ("head_learning_rate", "learning_rate", "finetune_head_learning_rate"):
        if not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 <= float(settings["dropout"]) < 1 or not 0 <= float(settings["warmup_ratio"]) < 1:
        raise ValueError("dropout/warmup_ratio must lie in [0, 1)")
    if not math.isfinite(float(settings["weight_decay"])) or settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    profiles = settings["profiles"]
    if not isinstance(profiles, (list, tuple)) or not profiles or len(set(profiles)) != len(profiles) or any(x not in PROFILES for x in profiles):
        raise ValueError("profiles must be a nonempty unique sequence chosen from A through F")
    settings["profiles"] = list(profiles)
    return settings


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(), "position": int(numpy_state[2]),
                      "has_gauss": int(numpy_state[3]), "cached_gaussian": float(numpy_state[4])},
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values["name"], np.asarray(values["keys"], dtype=np.uint32), values["position"],
                         values["has_gauss"], values["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact epoch resume is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _check_identity(state, signature, fingerprint, execution=None):
    if state.get("run_signature") != signature:
        raise ValueError("Checkpoint run signature mismatch; use a new output directory")
    if execution is not None and state.get("execution") != execution:
        raise ValueError("실행 환경/GPU 정밀도가 이전 epoch와 다릅니다. 같은 GPU 종류와 패키지 환경으로 재개하거나 "
                         "새 OUTPUT_DIR을 사용하세요. Exact epoch resume requires the same execution environment.")
    if state.get("configuration_fingerprint") != fingerprint:
        raise ValueError("Checkpoint config fingerprint mismatch; use a new output directory")


def _cached_batch(store, indices, device, legacy=False):
    documents = []
    for index in indices:
        # The legacy head only consumes CLS+mean, so a summary-aware store can
        # avoid rereading every token shard on every A/B/C epoch.
        windows = store.get_legacy(int(index)) if legacy and hasattr(store, "get_legacy") else store[int(index)]
        if not windows:
            raise ValueError("A feature-cache document must contain windows")
        documents.append([{"legacy_summary": window["legacy_summary"].to(device=device, dtype=torch.float32)}
                          if "legacy_summary" in window else
                          {"hidden": window["hidden"].to(device=device, dtype=torch.float32),
                           "content_mask": window["content_mask"].to(device=device, dtype=torch.bool)}
                          for window in windows])
    return documents


@torch.inference_mode()
def _predict_head(head, store, indices, batch_size, device):
    head.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        documents = _cached_batch(store, indices[start:start + batch_size], device, legacy=isinstance(head, LegacyHead))
        with _amp(device):
            predictions.append(head(documents).float().sigmoid().cpu().numpy())
    return np.concatenate(predictions)


def _head_from_base(base, profile, settings, fit_targets):
    hidden = int(base["encoder_config"]["hidden_size"])
    common = {"hidden_size": hidden, "num_labels": NUM_LABELS, "dropout": settings["dropout"],
              "attention_dim": settings["attention_dim"], "dependency": profile["dependency"]}
    if profile["head_type"] == "legacy":
        head = LegacyHead(**common)
        head.load_state_dict({key: value for key, value in base["model_state"].items() if not key.startswith("encoder.")}, strict=True)
    else:
        head = LabelAttentionHead(**common, fit_targets=fit_targets)
    return head


def _loss(head_or_model, inputs, labels, criterion, profile, head_phase=True, finetune_rdrop=False):
    def forward():
        return head_or_model(inputs) if head_phase else head_or_model(**inputs)
    first = forward()
    loss = criterion(first, labels, head_phase=head_phase)
    rdrop = profile["rdrop"] if head_phase or finetune_rdrop else 0.0
    if rdrop:
        second = forward()
        loss = 0.5 * (loss + criterion(second, labels, head_phase=head_phase)) + rdrop * bernoulli_symmetric_kl(first, second)
    return loss


def _run_head_trial(profile_id, base, store, fit_indices, stop_indices, targets,
                    selector, settings, directory, seed, signature, fingerprint, device, execution):
    profile = PROFILES[profile_id]
    path = directory / f"profile_{profile_id}.ckpt"
    state = _read_checkpoint(path) if path.exists() else None
    if state is not None:
        if state.get("format") != HEAD_FORMAT or state.get("profile") != profile_id:
            raise ValueError("Unsupported head trial checkpoint")
        _check_identity(state, signature, fingerprint, execution)
        if state["completed"]:
            return state
    _seed(seed)
    head = _head_from_base(base, profile, settings, targets).to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["head_learning_rate"], weight_decay=settings["weight_decay"])
    criterion = MultilabelLoss(profile["loss"], targets).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not native_bf16_supported())
    completed, history, best_score, best_epoch, stale = 0, [], None, 0, 0
    if state is not None:
        head.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state["rng"], generator)
        completed, history = state["completed_epochs"], list(state["history"])
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale_epochs"]
        best_state = state["best_state"]
    else:
        started = time.monotonic()
        best_score = selector.score(_predict_head(head, store, stop_indices, settings["head_batch_size"], device))
        best_state = _cpu_state(head)
        history.append({"epoch": 0, "phase": "head_baseline", "train_loss": None, **best_score,
                        "minutes": (time.monotonic() - started) / 60})

    def commit(epoch, done):
        result = {"format": HEAD_FORMAT, "format_version": 1, "profile": profile_id,
                  "execution": execution,
                  "run_signature": signature, "configuration_fingerprint": fingerprint,
                  "completed_epochs": epoch, "completed": done, "best_epoch": best_epoch,
                  "best_score": best_score, "best_state": best_state, "stale_epochs": stale,
                  "history": history, "model_state": _cpu_state(head), "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict(), "rng": _rng_state(generator)}
        _atomic_save(result, path)
        return result

    if state is None:
        state = commit(0, False)
    batch_size = settings["head_batch_size"]
    for epoch in range(completed + 1, settings["head_epochs"] + 1):
        started = time.monotonic()
        head.train()
        order = torch.randperm(len(fit_indices), generator=generator).numpy()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            documents = _cached_batch(store, np.asarray(fit_indices)[rows], device, legacy=profile["head_type"] == "legacy")
            labels = torch.as_tensor(targets[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device):
                loss = _loss(head, documents, labels, criterion, profile)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite head loss profile={profile_id} epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(rows)
            batch_number = start // batch_size + 1
            if settings["log_every"] > 0 and batch_number % settings["log_every"] == 0:
                print(f"V6-{profile_id} head epoch={epoch} batch={batch_number}/{math.ceil(len(order) / batch_size)} "
                      f"loss={loss_sum / (start + len(rows)):.5f} minutes={(time.monotonic() - started) / 60:.1f}", flush=True)
        score = selector.score(_predict_head(head, store, stop_indices, batch_size, device))
        history.append({"epoch": epoch, "phase": "head", "train_loss": loss_sum / len(targets), **score,
                        "minutes": (time.monotonic() - started) / 60})
        if better(score, best_score):
            best_score, best_epoch, best_state, stale = score, epoch, _cpu_state(head), 0
        else:
            stale += 1
        state = commit(epoch, epoch == settings["head_epochs"] or stale >= settings["head_patience"])
        print(f"V6-{profile_id} head epoch={epoch}/{settings['head_epochs']} stop_macro_F1={score['stop_macro_f1']:.6f} "
              f"stop_AP={score['stop_macro_ap']:.6f} best_epoch={best_epoch} minutes={history[-1]['minutes']:.1f}", flush=True)
        if state["completed"]:
            break
    return state


def _finetune_blocks(model, settings):
    count = settings["finetune_top_layers"]
    if count <= 0:
        return []
    blocks = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    if not isinstance(blocks, nn.ModuleList) or not len(blocks):
        raise ValueError("Partial fine-tuning requires BERT/RoBERTa encoder.layer")
    return list(blocks[-min(count, len(blocks)):])


def _configure_finetune(model, settings):
    model.encoder.requires_grad_(False)
    if settings["finetune_top_layers"] == -1:
        model.encoder.requires_grad_(True)
    else:
        for block in _finetune_blocks(model, settings):
            block.requires_grad_(True)
    if any(p.requires_grad for p in model.encoder.parameters()) and settings["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.encoder.gradient_checkpointing_disable()
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False


def _train_mode(model, settings):
    model.train()
    model.encoder.eval()
    if settings["finetune_top_layers"] == -1:
        model.encoder.train()
    else:
        for block in _finetune_blocks(model, settings):
            block.train()


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


def _from_saved(checkpoint, config):
    if checkpoint.get("format") not in (MODEL_FORMAT, LAST_FORMAT) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported ver6 neural checkpoint")
    architecture = dict(checkpoint["architecture"])
    architecture["window_batch_size"] = int(config.get("window_batch_size", architecture["window_batch_size"]))
    model = DialogueClassifier(_encoder(checkpoint["encoder_config"]), **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def predict_checkpoint(checkpoint_path, documents, tokenizer, config):
    checkpoint = _read_checkpoint(checkpoint_path)
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("Inference requires the selected best ver6 checkpoint")
    device = _device(config)
    model = _from_saved(checkpoint, config).to(device)
    del checkpoint
    probabilities = _predict(model, _loader(documents, None, tokenizer, config.get("inference_batch_size", 8)), device)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Inference produced nonfinite probabilities")
    return probabilities.astype(np.float32, copy=False)


def _best_payload(last):
    keep = ("format_version", "encoder_config", "architecture", "model_name", "init_epoch", "init_seed",
            "model_state", "seed", "run_signature", "configuration_fingerprint", "settings", "selected_profile", "execution")
    return {**{key: last[key] for key in keep}, "format": MODEL_FORMAT, "epoch": last["best_epoch"],
            "stop_ap": last["best_score"]["stop_macro_ap"], "stop_macro_f1": last["best_score"]["stop_macro_f1"]}


def _result(state, resumed, last_path):
    return {"best_epoch": state["best_epoch"], "best_stop_ap": state["best_score"]["stop_macro_ap"],
            "best_stop_macro_f1": state["best_score"]["stop_macro_f1"], "history": state["history"],
            "completed_epochs": state["completed_epochs"], "resumed": resumed, "last_checkpoint": str(last_path),
            "selected_profile": state["selected_profile"], "head_trials": state["head_trials"],
            "execution": state["execution"],
            "selection_scope": "Original Training fold stop: grouped threshold cross-fit macro-F1; AP secondary. "
                               "This is reused model/epoch selection, not model-level OOF or official Validation.",
            "finetune_rdrop": state["settings"]["finetune_rdrop"]}


def train_model(train_documents, train_targets, stop_documents, stop_targets, tokenizer, config,
                checkpoint_path, seed, init_checkpoint_path, run_signature, feature_store=None,
                fit_indices=None, stop_indices=None, stop_groups=None):
    train_targets = _targets(train_targets, train_documents, "fit")
    stop_targets = _targets(stop_targets, stop_documents, "stop")
    if feature_store is None or fit_indices is None or stop_indices is None or stop_groups is None:
        raise ValueError("ver6 requires a shared feature_store, fit_indices, stop_indices, and stop_groups")
    fit_indices, stop_indices = np.asarray(fit_indices), np.asarray(stop_indices)
    if len(fit_indices) != len(train_targets) or len(stop_indices) != len(stop_targets):
        raise ValueError("Feature index order must match supplied fit/stop targets")
    combined = np.concatenate([fit_indices, stop_indices])
    if combined.ndim != 1 or not np.issubdtype(combined.dtype, np.integer) or len(np.unique(combined)) != len(combined):
        raise ValueError("Fit and stop feature indices must be unique and disjoint")
    if combined.min() < 0 or combined.max() >= len(feature_store):
        raise ValueError("Feature indices are outside the store")
    if not isinstance(run_signature, str) or not run_signature:
        raise ValueError("run_signature must be a nonempty string")
    settings = _settings(config)
    device = _device(settings)
    execution = execution_identity(device)
    fingerprint_data = {"settings": settings, "seed": int(seed), "initial_checkpoint": str(Path(init_checkpoint_path).resolve()),
                        "fit_indices": fit_indices.tolist(), "stop_indices": stop_indices.tolist(), "stop_groups": list(map(str, stop_groups)),
                        "execution": execution}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    checkpoint_path = Path(checkpoint_path)
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.ckpt")
    last = _read_checkpoint(last_path) if last_path.exists() else None
    best_saved = _read_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    for saved in (last, best_saved):
        if saved is not None:
            _check_identity(saved, run_signature, fingerprint, execution)
    if last is not None:
        if last.get("format") != LAST_FORMAT:
            raise ValueError("Unsupported ver6 resume checkpoint")
        best_matches = (best_saved is not None and best_saved.get("epoch") == last["best_epoch"] and
                        best_saved.get("stop_macro_f1") == last["best_score"]["stop_macro_f1"] and
                        best_saved.get("stop_ap") == last["best_score"]["stop_macro_ap"])
        if not best_matches:
            if last["best_epoch"] != last["completed_epochs"]:
                raise ValueError("Best checkpoint is missing/inconsistent with completed epoch state")
            _atomic_save(_best_payload(last), checkpoint_path)
        if last["completed"]:
            return _result(last, True, last_path)
    elif best_saved is not None:
        raise ValueError("Best checkpoint exists without resume state")
    del best_saved
    selector = StopSelector(stop_targets, stop_groups, seed=seed, n_splits=settings["stop_crossfit_splits"])
    base = _read_checkpoint(init_checkpoint_path)
    if base.get("format_version") != 1 or "model_state" not in base or base.get("format") in (MODEL_FORMAT, LAST_FORMAT):
        raise ValueError("Warm start requires the original ver4 fold checkpoint")
    resumed = last is not None or any((checkpoint_path.parent / "head_trials" / f"profile_{profile}.ckpt").exists() for profile in settings["profiles"])
    trials, winner = [], None
    for profile_id in settings["profiles"]:
        state = _run_head_trial(profile_id, base, feature_store, fit_indices, stop_indices, train_targets, selector,
                                settings, checkpoint_path.parent / "head_trials", seed, run_signature, fingerprint, device, execution)
        trials.append({"profile": profile_id, **PROFILES[profile_id], "best_epoch": state["best_epoch"],
                       "best_stop_macro_f1": state["best_score"]["stop_macro_f1"], "best_stop_ap": state["best_score"]["stop_macro_ap"],
                       "completed_epochs": state["completed_epochs"], "history": state["history"]})
        if winner is None or better(state["best_score"], winner["best_score"]):
            winner = state
    profile_id, profile = winner["profile"], PROFILES[winner["profile"]]
    print(f"선택: V6-{profile_id}, stop macro-F1={winner['best_score']['stop_macro_f1']:.6f}. 이 후보만 encoder를 미세조정합니다.", flush=True)
    _seed(seed)
    if last is None:
        architecture = {"head_type": profile["head_type"], "num_labels": NUM_LABELS, "dropout": settings["dropout"],
                        "window_batch_size": settings["window_batch_size"], "attention_dim": settings["attention_dim"],
                        "special_token_ids": _special_ids(tokenizer), "dependency": profile["dependency"]}
        model = DialogueClassifier(_encoder(base["encoder_config"]), **architecture)
        model.encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in base["model_state"].items() if key.startswith("encoder.")}, strict=True)
        model.head.load_state_dict(winner["best_state"], strict=True)
        metadata = {"encoder_config": base["encoder_config"], "architecture": architecture,
                    "model_name": base.get("model_name"), "init_epoch": base.get("epoch"), "init_seed": base.get("seed")}
    else:
        if last["selected_profile"] != profile_id:
            raise ValueError("Selected head changed since partial fine-tune")
        model = _from_saved(last, settings)
        metadata = {key: last[key] for key in ("encoder_config", "architecture", "model_name", "init_epoch", "init_seed")}
    del base, state, winner
    model.to(device)
    _configure_finetune(model, settings)
    loader = _loader(train_documents, train_targets, tokenizer, settings["batch_size"], shuffle=True, seed=seed)
    stop_loader = _loader(stop_documents, None, tokenizer, settings["inference_batch_size"])
    criterion = MultilabelLoss(profile["loss"], train_targets).to(device)
    optimizer = _optimizer(model, {**settings, "head_learning_rate": settings["finetune_head_learning_rate"]})
    total_steps = math.ceil(len(loader) / settings["gradient_accumulation_steps"]) * max(1, settings["finetune_epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not native_bf16_supported())
    if last is None:
        started = time.monotonic()
        # Live encoder evaluation is required: cached fp16 feature rounding can
        # alter tight threshold ties, so compare live vs live for fine-tuning.
        best_score = selector.score(_predict(model, stop_loader, device))
        history = [{"epoch": 0, "phase": "selected_head", "train_loss": None, **best_score,
                    "minutes": (time.monotonic() - started) / 60}]
        completed, best_epoch = 0, 0
    else:
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        _restore_rng(last["rng"], loader.generator)
        best_score, best_epoch = last["best_score"], last["best_epoch"]
        history, completed = list(last["history"]), last["completed_epochs"]
    del last

    def commit(epoch, done, improved):
        state = {"format": LAST_FORMAT, "format_version": 1, **metadata, "model_state": _cpu_state(model),
                 "execution": execution,
                 "seed": int(seed), "run_signature": run_signature, "configuration_fingerprint": fingerprint,
                 "settings": settings, "selected_profile": profile_id, "head_trials": trials,
                 "completed_epochs": epoch, "completed": done, "best_epoch": best_epoch,
                 "best_score": best_score, "history": history, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(loader.generator)}
        _atomic_save(state, last_path)
        if improved:
            _atomic_save(_best_payload(state), checkpoint_path)
        return state

    if not last_path.exists():
        saved = commit(0, settings["finetune_epochs"] == 0, True)
    else:
        saved = None
    accumulation = settings["gradient_accumulation_steps"]
    for epoch in range(completed + 1, settings["finetune_epochs"] + 1):
        started = time.monotonic()
        _train_mode(model, settings)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count = 0.0, 0
        for step, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = (step // accumulation) * accumulation
            group_documents = min(accumulation * loader.batch_size, len(train_documents) - group_start * loader.batch_size)
            with _amp(device):
                loss = _loss(model, batch, labels, criterion, profile, head_phase=False, finetune_rdrop=settings["finetune_rdrop"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite fine-tune loss at epoch {epoch}")
            scaler.scale(loss * len(labels) / group_documents).backward()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if settings["log_every"] > 0 and (step + 1) % settings["log_every"] == 0:
                print(f"V6-{profile_id} finetune epoch={epoch} batch={step + 1}/{len(loader)} "
                      f"loss={loss_sum / count:.5f} minutes={(time.monotonic() - started) / 60:.1f}", flush=True)
        score = selector.score(_predict(model, stop_loader, device))
        history.append({"epoch": epoch, "phase": "finetune", "train_loss": loss_sum / count, **score,
                        "minutes": (time.monotonic() - started) / 60,
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "trainable_encoder_parameters": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                        "total_parameters": sum(p.numel() for p in model.parameters())})
        improved = better(score, best_score)
        if improved:
            best_score, best_epoch = score, epoch
        saved = commit(epoch, epoch == settings["finetune_epochs"], improved)
        print(f"V6-{profile_id} finetune epoch={epoch} stop_macro_F1={score['stop_macro_f1']:.6f} "
              f"stop_AP={score['stop_macro_ap']:.6f} best_finetune_epoch={best_epoch}", flush=True)
    return _result(saved, resumed, last_path)
