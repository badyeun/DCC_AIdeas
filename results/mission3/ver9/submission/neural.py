"""Full Korean DeBERTa training with complete dialogue windows and bounded epochs.

Unweighted BCE trains the entire public encoder plus CLS/mean window pooling.
Only grouped Training-stop cross-fit F1 selects an epoch. A wall-clock cutoff
can abandon a partial epoch without publishing its optimizer or model state.
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
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

if __package__:
    from .selection import StopSelector, better
    from .ver4_runtime.neural import NUM_LABELS, DialogueCollator, DialogueClassifier, _device, _loader, _optimizer, _targets, tokenize_dialogues
else:
    from selection import StopSelector, better
    from ver4_runtime.neural import NUM_LABELS, DialogueCollator, DialogueClassifier, _device, _loader, _optimizer, _targets, tokenize_dialogues


MODEL_FORMAT = "mission3_ver9_neural"
LAST_FORMAT = "mission3_ver9_last"
DEFAULTS = {
    "model_name": "team-lucid/deberta-v3-base-korean",
    "max_length": 512, "stride": 64, "max_epochs": 3, "patience": 2,
    "stop_crossfit_splits": 3, "learning_rate": 2e-5, "head_learning_rate": 1e-4,
    "dropout": 0.1, "batch_size": 4, "window_batch_size": 4,
    "gradient_accumulation_steps": 8, "inference_batch_size": 8,
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
    """Exact epoch resume requires the same numerical execution environment."""
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


def _settings(config):
    settings = {name: config.get(name, value) for name, value in DEFAULTS.items()}
    for name in ("max_epochs", "patience", "stop_crossfit_splits", "batch_size",
                 "window_batch_size", "gradient_accumulation_steps", "inference_batch_size", "max_length", "stride", "log_every"):
        minimum = 0 if name in ("max_epochs", "stride", "log_every") else 2 if name == "stop_crossfit_splits" else 1
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if settings["max_epochs"] > 3:
        raise ValueError("ver9 is bounded to max_epochs <= 3")
    if settings["max_length"] > 512 or settings["max_length"] < 3 or settings["stride"] >= settings["max_length"] - 2:
        raise ValueError("max_length must be in [3, 512]; stride must fit the semantic token window")
    for name in ("learning_rate", "head_learning_rate", "layerwise_lr_decay"):
        if isinstance(settings[name], bool) or not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 < settings["layerwise_lr_decay"] <= 1:
        raise ValueError("layerwise_lr_decay must be in (0, 1]")
    if not 0 <= float(settings["dropout"]) < 1 or not 0 <= float(settings["warmup_ratio"]) < 1:
        raise ValueError("dropout/warmup_ratio must lie in [0, 1)")
    if not math.isfinite(float(settings["weight_decay"])) or settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if not isinstance(settings["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be a boolean")
    if not isinstance(settings["model_name"], str) or not settings["model_name"]:
        raise ValueError("model_name must be nonempty")
    return settings


def _configure_training(model, settings):
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("The complete ver9 encoder must train from epoch one")
    if settings["gradient_checkpointing"]:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.encoder.config, "use_cache"):
        model.encoder.config.use_cache = False


def _read_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)


def _encoder(configuration):
    values = dict(configuration)
    return AutoModel.from_config(AutoConfig.for_model(values.pop("model_type"), **values))


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


def _check_identity(state, signature, fingerprint, execution):
    if state.get("run_signature") != signature:
        raise ValueError("Checkpoint run signature mismatch; use a new output directory")
    if state.get("execution") != execution:
        raise ValueError("Exact epoch resume requires the same execution environment, GPU and precision; use a new output directory")
    if state.get("configuration_fingerprint") != fingerprint:
        raise ValueError("Checkpoint config fingerprint mismatch; use a new output directory")


class BudgetDeadlineExceeded(RuntimeError):
    """The caller's absolute wall-clock training cutoff has been reached."""


def _check_deadline(deadline_epoch, reserve_seconds=0):
    if deadline_epoch is not None and time.time() + reserve_seconds >= deadline_epoch:
        raise BudgetDeadlineExceeded("Training phase deadline reached")


@torch.inference_mode()
def _predict(model, loader, device, deadline_epoch=None):
    model.eval()
    probabilities = []
    for batch in loader:
        _check_deadline(deadline_epoch)
        batch.pop("labels", None)
        batch = {key: value.to(device) for key, value in batch.items()}
        with _amp(device):
            logits = model(**batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Prediction logits contain nonfinite values")
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        _check_deadline(deadline_epoch)
    return np.concatenate(probabilities) if probabilities else np.empty((0, NUM_LABELS), dtype=np.float32)


def _from_saved(checkpoint, config):
    if checkpoint.get("format") not in (MODEL_FORMAT, LAST_FORMAT) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported ver9 neural checkpoint")
    architecture = dict(checkpoint["architecture"])
    architecture["window_batch_size"] = int(config.get("window_batch_size", architecture["window_batch_size"]))
    model = DialogueClassifier(_encoder(checkpoint["encoder_config"]), **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def predict_checkpoint(checkpoint_path, documents, tokenizer, config):
    """Reconstruct solely from saved config and weights, fully offline."""
    checkpoint = _read_checkpoint(checkpoint_path)
    if checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError("Inference requires the selected best ver9 checkpoint")
    device = _device(config)
    model = _from_saved(checkpoint, config).to(device)
    del checkpoint
    probabilities = _predict(model, _loader(documents, None, tokenizer, config.get("inference_batch_size", 8)),
                             device, deadline_epoch=config.get("deadline_epoch"))
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Inference produced nonfinite probabilities")
    return probabilities.astype(np.float32, copy=False)


def _best_payload(last):
    # Inference artifacts do not carry Adam states or a second model copy.
    payload = {key: value for key, value in last.items() if key not in ("optimizer", "scheduler", "scaler", "rng")}
    return {**payload, "format": MODEL_FORMAT, "epoch": last["best_epoch"],
            "stop_ap": last["best_score"]["stop_macro_ap"],
            "stop_macro_f1": last["best_score"]["stop_macro_f1"]}


def _summary(state):
    """Keep committed metadata without retaining an extra CPU encoder/Adam copy."""
    keys = ("best_epoch", "best_score", "history", "completed_epochs", "completed", "execution",
            "initialization", "initial_evaluated")
    return {key: state[key] for key in keys}


def _result(state, resumed, last_path, stopped_for_budget=False, budget_reason=None):
    return {"best_epoch": state["best_epoch"], "best_stop_ap": state["best_score"]["stop_macro_ap"],
            "best_stop_macro_f1": state["best_score"]["stop_macro_f1"], "history": state["history"],
            "completed_epochs": state["completed_epochs"], "completed": state["completed"],
            "resumed": resumed, "last_checkpoint": str(last_path), "execution": state["execution"],
            "best_is_initial_baseline": state["best_epoch"] == 0,
            "initial_evaluated": state["initial_evaluated"], "head_type": "cls_mean_window_attention",
            "initialization": state["initialization"], "loss": "unweighted_bce",
            "stopped_for_budget": bool(stopped_for_budget), "budget_reason": budget_reason,
            "selection_scope": "Training fold stop: grouped threshold cross-fit macro-F1; AP secondary. "
                               "Reused epoch selection, not model-level OOF or official Validation."}


def train_model(train_documents, train_targets, stop_documents, stop_targets, tokenizer, config,
                checkpoint_path, seed, initial_model_dir, run_signature, stop_groups=None, deadline_epoch=None):
    """Train/resume from local public weights, committing completed epochs only.

    ``deadline_epoch`` is an absolute ``time.time()`` timestamp for this training
    phase. It is intentionally excluded from the configuration identity: a new
    Colab session may resume the exact last committed epoch with a new cutoff.
    A partial epoch is abandoned in memory, never promoted to a completed one.
    """
    train_targets = _targets(train_targets, train_documents, "fit")
    stop_targets = _targets(stop_targets, stop_documents, "stop")
    if stop_groups is None or np.asarray(stop_groups).shape != (len(stop_documents),):
        raise ValueError("stop_groups must match stop documents")
    if not isinstance(run_signature, str) or not run_signature:
        raise ValueError("run_signature must be a nonempty string")
    if deadline_epoch is not None:
        if isinstance(deadline_epoch, bool) or not math.isfinite(float(deadline_epoch)):
            raise ValueError("deadline_epoch must be a finite wall-clock timestamp")
        deadline_epoch = float(deadline_epoch)
    settings = _settings(config)
    device = _device(settings)
    execution = execution_identity(device)
    fingerprint_data = {"settings": settings, "seed": int(seed),
                        "stop_groups": list(map(str, stop_groups)), "execution": execution}
    identity_hash = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode())
    for documents, targets in ((train_documents, train_targets), (stop_documents, stop_targets)):
        for document in documents:
            identity_hash.update(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
            identity_hash.update(b"\n")
        identity_hash.update(targets.tobytes())
    fingerprint = identity_hash.hexdigest()
    checkpoint_path = Path(checkpoint_path)
    last_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.ckpt")
    last = _read_checkpoint(last_path) if last_path.exists() else None
    best_saved = _read_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    for saved in (last, best_saved):
        if saved is not None:
            _check_identity(saved, run_signature, fingerprint, execution)
    if best_saved is not None and best_saved.get("format") != MODEL_FORMAT:
        raise ValueError("Unsupported ver9 best checkpoint")
    if last is not None:
        if last.get("format") != LAST_FORMAT:
            raise ValueError("Unsupported ver9 resume checkpoint")
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
    resumed = last is not None
    _seed(seed)
    if last is None:
        directory = Path(initial_model_dir)
        if not directory.is_dir():
            raise ValueError("initial_model_dir must be an already downloaded local public model directory")
        encoder = AutoModel.from_pretrained(str(directory), local_files_only=True,
                                            trust_remote_code=False, use_safetensors=True)
        architecture = {"num_labels": NUM_LABELS, "dropout": settings["dropout"],
                        "window_batch_size": settings["window_batch_size"]}
        _seed(seed + 17)
        model = DialogueClassifier(encoder, **architecture)
        metadata = {"encoder_config": encoder.config.to_dict(), "architecture": architecture,
                    "model_name": settings["model_name"],
                    "initialization": "local_public_encoder_and_fresh_cls_mean_window_attention_head"}
    else:
        model = _from_saved(last, settings)
        metadata = {key: last[key] for key in ("encoder_config", "architecture", "model_name", "initialization")}
    model.to(device)
    _configure_training(model, settings)
    loader = _loader(train_documents, train_targets, tokenizer, settings["batch_size"], shuffle=True, seed=seed)
    stop_loader = _loader(stop_documents, None, tokenizer, settings["inference_batch_size"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _optimizer(model, settings)
    accumulation = settings["gradient_accumulation_steps"]
    total_steps = math.ceil(len(loader) / accumulation) * max(1, settings["max_epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * settings["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not native_bf16_supported())
    if last is None:
        best_score = {"stop_macro_f1": None, "stop_macro_ap": None}
        history, completed, best_epoch, stale, initial_evaluated = [], 0, 0, 0, False
    else:
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        _restore_rng(last["rng"], loader.generator)
        best_score, best_epoch, stale = last["best_score"], last["best_epoch"], last["stale_epochs"]
        history, completed = list(last["history"]), last["completed_epochs"]
        initial_evaluated = last["initial_evaluated"]
    del last

    def commit(epoch, done, improved):
        state = {"format": LAST_FORMAT, "format_version": 1, **metadata, "model_state": _cpu_state(model),
                 "execution": execution, "seed": int(seed), "run_signature": run_signature,
                 "configuration_fingerprint": fingerprint, "settings": settings,
                 "completed_epochs": epoch, "completed": done, "best_epoch": best_epoch,
                 "best_score": best_score, "stale_epochs": stale, "history": list(history),
                 "initial_evaluated": initial_evaluated, "loss": "unweighted_bce",
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": _rng_state(loader.generator)}
        # Publish last first so an interrupted best write is repairable whenever
        # this newly committed epoch was the selected improvement.
        _atomic_save(state, last_path)
        if improved:
            _atomic_save(_best_payload(state), checkpoint_path)
        return _summary(state)

    if resumed:
        committed_state = {"best_epoch": best_epoch, "best_score": best_score, "history": list(history),
                           "completed_epochs": completed, "completed": False, "execution": execution,
                           "initialization": metadata["initialization"], "initial_evaluated": initial_evaluated}
    else:
        # Even if the first stop pass hits the cutoff, publish clearly marked
        # untrained weights and a resumable pre-evaluation RNG state.
        committed_state = commit(0, False, True)
    if not initial_evaluated:
        started = time.monotonic()
        try:
            best_score = selector.score(_predict(model, stop_loader, device, deadline_epoch))
            _check_deadline(deadline_epoch)
        except BudgetDeadlineExceeded:
            return _result(committed_state, resumed, last_path, True, "initial_stop_evaluation")
        history = [{"epoch": 0, "phase": "initial_public_encoder", "train_loss": None,
                    **best_score, "minutes": (time.monotonic() - started) / 60}]
        initial_evaluated = True
        committed_state = commit(0, settings["max_epochs"] == 0, True)
    if settings["max_epochs"] == 0:
        return _result(committed_state, resumed, last_path)

    for epoch in range(completed + 1, settings["max_epochs"] + 1):
        # The slowest measured completed epoch includes training and stop
        # evaluation. Keep 120 s for checkpoint publication; never start another
        # epoch merely because some wall time remains.
        measured = [row["minutes"] * 60 for row in history if row["epoch"] > 0]
        estimated_next = max(measured, default=0.0) * 1.10
        try:
            _check_deadline(deadline_epoch, reserve_seconds=estimated_next + 120)
        except BudgetDeadlineExceeded:
            return _result(committed_state, resumed, last_path, True, "insufficient_time_for_next_epoch")
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count = 0.0, 0
        try:
            for step, batch in enumerate(loader):
                _check_deadline(deadline_epoch, reserve_seconds=120)
                labels = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                group_start = (step // accumulation) * accumulation
                group_documents = min(accumulation * loader.batch_size,
                                      len(train_documents) - group_start * loader.batch_size)
                with _amp(device):
                    loss = criterion(model(**batch).float(), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, batch={step}")
                scaler.scale(loss * len(labels) / group_documents).backward()
                loss_sum += float(loss.detach()) * len(labels)
                count += len(labels)
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                _check_deadline(deadline_epoch, reserve_seconds=120)
                if settings["log_every"] > 0 and (step + 1) % settings["log_every"] == 0:
                    print(f"V9 seed={seed} epoch={epoch} batch={step + 1}/{len(loader)} "
                          f"loss={loss_sum / count:.5f}", flush=True)
            evaluation_cutoff = None if deadline_epoch is None else deadline_epoch - 120
            score = selector.score(_predict(model, stop_loader, device, evaluation_cutoff))
            _check_deadline(deadline_epoch, reserve_seconds=120)
        except BudgetDeadlineExceeded:
            # The model/optimizer may contain partial updates. Do not call
            # commit here: exact restart uses the previous complete epoch.
            optimizer.zero_grad(set_to_none=True)
            return _result(committed_state, resumed, last_path, True, "partial_epoch_abandoned")
        history.append({"epoch": epoch, "phase": "full_encoder_finetune", "train_loss": loss_sum / count,
                        **score, "minutes": (time.monotonic() - started) / 60})
        improved = better(score, best_score)
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        done = epoch == settings["max_epochs"] or stale >= settings["patience"]
        committed_state = commit(epoch, done, improved)
        print(f"V9 epoch={epoch}/{settings['max_epochs']} loss={loss_sum / count:.5f} "
              f"stop_macro_F1={score['stop_macro_f1']:.6f} stop_AP={score['stop_macro_ap']:.6f} "
              f"best_epoch={best_epoch}", flush=True)
        if done:
            return _result(committed_state, resumed, last_path)
    raise RuntimeError("Resume checkpoint marked incomplete after the configured epoch budget")

