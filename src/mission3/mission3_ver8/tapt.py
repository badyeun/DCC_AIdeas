"""Bounded masked-language adaptation of FIT text only, with optimizer-step resume."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup

if __package__:
    from . import neural
    from .ver4_runtime.data import write_json
else:
    import neural
    from ver4_runtime.data import write_json


DEFAULTS = {"tapt_steps": 500, "tapt_max_length": 256, "tapt_batch_size": 4,
            "tapt_accumulation_steps": 8, "tapt_learning_rate": 2e-5,
            "tapt_mask_probability": 0.15, "tapt_save_every": 100}


def settings(config):
    values = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    for key in ("tapt_steps", "tapt_max_length", "tapt_batch_size", "tapt_accumulation_steps", "tapt_save_every"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 4 <= values["tapt_max_length"] <= 512:
        raise ValueError("tapt_max_length must be in [4, 512]")
    if not 0 < values["tapt_mask_probability"] < 1:
        raise ValueError("tapt_mask_probability must be in (0, 1)")
    if not math.isfinite(values["tapt_learning_rate"]) or values["tapt_learning_rate"] <= 0:
        raise ValueError("tapt_learning_rate must be positive and finite")
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(directory):
    return {p.relative_to(directory).as_posix(): _sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def masked_batch(windows, tokenizer, probability):
    """Dynamic 15% MLM, 80/10/10 replacement; never mask padding/special tokens."""
    batch = dict(tokenizer.pad(windows, padding=True, return_tensors="pt"))
    ids = batch["input_ids"]
    allowed = batch["attention_mask"].bool()
    for token_id in tokenizer.all_special_ids:
        allowed &= ids.ne(token_id)
    chosen = (torch.rand(ids.shape) < probability) & allowed
    # Very short calls must not produce an all-ignored cross entropy / NaN.
    if not chosen.any():
        eligible = allowed.nonzero()
        if not len(eligible):
            raise ValueError("MLM batch has no semantic tokens")
        row, column = eligible[torch.randint(len(eligible), ())]
        chosen[row, column] = True
    labels = ids.clone()
    labels[~chosen] = -100
    replacement = torch.rand(ids.shape)
    ids = ids.clone()
    ids[chosen & (replacement < .8)] = tokenizer.mask_token_id
    random_positions = chosen & (replacement >= .8) & (replacement < .9)
    random_ids = torch.randint(len(tokenizer), ids.shape)
    ids[random_positions] = random_ids[random_positions]
    batch["input_ids"], batch["labels"] = ids, labels
    return batch


def adapt(fit_texts, fit_filenames, tokenizer, initial_dir, output_dir, config, *, run_signature, seed):
    """No target, stop, calibration, Validation or review argument exists here.

    The run identity includes exact fit rows, rendered text, tokenizer, settings
    and parent provenance. Incomplete runs require the same numerical environment.
    Completed exported encoders can be reused without the optimizer state.
    """
    cfg = settings(config)
    if not fit_texts or len(fit_texts) != len(fit_filenames) or len(set(fit_filenames)) != len(fit_filenames):
        raise ValueError("TAPT requires nonempty, unique fit filenames aligned with text")
    if tokenizer.mask_token_id is None:
        raise ValueError("TAPT requires a pretrained MLM tokenizer")
    identity = {"parent_signature": run_signature, "settings": cfg, "seed": seed,
                "gradient_checkpointing": config.get("gradient_checkpointing", True),
                "fit_filenames": list(fit_filenames), "fit_text_sha256": hashlib.sha256(
                    json.dumps(list(fit_texts), ensure_ascii=False).encode()).hexdigest(),
                "tokenizer_vocab_sha256": hashlib.sha256(
                    json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    encoder_dir, done_path, last_path = root / "encoder", root / "tapt_done.json", root / "tapt_last.ckpt"
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("signature") != signature or done.get("files") != _files(encoder_dir):
            raise ValueError("Completed TAPT data/settings/encoder mismatch")
        print("완료된 fit-only TAPT encoder 재사용", flush=True)
        return encoder_dir, done["report"]
    write_json(root / "tapt_run.json", {"signature": signature, **identity})
    device = neural._device(config)
    execution = neural.execution_identity(device)
    last = neural._read_checkpoint(last_path) if last_path.is_file() else None
    if last is not None and (last.get("signature") != signature or last.get("execution") != execution):
        raise ValueError("TAPT resume requires the same fit data, settings, code and execution environment")
    docs = neural.tokenize_dialogues(fit_texts, tokenizer, cfg["tapt_max_length"], 0)
    special = set(tokenizer.all_special_ids)
    windows = [window for document in docs for window in document
               if any(token not in special for token in window["input_ids"])]
    if not windows:
        raise ValueError("TAPT fit text has no semantic tokens")
    del docs
    neural._seed(seed)
    model = AutoModelForMaskedLM.from_pretrained(initial_dir, local_files_only=True, trust_remote_code=False)
    if last is not None:
        model.load_state_dict(last["model_state"], strict=True)
    model.to(device).train()
    model.requires_grad_(True)
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["tapt_learning_rate"], weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * cfg["tapt_steps"]), cfg["tapt_steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not neural.native_bf16_supported())
    rng_marker = torch.Generator().manual_seed(seed)
    completed, loss_sum, token_count, elapsed = 0, 0., 0, 0.
    resumed = last is not None
    if last is not None:
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        neural._restore_rng(last["rng"], rng_marker)
        completed, loss_sum, token_count, elapsed = (last[k] for k in ("completed_steps", "loss_sum", "token_count", "minutes"))
    del last
    effective = cfg["tapt_batch_size"] * cfg["tapt_accumulation_steps"]
    steps_per_pass = math.ceil(len(windows) / effective)
    order, previous_pass = None, None
    started = time.monotonic()
    for step in range(completed, cfg["tapt_steps"]):
        pass_number, within = divmod(step, steps_per_pass)
        if previous_pass != pass_number:
            generator = torch.Generator().manual_seed(seed + pass_number)
            order = torch.randperm(len(windows), generator=generator).tolist()
            previous_pass = pass_number
        indices = order[within * effective:(within + 1) * effective]
        # Generate every microbatch mask before backward, to weight by masked tokens.
        batches = [masked_batch([windows[i] for i in indices[start:start + cfg["tapt_batch_size"]]],
                                tokenizer, cfg["tapt_mask_probability"])
                   for start in range(0, len(indices), cfg["tapt_batch_size"])]
        masked_counts = [int(batch["labels"].ne(-100).sum()) for batch in batches]
        total_masked = sum(masked_counts)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.
        for batch, masked in zip(batches, masked_counts):
            batch = {key: value.to(device) for key, value in batch.items()}
            with neural._amp(device):
                loss = model(**batch).loss.float()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite TAPT loss at step {step + 1}")
            scaler.scale(loss * masked / total_masked).backward()
            step_loss += float(loss.detach()) * masked
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed, loss_sum, token_count = step + 1, loss_sum + step_loss, token_count + total_masked
        if completed % cfg["tapt_save_every"] == 0 or completed == cfg["tapt_steps"]:
            state = {"format": "mission3_ver8_tapt_last", "signature": signature, "execution": execution,
                     "model_state": neural._cpu_state(model), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "rng": neural._rng_state(rng_marker), "completed_steps": completed,
                     "loss_sum": loss_sum, "token_count": token_count,
                     "minutes": elapsed + (time.monotonic() - started) / 60}
            neural._atomic_save(state, last_path)
            del state
        if completed == 1 or completed % 25 == 0 or completed == cfg["tapt_steps"]:
            print(f"TAPT fit-only step={completed}/{cfg['tapt_steps']} masked_loss={step_loss / total_masked:.5f}", flush=True)
    encoder_dir.mkdir(exist_ok=True)
    model.base_model.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(encoder_dir)
    report = {"scope": "Only the supervised fit transcripts, with no targets; fixed MLM budget",
              "fit_samples": len(fit_texts), "semantic_windows": len(windows), "completed_steps": completed,
              "masked_token_loss": loss_sum / token_count, "masked_tokens_seen": token_count,
              "minutes": elapsed + (time.monotonic() - started) / 60, "resumed": resumed,
              "settings": cfg, "execution": execution,
              "stop_or_calibration_or_validation_used": False}
    write_json(done_path, {"signature": signature, "report": report, "files": _files(encoder_dir)})
    if not config.get("keep_completed_resume_states", False):
        last_path.unlink(missing_ok=True)
    return encoder_dir, report
