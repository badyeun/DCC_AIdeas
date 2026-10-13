"""Fixed-step task adaptation using only the caller's Training text.

The pilot is part of the prescribed optimizer steps. A partial adaptation is
never returned as a completed encoder. MLM draws sample windows; unlike the
later supervised full refit, they need not visit every Training dialogue.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import time

import torch
from torch.utils.data import DataLoader

from . import data, neural
from .budget import BudgetExceeded


DEFAULTS = {
    "tapt_steps": 800, "tapt_learning_rate": 5e-5, "tapt_batch_size": 4,
    "tapt_gradient_accumulation": 4, "tapt_mask_probability": .15,
    "tapt_max_length": 256, "tapt_pilot_steps": 10,
    "tapt_fit_time_limit_seconds": 900., "tapt_full_time_limit_seconds": 1200.,
}


def validate_recipe(settings):
    result = {key: settings.get(key, default) for key, default in DEFAULTS.items()}
    for key in ("tapt_steps", "tapt_batch_size", "tapt_gradient_accumulation", "tapt_max_length", "tapt_pilot_steps"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not 4 <= result["tapt_max_length"] <= 512:
        raise ValueError("tapt_max_length must lie in [4, 512]")
    for key in ("tapt_learning_rate", "tapt_mask_probability", "tapt_fit_time_limit_seconds", "tapt_full_time_limit_seconds"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be positive and finite")
    if result["tapt_mask_probability"] > .15:
        raise ValueError("The TAPT masking probability cannot exceed 0.15")
    if result["tapt_fit_time_limit_seconds"] > 900 or result["tapt_full_time_limit_seconds"] > 1200:
        raise ValueError("TAPT phase caps cannot exceed 15/20 minutes")
    return result


def prepare_masked_model(model_id, revision, asset_dir):
    """Load the public pretrained MLM head, never a freshly random MLM head."""
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("The MLM revision must be explicit")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    resolved = getattr(config, "_commit_hash", None)
    if resolved is None and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        resolved = revision
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError("The immutable MLM model revision was not resolved")
    options = {"revision": resolved, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **options)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("TAPT requires a fast tokenizer retaining all overflow windows")
    model = AutoModelForMaskedLM.from_pretrained(model_id, config=config, **options)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not getattr(model, "supports_gradient_checkpointing", False):
        raise ValueError("The masked model must support gradient checkpointing")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    asset_dir = Path(asset_dir)
    for name in ("encoder", "tokenizer"):
        (asset_dir / name).mkdir(parents=True, exist_ok=True)
    config.save_pretrained(asset_dir / "encoder")
    tokenizer.save_pretrained(asset_dir / "tokenizer")
    return model, tokenizer, resolved


class WindowDataset:
    def __init__(self, texts, tokenizer, max_length):
        encoded = neural.EncodeDataset(texts, None, tokenizer, max_length, overlap=0)
        special = set(tokenizer.all_special_ids) | {tokenizer.pad_token_id, tokenizer.mask_token_id}
        self.windows, self.skipped_empty_windows = [], 0
        for document in encoded.documents:
            for window in document:
                if any(token not in special for token in window["input_ids"]):
                    self.windows.append(window)
                else:
                    self.skipped_empty_windows += 1
        if not self.windows:
            raise ValueError("TAPT requires at least one non-special text token")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


class MaskedCollator:
    """Dynamic 80/10/10 masking; specials and padding never become targets."""
    def __init__(self, tokenizer, probability, seed):
        if not 0 < probability <= 1:
            raise ValueError("Mask probability must lie in (0, 1]")
        self.tokenizer, self.probability = tokenizer, probability
        self.generator = torch.Generator().manual_seed(seed)
        special = set(tokenizer.all_special_ids) | {tokenizer.pad_token_id, tokenizer.mask_token_id}
        if tokenizer.pad_token_id is None or tokenizer.mask_token_id is None:
            raise ValueError("TAPT tokenizer must define padding and mask tokens")
        self.special = special
        self.content_vocab = torch.tensor([value for value in range(len(tokenizer)) if value not in special])
        if not len(self.content_vocab):
            raise ValueError("The tokenizer requires a non-special vocabulary")

    def __call__(self, examples):
        if not examples:
            raise ValueError("MLM batches cannot be empty")
        keys = set(examples[0])
        if any(set(example) != keys for example in examples):
            raise ValueError("MLM windows must use identical input fields")
        length = max(len(example["input_ids"]) for example in examples)
        batch = {key: torch.full((len(examples), length),
                                self.tokenizer.pad_token_id if key == "input_ids" else 0,
                                dtype=torch.long) for key in keys}
        for row, example in enumerate(examples):
            for key, values in example.items():
                batch[key][row, :len(values)] = torch.tensor(values)
        original = batch["input_ids"].clone()
        eligible = batch["attention_mask"].bool()
        for token in self.special:
            eligible &= original != token
        selected = (torch.rand(original.shape, generator=self.generator) < self.probability) & eligible
        for row in range(len(examples)):
            positions = eligible[row].nonzero(as_tuple=False).flatten()
            if not len(positions):
                raise ValueError("Every MLM window needs a non-special token")
            if not bool(selected[row].any()):
                pick = torch.randint(len(positions), (1,), generator=self.generator)
                selected[row, positions[pick]] = True
        labels = original.clone().masked_fill(~selected, -100)
        replacements = torch.rand(original.shape, generator=self.generator)
        batch["input_ids"][selected & (replacements < .8)] = self.tokenizer.mask_token_id
        random_positions = selected & (replacements >= .8) & (replacements < .9)
        count = int(random_positions.sum())
        if count:
            draws = torch.randint(len(self.content_vocab), (count,), generator=self.generator)
            batch["input_ids"][random_positions] = self.content_vocab[draws]
        batch["labels"] = labels
        return batch


def encoder_from_masked_model(model):
    """Keep ordinary AutoModel checkpoint structure, including its unused pooler."""
    from transformers import AutoModel
    encoder = AutoModel.from_config(model.config, trust_remote_code=False)
    copied = encoder.load_state_dict(model.base_model.state_dict(), strict=False)
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    if copied.unexpected_keys or set(copied.missing_keys) - allowed_missing:
        raise ValueError("Adapted MLM encoder does not match the standard inference encoder")
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return encoder


def adapt_encoder(texts, settings, asset_dir, budget, scope):
    """Return an encoder only after every prescribed MLM optimizer step completes."""
    if scope not in ("fit_only", "all_training"):
        raise ValueError("MLM scope must be fit_only or all_training")
    recipe = validate_recipe(settings)
    texts = list(texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("MLM inputs must be nonempty Training text strings")
    started = time.monotonic()
    cap = recipe["tapt_fit_time_limit_seconds" if scope == "fit_only" else "tapt_full_time_limit_seconds"]
    report_path = Path(asset_dir).parent / "tapt_report.json"
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8")); digest.update(b"\0")
    report = {"status": "initializing", "enabled": True, "scope": scope, "input_text_count": len(texts),
              "text_digest": digest.hexdigest(), "recipe": recipe, "seed": settings["seed"],
              "target_optimizer_steps": recipe["tapt_steps"], "completed_optimizer_steps": 0,
              "window_draws": 0, "attended_tokens": 0, "masked_tokens": 0,
              "phase_limit_seconds": cap, "all_dialogues_seen_is_not_guaranteed": True}
    data.write_json(report_path, report)

    def check(stage):
        budget.check(stage, training=True)
        if time.monotonic() - started >= cap:
            raise BudgetExceeded("TAPT phase cap: " + stage)

    model = optimizer = None
    loss_total, microbatches, timings = 0., 0, []
    try:
        check("prepare public MLM")
        neural.seed_everything(settings["seed"])
        model, tokenizer, revision = prepare_masked_model(settings["model_id"], settings["revision"], asset_dir)
        check("tokenize MLM text")
        dataset = WindowDataset(texts, tokenizer, recipe["tapt_max_length"])
        loader = DataLoader(dataset, batch_size=recipe["tapt_batch_size"], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(settings["seed"]),
                            collate_fn=MaskedCollator(tokenizer, recipe["tapt_mask_probability"], settings["seed"] + 1))
        report.update(source_revision=revision, available_windows=len(dataset),
                      skipped_no_text_windows=dataset.skipped_empty_windows, status="training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and not settings.get("allow_cpu", False):
            raise RuntimeError("TAPT requires the selected Colab GPU")
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["tapt_learning_rate"], weight_decay=.01)
        cuda = device.type == "cuda"
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=cuda and dtype == torch.float16)
        iterator = iter(loader)
        pilot = min(recipe["tapt_pilot_steps"], recipe["tapt_steps"])
        preparation_seconds = time.monotonic() - started
        for update in range(recipe["tapt_steps"]):
            check("MLM optimizer update")
            if cuda:
                torch.cuda.synchronize()
            update_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(recipe["tapt_gradient_accumulation"]):
                check("MLM microbatch")
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                report["window_draws"] += len(batch["input_ids"])
                report["attended_tokens"] += int(batch["attention_mask"].sum())
                report["masked_tokens"] += int((batch["labels"] != -100).sum())
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=cuda):
                    loss = model(**batch).loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("MLM loss is nonfinite")
                scaler.scale(loss / recipe["tapt_gradient_accumulation"]).backward()
                loss_total += float(loss.detach()); microbatches += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            if cuda:
                torch.cuda.synchronize()
            timings.append(time.monotonic() - update_start)
            report["completed_optimizer_steps"] = update + 1
            check("completed MLM optimizer update")
            if update + 1 == pilot:
                per_update = max(sum(timings) / len(timings), max(timings[-min(3, len(timings)):]))
                remaining_estimate = per_update * (recipe["tapt_steps"] - update - 1)
                repeat_estimate = preparation_seconds + per_update * recipe["tapt_steps"]
                report.update(pilot_optimizer_steps=pilot, pilot_seconds=sum(timings),
                              estimated_repeat_seconds=repeat_estimate)
                safety = settings.get("budget_safety_factor", 1.25)
                after = settings.get("_tapt_supervised_forecast_seconds", 0.)
                after += settings.get("_tapt_refit_forecast_seconds", 0.)
                if scope == "fit_only":
                    after += repeat_estimate
                if remaining_estimate * safety > cap - (time.monotonic() - started):
                    raise BudgetExceeded("TAPT pilot forecast exceeds phase cap")
                if not budget.can_start(remaining_estimate + after, training=True):
                    raise BudgetExceeded("TAPT pilot cannot preserve supervised/final work")
            if (update + 1) % 100 == 0:
                report.update(mean_mlm_loss=loss_total / microbatches, seconds=time.monotonic() - started)
                data.write_json(report_path, report)
                print(f"TAPT {scope}: {update + 1}/{recipe['tapt_steps']} optimizer steps", flush=True)
        check("finish prescribed MLM recipe")
        encoder = encoder_from_masked_model(model)
        encoder.config.save_pretrained(Path(asset_dir) / "encoder")
        check("complete adapted encoder")
        report.update(status="complete", mean_mlm_loss=loss_total / microbatches,
                      seconds=time.monotonic() - started)
        data.write_json(report_path, report)
        return encoder, tokenizer, revision, report
    except BaseException as error:
        report.update(status="budget_exhausted" if isinstance(error, BudgetExceeded) else "failed",
                      seconds=time.monotonic() - started, mean_mlm_loss=loss_total / microbatches if microbatches else None,
                      error_type=type(error).__name__)
        data.write_json(report_path, report)
        raise
    finally:
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
