"""Measure fresh-model batch candidates while keeping effective batch 16 fixed."""
from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import data, trainer

EFFECTIVE_BATCH = 16
WARMUP_EXAMPLES = 16
MEASURED_EXAMPLES = 64
PREDICTION_EXAMPLES = 64
MIN_HEADROOM_FRACTION = .10
BATCH_CANDIDATES = {"qwen": (8, 4, 2, 1), "control": (16, 8, 4)}


class _TrainingBudget:
    """Benchmark inference also spends only the time available before export."""
    def __init__(self, budget):
        self.budget = budget

    def check(self, stage, training=False):
        return self.budget.check(stage, training=True)


def _indices(values, rows, minimum, name):
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) < minimum
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= rows)
            or len(np.unique(values)) != len(values)):
        raise ValueError(f"{name} needs at least {minimum} unique valid row indices")
    return values.astype(np.int64, copy=False)


def representative_indices(texts, indices, count, seed):
    """Sample evenly across transcript lengths, then fix the batch ordering."""
    ordered = np.asarray(sorted(map(int, indices), key=lambda row: (len(texts[row]), row)))
    positions = np.floor((np.arange(count) + .5) * len(ordered) / count).astype(np.int64)
    selected = ordered[positions]
    return np.random.default_rng(seed).permutation(selected)


def _digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _release(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _memory(device):
    if torch.device(device).type != "cuda":
        return {"peak_vram_bytes": 0, "peak_reserved_bytes": 0,
                "device_total_bytes": None, "headroom_bytes": None,
                "headroom_fraction": None, "memory_safe": True}
    trainer._synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    # CUDA free excludes our reserved pool and other processes. Estimate free
    # space at our peak as well, so cached pages are not counted twice.
    headroom = min(free, free + reserved - peak_reserved)
    return {"peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(peak_reserved), "device_total_bytes": int(total),
            "headroom_bytes": int(headroom), "headroom_fraction": headroom / total,
            "memory_safe": headroom >= total * MIN_HEADROOM_FRACTION}


def _stress_loader(kind, tokenizer, text, target, batch_size, max_length):
    """A full update at maximum length, including optimizer-state allocation.

    Extra synthetic padding tokens are deliberately attended to: kernels that
    unpad masked inputs must also exercise the configured maximum sequence.
    These disposable weights are never kept for either experiment stage.
    """
    source = trainer._Collate(tokenizer, kind, max_length)([(text, target)])
    tokens = source["input_ids"]
    missing = max_length - tokens.shape[1]
    tokens = torch.nn.functional.pad(tokens, (0, missing), value=tokenizer.pad_token_id)
    batch = {"input_ids": tokens.repeat(batch_size, 1),
             "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
             "labels": source["labels"].repeat(batch_size, 1)}
    if "token_type_ids" in source:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return [batch] * (EFFECTIVE_BATCH // batch_size)


def _trial(kind, spec, texts, targets, fit, warmup, measured, development,
           settings, device, budget, batch_size, root):
    """Do not return models, tensors, optimizers, logits, or exception objects."""
    budget.check("batch benchmark model load", training=True)
    started = time.monotonic()
    model, tokenizer = trainer.create_model(spec, settings, device)
    setup_seconds = time.monotonic() - started
    budget.check("batch benchmark model loaded", training=True)
    accumulation = EFFECTIVE_BATCH // batch_size
    maximum = settings[kind + "_max_length"]
    lengths_path = Path(root) / (kind + "_token_lengths.json")
    if not lengths_path.exists():
        budget.check("Training token lengths", training=True)
        data.write_json(lengths_path, trainer.length_statistics(tokenizer, texts, kind, maximum))
        budget.check("Training token lengths complete", training=True)
    optimizer = trainer.build_optimizer(model, kind, settings)
    scheduler = trainer.build_scheduler(optimizer, 6, settings["warmup_ratio"])
    scaler = trainer.make_scaler(device)
    weights = trainer.positive_weights(targets[fit])
    memory = []
    common = dict(device=device, pos_weight=weights, accumulation_steps=accumulation,
                  scheduler=scheduler, scaler=scaler, budget=budget,
                  gradient_clip=settings.get("gradient_clip", 1.0))

    def train(loader):
        budget.check("batch benchmark training", training=True)
        stats = trainer.train_epoch(model, loader, optimizer, **common)
        memory.append(_memory(device))
        if not stats["completed_epoch"]:
            raise RuntimeError("Batch benchmark did not finish its fixed sample set")
        return stats

    def loader(indices, with_targets=True):
        return trainer.make_loader(tokenizer, texts, targets if with_targets else None,
            indices, kind=kind, batch_size=batch_size, max_length=maximum,
            shuffle=False, seed=settings["seed"], pin_memory=torch.device(device).type == "cuda")

    stress_row = max(map(int, fit), key=lambda row: len(texts[row]))
    stress = train(_stress_loader(kind, tokenizer, texts[stress_row], targets[stress_row], batch_size, maximum))
    warmup_stats = train(loader(warmup))
    stats = train(loader(measured))
    if (stress["examples"] != EFFECTIVE_BATCH or warmup_stats["examples"] != WARMUP_EXAMPLES
            or stats["examples"] != MEASURED_EXAMPLES
            or stats["successful_updates"] != MEASURED_EXAMPLES // EFFECTIVE_BATCH):
        raise RuntimeError("Batch benchmark compared unequal training workloads")
    prediction = trainer.predict_logits(model, loader(development, with_targets=False),
        device=device, budget=_TrainingBudget(budget), return_stats=True)
    memory.append(_memory(device))
    if prediction["examples"] != PREDICTION_EXAMPLES:
        raise RuntimeError("Batch benchmark compared unequal prediction workloads")
    seconds = stats["elapsed_seconds"] / MEASURED_EXAMPLES
    predict_seconds = prediction["seconds_per_example"]
    if not all(math.isfinite(value) and value > 0 for value in (seconds, predict_seconds)):
        raise ValueError("Batch benchmark timings must be finite and positive")
    cuda_memory = [item for item in memory if item["device_total_bytes"] is not None]
    safe = all(item["memory_safe"] for item in memory)
    result = {"status": "success" if safe else "insufficient_memory_headroom",
        "batch_size": batch_size, "gradient_accumulation": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH, "seconds_per_example": seconds,
        "examples_per_second": 1 / seconds, "seconds_per_microstep": seconds * batch_size,
        "seconds_per_prediction": predict_seconds, "setup_seconds": setup_seconds,
        "measured_examples": MEASURED_EXAMPLES, "warmup_examples": WARMUP_EXAMPLES,
        "stress_examples": EFFECTIVE_BATCH, "stress_max_length": maximum,
        "measured_optimizer_updates": stats["successful_updates"],
        "overflow_retries": sum(item.get("overflow_retries", 0) for item in (stress, warmup_stats, stats)),
        "final_scaler_scale": stats.get("final_scaler_scale"),
        "prediction_examples": PREDICTION_EXAMPLES,
        "peak_vram_bytes": max(item["peak_vram_bytes"] for item in memory),
        "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in memory),
        "device_total_bytes": cuda_memory[0]["device_total_bytes"] if cuda_memory else None,
        "headroom_fraction": min(item["headroom_fraction"] for item in cuda_memory) if cuda_memory else None,
        "memory_check": "cuda_peak_reserved_and_free" if cuda_memory else "cpu_test_only",
        "pilot_weights_discarded": True}
    return result


def _is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and any(phrase in str(error).lower()
            for phrase in ("cuda out of memory", "cuda error: out of memory")))


def autotune(kind, spec, texts, targets, fit, development, settings, device, budget, root):
    """Return updated settings and a measured, auditable batch-selection report.

    Candidate models always start from the pinned pretrained weights. Only the
    supplied fit labels enter loss or positive weights; development is timed
    without labels. A later caller creates another fresh model for training.
    """
    if kind not in BATCH_CANDIDATES or trainer._kind(spec) != kind:
        raise ValueError("Batch benchmark requires matching control or qwen model kind")
    targets = trainer._targets(targets, len(texts))
    fit = _indices(fit, len(texts), WARMUP_EXAMPLES + MEASURED_EXAMPLES, "Training benchmark")
    development = _indices(development, len(texts), PREDICTION_EXAMPLES, "Development benchmark")
    if np.intersect1d(fit, development).size:
        raise ValueError("Training and development benchmark rows must be disjoint")
    ordered = representative_indices(texts, fit, WARMUP_EXAMPLES + MEASURED_EXAMPLES, settings["seed"])
    warmup, measured = ordered[:WARMUP_EXAMPLES], ordered[WARMUP_EXAMPLES:]
    prediction = representative_indices(texts, development, PREDICTION_EXAMPLES, settings["seed"])
    report = {"kind": kind, "status": "running", "selection": "fastest_safe_seconds_per_example",
        "effective_batch_size": EFFECTIVE_BATCH, "minimum_headroom_fraction": MIN_HEADROOM_FRACTION,
        "candidate_batches": list(BATCH_CANDIDATES[kind]), "warmup_examples": WARMUP_EXAMPLES,
        "measured_examples": MEASURED_EXAMPLES, "prediction_examples": PREDICTION_EXAMPLES,
        "warmup_indices_digest": _digest(warmup), "measured_indices_digest": _digest(measured),
        "prediction_indices_digest": _digest(prediction), "pilot_weights_discarded": True,
        "trials": []}
    path = Path(root) / (kind + "_batch_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_json(path, report)
    _release(device)
    for batch_size in BATCH_CANDIDATES[kind]:
        print(f"{kind}: measuring batch {batch_size} x accumulation {EFFECTIVE_BATCH // batch_size}", flush=True)
        try:
            trial = _trial(kind, spec, texts, targets, fit, warmup, measured, prediction,
                           settings, device, budget, batch_size, root)
        except BaseException as error:
            if not _is_oom(error):
                report["status"] = "interrupted"
                report["error_type"] = type(error).__name__
                data.write_json(path, report)
                raise
            trial = {"status": "out_of_memory", "batch_size": batch_size,
                "gradient_accumulation": EFFECTIVE_BATCH // batch_size,
                "effective_batch_size": EFFECTIVE_BATCH, "error_type": type(error).__name__,
                "pilot_weights_discarded": True}
            # Drop training frames (and their GPU tensors) BEFORE collecting or
            # trying the next candidate. Never retain the raw OOM payload.
            error.__traceback__ = None
            error.__context__ = None
            error.__cause__ = None
        finally:
            _release(device)
        trial.update(warmup_indices_digest=report["warmup_indices_digest"],
                     measured_indices_digest=report["measured_indices_digest"],
                     prediction_indices_digest=report["prediction_indices_digest"])
        report["trials"].append(trial)
        data.write_json(path, report)
    eligible = [trial for trial in report["trials"] if trial["status"] == "success"]
    if not eligible:
        report["status"] = "no_safe_candidate"
        data.write_json(path, report)
        raise RuntimeError(f"No safe {kind} batch candidate; inspect {path.name}")
    selected = min(eligible, key=lambda trial: (trial["seconds_per_example"], trial["batch_size"]))
    report.update({key: value for key, value in selected.items() if key != "status"})
    report["status"] = "selected"
    report["probe_examples"] = MEASURED_EXAMPLES
    selected_settings = dict(settings)
    selected_settings[kind + "_batch_size"] = selected["batch_size"]
    selected_settings[kind + "_gradient_accumulation"] = selected["gradient_accumulation"]
    data.write_json(path, report)
    print(f"{kind}: selected batch {selected['batch_size']}; {selected['examples_per_second']:.2f} examples/s", flush=True)
    return selected_settings, report
