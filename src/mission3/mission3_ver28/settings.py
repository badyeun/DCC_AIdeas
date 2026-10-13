"""Explicit, reproducible experiment settings; paths never select a recipe."""
from __future__ import annotations

import math


def default_settings():
    return {
        "version": "ver28", "run_mode": "screening", "candidate": "kure",
        "source_run": "", "resume_run": "", "input_zip": "",
        "input_root": "/kaggle/input", "train_dir": "", "validation_dir": "",
        "output_dir": "/kaggle/working/results/mission3/ver28",
        "work_dir": "/kaggle/temp/mission3_ver28", "cache_dir": "/kaggle/temp/hf_ver28",
        "run_validation": False, "seed": 42, "max_epochs": 3,
        "kure_max_length": 1024, "control_max_length": 512, "qwen_embedding_max_length": 1024,
        "kure_lr": 2e-5, "control_lr": 2e-5, "qwen_embedding_lr": 2e-5,
        "head_lr": 1e-4, "weight_decay": .01, "warmup_ratio": .05,
        "effective_batch_size": 16, "batch_candidates": [1, 2, 4, 8, 16],
        "benchmark_examples": 64, "checkpoint_every_updates": 100, "log_every": 50,
        "gradient_clip": 1., "dropout": .1, "allow_cpu": False,
        "time_budget_minutes": 480, "finalization_reserve_seconds": 1500,
        "budget_safety_factor": 1.35, "bootstrap_iterations": 500,
    }


RUNTIME_KEYS = {"source_run", "resume_run", "input_zip", "input_root", "train_dir", "validation_dir",
                "output_dir", "work_dir", "cache_dir", "run_mode", "run_validation",
                "time_budget_minutes", "finalization_reserve_seconds", "budget_safety_factor",
                "deadline_monotonic", "invocation_started_monotonic", "log_every", "allow_cpu"}


def recipe_settings(settings):
    return {key: value for key, value in settings.items() if key not in RUNTIME_KEYS}


def validate_settings(settings):
    defaults = default_settings()
    if not isinstance(settings, dict) or set(settings) - set(defaults) - {
            "deadline_monotonic", "invocation_started_monotonic"}:
        raise ValueError("Unknown ver28 settings")
    merged = {**defaults, **settings}
    for key, value in defaults.items():
        if type(merged[key]) is not type(value) and not (
                type(value) is float and type(merged[key]) in (int, float)):
            raise ValueError("Invalid setting type: " + key)
    if merged["run_mode"] not in ("screening", "full"):
        raise ValueError("run_mode must be screening or full")
    if merged["candidate"] not in ("kure", "qwen_embedding"):
        raise ValueError("candidate must be kure or qwen_embedding")
    if merged["run_validation"] != (merged["run_mode"] == "full"):
        raise ValueError("Official Validation is evaluated only in full mode")
    if merged["run_mode"] == "screening" and merged["source_run"]:
        raise ValueError("screening must not use a full-training source_run")
    if merged["run_mode"] == "full" and not (merged["source_run"] or merged["resume_run"]):
        raise ValueError("full requires a completed screening source_run or a full resume_run")
    for key, value in defaults.items():
        if key not in RUNTIME_KEYS | {"candidate"} and merged[key] != value:
            raise ValueError("Pinned experiment setting changed: " + key)
    if (not 30 <= merged["time_budget_minutes"] <= 480
            or not 1200 <= merged["finalization_reserve_seconds"] <= 1800
            or merged["finalization_reserve_seconds"] >= merged["time_budget_minutes"] * 60
            or not math.isfinite(merged["budget_safety_factor"])
            or merged["budget_safety_factor"] < 1.25 or merged["log_every"] < 1):
        raise ValueError("Invalid time budget or save reserve")
    for key in ("deadline_monotonic", "invocation_started_monotonic"):
        if key in merged and (type(merged[key]) not in (int, float)
                              or not math.isfinite(merged[key]) or merged[key] < 0):
            raise ValueError("Invalid invocation timestamp")
    return merged
