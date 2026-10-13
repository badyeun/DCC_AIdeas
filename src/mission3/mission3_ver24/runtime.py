"""Budgeted, matched encoder comparison using untouched Training labels."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from . import data, trainer
from .bootstrap import prepare_data
from .budget import Budget, BudgetExceeded

ORDER = ("medical", "control")
NAUSEA = data.TARGET_SYMPTOMS.index("오심")


class _TrainingBudget:
    """Keep internal inference out of the final export/evaluation reserve."""
    def __init__(self, budget):
        self.budget = budget

    def check(self, stage, training=False):
        return self.budget.check(stage, training=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_settings(settings):
    result = dict(settings)
    pinned = {"version": "ver24", "seed": 42, "holdout_size": .2, "max_length": 512,
              "learning_rate": 2e-5, "weight_decay": .01, "dropout": .1, "warmup_ratio": .1}
    for key, value in pinned.items():
        if type(result.get(key)) is not type(value) or result[key] != value:
            raise ValueError("Pinned experiment setting changed: " + key)
    for key in ("batch_size", "gradient_accumulation", "max_epochs", "pilot_optimizer_steps", "log_every"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if result["max_epochs"] > 3 or result["pilot_optimizer_steps"] < 10:
        raise ValueError("Use at most three epochs and at least ten measured pilot updates")
    for key in ("time_budget_minutes", "finalization_reserve_seconds", "budget_safety_factor"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    if (result["time_budget_minutes"] > 180 or result["budget_safety_factor"] < 1.25
            or result["finalization_reserve_seconds"] < 600
            or result["finalization_reserve_seconds"] >= result["time_budget_minutes"] * 60):
        raise ValueError("Use at most 180 minutes, sufficient export reserve, and a conservative safety factor")
    for key in ("allow_cpu", "run_validation"):
        if type(result.get(key)) is not bool:
            raise ValueError(key + " must be boolean")
    for key in ("input_zip", "output_dir", "work_dir", "train_dir", "validation_dir"):
        if not isinstance(result.get(key), str):
            raise ValueError(key + " must be a path string")
    if any(any(word in key.lower() for word in ("threshold", "calibration", "logit_shift", "label_mask"))
           for key in result):
        raise ValueError("Post-hoc decision changes and label masks are not supported")
    return result


def dataset_digest(texts, targets):
    digest = hashlib.sha256()
    for text, labels in zip(texts, targets):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _hash(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            if budget:
                budget.check("artifact hash")
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device(settings):
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if not settings["allow_cpu"]:
        raise RuntimeError("Colab에서 GPU 런타임을 선택하세요")
    return "cpu"


def _create(spec, settings, device):
    model, tokenizer = trainer.create_model(spec["model_id"], spec["revision"], seed=settings["seed"],
        cache_dir=str(Path(settings["work_dir"]) / "hf_cache"), dropout=settings["dropout"])
    model.to(device)
    return model, tokenizer


def _loader(tokenizer, texts, targets, indices, settings, *, shuffle=False, device="cpu"):
    return trainer.make_loader(tokenizer, texts, targets=targets, indices=indices,
        batch_size=settings["batch_size"], max_length=settings["max_length"], shuffle=shuffle,
        seed=settings["seed"], pin_memory=device == "cuda")


def _optimizer(model, loader, epochs, settings, device):
    optimizer = trainer.build_optimizer(model, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    updates = math.ceil(len(loader) / settings["gradient_accumulation"]) * epochs
    scheduler = trainer.build_scheduler(optimizer, total_updates=updates, warmup_ratio=settings["warmup_ratio"])
    return optimizer, scheduler, trainer.make_scaler(device)


def _progress_callback(name, every):
    def report(progress):
        if progress["successful_updates"] % every == 0:
            print(f"{name}: {progress['examples']} rows, {progress['elapsed_seconds'] / 60:.1f} minutes", flush=True)
    return report


def plan_epochs(probes, fit_count, full_count, holdout_count, settings, budget):
    """Reserve BOTH fits, worst-case full refit, setup, and final export."""
    batch = settings["batch_size"]
    accum = settings["gradient_accumulation"]
    fit_updates = math.ceil(math.ceil(fit_count / batch) / accum)
    full_updates = math.ceil(math.ceil(full_count / batch) / accum)
    estimates = {}
    for name in ORDER:
        probe = probes[name]
        step = probe["seconds_per_update"]
        predict = probe["seconds_per_prediction"]
        if not all(math.isfinite(v) and v > 0 for v in (step, predict)):
            raise ValueError("Pilot timing must be finite and positive")
        estimates[name] = {"fit_epoch_seconds": step * fit_updates,
            "holdout_seconds": predict * holdout_count, "full_epoch_seconds": step * full_updates}
    for epochs in range(settings["max_epochs"], 0, -1):
        fits = sum(row["fit_epoch_seconds"] + row["holdout_seconds"] for row in estimates.values()) * epochs
        refit = max(row["full_epoch_seconds"] for row in estimates.values()) * epochs
        # Include repeated model loading/tokenization/checkpoint startup before the export reserve.
        requested = fits + refit + 240
        if budget.can_start(requested, training=True):
            return {"common_epochs": epochs, "estimates": estimates,
                    "estimated_work_seconds": requested, "remaining_seconds": budget.remaining(),
                    "safety_factor": budget.safety_factor,
                    "finalization_reserve_seconds": budget.finalization_reserve_s}
    raise BudgetExceeded("insufficient time for one matched epoch plus full Training refit")


def choose_candidate(records):
    control, medical = records["control"], records["medical"]
    better = (medical["macro_f1"] > control["macro_f1"]
              and medical["nausea_f1"] >= control["nausea_f1"])
    selected = "medical" if better else "control"
    return {"selected_candidate": selected, "selected_epochs": records[selected]["selected_epochs"],
        "medical_improved": better,
        "rule": "medical macro F1 > control and medical nausea F1 >= control; otherwise control",
        "selection_data": "Training internal holdout only",
        "macro_f1_delta": medical["macro_f1"] - control["macro_f1"],
        "nausea_f1_delta": medical["nausea_f1"] - control["nausea_f1"],
        "historical_best_comparison": "not evaluated here; control is a fresh single KLUE model, not the eleven-model ensemble"}


def _probe(spec, texts, targets, fit, holdout, settings, device, budget):
    budget.check("load pilot", training=True)
    model, tokenizer = _create(spec, settings, device)
    loader = _loader(tokenizer, texts, targets, fit, settings, shuffle=True, device=device)
    validation = _loader(tokenizer, texts, None, holdout, settings, device=device)
    optimizer, scheduler, scaler = _optimizer(model, loader, settings["max_epochs"], settings, device)
    stats = trainer.train_epoch(model, loader, optimizer, device=device,
        pos_weight=trainer.positive_weights(targets[fit]), accumulation_steps=settings["gradient_accumulation"],
        scheduler=scheduler, scaler=scaler, budget=budget, max_updates=settings["pilot_optimizer_steps"] + 1)
    measured = stats["measured_update_seconds"]
    if len(measured) < settings["pilot_optimizer_steps"]:
        raise ValueError("Pilot requires at least eleven successful optimizer updates")
    prediction = trainer.predict_logits(model, validation, device=device, budget=_TrainingBudget(budget),
        max_batches=32, return_stats=True)
    result = {"successful_updates": stats["successful_updates"], "measured_updates": len(measured),
        "seconds_per_update": float(max(np.mean(measured), np.percentile(measured, 75))),
        "seconds_per_prediction": float(prediction["seconds_per_example"]),
        "probe_examples": stats["examples"], "prediction_examples": prediction["examples"],
        "pilot_weights_discarded": True}
    del model, tokenizer, loader, validation, optimizer, scheduler, scaler
    _release()
    return result


def _fit_candidate(name, spec, texts, targets, fit, holdout, settings, plan, device, budget, root):
    model, tokenizer = _create(spec, settings, device)
    loader = _loader(tokenizer, texts, targets, fit, settings, shuffle=True, device=device)
    validation = _loader(tokenizer, texts, None, holdout, settings, device=device)
    optimizer, scheduler, scaler = _optimizer(model, loader, plan["common_epochs"], settings, device)
    weights = trainer.positive_weights(targets[fit])
    best, best_logits, history = None, None, []
    for epoch in range(1, plan["common_epochs"] + 1):
        print(f"{name}: internal fit epoch {epoch}/{plan['common_epochs']}", flush=True)
        stats = trainer.train_epoch(model, loader, optimizer, device=device, pos_weight=weights,
            accumulation_steps=settings["gradient_accumulation"], scheduler=scheduler, scaler=scaler, budget=budget,
            on_update=_progress_callback(name, settings["log_every"]))
        if not stats["completed_epoch"]:
            raise RuntimeError("Candidate epoch did not finish")
        logits = trainer.predict_logits(model, validation, device=device, budget=_TrainingBudget(budget))
        metrics = data.score_predictions(targets[holdout], trainer.fixed_decisions(logits))
        record = {"candidate": name, "macro_f1": metrics["macro_f1"],
            "nausea_f1": metrics["per_symptom"]["오심"]["f1"], "selected_epochs": epoch,
            "train_seconds": stats["elapsed_seconds"], "stop_reason": "completed_matched_epochs"}
        history.append({**record, "metrics": metrics})
        if best is None or record["macro_f1"] > best["macro_f1"]:
            best, best_logits = record, logits.copy()
            data.write_json(root / (name + "_training_metrics.json"), metrics)
            _save_npz(root / (name + "_holdout_predictions.npz"), indices=holdout, targets=targets[holdout],
                      logits=best_logits, predictions=trainer.fixed_decisions(best_logits))
        data.write_json(root / (name + "_history.json"), history)
        print(f"{name}: Macro F1={record['macro_f1']:.6f}, nausea F1={record['nausea_f1']:.6f}", flush=True)
    del model, tokenizer, loader, validation, optimizer, scheduler, scaler
    _release()
    best["positive_weights"] = weights.tolist()
    best["actual_epoch_seconds"] = max(row["train_seconds"] for row in history)
    return best, best_logits


def _save_comparison(root, records, logits, targets, holdout):
    columns = ["candidate", "macro_f1", "nausea_f1", "selected_epochs", "stop_reason"]
    with (root / "candidate_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: records[name][key] for key in columns} for name in ORDER)
    before, after = [trainer.fixed_decisions(logits[name]) for name in ("control", "medical")]
    per_label = {}
    truth = targets[holdout]
    for j, name in enumerate(data.TARGET_SYMPTOMS):
        per_label[name] = {"fn_to_tp": int(((truth[:, j] == 1) & (before[:, j] == 0) & (after[:, j] == 1)).sum()),
            "tp_to_fn": int(((truth[:, j] == 1) & (before[:, j] == 1) & (after[:, j] == 0)).sum()),
            "fp_to_tn": int(((truth[:, j] == 0) & (before[:, j] == 1) & (after[:, j] == 0)).sum()),
            "tn_to_fp": int(((truth[:, j] == 0) & (before[:, j] == 0) & (after[:, j] == 1)).sum())}
    data.write_json(root / "paired_error_changes.json", {"direction": "control to medical", "per_symptom": per_label})
    probabilities = {name: 1 / (1 + np.exp(-np.clip(logits[name][:, NAUSEA], -80, 80))) for name in ORDER}
    disagreement = np.minimum(abs(probabilities["medical"] - truth[:, NAUSEA]),
                              abs(probabilities["control"] - truth[:, NAUSEA]))
    # Models never saw these holdout rows during fitting. Ranking is for review only.
    indices = np.argsort(-disagreement, kind="stable")[:200]
    with (root / "training_review_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["training_row_index", "original_nausea_label", "medical_probability", "control_probability",
                         "both_disagree_with_original", "review_priority"])
        for i in indices:
            writer.writerow([int(holdout[i]), int(truth[i, NAUSEA]), float(probabilities["medical"][i]),
                float(probabilities["control"][i]), bool(before[i, NAUSEA] != truth[i, NAUSEA]
                and after[i, NAUSEA] != truth[i, NAUSEA]), float(disagreement[i])])
    data.write_json(root / "review_note.json", {"source": "Training internal holdout; zero-based sorted Training JSON row index",
        "labels_changed": 0, "samples_removed": 0, "label_errors_confirmed": 0,
        "warning": "Agreement against a provided label is not proof of label error. Review dialogue evidence manually. Do not use this file as a training mask."})


def _refit(spec, selection, record, texts, targets, settings, plan, device, budget, root):
    epochs = selection["selected_epochs"]
    estimate = max(plan["estimates"][selection["selected_candidate"]]["full_epoch_seconds"],
                   record["actual_epoch_seconds"] * len(texts) / record["fitted_samples"]) * epochs + 60
    if not budget.can_start(estimate, training=True):
        raise BudgetExceeded("insufficient remaining time for full Training refit")
    model, tokenizer = _create(spec, settings, device)
    loader = _loader(tokenizer, texts, targets, None, settings, shuffle=True, device=device)
    # Preserve the selected fit's relative LR schedule; stopping epoch may be earlier than the horizon.
    optimizer, scheduler, scaler = _optimizer(model, loader, plan["common_epochs"], settings, device)
    weights = trainer.positive_weights(targets)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"Full Training refit epoch {epoch}/{epochs}", flush=True)
        stats = trainer.train_epoch(model, loader, optimizer, device=device, pos_weight=weights,
            accumulation_steps=settings["gradient_accumulation"], scheduler=scheduler, scaler=scaler, budget=budget,
            on_update=_progress_callback("full refit", settings["log_every"]))
        if not stats["completed_epoch"]:
            raise RuntimeError("Full Training epoch did not finish")
        history.append({"epoch": epoch, "examples": stats["examples"], "seconds": stats["elapsed_seconds"]})
        data.write_json(root / "final_history.json", history)
    metadata = {**selection, "version": "ver24", "model_spec": spec, "training_scope": "all_training",
        "fitted_samples": len(texts), "scheduler_horizon_epochs": plan["common_epochs"],
        "positive_weights": weights.tolist(), "max_length": settings["max_length"],
        "batch_size": settings["batch_size"], "seed": settings["seed"],
        "decision": "fixed binary logits >= 0; no threshold search, calibration, or forced positives",
        "input": "utterances[].text only; NFC whitespace normalization; first 512 tokenizer tokens",
        "target_symptoms": list(data.TARGET_SYMPTOMS), "training_dataset_digest": dataset_digest(texts, targets)}
    budget.check("export final model")
    trainer.export_model(root / "final_model", model, tokenizer, metadata)
    # Compare the exact emitted checkpoint against the in-memory model, then use only the reload for Validation.
    sample_texts = [texts[index] for index in np.linspace(0, len(texts) - 1, min(16, len(texts)), dtype=int)]
    sample_loader = _loader(tokenizer, sample_texts, None, None, settings, device=device)
    expected = trainer.predict_logits(model, sample_loader, device=device, budget=budget)
    del model, tokenizer, loader, sample_loader, optimizer, scheduler, scaler
    _release()
    loaded, tokenizer, saved = trainer.load_export(root / "final_model", device=device)
    loaded_loader = _loader(tokenizer, sample_texts, None, None, settings, device=device)
    actual = trainer.predict_logits(loaded, loaded_loader, device=device, budget=budget)
    if (not np.allclose(expected, actual, rtol=1e-4, atol=1e-4)
            or not np.array_equal(trainer.fixed_decisions(expected), trainer.fixed_decisions(actual))):
        raise ValueError("Exported model does not reproduce the saved model predictions")
    data.write_json(root / "portable_verification.json", {"samples": len(sample_texts), "decisions_equal": True,
        "max_abs_logit_difference": float(abs(expected - actual).max()), "local_files_only": True})
    return loaded, tokenizer, metadata


def run(settings, runtime_sources=None):
    settings = validate_settings(settings)
    started = time.monotonic()
    deadline = min(float(settings.get("deadline_monotonic", started + 180 * 60)),
                   started + settings["time_budget_minutes"] * 60)
    budget = Budget(deadline, settings["finalization_reserve_seconds"], settings["budget_safety_factor"])
    device = _device(settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    root = Path(settings["output_dir"]).expanduser().resolve() / stamp
    root.mkdir(parents=True, exist_ok=False)
    stage = "prepare_data"
    data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
    data.write_json(root / "settings.json", settings)
    try:
        budget.check(stage, training=True)
        train_dir, validation_dir = prepare_data(settings["input_zip"], settings["work_dir"],
            train_dir=settings["train_dir"], validation_dir=settings["validation_dir"])
        _, texts, targets = data.load_dataset(train_dir, include_targets=True)
        fit, holdout = data.make_split(texts, targets, settings["holdout_size"], settings["seed"])
        _save_npz(root / "split.npz", fit=fit, holdout=holdout)
        data.write_json(root / "training_data.json", {"samples": len(texts), "fit_samples": len(fit),
            "holdout_samples": len(holdout), "digest": dataset_digest(texts, targets),
            "labels_changed": 0, "samples_removed": 0})
        data.write_json(root / "training_diagnostics.json", data.training_diagnostics(texts, targets))
        specs = read_json(Path(__file__).with_name("model_specs.json"))
        data.write_json(root / "model_specs.json", specs)
        probes = {}
        for name in ORDER:
            stage = name + "_pilot"
            data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
            print(f"Measuring {name} GPU throughput; pilot weights will be discarded.", flush=True)
            probes[name] = _probe(specs[name], texts, targets, fit, holdout, settings, device, budget)
            data.write_json(root / "pilots.json", probes)
        plan = plan_epochs(probes, len(fit), len(texts), len(holdout), settings, budget)
        data.write_json(root / "budget_plan.json", plan)
        print(f"Budget permits {plan['common_epochs']} matched epoch(s) and a fresh full Training refit.", flush=True)
        records, logits = {}, {}
        for name in ORDER:
            stage = name + "_fit"
            data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
            records[name], logits[name] = _fit_candidate(name, specs[name], texts, targets, fit, holdout,
                settings, plan, device, budget, root)
            records[name]["fitted_samples"] = len(fit)
        stage = "selection"
        _save_comparison(root, records, logits, targets, holdout)
        selection = choose_candidate(records)
        selection["common_epochs"] = plan["common_epochs"]
        data.write_json(root / "final_selection.json", selection)
        selected = selection["selected_candidate"]
        data.write_json(root / "training_metrics.json", data.score_predictions(targets[holdout],
            trainer.fixed_decisions(logits[selected])))
        _save_npz(root / "holdout_predictions.npz", indices=holdout, targets=targets[holdout],
            logits=logits[selected], predictions=trainer.fixed_decisions(logits[selected]))
        stage = "full_training_refit"
        data.write_json(root / "runtime.json", {"status": "running", "stage": stage})
        model, tokenizer, metadata = _refit(specs[selected], selection, records[selected], texts, targets,
            settings, plan, device, budget, root)
        data.write_json(root / "mission3_ver24_model.ckpt", {"format": "mission3_ver24_pointer_v1",
            "model_directory": "final_model", **metadata})
        stage = "final_validation"
        if settings["run_validation"]:
            # Official Validation is opened only after selection and complete full Training refit.
            budget.check(stage)
            paths, validation_texts, validation_targets = data.load_dataset(validation_dir, include_targets=True)
            loader = _loader(tokenizer, validation_texts, None, None, settings, device=device)
            final_logits = trainer.predict_logits(model, loader, device=device, budget=budget)
            predictions = trainer.fixed_decisions(final_logits)
            data.write_predictions(paths, predictions, root / "mission3.csv")
            data.write_json(root / "validation_metrics.json", data.score_predictions(validation_targets, predictions))
            _save_npz(root / "validation_predictions.npz", logits=final_logits,
                targets=validation_targets, predictions=predictions)
            del loader
        del model, tokenizer
        _release()
        stage = "finalize"
        sources = runtime_sources or {path.name: path.read_text(encoding="utf-8") for path in Path(__file__).parent.iterdir()
            if path.suffix in {".py", ".txt", ".json", ".md"} and not path.name.startswith("test_")}
        submission = root / "submission"
        submission.mkdir()
        for name, content in sources.items():
            if Path(name).name != name:
                raise ValueError("Unsafe runtime source name")
            budget.check(stage)
            (submission / name).write_text(content, encoding="utf-8")
        data.write_json(root / "runtime.json", {"status": "complete", "stage": "complete", "device": device,
            "elapsed_worker_seconds": time.monotonic() - started, "medical_improved": selection["medical_improved"]})
        hashes = {path.relative_to(root).as_posix(): _hash(path, budget)
                  for path in sorted(root.rglob("*")) if path.is_file()}
        budget.check("publish completion marker")
        data.write_json(root / "run_complete.json", {"version": "ver24", "status": "complete",
            "hashes": hashes, "training_scope": "all_training", "fitted_samples": len(texts),
            "validation_evaluated": settings["run_validation"]})
        return root
    except Exception as error:
        data.write_json(root / "runtime.json", {"status": "incomplete", "stage": stage,
            "error_type": type(error).__name__, "elapsed_worker_seconds": time.monotonic() - started})
        raise


def _model_directory(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_dir():
        return checkpoint
    manifest = read_json(checkpoint)
    if manifest.get("format") != "mission3_ver24_pointer_v1" or manifest.get("model_directory") != "final_model":
        raise ValueError("Unrecognized ver24 model pointer")
    result = checkpoint.parent / "final_model"
    if result.is_symlink() or not result.resolve().is_relative_to(checkpoint.parent):
        raise ValueError("Model directory escapes the result folder")
    return result


def inference(checkpoint, label_dir, output):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, metadata = trainer.load_export(_model_directory(checkpoint), device=device)
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    loader = trainer.make_loader(tokenizer, texts, batch_size=metadata["batch_size"], max_length=metadata["max_length"])
    logits = trainer.predict_logits(model, loader, device=device)
    data.write_predictions(paths, trainer.fixed_decisions(logits), output)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    inference(args.checkpoint, args.label_dir, args.output)


if __name__ == "__main__":
    main()
