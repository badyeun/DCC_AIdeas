"""Ordered Training audit, weighted CE trials, and matched contrastive ablations."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import shutil
import time
import uuid

import numpy as np

from . import data, runtime
from .audit import write_audit


NAUSEA = data.TARGET_SYMPTOMS[6]
POLICY = "Training Macro F1 first, with no regression in baseline Macro F1 or nausea F1"
BASELINE = "stage2_ce_w1"


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(root, relative):
    candidate = root / relative
    path = candidate.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Saved artifact path escapes its run directory")
    if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root.parent):
        raise ValueError("Symlink artifacts cannot be resumed")
    return path


def _hashes(root, manifest, extra=()):
    weights = _safe(root, manifest["weights"])
    assets = _safe(root, manifest["assets"])
    if not weights.is_file() or not assets.is_dir():
        raise FileNotFoundError("Completed model is missing weights or assets")
    files = [weights, *sorted(p for p in assets.rglob("*") if p.is_file()),
             *(_safe(root, name) for name in extra)]
    if not any(path.parent.name == "tokenizer" for path in files):
        raise FileNotFoundError("Completed model has no tokenizer assets")
    return {str(path.relative_to(root.resolve())).replace("\\", "/"): _hash(path) for path in files}


def _verify_hashes(root, hashes):
    if not hashes:
        raise ValueError("A completion record must contain artifact hashes")
    for name, expected in hashes.items():
        path = _safe(root, name)
        if not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Saved artifact is missing or changed: {name}")


def validate_settings(settings):
    settings = dict(settings)
    if settings.get("version") != "ver14" or settings.get("backend") != "neural":
        raise ValueError("This runner supports ver14 neural experiments only")
    weights = settings.get("nausea_weight_candidates", [1.0, 2.0, 3.0])
    if (not isinstance(weights, list) or not weights or weights[0] != 1.0
            or any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 1 for w in weights)
            or len(weights) != len(set(weights))):
        raise ValueError("nausea_weight_candidates must be distinct finite weights >=1, starting with 1")
    settings["nausea_weight_candidates"] = [float(w) for w in weights]
    for key in ("batch_size", "gradient_accumulation", "stage3_batch_size", "stage3_gradient_accumulation",
                "max_epochs", "patience", "encoder_window_batch", "max_length", "log_every"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["batch_size"] * settings["gradient_accumulation"] != (
            settings["stage3_batch_size"] * settings["stage3_gradient_accumulation"]):
        raise ValueError("Both stages must keep the same optimizer effective batch size")
    if settings.get("refit_full_training") is not True:
        raise ValueError("ver14 always refits its one selected candidate on all Training")
    if type(settings.get("seed")) is not int or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < settings.get("holdout_size", 0) < 1:
        raise ValueError("holdout_size must lie strictly between zero and one")
    for key in ("learning_rate", "head_learning_rate"):
        if (isinstance(settings.get(key), bool) or not isinstance(settings.get(key), Real)
                or not math.isfinite(settings[key]) or settings[key] <= 0):
            raise ValueError(f"{key} must be positive and finite")
    if type(settings.get("overlap")) is not int or not 0 <= settings["overlap"] < settings["max_length"] - 2:
        raise ValueError("overlap must fit within a token window")
    if not isinstance(settings.get("resume_run", ""), str):
        raise ValueError("resume_run must be an empty string or a prior ver14 run path")
    return settings


def _candidate(settings, name, stage, weight, attention, contrastive, batch, accumulation):
    return {**settings, "candidate_id": name, "stage": stage,
            "ce_views": 1 if stage == 2 else 2,
            "nausea_positive_weight": weight, "label_attention": attention,
            "contrastive_weight": contrastive, "batch_size": batch,
            "gradient_accumulation": accumulation, "refit_full_training": False,
            "run_validation": False, "resume_run": ""}


def stage2_plan(settings):
    return [_candidate(settings, "stage2_ce_w" + f"{weight:g}".replace(".", "p"), 2,
                       weight, False, 0.0, settings["batch_size"], settings["gradient_accumulation"])
            for weight in settings["nausea_weight_candidates"]]


def stage3_plan(settings, selected_weight):
    return [_candidate(settings, name, 3, selected_weight, attention, contrastive,
                       settings["stage3_batch_size"], settings["stage3_gradient_accumulation"])
            for name, attention, contrastive in (
                ("stage3_attention_off_ce", False, 0.0),
                ("stage3_attention_on_ce", True, 0.0),
                ("stage3_attention_on_supcon", True, 0.1))]


def select_candidate(records):
    """Select by Training only; a newly trained baseline remains an eligible control."""
    baseline = next((r for r in records if r["candidate_id"] == BASELINE), None)
    if baseline is None:
        raise ValueError("The unweighted Training baseline must be evaluated first")
    base_macro = baseline["metrics"]["macro_f1"]
    base_nausea = baseline["metrics"]["per_symptom"][NAUSEA]["f1"]
    eligible = []
    for record in records:
        metrics = record["metrics"]
        macro, nausea = metrics["macro_f1"], metrics["per_symptom"][NAUSEA]["f1"]
        if not all(isinstance(x, Real) and math.isfinite(x) and 0 <= x <= 1 for x in (macro, nausea)):
            raise ValueError("Candidate selection metrics must be finite F1 scores")
        if macro >= base_macro and nausea >= base_nausea:
            eligible.append(record)
    return max(eligible, key=lambda r: (r["metrics"]["macro_f1"], r["metrics"]["per_symptom"][NAUSEA]["f1"]))


def _manifest(value):
    return {**value, "format": "mission3_categorical_v1", "labels": list(data.TARGET_SYMPTOMS),
            "version": "ver14", "decision_rule": "categorical predict/argmax"}


def _training_settings(settings):
    return {k: v for k, v in settings.items() if k not in {
        "input_zip", "train_dir", "validation_dir", "work_dir", "output_dir", "resume_run", "log_every"}}


def _run_candidate(settings, paths, texts, targets, fit, holdout, root):
    name = settings["candidate_id"]
    candidate_root = root / "experiments" / name
    completed = candidate_root / "completed.json"
    signature = _digest(_training_settings(settings))
    if completed.is_file():
        record = _read(completed)
        if (record["settings_digest"] != signature or _digest(record["settings"]) != signature
                or record["settings"] != _training_settings(settings)
                or record["candidate_id"] != name or record["stage"] != settings["stage"]):
            raise ValueError("Resume candidate configuration mismatch")
        if Path(record["directory"]).parts[:2] != ("experiments", name):
            raise ValueError("Resume candidate directory mismatch")
        _verify_hashes(root, record["artifact_hashes"])
        local_manifest = _read(_safe(root, record["directory"] + "/candidate_model.ckpt"))
        expected_manifest = {**local_manifest,
            "weights": record["directory"] + "/" + local_manifest["weights"],
            "assets": record["directory"] + "/" + local_manifest["assets"]}
        if record["manifest"] != expected_manifest:
            raise ValueError("Resume candidate manifest mismatch")
        print(f"Reusing completed Training candidate: {name}", flush=True)
    else:
        attempt = candidate_root / _stamp()
        attempt.mkdir(parents=True, exist_ok=False)
        local = Path(settings["work_dir"]) / root.name / name / attempt.name
        local.mkdir(parents=True, exist_ok=False)
        data.write_json(attempt / "settings.json", settings)
        print(f"Stage {settings['stage']}: {name}; nausea positive weight={settings['nausea_positive_weight']:g}", flush=True)
        started = time.monotonic()
        manifest, metrics = runtime.train_neural(paths, texts, targets, fit, holdout, local, attempt, settings)
        manifest = _manifest(manifest)
        data.write_json(attempt / "candidate_model.ckpt", manifest)
        data.write_json(attempt / "training_metrics.json", metrics)
        relative = attempt.relative_to(root).as_posix()
        root_manifest = {**manifest, "weights": f"{relative}/{manifest['weights']}",
                         "assets": f"{relative}/{manifest['assets']}"}
        record = {"candidate_id": name, "stage": settings["stage"], "settings_digest": signature,
                  "settings": _training_settings(settings), "manifest": root_manifest,
                  "directory": relative, "metrics": metrics, "seconds": time.monotonic() - started}
        record["artifact_hashes"] = _hashes(root, root_manifest, (
            f"{relative}/holdout_predictions.npz", f"{relative}/selection_history.json",
            f"{relative}/candidate_model.ckpt", f"{relative}/training_metrics.json", f"{relative}/settings.json"))
        data.write_json(completed, record)
    with np.load(_safe(root, record["directory"] + "/holdout_predictions.npz"), allow_pickle=False) as values:
        if not np.array_equal(values["indices"], holdout) or not np.array_equal(values["targets"], targets[holdout]):
            raise ValueError("Cached candidate targets or split do not match Training")
        actual = data.score_predictions(targets[holdout], values["predictions"])
    if actual != record["metrics"]:
        raise ValueError("Cached candidate metrics do not match its saved discrete predictions")
    return record


def _write_table(root, records):
    fields = ["candidate", "stage", "nausea_positive_weight", "label_attention", "contrastive_weight",
              "batch_size", "ce_views", "selected_epochs", "macro_f1", "nausea_f1", "nausea_precision",
              "nausea_recall", "nausea_fp", "nausea_fn", "seconds"]
    temporary = root / "candidate_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            n = r["metrics"]["per_symptom"][NAUSEA]
            writer.writerow({"candidate": r["candidate_id"], "stage": r["stage"],
                             **{k: r["settings"][k] for k in fields[2:6]},
                             "batch_size": r["settings"]["batch_size"],
                             "ce_views": r["settings"]["ce_views"],
                             "selected_epochs": r["manifest"]["selected_epochs"],
                             "macro_f1": r["metrics"]["macro_f1"], "nausea_f1": n["f1"],
                             "nausea_precision": n["precision"], "nausea_recall": n["recall"],
                             "nausea_fp": n["false_positive"], "nausea_fn": n["false_negative"],
                             "seconds": r["seconds"]})
    temporary.replace(root / "candidate_summary.csv")


def _save_environment(root):
    import torch
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ("torch", "transformers", "numpy", "scikit-learn", "tokenizers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    data.write_json(root / ("environment_" + _stamp() + ".json"), {
        "python": platform.python_version(), "packages": packages, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})


def _final_model(settings, selected, texts, targets, root):
    completed = root / "final_completion.json"
    checkpoint = root / "mission3_ver14_model.ckpt"
    if completed.is_file():
        saved = _read(completed)
        if saved["candidate_id"] != selected["candidate_id"]:
            raise ValueError("Final candidate changed during resume")
        _verify_hashes(root, saved["artifact_hashes"])
        return checkpoint
    attempt = root / "final_model" / _stamp()
    attempt.mkdir(parents=True, exist_ok=False)
    final_settings = {**settings, **selected["settings"], "refit_full_training": True}
    data.write_json(attempt / "settings.json", final_settings)
    manifest, expected_probe = runtime.refit_selected(
        texts, targets, final_settings, selected["manifest"]["selected_epochs"], attempt)
    manifest = _manifest(manifest)
    local_checkpoint = attempt / "mission3_ver14_model.ckpt"
    data.write_json(local_checkpoint, manifest)
    actual_probe = runtime.predict_checkpoint(local_checkpoint, texts[:len(expected_probe)])
    if not np.array_equal(expected_probe, actual_probe):
        raise ValueError("Reloaded final model predictions differ from its in-memory model")
    relative = attempt.relative_to(root).as_posix()
    manifest["weights"] = f"{relative}/{manifest['weights']}"
    manifest["assets"] = f"{relative}/{manifest['assets']}"
    manifest["selected_candidate"] = selected["candidate_id"]
    data.write_json(checkpoint, manifest)
    data.write_json(root / "final_history.json", _read(attempt / "final_history.json"))
    data.write_json(completed, {"candidate_id": selected["candidate_id"],
                               "reloaded_probe_samples": len(expected_probe),
                               "artifact_hashes": _hashes(root, manifest, (checkpoint.name, "final_history.json"))})
    return checkpoint


def run(settings, runtime_sources=None):
    started = time.monotonic()
    settings = validate_settings(settings)
    sources = runtime_sources or {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.iterdir()
                                 if p.is_file() and p.suffix in (".py", ".md", ".txt")}
    sources = {k: v for k, v in sources.items() if not k.startswith("test_") and k != "build_notebook.py"}
    # Source and settings fingerprints make resuming completed candidates explicit and reproducible.
    source_hash = _digest({k: v for k, v in sources.items() if not k.endswith(".md")})
    settings_hash = _digest(_training_settings(settings))
    output_base = Path(settings["output_dir"]).expanduser().resolve()
    resume = settings.get("resume_run", "")
    root = Path(resume).expanduser().resolve() if resume else output_base / _stamp()
    if root == output_base or not root.is_relative_to(output_base):
        raise ValueError("resume_run must be a run directory inside output_dir")
    if resume and not (root / "run_identity.json").is_file():
        raise ValueError("resume_run is not an initialized ver14 run")
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    print("Result run:", root, flush=True)
    print("Resume a disconnected session by setting resume_run to this exact directory.", flush=True)
    can_write_status = not bool(resume)
    try:
        train_dir, validation_dir = runtime.prepare_data(settings["input_zip"], settings["work_dir"],
                                                        settings.get("train_dir", ""), settings.get("validation_dir", ""))
        paths, texts, targets = data.load_dataset(train_dir, include_targets=True)
        fit, holdout = data.make_split(texts, targets, settings["holdout_size"], settings["seed"])
        identity = {"version": "ver14", "training_digest": runtime.dataset_digest(texts, targets),
                    "source_digest": source_hash, "settings_digest": settings_hash,
                    "training_samples": len(texts), "fit_samples": len(fit), "holdout_samples": len(holdout),
                    "model_selection": "Training internal holdout only", "decision_rule": "categorical predict/argmax",
                    "selection_policy": POLICY}
        if resume:
            if _read(root / "run_identity.json") != identity:
                raise ValueError("Resume source, data, settings, or selection protocol changed")
            with np.load(root / "training_split.npz", allow_pickle=False) as split:
                if not np.array_equal(split["fit"], fit) or not np.array_equal(split["holdout"], holdout):
                    raise ValueError("Resume Training split changed")
            can_write_status = True
        else:
            np.savez_compressed(root / "training_split.npz", fit=fit, holdout=holdout)
            data.write_json(root / "run_identity.json", identity)
            data.write_json(root / "settings.json", settings)
            submission = root / "submission"
            submission.mkdir()
            for name, source in sources.items():
                if Path(name).name != name:
                    raise ValueError("Runtime source names must be flat files")
                (submission / name).write_text(source, encoding="utf-8")
        if (root / "run_complete.json").is_file():
            _verify_hashes(root, _read(root / "run_complete.json")["artifact_hashes"])
            print("This run is already complete; verified its final artifacts.", flush=True)
            return root
        _save_environment(root)
        audit_path = root / "audit_stage.json"
        if not audit_path.exists():
            print("Stage 1: Training evidence audit; original targets are retained.", flush=True)
            data.write_json(audit_path, write_audit(paths, texts, targets, fit, holdout, root / "audit"))
            data.write_json(root / "training_diagnostics.json", data.training_diagnostics(texts, targets))
        records = []
        plan = {"stage1": "Training evidence audit without relabeling or sample removal",
                "stage2": stage2_plan(settings), "stage3": "Matched ablations using the selected stage2 weight",
                "selection_policy": POLICY, "official_validation": "Only after final selection, refit and CSV save"}
        data.write_json(root / "experiment_plan.json", plan)
        for candidate in plan["stage2"]:
            records.append(_run_candidate(candidate, paths, texts, targets, fit, holdout, root))
            _write_table(root, records)
        stage2_winner = select_candidate(records)
        data.write_json(root / "stage2_selection.json", {"candidate_id": stage2_winner["candidate_id"],
                        "selection_policy": POLICY, "metrics": stage2_winner["metrics"]})
        plan["stage3"] = stage3_plan(settings, stage2_winner["settings"]["nausea_positive_weight"])
        data.write_json(root / "experiment_plan.json", plan)
        for candidate in plan["stage3"]:
            records.append(_run_candidate(candidate, paths, texts, targets, fit, holdout, root))
            _write_table(root, records)
        selected = select_candidate(records)
        data.write_json(root / "final_selection.json", {"candidate_id": selected["candidate_id"],
                        "selection_policy": POLICY, "metrics": selected["metrics"],
                        "selected_epochs": selected["manifest"]["selected_epochs"],
                        "baseline_selected": selected["candidate_id"] == BASELINE,
                        "candidate_settings": selected["settings"]})
        data.write_json(root / "training_metrics.json", selected["metrics"])
        saved = _safe(root, selected["directory"])
        shutil.copy2(saved / "holdout_predictions.npz", root / "holdout_predictions.npz")
        with np.load(root / "holdout_predictions.npz", allow_pickle=False) as values:
            data.export_errors([paths[i] for i in holdout], [texts[i] for i in holdout], targets[holdout],
                               values["predictions"], root / "training_errors")
        print("Selected from Training:", selected["candidate_id"], flush=True)
        checkpoint = _final_model(settings, selected, texts, targets, root)
        completion = _read(root / "final_completion.json")["artifact_hashes"]
        if settings.get("run_validation", True):
            validation_paths, predicted = runtime.inference(checkpoint, validation_dir, root / "mission3.csv")
            # This is the first use of official targets. They never affect model/epoch/weight selection.
            scored_paths, _, truth = data.load_dataset(validation_dir, include_targets=True)
            if [p.name for p in scored_paths] != [p.name for p in validation_paths]:
                raise ValueError("Official evaluation row order mismatch")
            metrics = data.score_predictions(truth, predicted)
            data.write_json(root / "validation_metrics.json", metrics)
            for name in ("mission3.csv", "validation_metrics.json"):
                completion[name] = _hash(root / name)
            print(f"Final Validation Macro F1={metrics['macro_f1']:.6f}; nausea F1={metrics['per_symptom'][NAUSEA]['f1']:.6f}", flush=True)
        data.write_json(root / "runtime.json", {"status": "complete", "seconds": time.monotonic() - started,
                        "seconds_scope": "current invocation; candidate times are in candidate_summary.csv",
                        "candidate_count": len(records), "selected_candidate": selected["candidate_id"]})
        data.write_json(root / "run_complete.json", {"artifact_hashes": completion})
    except BaseException as error:
        if can_write_status:
            data.write_json(root / "runtime.json", {"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                            "seconds": time.monotonic() - started, "error_type": type(error).__name__})
        raise
    return root
