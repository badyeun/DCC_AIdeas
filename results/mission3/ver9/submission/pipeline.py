"""Training-only DeBERTa/NB-LR candidate and portable guarded baseline ensemble."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

if __package__:
    from . import assets, calibration, data, neural, pretrained, sparse
    from .budget import TimeBudget, BudgetExceeded
    from .prediction_cache import PredictionCache
    from .ver4_runtime import pipeline as base_pipeline, neural as base_neural
    from .ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, write_predictions
else:
    import assets, calibration, data, neural, pretrained, sparse
    from budget import TimeBudget, BudgetExceeded
    from prediction_cache import PredictionCache
    from ver4_runtime import pipeline as base_pipeline, neural as base_neural
    from ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, write_predictions

DEFAULT_CONFIG = {
    **neural.DEFAULTS, "seed": 42, "max_models": 2, "require_cuda": True,
    "min_calibration_gain": 0.002, "min_calibration_improved_folds": 3,
    "final_reserve_seconds": 1500, "calibration_reserve_seconds": 600,
    "keep_completed_resume_states": False,
}
CANDIDATE_NEURAL_WEIGHT = 0.75  # Declared before any calibration/Validation results.
MANIFEST = "mission3_ver9_model.ckpt"  # JSON ensemble manifest; weights are separate .ckpt files.
RUNTIME_FILES = (
    "__init__.py", "assets.py", "budget.py", "bootstrap.py", "calibration.py", "data.py",
    "model.py", "neural.py", "pipeline.py", "prediction_cache.py", "pretrained.py",
    "selection.py", "sparse.py", "train.py", "inference.py", "run.py",
    "requirements.txt", "README.md", "RESEARCH.md",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_code(output_dir):
    source, destination = Path(__file__).parent, Path(output_dir) / "submission"
    destination.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        shutil.copyfile(source / name, destination / name)
    for path in source.joinpath("ver4_runtime").iterdir():
        if path.is_file():
            target = destination / "ver4_runtime" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)


def _safe_file(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in str(name):
        raise ValueError("Model paths must remain inside the portable result folder")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or not result.exists():
        raise ValueError(f"Missing or unsafe model artifact: {name}")
    return result


def load_manifest(checkpoint_path):
    path = Path(checkpoint_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mission3_ver9" or manifest.get("labels") != list(TARGET_SYMPTOMS):
        raise ValueError("Unsupported ver9 manifest or label order")
    base_path = _safe_file(path.parent, manifest["base_manifest"])
    base = base_pipeline.load_manifest(base_path)
    required = {manifest["base_manifest"], *[str(Path(manifest["base_manifest"]).parent / n) for n in [*base["folds"], base["sparse"]]]}
    required.update(p.relative_to(path.parent).as_posix() for p in (base_path.parent / base["tokenizer"]).rglob("*") if p.is_file())
    if manifest["candidate_models"]:
        required.update(manifest["candidate_models"])
        required.add(manifest["candidate_sparse"])
        tokenizer = _safe_file(path.parent, manifest["tokenizer"])
        if not tokenizer.is_dir() or not (tokenizer / "tokenizer.json").is_file():
            raise ValueError("Missing candidate fast tokenizer")
        required.update(p.relative_to(path.parent).as_posix() for p in tokenizer.rglob("*") if p.is_file())
    elif any(manifest["selection"]["candidate_weights"]):
        raise ValueError("Selected candidate has no trained checkpoint")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("Manifest must cover all model artifacts with SHA256 checksums")
    for name, digest in hashes.items():
        if assets._sha256(_safe_file(path.parent, name)) != digest:
            raise ValueError(f"Model artifact checksum mismatch: {name}")
    return manifest


def _config(supplied):
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown ver9 options: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    for name in ("final_reserve_seconds", "calibration_reserve_seconds", "min_calibration_gain"):
        if isinstance(config[name], bool) or not isinstance(config[name], (int, float)) or not math.isfinite(config[name]):
            raise ValueError(f"{name} must be a finite number")
    for name in ("seed", "min_calibration_improved_folds"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(config["max_models"]) is not int or not 1 <= config["max_models"] <= 2:
        raise ValueError("max_models must be 1 or 2")
    if config["model_name"] != pretrained.MODEL_ID:
        raise ValueError("ver9 requires the pinned Korean DeBERTa-v3 model")
    if config["final_reserve_seconds"] < 1500 or config["calibration_reserve_seconds"] < 600:
        raise ValueError("Reserve at least 25 minutes for final inference and 10 for calibration")
    if config["min_calibration_gain"] < .002 or config["min_calibration_improved_folds"] < 3:
        raise ValueError("Do not relax the predeclared adoption guard")
    neural._settings(config)
    return config


def _identity(config, base_manifest, base_dir, pretrained_metadata):
    source = Path(__file__).parent
    code = {p.relative_to(source).as_posix(): assets._sha256(p)
            for p in [*[source / name for name in RUNTIME_FILES if name.endswith(".py")],
                      *sorted((source / "ver4_runtime").glob("*.py"))]}
    files = [*base_manifest["folds"], base_manifest["sparse"], "mission3_ver4_model.ckpt"]
    files += [p.relative_to(base_dir).as_posix() for p in (base_dir / base_manifest["tokenizer"]).rglob("*") if p.is_file()]
    hashes = {name: assets._sha256(base_dir / name) for name in files}
    identity = {"config": config, "base_signature": base_manifest["signature"], "base_artifact_sha256": hashes,
                "splits_sha256": assets._sha256(base_dir / "splits.json"),
                "calibration_sha256": assets._sha256(base_dir / "calibration_predictions.npz"),
                "pretrained": {k: pretrained_metadata[k] for k in ("model_id", "revision", "files")},
                "code": code, "versions": assets._versions(), "candidate_neural_weight": CANDIDATE_NEURAL_WEIGHT}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return signature, identity


def train_pipeline(train_label_dir, base_dir, output_dir, config=None, pretrained_dir=None,
                   start_epoch=None, deadline_epoch=None, attempt_marker=None):
    """No Validation path is accepted. All choices finish within Training partitions."""
    started = time.time()
    if (start_epoch is None) != (deadline_epoch is None):
        raise ValueError("Pass the shared start_epoch and deadline_epoch together")
    budget = TimeBudget(start_epoch=start_epoch, deadline_epoch=deadline_epoch)
    config = _config(config or {})
    if config["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Colab GPU runtime is required; CPU training would exceed the budget")
    base_dir, output_dir = Path(base_dir).resolve(), Path(output_dir).resolve()
    if output_dir == base_dir or output_dir.is_relative_to(base_dir) or base_dir.is_relative_to(output_dir):
        raise ValueError("Output must be separate from original ver4 inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pretrained.validate_pretrained(pretrained_dir)
    if metadata["model_id"] != pretrained.MODEL_ID or metadata["revision"] != pretrained.MODEL_REVISION:
        raise ValueError("Unexpected public pretrained model/revision")
    base, paths, texts, targets, fit, stop, cal, groups, base_probs, base_sparse = assets._read_base(train_label_dir, base_dir, 1)
    signature, identity = _identity(config, base, base_dir, metadata)
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text())["signature"] != signature:
            raise ValueError("Output contains a different experiment; choose a new output directory")
    else:
        if (output_dir / MANIFEST).exists():
            raise ValueError("Existing output lacks a verified run identity")
        write_json(identity_path, {"signature": signature, **identity})
    completed_path = output_dir / "training_metrics.json"
    if completed_path.exists() and json.loads(completed_path.read_text()).get("complete"):
        load_manifest(output_dir / MANIFEST)
        if attempt_marker is not None:
            write_json(attempt_marker, {"signature": signature, "output_dir": str(output_dir)})
        print("Completed ver9 run verified; reusing frozen model selection.", flush=True)
        return json.loads(completed_path.read_text())
    export_code(output_dir)
    budget.check(config["final_reserve_seconds"] + config["calibration_reserve_seconds"], phase="portable base preparation")
    assets._export_base(base_dir, output_dir, base, identity["base_artifact_sha256"])
    manifest = {"format": "mission3_ver9", "format_version": 1, "labels": list(TARGET_SYMPTOMS),
                "signature": signature, "config": config, "base_manifest": "base/mission3_ver4_model.ckpt",
                "selected_name": "ver4_baseline", "selection": {**base["calibrator"], "candidate_weights": [0.] * 9, "blend_space": "raw"},
                "candidate_models": [], "candidate_neural_weight": CANDIDATE_NEURAL_WEIGHT,
                "artifact_sha256": {"base/" + k: v for k, v in identity["base_artifact_sha256"].items()},
                "status": "baseline_ready", "versions": assets._versions()}
    # Atomic early fallback survives a later candidate timeout or exception.
    write_json(output_dir / MANIFEST, manifest)
    if attempt_marker is not None:
        write_json(attempt_marker, {"signature": signature, "output_dir": str(output_dir)})
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, use_fast=True, local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    documents = neural.tokenize_dialogues(data.candidate_texts(texts, tokenizer), tokenizer, config["max_length"], config["stride"])
    filenames = [paths[i].name for i in cal]
    cal_documents = [documents[i] for i in cal]
    splits = json.loads((base_dir / "splits.json").read_text(encoding="utf-8"))
    lookup = {path.name: i for i, path in enumerate(paths)}
    histories, candidate_models, cal_predictions = [], [], []
    training_cutoff = budget.deadline_epoch - config["final_reserve_seconds"] - config["calibration_reserve_seconds"] - 20
    first_duration = None
    for fold_number in range(1, config["max_models"] + 1):
        available = training_cutoff - time.time()
        # The optional second fold gets a full measured run allowance + 20% margin.
        if available <= 120 or (fold_number > 1 and (first_duration is None or available < first_duration * 1.2 + 120)):
            print(f"Fold {fold_number} skipped: preserving final inference/calibration time.", flush=True)
            break
        saved = splits["folds"][fold_number - 1]
        fit = np.asarray([lookup[name] for name in saved["fit"]])
        stop = np.asarray([lookup[name] for name in saved["stop"]])
        checkpoint = output_dir / f"candidate_{fold_number}.ckpt"
        fold_started = time.time()
        history = neural.train_model([documents[i] for i in fit], targets[fit], [documents[i] for i in stop], targets[stop],
                                     tokenizer, config, checkpoint, config["seed"] + fold_number * 1009,
                                     pretrained_dir, signature, stop_groups=groups[stop], deadline_epoch=training_cutoff)
        histories.append({"fold": fold_number, **history})
        write_json(output_dir / "candidate_history.json", histories)
        assets._release()
        if history["best_epoch"] > 0:
            prediction_config = {**config, "deadline_epoch": budget.deadline_epoch - config["final_reserve_seconds"] - 30}
            probabilities = assets._cached_predictions(output_dir / f"cal_candidate_{fold_number}.npz", signature, filenames,
                assets._sha256(checkpoint), lambda: neural.predict_checkpoint(checkpoint, cal_documents, tokenizer, prediction_config))
            cal_predictions.append(probabilities)
            candidate_models.append(checkpoint.name)
            assets._release()
        first_duration = time.time() - fold_started if first_duration is None else first_duration
        if history["stopped_for_budget"]:
            break
    if not candidate_models:
        report = {"complete": True, "status": "no_trained_candidate", "selected_name": "ver4_baseline", "histories": histories,
                  "training_seconds": time.time() - started, "reason": "No trained epoch improved the initial head, or the time budget was exhausted."}
        manifest["status"] = report["status"]
        write_json(output_dir / MANIFEST, manifest)
        write_json(completed_path, report)
        return report
    budget.check(config["final_reserve_seconds"] + 60, phase="NB-LR and Training calibration")
    noncal = np.setdiff1d(np.arange(len(paths)), cal)
    sparse_model = sparse.fit_model([texts[i] for i in noncal], targets[noncal], {"seed": config["seed"]})
    temporary = output_dir / "nb_ratio_lr.joblib.tmp"
    joblib.dump(sparse_model, temporary, compress=3)
    temporary.replace(output_dir / "nb_ratio_lr.joblib")
    sparse_probs = sparse.predict_model(sparse_model, [texts[i] for i in cal])
    candidate = CANDIDATE_NEURAL_WEIGHT * np.mean(cal_predictions, axis=0) + (1 - CANDIDATE_NEURAL_WEIGHT) * sparse_probs
    selection, comparison = calibration.compare_candidates(targets[cal], base_probs, base_sparse, candidate, groups[cal],
        seed=config["seed"] + 2, min_gain=config["min_calibration_gain"], min_improved_folds=config["min_calibration_improved_folds"])
    standalone, diagnostic = calibration.fit_candidate_only_crossfit(targets[cal], candidate, groups[cal], seed=config["seed"] + 2)
    # The fallback is the exact existing submitted ver4 calibrator, not a re-tuned variant.
    if comparison["selected_name"] == "ver4_baseline":
        selection = manifest["selection"]
    manifest.update(status="selection_frozen", selected_name=comparison["selected_name"], selection=selection,
                    candidate_models=candidate_models, candidate_sparse="nb_ratio_lr.joblib", tokenizer="tokenizer",
                    candidate_calibrator=standalone)
    for name in [*candidate_models, "nb_ratio_lr.joblib", *[p.relative_to(output_dir).as_posix() for p in (output_dir / "tokenizer").rglob("*") if p.is_file()]]:
        manifest["artifact_sha256"][name] = assets._sha256(output_dir / name)
    write_json(output_dir / "calibration_comparison.json", comparison)
    write_json(output_dir / "candidate_calibration_metrics.json", diagnostic)
    with (output_dir / "calibration_predictions.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, filenames=np.asarray(filenames), targets=targets[cal], candidate=candidate,
                            candidate_neural=np.mean(cal_predictions, axis=0), nb_ratio_lr=sparse_probs)
    (output_dir / "calibration_predictions.npz.tmp").replace(output_dir / "calibration_predictions.npz")
    write_json(output_dir / MANIFEST, manifest)
    report = {"complete": True, "status": "selection_frozen", "selected_name": comparison["selected_name"], "histories": histories,
              "candidate_models": candidate_models, "candidate_neural_weight": CANDIDATE_NEURAL_WEIGHT,
              "calibration": comparison, "candidate_only": diagnostic, "training_seconds": time.time() - started,
              "total_elapsed_seconds": time.time() - budget.start_epoch, "remaining_seconds": budget.remaining_seconds(),
              "official_validation_used_for_selection": False}
    write_json(completed_path, report)
    if not config["keep_completed_resume_states"]:
        for history in histories:
            # Only remove this experiment's optimizer states after final selection is durable.
            if history["completed"]:
                (output_dir / f"candidate_{history['fold']}_last.ckpt").unlink(missing_ok=True)
    return report


def _raw_base(root, base, paths, texts, config, cache_dir):
    tokenizer_dir = root / base["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    documents = base_neural.tokenize_dialogues(texts, tokenizer, config["max_length"], config["stride"])
    cache = PredictionCache(cache_dir, paths, texts, tokenizer_dir, TARGET_SYMPTOMS,
                            neural.execution_identity(config["device"])) if cache_dir else None
    sources = [Path(base_pipeline.__file__), Path(base_neural.__file__)]
    outputs = []
    for name in base["folds"]:
        print("Base prediction:", name, flush=True)
        compute = lambda name=name: base_neural.predict_checkpoint(root / name, documents, tokenizer, config)
        outputs.append(cache.get("ver4-neural", root / name, config, sources, compute) if cache else compute())
        assets._release()
    return np.mean(np.asarray(outputs, dtype=np.float64), axis=0), base_pipeline.predict_sparse(joblib.load(root / base["sparse"]), texts)


def predict_pipeline(checkpoint_path, label_dir, output, device=None, batch_size=None, cache_dir=None, candidate_output=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = load_manifest(checkpoint_path)
    root = checkpoint_path.parent
    base_path = _safe_file(root, manifest["base_manifest"])
    base = base_pipeline.load_manifest(base_path)
    assets._check_versions(base)
    config = {**base["config"], "device": device or ("cuda" if torch.cuda.is_available() else "cpu")}
    if batch_size is not None:
        config["inference_batch_size"] = batch_size
    # This loader ignores symptom targets and all metadata, even if present.
    paths, texts, _ = load_dataset(Path(label_dir), include_targets=False)
    base_probs, base_sparse = _raw_base(base_path.parent, base, paths, texts, config, cache_dir)
    candidate = None
    if any(manifest["selection"]["candidate_weights"]) or candidate_output:
        if not manifest["candidate_models"]:
            if any(manifest["selection"]["candidate_weights"]):
                raise ValueError("Selected candidate has no trained checkpoint")
            print("No trained candidate; standalone diagnostic skipped.", flush=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(_safe_file(root, manifest["tokenizer"]), use_fast=True, local_files_only=True)
            candidate_config = {**manifest["config"], "device": config["device"], "inference_batch_size": config["inference_batch_size"]}
            documents = neural.tokenize_dialogues(data.candidate_texts(texts, tokenizer), tokenizer,
                                                 candidate_config["max_length"], candidate_config["stride"])
            outputs = []
            for name in manifest["candidate_models"]:
                outputs.append(neural.predict_checkpoint(_safe_file(root, name), documents, tokenizer, candidate_config))
                assets._release()
            sparse_probs = sparse.predict_model(joblib.load(_safe_file(root, manifest["candidate_sparse"])), texts)
            weight = manifest["candidate_neural_weight"]
            candidate = weight * np.mean(outputs, axis=0) + (1 - weight) * sparse_probs
    probabilities = calibration.apply_selected(base_probs, base_sparse, candidate, manifest["selection"])
    # Publish complete CSVs atomically; a deadline cannot leave a truncated final CSV.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    predictions = write_predictions(paths, probabilities, manifest["selection"]["thresholds"], temporary)
    temporary.replace(output)
    if candidate_output and candidate is not None:
        destination = Path(candidate_output)
        temporary = destination.with_name(destination.name + ".tmp")
        write_predictions(paths, candidate, manifest["candidate_calibrator"]["thresholds"], temporary)
        temporary.replace(destination)
    print(f"Saved {len(paths):,} predictions: {output}", flush=True)
    return paths, probabilities, predictions


evaluate_csv = base_pipeline.evaluate_csv
