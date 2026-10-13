"""Kaggle screening, immutable selection, resumable refit and final evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import time
import traceback
import uuid

import numpy as np

from . import analysis, bootstrap, data, trainer
from .budget import Budget, BudgetExceeded
from .settings import recipe_settings, validate_settings

FORMAT = "mission3_ver28_v1"
EXPECTED_TRAINING_SAMPLES = 29200
EXPECTED_VALIDATION_SAMPLES = 3640
EXPECTED_DATA_DIGEST = "71b751de9931f10a6d5e2b17d38bf36d0f986baef82e47f7d98159a30348aa6f"
EXPECTED_SPLIT_DIGEST = "deac9096788901eb8652169e8f97c98f35b21b08f307f73751a253646b75efa7"
CORE_SOURCES = {"trainer.py", "runtime.py", "data.py", "analysis.py", "settings.py", "bootstrap.py", "budget.py"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def find_run(path):
    root = Path(path).expanduser().resolve()
    if (root / "run_state.json").is_file():
        return root
    matches = list(root.rglob("run_state.json")) if root.is_dir() else []
    if len(matches) != 1:
        raise ValueError("Specify the exact ver28 run folder containing run_state.json")
    return matches[0].parent


def write_manifest(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".tmp":
            if path.is_symlink():
                raise ValueError("Run artifacts must not contain symlinks")
            files[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": file_hash(path)}
    data.write_json(root / "artifact_manifest.json", {"format": FORMAT, "files": files})


def verify_run(root):
    manifest = read_json(root / "artifact_manifest.json")
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), dict):
        raise ValueError("Source run manifest is not ver28")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Source run must not contain symlinks")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".tmp"}
    if actual != set(manifest["files"]):
        raise ValueError("Source run manifest has missing or unexpected files")
    for name, metadata in manifest["files"].items():
        path = bootstrap._target(root, name)
        if not path.is_file() or path.stat().st_size != metadata["size"] or file_hash(path) != metadata["sha256"]:
            raise ValueError("Source run artifact integrity failed: " + name)
    for name in ("run_state.json", "settings.json"):
        if name not in manifest["files"]:
            raise ValueError("Source run manifest omits required metadata")
    state = read_json(root / "run_state.json")
    if state.get("format") != FORMAT:
        raise ValueError("Source run state is not ver28")
    if state["status"] in ("screening_complete", "complete"):
        for name in ("split_manifest.json", "selection.json", "resolved_models.json"):
            if name not in manifest["files"]:
                raise ValueError("Completed source run omits selection metadata")
    return state


def resolve_inputs(settings):
    if settings["train_dir"] or settings["validation_dir"]:
        return bootstrap.prepare_data("", settings["work_dir"], settings["train_dir"], settings["validation_dir"])
    explicit = Path(settings["input_zip"]).expanduser() if settings["input_zip"] else None
    if explicit and explicit.is_file():
        return bootstrap.prepare_data(explicit, settings["work_dir"])
    search_root = explicit or Path(settings["input_root"])
    if not search_root.is_dir():
        raise FileNotFoundError("Kaggle input not found; attach the private ver28 input Dataset")
    archives = list(search_root.rglob("mission3_ver28_inputs.zip"))
    if len(archives) == 1:
        return bootstrap.prepare_data(archives[0], settings["work_dir"])
    if len(archives) > 1:
        raise ValueError("Multiple ver28 input ZIPs found; set INPUT_ZIP explicitly")
    # Kaggle may unzip Dataset uploads automatically; verify those exact bytes too.
    candidates = []
    for marker in search_root.rglob("bundle_manifest.json"):
        manifest = read_json(marker)
        base = marker.parent
        train = base / bootstrap.TRAIN_PREFIX
        valid = base / bootstrap.VALID_PREFIX
        if manifest.get("format") == "mission3_ver5_inputs_v1" and train.is_dir() and valid.is_dir():
            candidates.append((base, train, valid, manifest))
    if len(candidates) != 1:
        raise ValueError("Attach exactly one ver28 input Dataset, or set INPUT_ZIP to its extracted folder")
    base, train, valid, manifest = candidates[0]
    selected = [row for row in manifest["files"] if row["path"].startswith((bootstrap.TRAIN_PREFIX, bootstrap.VALID_PREFIX))]
    expected = {row["path"] for row in selected}
    if len(expected) != len(selected):
        raise ValueError("Duplicate input manifest rows")
    actual = {path.relative_to(base).as_posix() for root in (train, valid) for path in root.rglob("*.json")}
    if actual != expected:
        raise ValueError("Extracted Dataset has missing or extra label JSON files")
    for row in selected:
        path = bootstrap._target(base.resolve(), row["path"])
        if path.stat().st_size != row["size"] or file_hash(path) != row["sha256"]:
            raise ValueError("Extracted Dataset content hash mismatch")
    return train, valid


def resolve_spec(kind, spec, settings, budget, output):
    from huggingface_hub import HfApi, snapshot_download
    budget.check("pretrained model download", training=True)
    spec = dict(spec)
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(spec["model_id"], revision=revision).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Pretrained revision was not resolved to an immutable commit")
    # Persist the resolved identity even if a long download is interrupted.
    spec.update(kind=kind, revision=revision)
    specs = read_json(output / "resolved_models.json") if (output / "resolved_models.json").exists() else {}
    specs[kind] = {key: value for key, value in spec.items() if key != "snapshot_path"}
    data.write_json(output / "resolved_models.json", specs)
    snapshot = snapshot_download(spec["model_id"], revision=revision, cache_dir=settings["cache_dir"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        ignore_patterns=["onnx/*", "openvino/*", "original/*"])
    spec["snapshot_path"] = str(snapshot)
    notices = output / "licenses" / kind
    notices.mkdir(parents=True, exist_ok=True)
    for path in Path(snapshot).iterdir():
        if path.is_file() and (path.name.startswith("LICENSE") or path.name == "README.md"):
            shutil.copyfile(path, notices / path.name)
    pooling = Path(snapshot) / "1_Pooling" / "config.json"
    if pooling.is_file():
        shutil.copyfile(pooling, notices / "upstream_pooling_config.json")
    data.write_json(notices / "provenance.json", specs[kind])
    budget.check("pretrained model ready", training=True)
    return spec


def choose_candidate(targets, logits, candidate):
    candidates = {"control": logits["control"], candidate: logits[candidate],
                  "mean": analysis.ensemble_logits(logits["control"], logits[candidate])}
    scores = {name: analysis.metrics(targets, values) for name, values in candidates.items()}
    # Prefer a simpler existing control for exact-score ties.
    order = ("control", candidate, "mean")
    winner = max(order, key=lambda name: scores[name]["macro_f1"])
    comparison = analysis.paired_bootstrap_ci(targets, logits["control"], candidates[winner])
    delta = comparison["macro_f1_delta"]
    return {"selected_recipe": winner, "members": ["control", candidate] if winner == "mean" else [winner],
            "scores": scores, "paired_comparison": comparison,
            "promising": delta["estimate"] >= .005 and delta["lower"] > 0,
            "decision_rule": "independent sigmoid >= 0.5; ensemble is fixed equal probability mean",
            "limitations": "Same-fit comparison. Development and confirmation have prior experiment exposure; "
                            "bootstrap excludes repeated selection and training-seed variation. Not an official 0.7 result."}, candidates


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _predict(root, kind, texts, ids, device, budget):
    batch_size = read_json(root / "models" / kind / "batch_benchmark.json")["selected_batch_size"]
    values = trainer.predict_checkpoint(root / "models" / kind / "best_model", texts, ids,
        device=device, batch_size=batch_size, budget=budget)
    _release()
    return analysis._logits(values, len(texts) if ids is None else len(ids))


def _finish(output, state, status, started, **extra):
    state.update(status=status, elapsed_seconds=time.monotonic() - started, **extra)
    data.write_json(output / "run_state.json", state)
    write_manifest(output)
    result = {"run_dir": str(output), "status": status, "submission_ready": status == "complete"}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _verify_source_recipe(source, settings, split_manifest):
    if recipe_settings(read_json(source / "settings.json")) != recipe_settings(settings):
        raise ValueError("Source and current experiment recipes differ")
    if ((source / "split_manifest.json").exists()
            and read_json(source / "split_manifest.json")["split_digest"] != split_manifest["split_digest"]):
        raise ValueError("Source and current Training split differ")


def _verify_reload(output, kind, texts, device, budget):
    root = output / "models" / kind
    with np.load(root / "reload_reference.npz", allow_pickle=False) as reference:
        indices, expected = reference["indices"], reference["logits"]
    observed = _predict(output, kind, texts, indices, device, budget)
    if not (np.allclose(observed, expected, rtol=1e-4, atol=1e-5)
            and np.array_equal(observed >= 0, expected >= 0)):
        raise ValueError("Saved model does not reproduce the live model reference")
    data.write_json(root / "reload_verification.json", {"passed": True, "samples": len(indices),
        "max_abs_logit_difference": float(np.max(np.abs(observed - expected))), "decisions_identical": True})


def run(settings, runtime_sources=None):
    import torch
    settings = validate_settings(settings)
    started = settings.get("invocation_started_monotonic", time.monotonic())
    deadline = settings.get("deadline_monotonic", started + settings["time_budget_minutes"] * 60)
    budget = Budget(deadline, settings["finalization_reserve_seconds"], settings["budget_safety_factor"])
    output = Path(settings["output_dir"]).expanduser().resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    state = {"format": FORMAT, "status": "starting", "run_mode": settings["run_mode"],
             "candidate": settings["candidate"], "submission_ready": False, "completed_models": {}}
    sources = runtime_sources or {}
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not settings["allow_cpu"]:
            raise RuntimeError("CUDA GPU required; choose Kaggle Accelerator GPU")
        resume = None
        if settings["resume_run"]:
            resume = find_run(settings["resume_run"])
            old_state = verify_run(resume)
            if old_state["status"] != "paused" or old_state["run_mode"] != settings["run_mode"]:
                raise ValueError("RESUME_RUN must be a paused ver28 run of the same mode")
            for name in CORE_SOURCES & sources.keys():
                if (resume / "source_snapshot" / name).read_text(encoding="utf-8") != sources[name]:
                    raise ValueError("Resume source code differs: " + name)
            shutil.copytree(resume, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp"))
            state = {**old_state, "status": "resuming", "resumed_from": str(resume)}
        data.write_json(output / "settings.json", settings)
        data.write_json(output / "run_state.json", state)
        for name, content in sources.items():
            path = output / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        packages = {}
        for name in ("torch", "transformers", "numpy", "scikit-learn", "scipy", "accelerate"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        data.write_json(output / "environment.json", {"packages": packages, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "gpu_count": torch.cuda.device_count(), "used_gpu_count": int(device == "cuda")})
        budget.check("data preparation", training=True)
        train_dir, valid_dir = resolve_inputs(settings)
        paths, texts, targets = data.load_dataset(train_dir, include_targets=True)
        if len(texts) != EXPECTED_TRAINING_SAMPLES or data.dataset_digest(texts, targets) != EXPECTED_DATA_DIGEST:
            raise ValueError("Original Training count or text/target digest does not match the verified dataset")
        fit, development, confirmation, split_manifest = data.three_way_split(texts, targets, seed=settings["seed"])
        if split_manifest["split_digest"] != EXPECTED_SPLIT_DIGEST:
            raise ValueError("Internal split differs from ver25-ver27")
        if resume:
            _verify_source_recipe(resume, settings, split_manifest)
        data.write_json(output / "split_manifest.json", split_manifest)
        save_npz(output / "split_indices.npz", fit=fit, development=development, confirmation=confirmation)
        specs = read_json(Path(__file__).with_name("model_specs.json"))
        if (output / "resolved_models.json").exists():
            specs.update(read_json(output / "resolved_models.json"))
        selection = None
        if settings["run_mode"] == "full":
            if resume and (output / "selection.json").is_file():
                selection = read_json(output / "selection.json")
            else:
                source_path = settings["source_run"] or (read_json(resume / "settings.json")["source_run"] if resume else "")
                if not source_path:
                    raise ValueError("Full preparation has no frozen selection yet; attach completed screening and set SOURCE_RUN")
                source = find_run(source_path)
                source_state = verify_run(source)
                if source_state["status"] != "screening_complete" or source_state["run_mode"] != "screening":
                    raise ValueError("SOURCE_RUN must be completed ver28 screening")
                _verify_source_recipe(source, settings, split_manifest)
                for name in CORE_SOURCES & sources.keys():
                    if (source / "source_snapshot" / name).read_text(encoding="utf-8") != sources[name]:
                        raise ValueError("Full source implementation differs from screening: " + name)
                selection = read_json(source / "selection.json")
                specs.update(read_json(source / "resolved_models.json"))
                data.write_json(output / "selection.json", selection)
                data.write_json(output / "source_screening.json", {"source_run": str(source),
                    "artifact_manifest_sha256": file_hash(source / "artifact_manifest.json")})
            members = selection["members"]
            if not members or set(members) - {"control", settings["candidate"]}:
                raise ValueError("Invalid selected model members")
            data.write_json(output / "resolved_models.json", {kind: specs[kind] for kind in members})
        else:
            members = ["control", settings["candidate"]]
        for kind in members:
            if kind in state["completed_models"]:
                continue
            budget.check("model training", training=True)
            spec = resolve_spec(kind, specs[kind], settings, budget, output)
            model_dir = output / "models" / kind
            epochs = settings["max_epochs"] if selection is None else selection["selected_epochs"][kind]
            if type(epochs) is not int or not 1 <= epochs <= settings["max_epochs"]:
                raise ValueError("Invalid frozen selected epoch")
            print(f"Training {kind}: rows={len(fit) if selection is None else len(texts)}, epochs={epochs}", flush=True)
            result = trainer.train_model(kind, spec, settings, texts, targets,
                fit if selection is None else np.arange(len(texts)), development if selection is None else None,
                model_dir, budget, epochs, resume=(model_dir / "training_identity.json").is_file())
            _release()
            if result["status"] == "paused":
                return _finish(output, state, "paused", started, pause_stage="training " + kind)
            if result["status"] != "complete":
                raise ValueError("Trainer returned an unexpected status")
            state["completed_models"][kind] = result
            data.write_json(output / "run_state.json", state)
        for kind in members:
            if not (output / "models" / kind / "reload_verification.json").exists():
                _verify_reload(output, kind, texts, device, budget)
        if selection is None:
            budget.check("development candidate comparison", training=True)
            logits = {}
            for kind in members:
                epoch = state["completed_models"][kind]["best_epoch"]
                logits[kind] = np.load(output / "models" / kind / f"epoch_{epoch}_development_logits.npy", allow_pickle=False)
            selection, candidates = choose_candidate(targets[development], logits, settings["candidate"])
            selection.update(format=FORMAT, candidate=settings["candidate"],
                selected_epochs={kind: int(state["completed_models"][kind]["best_epoch"]) for kind in members},
                split_digest=split_manifest["split_digest"], training_rows=len(fit),
                confirmation_policy="Report only; never revise this frozen selection. Already exposed in ver25.")
            # Freeze BEFORE looking at confirmation targets/predictions.
            data.write_json(output / "selection.json", selection)
            save_npz(output / "development_predictions.npz", targets=targets[development], indices=development,
                     **candidates)
            data.write_json(output / "paired_errors.json", analysis.paired_error_changes(
                targets[development], logits["control"], candidates[selection["selected_recipe"]]))
            budget.check("confirmation report", training=True)
            confirm = {kind: _predict(output, kind, texts, confirmation, device, budget)
                       for kind in set(selection["members"]) | {"control"}}
            chosen = (analysis.ensemble_logits(confirm["control"], confirm[settings["candidate"]])
                      if selection["selected_recipe"] == "mean" else confirm[selection["members"][0]])
            save_npz(output / "confirmation_predictions.npz", targets=targets[confirmation],
                     indices=confirmation, selected=chosen, control=confirm["control"])
            data.write_json(output / "confirmation_report.json", {
                "selected": analysis.metrics(targets[confirmation], chosen),
                "control": analysis.metrics(targets[confirmation], confirm["control"]),
                "paired_comparison": analysis.paired_bootstrap_ci(targets[confirmation], confirm["control"], chosen),
                "selection_changed": False, "prior_exposure": "Used in ver25; not an untouched test set."})
            data.write_json(output / "screening_complete.json", {"format": FORMAT, "submission_ready": False,
                "selection": selection["selected_recipe"], "next_action": "Review results, then run full separately."})
            return _finish(output, state, "screening_complete", started)
        # Frozen full-training recipe; official Validation cannot change it.
        budget.check("official Validation inference", training=True)
        valid_paths, valid_texts, valid_targets = data.load_dataset(valid_dir, include_targets=True)
        if len(valid_texts) != EXPECTED_VALIDATION_SAMPLES:
            raise ValueError("Official Validation sample count does not match the original dataset")
        predictions = {kind: _predict(output, kind, valid_texts, None, device, budget) for kind in members}
        final_logits = (analysis.ensemble_logits(predictions["control"], predictions[settings["candidate"]])
                        if len(members) == 2 else predictions[members[0]])
        data.write_predictions(valid_paths, final_logits >= 0, output / "validation_predictions.csv")
        save_npz(output / "validation_predictions.npz", targets=valid_targets, logits=final_logits,
                 filenames=np.asarray([path.name for path in valid_paths]))
        scores = analysis.metrics(valid_targets, final_logits)
        data.write_json(output / "validation_metrics.json", scores)
        data.write_json(output / "submission_manifest.json", {"format": FORMAT, "label_order": list(data.TARGET_SYMPTOMS),
            "members": {kind: "models/" + kind + "/best_model" for kind in members},
            "ensemble": "equal_probability_mean" if len(members) == 2 else "single",
            "decision_threshold": .5, "selected_epochs": selection["selected_epochs"],
            "training_samples": len(texts), "validation_macro_f1": scores["macro_f1"],
            "above_0_7": scores["macro_f1"] > .7, "selection_frozen_before_official_validation": True})
        data.write_json(output / "run_complete.json", {"format": FORMAT, "submission_ready": True,
            "macro_f1": scores["macro_f1"], "above_0_7": scores["macro_f1"] > .7})
        (output / "INFERENCE_README.md").write_text(
            "# ver28 inference\n\nKeep this complete run directory, including models, licenses and source_snapshot.\n"
            "From this directory, install source_snapshot/requirements.txt in a compatible PyTorch environment, then run:\n\n"
            "```bash\npython -m source_snapshot.inference --run-dir . --label-dir /path/to/label_jsons "
            "--output predictions.csv --device cuda --batch-size 1\n```\n\n"
            "Use --device cpu if CUDA is unavailable. Inference reads only utterance text, never symptom labels.\n"
            "The models and tokenizers load locally without Hugging Face access.\n", encoding="utf-8")
        return _finish(output, state, "complete", started, submission_ready=True)
    except BudgetExceeded as error:
        return _finish(output, state, "paused", started, pause_stage=error.stage)
    except Exception as error:
        # Raw exception values may contain source texts; retain code positions only.
        frames = [{"file": Path(frame.f_code.co_filename).name, "line": line,
                   "function": frame.f_code.co_name} for frame, line in traceback.walk_tb(error.__traceback__)]
        data.write_json(output / "error.json", {"error_type": type(error).__name__, "frames": frames[-15:]})
        for name in ("run_complete.json", "screening_complete.json"):
            marker = output / name
            if marker.is_file():
                marker.unlink()
        _finish(output, state, "failed", started, error_type=type(error).__name__)
        raise
