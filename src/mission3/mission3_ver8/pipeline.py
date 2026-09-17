"""Training-only error audit and matched A/B/C trials, then one-candidate calibration."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

if __package__:
    from . import audit, calibration, data, neural, pretrained, tapt
    from .selection import StopSelector, better
    from .prediction_cache import PredictionCache
    from .ver4_runtime import pipeline as base_pipeline
    from .ver4_runtime import neural as base_neural
    from .ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group, write_json, write_predictions
else:
    import audit
    import tapt
    import calibration
    import data
    import pretrained
    from selection import StopSelector, better
    import neural
    from prediction_cache import PredictionCache
    from ver4_runtime import pipeline as base_pipeline
    from ver4_runtime import neural as base_neural
    from ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group, write_json, write_predictions


DEFAULT_CONFIG = {
    **neural.DEFAULTS, **tapt.DEFAULTS, "seed": 42, "init_fold": 1, "require_cuda": True,
    "profiles": ["B"], "audit_per_error": 20, "keep_completed_resume_states": False,
    "bundle_base": False,
    "min_calibration_gain": 0.002, "min_calibration_improved_folds": 3,
}
RUNTIME_FILES = (
    "audit.py", "tapt.py",
    "__init__.py", "pipeline.py", "neural.py", "data.py", "selection.py", "pretrained.py",
    "calibration.py", "prediction_cache.py", "train.py", "inference.py",
    "bootstrap.py", "requirements.txt", "README.md", "RESEARCH.md",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "numpy", "scikit-learn", "iterative-stratification")}


def _check_versions(base_manifest):
    current = _versions()
    # sklearn objects in sparse.joblib require the original compatible environment.
    for name in ("transformers", "numpy", "scikit-learn"):
        saved = base_manifest.get("versions", {}).get(name)
        if saved and current[name] != saved:
            raise RuntimeError(f"{name}: ver4 저장 버전={saved}, 현재={current[name]}. requirements.txt를 설치하세요.")
    return current


def _read_base(train_label_dir, base_dir, init_fold):
    """Verify the exact original training run before trusting cached predictions."""
    checkpoint = base_dir / "mission3_ver4_model.ckpt"
    manifest = base_pipeline.load_manifest(checkpoint)
    _check_versions(manifest)
    print("기존 ver4 확인 완료. 서울 Training JSON을 읽습니다...", flush=True)
    paths, texts, targets = load_dataset(train_label_dir, include_targets=True)
    print(f"Training {len(paths):,}개 로드 완료. 원래 데이터·분할·보정 캐시를 검증합니다...", flush=True)
    fingerprint = base_pipeline._fingerprint(paths, texts, targets, manifest["config"])
    if fingerprint != manifest.get("signature"):
        raise ValueError("ver4와 Training 데이터/전처리/코드가 다릅니다. 원래 Training과 ver4 결과 묶음을 사용하세요.")
    splits = json.loads((base_dir / "splits.json").read_text(encoding="utf-8"))
    lookup = {p.name: i for i, p in enumerate(paths)}
    all_names = set(lookup)
    group_ids = np.asarray([text_group(text) for text in texts])

    def indices(names, purpose):
        if not names or len(names) != len(set(names)) or not set(names) <= all_names:
            raise ValueError(f"Invalid saved {purpose} filenames")
        return np.asarray([lookup[name] for name in names], dtype=np.int64)

    cal = indices(splits["calibration"], "calibration")
    if len(splits["folds"]) != len(manifest["folds"]) or not 1 <= init_fold <= len(splits["folds"]):
        raise ValueError("init_fold / saved folds mismatch")
    fold_indices = []
    for number, fold in enumerate(splits["folds"], 1):
        fit, stop = indices(fold["fit"], "fit"), indices(fold["stop"], "stop")
        sets = [set(fit), set(stop), set(cal)]
        groups = [set(group_ids[x]) for x in (fit, stop, cal)]
        if any(sets[i] & sets[j] or groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"fold {number}: sample or text-group leakage")
        if set.union(*sets) != set(range(len(paths))):
            raise ValueError(f"fold {number}: saved partitions do not cover Training")
        fold_indices.append((fit, stop))
    # Verify the stored fold identities against the ORIGINAL deterministic run.
    # Preserve the old models' exclusions and the comparable single-fold stop.
    _, expected_cal, expected_folds, _ = base_pipeline.make_splits(
        texts, targets, manifest["config"]["calibration_size"],
        manifest["config"]["n_folds"], manifest["config"]["seed"],
    )
    if not np.array_equal(cal, expected_cal) or any(
        not np.array_equal(actual, expected)
        for pair, reference in zip(fold_indices, expected_folds)
        for actual, expected in zip(pair, reference)
    ):
        raise ValueError("Saved fold identity differs from the original ver4 run; base reuse would leak.")
    for number, name in enumerate(manifest["folds"], 1):
        saved = torch.load(base_dir / name, map_location="cpu", weights_only=True, mmap=True)
        if (saved.get("format_version") != 1
                or saved.get("seed") != manifest["config"]["seed"] + number * 1009
                or saved.get("architecture", {}).get("num_labels") != len(TARGET_SYMPTOMS)):
            raise ValueError(f"Saved checkpoint fold identity mismatch: {name}")
        del saved
    with np.load(base_dir / "calibration_predictions.npz", allow_pickle=False) as cache:
        names = cache["filenames"].tolist()
        if len(set(names)) != len(names) or set(names) != set(splits["calibration"]):
            raise ValueError("ver4 calibration cache filenames mismatch")
        order = {name: i for i, name in enumerate(names)}
        aligned = [order[paths[i].name] for i in cal]
        raw_neural = np.array(cache["neural"][aligned], dtype=np.float64)
        raw_sparse = np.array(cache["sparse"][aligned], dtype=np.float64)
        if not np.array_equal(cache["targets"][aligned], targets[cal]):
            raise ValueError("ver4 calibration cache targets mismatch")
    for values in (raw_neural, raw_sparse):
        if values.shape != (len(cal), len(TARGET_SYMPTOMS)) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid ver4 raw calibration probabilities")
    fit, stop = fold_indices[init_fold - 1]
    return manifest, paths, texts, targets, fit, stop, cal, group_ids, raw_neural, raw_sparse


def _run_signature(config, base_manifest, base_dir, pretrained_metadata):
    source_dir = Path(__file__).parent
    code = hashlib.sha256()
    own_code = [source_dir / name for name in RUNTIME_FILES if name.endswith(".py")]
    for path in sorted([*own_code, *source_dir.joinpath("ver4_runtime").glob("*.py")]):
        code.update(str(path.relative_to(source_dir)).encode())
        code.update(path.read_bytes())
    assets = [base_dir / name for name in [*base_manifest["folds"], base_manifest["sparse"], "mission3_ver4_model.ckpt"]]
    assets += [path for path in (base_dir / base_manifest["tokenizer"]).rglob("*") if path.is_file()]
    asset_hashes = {}
    for path in sorted(assets):
        name = str(path.relative_to(base_dir))
        print("기본 모델 무결성 확인:", name, flush=True)
        asset_hashes[name] = _sha256(path)
    value = {
        "config": config, "base_signature": base_manifest["signature"],
        "public_pretrained": {key: pretrained_metadata[key] for key in ("model_id", "revision", "files")},
        "base_artifact_sha256": asset_hashes,
        "calibration_cache_sha256": _sha256(base_dir / "calibration_predictions.npz"),
        "splits_sha256": _sha256(base_dir / "splits.json"), "code_sha256": code.hexdigest(),
        "versions": _versions(),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


def _copy_file(source, target, expected_hash):
    """An interrupted copy never appears under its final filename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size or _sha256(target) != expected_hash:
            raise ValueError(f"기존 묶음 파일 내용이 다릅니다: {target}")
        return
    temporary = target.with_name(target.name + ".tmp")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        temporary.unlink()
        raise ValueError(f"복사 중 기본 모델 파일이 변경되었습니다: {source}")
    temporary.replace(target)


def _export_base(base_dir, output_dir, manifest, asset_hashes):
    destination = output_dir / "base"
    destination.mkdir(exist_ok=True)
    # Copy to make the resulting ver8 folder portable without its old sibling.
    for name in [*manifest["folds"], manifest["sparse"], "mission3_ver4_model.ckpt"]:
        print("Bundle:", name, flush=True)
        _copy_file(base_dir / name, destination / name, asset_hashes[name])
    tokenizer_dir = base_dir / manifest["tokenizer"]
    for path in tokenizer_dir.rglob("*"):
        if path.is_file():
            _copy_file(path, destination / manifest["tokenizer"] / path.relative_to(tokenizer_dir),
                       asset_hashes[str(path.relative_to(base_dir))])
    copied = base_pipeline.load_manifest(destination / "mission3_ver4_model.ckpt")
    if copied["signature"] != manifest["signature"]:
        raise ValueError("Portable base bundle signature mismatch")


def _export_code(output_dir):
    source = Path(__file__).parent
    submission = output_dir / "submission"
    submission.mkdir(exist_ok=True)
    for name in RUNTIME_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, submission / name)
    for path in source.joinpath("ver4_runtime").iterdir():
        if path.is_file():
            destination = submission / "ver4_runtime" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)


def _checked_probabilities(values, rows):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (rows, len(TARGET_SYMPTOMS)) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError("Invalid cached/model probabilities")
    return values


def _cached_predictions(path, signature, filenames, weights_digest, compute):
    """Bind a Training diagnostic cache to its exact model and held-out rows."""
    if path.is_file():
        with np.load(path, allow_pickle=False) as cache:
            if (cache["signature"].item() != signature or cache["filenames"].tolist() != filenames
                    or cache["weights_sha256"].item() != weights_digest):
                raise ValueError("Training prediction cache mismatch")
            values = np.array(cache["probabilities"])
    else:
        values = _checked_probabilities(compute(), len(filenames))
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, probabilities=values, filenames=np.asarray(filenames),
                                signature=np.asarray(signature), weights_sha256=np.asarray(weights_digest))
        temporary.replace(path)
    return _checked_probabilities(values, len(filenames))


def _tokenizer_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): _sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def choose_profile(trials):
    """Stable stop-only selection; calibration and Validation cannot enter here."""
    winner, best = None, None
    for profile in sorted(trials):
        history = trials[profile]
        score = {"stop_macro_f1": history["best_stop_macro_f1"], "stop_macro_ap": history["best_stop_ap"]}
        if better(score, best):
            winner, best = profile, score
    if winner is None:
        raise ValueError("At least one completed trial is required")
    return winner


def train_pipeline(train_label_dir, base_dir, output_dir, config=None, pretrained_dir=None):
    """Compare matched classifiers on one Training fold; calibrate the stop winner.

    Official Validation is deliberately not an argument. The old task encoder is
    used only for the baseline comparison/ensemble, never as initialization.
    """
    started = time.monotonic()
    supplied = config or {}
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    profiles = config["profiles"]
    if (not isinstance(profiles, list) or not profiles or len(set(profiles)) != len(profiles)
            or any(profile not in ("A", "B", "C") for profile in profiles)
            or profiles != sorted(profiles)):
        raise ValueError("profiles must be an ordered nonempty subset of A, B, C")
    if type(config["audit_per_error"]) is not int or config["audit_per_error"] < 1:
        raise ValueError("audit_per_error must be a positive integer")
    if not isinstance(config["keep_completed_resume_states"], bool):
        raise ValueError("keep_completed_resume_states must be boolean")
    tapt.settings(config)
    if not isinstance(config["bundle_base"], bool):
        raise ValueError("bundle_base must be true or false")
    if not np.isfinite(config["min_calibration_gain"]) or config["min_calibration_gain"] < 0:
        raise ValueError("min_calibration_gain must be finite and nonnegative")
    if type(config["min_calibration_improved_folds"]) is not int or config["min_calibration_improved_folds"] < 1:
        raise ValueError("min_calibration_improved_folds must be a positive integer")
    neural._settings(config)
    if pretrained_dir is None:
        raise ValueError("공개 사전학습 모델을 먼저 준비하고 pretrained_dir를 지정하세요.")
    train_label_dir, base_dir, output_dir, pretrained_dir = (Path(x).expanduser().resolve() for x in
                                             (train_label_dir, base_dir, output_dir, pretrained_dir))
    if train_label_dir.name != "TL_서울_구급" or any("validation" in part.lower() for part in train_label_dir.parts):
        raise ValueError("학습 입력은 서울 Training/TL_서울_구급 폴더여야 합니다.")
    for protected in (base_dir, pretrained_dir, train_label_dir):
        if output_dir == protected or output_dir.is_relative_to(protected) or protected.is_relative_to(output_dir):
            raise ValueError("ver8 출력은 기존 ver4/사전학습 모델/Training과 별도 폴더여야 합니다.")
    if config["require_cuda"] and (config["device"] != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("ver8 학습에는 GPU 런타임을 선택하세요.")
    pretrained_metadata = pretrained.validate_pretrained(pretrained_dir)
    if pretrained_metadata["model_id"] != config["model_name"]:
        raise ValueError("준비한 공개 encoder와 config.model_name이 다릅니다.")
    (base_manifest, paths, texts, targets, fit, stop, cal, groups,
     base_probs, sparse_probs) = _read_base(train_label_dir, base_dir, int(config["init_fold"]))
    signature, provenance = _run_signature(config, base_manifest, base_dir, pretrained_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")).get("signature") != signature:
        raise ValueError("ver8 출력 폴더의 데이터/설정/초기 가중치/코드가 다릅니다. 새 OUTPUT_DIR를 지정하세요.")
    if not run_path.exists() and any(output_dir.glob("*.ckpt")):
        raise ValueError("run.json 없는 기존 checkpoint가 있습니다. 새 OUTPUT_DIR를 지정하세요.")
    write_json(run_path, {"signature": signature, "config": config, "provenance": provenance})
    manifest_path = output_dir / "mission3_ver8_model.ckpt"
    report_path = output_dir / "mission3_ver8_metrics.json"
    if manifest_path.is_file() and report_path.is_file():
        saved = load_manifest(manifest_path, base_dir=None if config["bundle_base"] else base_dir)
        if saved["signature"] != signature:
            raise ValueError("Final ver8 signature mismatch")
        print("완료된 ver8 재사용: 추가 학습·보정 탐색을 건너뜁니다.", flush=True)
        return manifest_path, json.loads(report_path.read_text(encoding="utf-8"))

    write_json(output_dir / "splits.json", {
        "fit": [paths[i].name for i in fit], "stop": [paths[i].name for i in stop],
        "calibration": [paths[i].name for i in cal], "init_fold": config["init_fold"],
        "source": "Exact stored ver4 fold for a comparable independent encoder experiment",
    })
    audit_summary = audit.write_audit(
        output_dir / "error_analysis", [paths[i] for i in cal], [texts[i] for i in cal], targets[cal],
        base_pipeline.apply_calibrator(base_probs, sparse_probs, base_manifest["calibrator"]),
        base_manifest["calibrator"]["thresholds"], seed=config["seed"], per_error=config["audit_per_error"],
    )
    print(f"Training 오답 검토 template: {audit_summary['review_rows']}건 저장; 자동 정답 수정 없음", flush=True)
    print(f"새 encoder 전체 학습 | fit={len(fit):,}, stop={len(stop):,}, calibration={len(cal):,}", flush=True)
    print(f"{config['model_name']} | 최대 {config['max_epochs']} epoch | 발화 경계 SEP 보존", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, use_fast=True, local_files_only=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    rendered = data.candidate_texts(texts, tokenizer)
    documents = neural.tokenize_dialogues(rendered, tokenizer, config["max_length"], config["stride"])
    candidate_started = time.monotonic()
    trials, tapt_report = {}, None
    for profile in profiles:
        trial_dir = output_dir / "trials" / profile
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_path, done_path = trial_dir / "best.ckpt", trial_dir / "done.json"
        trial_config = {**config, "head_type": "window" if profile == "A" else "token",
                        "initialization_type": "fit_only_tapt" if profile == "C" else "public"}
        trial_signature = hashlib.sha256((signature + ":" + profile).encode()).hexdigest()
        if trial_path.is_file() and done_path.is_file():
            record = json.loads(done_path.read_text(encoding="utf-8"))
            if record.get("signature") != trial_signature or record.get("weights_sha256") != _sha256(trial_path):
                raise ValueError("Trial completion signature/weights mismatch")
            history = record["training"]
            print(f"완료된 {profile} 후보 재사용", flush=True)
            if profile == "C":
                tapt_report = record["tapt"]
        else:
            initial_dir = pretrained_dir
            if profile == "C":
                initial_dir, tapt_report = tapt.adapt(
                    [rendered[i] for i in fit], [paths[i].name for i in fit], tokenizer,
                    pretrained_dir, output_dir / "tapt", config, run_signature=signature,
                    seed=int(config["seed"]) + 7001,
                )
                _release()
            print(f"Profile {profile}: head={trial_config['head_type']}, TAPT={profile == 'C'}", flush=True)
            history = neural.train_model(
                [documents[i] for i in fit], targets[fit], [documents[i] for i in stop], targets[stop],
                tokenizer, trial_config, trial_path, seed=int(config["seed"]) + 5003,
                initial_model_dir=initial_dir, run_signature=trial_signature, stop_groups=groups[stop],
            )
            history["resume_state_retained"] = config["keep_completed_resume_states"]
            write_json(done_path, {"signature": trial_signature, "training": history,
                                  "weights_sha256": _sha256(trial_path),
                                  "tapt": tapt_report if profile == "C" else None})
        if not config["keep_completed_resume_states"]:
            trial_path.with_name("best_last.ckpt").unlink(missing_ok=True)
        trials[profile] = history
        write_json(output_dir / "trial_progress.json", {"signature": signature, "trials": trials})
        _release()
    # Choose exactly ONE candidate on stop before touching candidate calibration.
    selected_profile = choose_profile(trials)
    history = trials[selected_profile]
    candidate_path = output_dir / "trials" / selected_profile / "best.ckpt"
    write_json(output_dir / "candidate_done.json", {
        "signature": signature, "selected_profile": selected_profile, "training": history,
        "candidate": candidate_path.relative_to(output_dir).as_posix(), "weights_sha256": _sha256(candidate_path),
    })
    candidate_minutes = (time.monotonic() - candidate_started) / 60
    _release()

    # Only the corresponding old fold excluded this stop set from gradient
    # training. The other old folds must NEVER form an ensemble stop benchmark.
    baseline_started = time.monotonic()
    fold_name = base_manifest["folds"][int(config["init_fold"]) - 1]
    base_tokenizer = AutoTokenizer.from_pretrained(base_dir / base_manifest["tokenizer"],
                                                  use_fast=True, local_files_only=True)
    base_config = {**base_manifest["config"], "device": config["device"],
                   "inference_batch_size": config["inference_batch_size"]}
    base_stop_documents = base_neural.tokenize_dialogues([texts[i] for i in stop], base_tokenizer,
                                 base_config["max_length"], base_config["stride"])
    baseline_stop_probs = _cached_predictions(
        output_dir / "baseline_single_fold_stop.npz", signature, [paths[i].name for i in stop],
        provenance["base_artifact_sha256"][fold_name],
        lambda: _predict_base_checkpoint(base_dir / fold_name, base_stop_documents, base_tokenizer, base_config),
    )
    baseline_stop = StopSelector(targets[stop], groups[stop], seed=int(config["seed"]) + 5003,
                                n_splits=config["stop_crossfit_splits"]).score(baseline_stop_probs)
    baseline_stop.update(fold=config["init_fold"],
                         scope="Single original fold on its own stop set; repeated epoch-selection estimate")
    baseline_minutes = (time.monotonic() - baseline_started) / 60
    del base_stop_documents, base_tokenizer
    _release()
    calibration_started = time.monotonic()
    candidate_digest = _sha256(candidate_path)
    candidate_probs = _cached_predictions(
        output_dir / "candidate_calibration.npz", signature, [paths[i].name for i in cal], candidate_digest,
        lambda: neural.predict_checkpoint(candidate_path, [documents[i] for i in cal], tokenizer, config),
    )
    _release()
    cal_settings = {"seed": int(base_manifest["config"]["seed"]) + 2,
                    "n_splits": int(base_manifest["config"]["calibration_folds"])}
    calibrator, report = calibration.calibrate_crossfit(
        targets[cal], base_probs, sparse_probs, candidate_probs, groups[cal], **cal_settings,
        min_gain=float(config["min_calibration_gain"]),
        min_improved_folds=config["min_calibration_improved_folds"],
    )
    standalone_calibrator, standalone_report = calibration.fit_candidate_only_crossfit(
        targets[cal], candidate_probs, groups[cal], **cal_settings,
    )
    report.update(standalone_report)
    report.update({
        "signature": signature, "base_signature": base_manifest["signature"],
        "training_samples": len(paths), "candidate_fit_samples": len(fit), "candidate_stop_samples": len(stop),
        "candidate_training": history, "calibrator": calibrator, "config": config,
        "selected_profile": selected_profile, "trials": trials, "tapt": tapt_report,
        "error_analysis": audit_summary,
        "profile_selection_scope": "Training stop only, macro F1 then AP; exact ties prefer A, then B, then C",
        "candidate_only_calibrator": standalone_calibrator, "public_pretrained": provenance["public_pretrained"],
        "baseline_single_fold_stop": baseline_stop,
        "candidate_minus_single_fold_stop_macro_f1": history["best_stop_macro_f1"] - baseline_stop["stop_macro_f1"],
        "versions": _versions(), "validation_used_for_training_or_selection": False,
        "new_models_trained": len(trials), "base_models_retrained": 0,
        "timing_minutes_this_invocation": {
            "candidate_training_or_reuse": candidate_minutes, "baseline_stop_or_reuse": baseline_minutes,
            "calibration_prediction_and_selection": (time.monotonic() - calibration_started) / 60,
        },
    })
    if config["bundle_base"]:
        _export_base(base_dir, output_dir, base_manifest, provenance["base_artifact_sha256"])
    else:
        print("기존 ZIP의 ver4를 복사 없이 재사용합니다.", flush=True)
    _export_code(output_dir)
    manifest = {
        "format": "mission3_ver8", "version": 1, "signature": signature,
        "target_symptoms": list(TARGET_SYMPTOMS), "portable": config["bundle_base"],
        "base_manifest": ("base/mission3_ver4_model.ckpt" if config["bundle_base"] else
                          str(base_dir / "mission3_ver4_model.ckpt")),
        "base_signature": base_manifest["signature"], "candidate": candidate_path.relative_to(output_dir).as_posix(),
        "selected_profile": selected_profile,
        "candidate_sha256": candidate_digest, "candidate_tokenizer": "tokenizer",
        "candidate_tokenizer_sha256": _tokenizer_hashes(output_dir / "tokenizer"),
        "config": config, "calibrator": calibrator, "versions": _versions(),
        "candidate_only_calibrator": standalone_calibrator, "selected_name": report["selected_name"],
    }
    report["timing_minutes_this_invocation"]["total_pipeline"] = (time.monotonic() - started) / 60
    write_json(report_path, report)
    _atomic_torch_save(manifest_path, manifest)
    print(f"Training 보정 교차평가: ver4={report['baseline_crossfit_macro_f1']:.6f}, "
          f"새 모델 단독={report['candidate_standalone_crossfit_macro_f1']:.6f}, "
          f"혼합 후보={report['candidate_crossfit_macro_f1']:.6f}, 선택={report['selected_name']}", flush=True)
    print("Training 내부 선택용 점수입니다. 공식 Validation은 선택 완료 후 별도 평가합니다.", flush=True)
    return manifest_path, report


def load_manifest(checkpoint_path, base_dir=None):
    """Validate a portable bundle or an explicit external-base experiment.

    ``base_dir`` relocates an external ver4 dependency without rewriting the
    saved manifest. Portable manifests always retain their internal base.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if manifest.get("format") != "mission3_ver8" or manifest.get("version") != 1:
        raise ValueError("mission3_ver8_model.ckpt를 지정하세요.")
    if manifest.get("target_symptoms") != list(TARGET_SYMPTOMS):
        raise ValueError("ver8 target label order mismatch")
    root = checkpoint_path.parent
    candidate = (root / manifest["candidate"]).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(f"ver8 결과 폴더 전체를 함께 보관하세요: {manifest['candidate']}")
    if _sha256(candidate) != manifest.get("candidate_sha256"):
        raise ValueError("ver8 candidate weights hash mismatch")
    tokenizer_dir = (root / manifest["candidate_tokenizer"]).resolve()
    if (not tokenizer_dir.is_relative_to(root) or not tokenizer_dir.is_dir()
            or _tokenizer_hashes(tokenizer_dir) != manifest.get("candidate_tokenizer_sha256")):
        raise ValueError("ver8 candidate tokenizer hash/path mismatch")
    portable = manifest.get("portable", True)
    if not isinstance(portable, bool):
        raise ValueError("Invalid ver8 portable flag")
    if portable:
        if base_dir is not None:
            raise ValueError("--base_dir is only supported for external-base ver8 experiments")
        base_path = (root / manifest["base_manifest"]).resolve()
        if not base_path.is_relative_to(root) or not base_path.is_file():
            raise FileNotFoundError(f"ver8 결과 폴더 전체를 함께 보관하세요: {manifest['base_manifest']}")
    else:
        base_path = Path(manifest["base_manifest"])
        if not base_path.is_absolute():
            raise ValueError("External ver4 manifest path must be absolute")
        if base_dir is not None:
            base_path = Path(base_dir).expanduser().resolve() / "mission3_ver4_model.ckpt"
        if not base_path.is_file():
            raise FileNotFoundError("기존 ver4 묶음을 다시 준비하고 --base_dir에 폴더를 지정하세요: "
                                    f"{base_path}")
        manifest["base_manifest"] = str(base_path.resolve())
    base = base_pipeline.load_manifest(base_path)
    if base.get("signature") != manifest.get("base_signature"):
        raise ValueError("ver8 references a different ver4 base")
    return manifest


def _predict_base_checkpoint(path, documents, tokenizer, config):
    # Keep archived ver4 source bytes intact for its data/code fingerprint.
    # Its old support check can count emulated BF16 on T4; use native precision
    # during this synchronous inference call and restore the original helper.
    original = base_neural._amp
    base_neural._amp = neural._amp
    try:
        return base_pipeline.predict_checkpoint(path, documents, tokenizer, config)
    finally:
        base_neural._amp = original


def _cache_execution(device):
    # Keep the original cache schema so the unchanged ver4 branch is reusable.
    execution = {"versions": {**_versions(), **{name: importlib.metadata.version(name)
                 for name in ("scipy", "joblib")}}, "cuda_version": torch.version.cuda}
    if device == "cuda":
        execution.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                         bf16=neural.native_bf16_supported())
    return execution


def predict_pipeline(checkpoint_path, label_dir, output, device=None, batch_size=None, cache_dir=None,
                     base_dir=None, candidate_output=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    if candidate_output is not None and Path(candidate_output).resolve() == Path(output).resolve():
        raise ValueError("candidate_output must differ from final output")
    manifest = load_manifest(checkpoint_path, base_dir=base_dir)
    root = checkpoint_path.parent
    base_path = root / manifest["base_manifest"]
    base_manifest = base_pipeline.load_manifest(base_path)
    _check_versions(base_manifest)
    base_root = base_path.parent
    config = dict(manifest["config"])
    config["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        config["inference_batch_size"] = int(batch_size)
    paths, texts, _ = load_dataset(Path(label_dir), include_targets=False)
    base_config = {**base_manifest["config"], "device": config["device"],
                   "inference_batch_size": config["inference_batch_size"]}
    base_tokenizer = AutoTokenizer.from_pretrained(base_root / base_manifest["tokenizer"],
                                                 use_fast=True, local_files_only=True)
    execution = _cache_execution(config["device"])
    base_cache = (PredictionCache(cache_dir, paths, texts, base_root / base_manifest["tokenizer"],
                                 TARGET_SYMPTOMS, execution) if cache_dir is not None else None)
    source = Path(__file__).parent

    def predict(cache, kind, path, settings, sources, compute):
        if cache is None:
            return compute()
        return cache.get(kind, path, settings, [source / name for name in sources], compute)

    alpha = np.asarray(manifest["calibrator"]["neural_weights"])
    beta = np.asarray(manifest["calibrator"]["candidate_weights"])
    base_probs = np.zeros((len(paths), len(TARGET_SYMPTOMS)), dtype=np.float64)
    sparse_probs = np.zeros_like(base_probs)
    candidate_probs = None
    if np.any((1 - beta) * alpha > 0):
        base_documents = base_neural.tokenize_dialogues(texts, base_tokenizer,
                                      base_config["max_length"], base_config["stride"])
        for name in base_manifest["folds"]:
            print("Predict base:", name, flush=True)
            base_probs += predict(base_cache, "ver4-neural", base_root / name, base_config,
                                  ["ver4_runtime/neural.py"],
                                  lambda: _predict_base_checkpoint(base_root / name, base_documents, base_tokenizer, base_config))
            _release()
        base_probs /= len(base_manifest["folds"])
        del base_documents
    if np.any((1 - beta) * (1 - alpha) > 0):
        sparse_probs = predict(base_cache, "ver4-sparse", base_root / base_manifest["sparse"], base_config,
                               ["ver4_runtime/pipeline.py"],
                               lambda: base_pipeline.predict_sparse(joblib.load(base_root / base_manifest["sparse"]), texts))
    if np.any(beta > 0) or candidate_output is not None:
        # New tokenizer + SEP rendering must never replace the original baseline inputs.
        tokenizer_dir = root / manifest["candidate_tokenizer"]
        candidate_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
        rendered = data.candidate_texts(texts, candidate_tokenizer)
        candidate_documents = neural.tokenize_dialogues(rendered, candidate_tokenizer,
                                                        config["max_length"], config["stride"])
        candidate_cache = (PredictionCache(cache_dir, paths, rendered, tokenizer_dir, TARGET_SYMPTOMS, execution)
                           if cache_dir is not None else None)
        print("Predict:", manifest["candidate"], flush=True)
        candidate_probs = predict(candidate_cache, "ver8-neural", root / manifest["candidate"], config,
                                  ["neural.py", "data.py", "ver4_runtime/neural.py"],
                                  lambda: neural.predict_checkpoint(root / manifest["candidate"], candidate_documents,
                                                                    candidate_tokenizer, config))
        _release()
    probabilities = calibration.apply_calibrator(base_probs, sparse_probs, candidate_probs, manifest["calibrator"])
    predictions = write_predictions(paths, probabilities, manifest["calibrator"]["thresholds"], Path(output))
    if candidate_output is not None:
        write_predictions(paths, candidate_probs, manifest["candidate_only_calibrator"]["thresholds"], Path(candidate_output))
        print(f"새 모델 단독 예측 저장: {candidate_output} (Training에서 고정한 임계값)", flush=True)
    print(f"{len(paths):,} predictions saved: {output}", flush=True)
    return paths, probabilities, predictions


evaluate_csv = base_pipeline.evaluate_csv
