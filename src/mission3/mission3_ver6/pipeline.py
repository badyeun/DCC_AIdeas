"""Reuse ver4; fit one complementary model without touching official Validation."""
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
    from . import calibration, features, neural
    from .prediction_cache import PredictionCache
    from .ver4_runtime import pipeline as base_pipeline
    from .ver4_runtime import neural as base_neural
    from .ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group, write_json, write_predictions
else:
    import calibration
    import features
    import neural
    from prediction_cache import PredictionCache
    from ver4_runtime import pipeline as base_pipeline
    from ver4_runtime import neural as base_neural
    from ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group, write_json, write_predictions


DEFAULT_CONFIG = {
    **neural.DEFAULTS, "seed": 42, "init_fold": 1, "require_cuda": True,
    "bundle_base": False, "feature_cache_dir": None,
    "min_calibration_gain": 0.002, "min_calibration_improved_folds": 3,
}
RUNTIME_FILES = (
    "__init__.py", "pipeline.py", "neural.py", "losses.py", "selection.py", "features.py",
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


class _LazyFeatureStore:
    """Completed head trials can resume fine-tuning without rebuilding features."""

    def __init__(self, size, directory, factory):
        self.size, self.directory, self.factory = size, Path(directory).expanduser().resolve(), factory
        self.store, self.minutes = None, 0.0

    def __len__(self):
        return self.size

    def _materialize(self):
        if self.store is None:
            started = time.monotonic()
            self.store = self.factory()
            self.minutes = (time.monotonic() - started) / 60
            self.directory = self.store.directory
        return self.store

    def __getitem__(self, index):
        return self._materialize()[index]

    def get_legacy(self, index):
        return self._materialize().get_legacy(index)

    def report(self):
        return {"directory": str(self.directory), "documents": self.size, "minutes": self.minutes,
                "features_accessed_this_run": self.store is not None,
                "scope": "original fold fit+stop inputs only; no labels in cache"}

    def close(self):
        if self.store is not None:
            self.store.close()


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
    # We still use the stored indices below; a new split after warm-start is unsafe.
    _, expected_cal, expected_folds, _ = base_pipeline.make_splits(
        texts, targets, manifest["config"]["calibration_size"],
        manifest["config"]["n_folds"], manifest["config"]["seed"],
    )
    if not np.array_equal(cal, expected_cal) or any(
        not np.array_equal(actual, expected)
        for pair, reference in zip(fold_indices, expected_folds)
        for actual, expected in zip(pair, reference)
    ):
        raise ValueError("Saved fold identity differs from the original ver4 run; warm-start would leak.")
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


def _run_signature(config, base_manifest, base_dir, init_checkpoint):
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
        "initial_checkpoint_sha256": asset_hashes[str(init_checkpoint.relative_to(base_dir))],
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
    # Copy to make the resulting ver6 folder portable without its old sibling.
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


def train_pipeline(train_label_dir, base_dir, output_dir, config=None):
    """Only TL_서울_구급 is accepted. Validation is deliberately not an argument."""
    supplied = config or {}
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    if not isinstance(config["bundle_base"], bool):
        raise ValueError("bundle_base must be true or false")
    if not np.isfinite(config["min_calibration_gain"]) or config["min_calibration_gain"] < 0:
        raise ValueError("min_calibration_gain must be finite and nonnegative")
    if type(config["min_calibration_improved_folds"]) is not int or config["min_calibration_improved_folds"] < 1:
        raise ValueError("min_calibration_improved_folds must be a positive integer")
    neural._settings(config)  # Reject invalid trials before hashing/extracting large inputs.
    train_label_dir, base_dir, output_dir = (Path(x).expanduser().resolve() for x in
                                          (train_label_dir, base_dir, output_dir))
    if train_label_dir.name != "TL_서울_구급" or any("validation" in part.lower() for part in train_label_dir.parts):
        raise ValueError("학습 입력은 서울 Training/TL_서울_구급 폴더여야 합니다.")
    if output_dir == base_dir or output_dir.is_relative_to(base_dir) or base_dir.is_relative_to(output_dir):
        raise ValueError("ver6 출력은 기존 ver4와 별도 폴더여야 합니다.")
    if config["require_cuda"] and (config["device"] != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("ver6 학습에는 GPU 런타임을 선택하세요.")
    (base_manifest, paths, texts, targets, fit, stop, cal, groups,
     base_probs, sparse_probs) = _read_base(train_label_dir, base_dir, int(config["init_fold"]))
    init_checkpoint = base_dir / base_manifest["folds"][int(config["init_fold"]) - 1]
    signature, provenance = _run_signature(config, base_manifest, base_dir, init_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")).get("signature") != signature:
        raise ValueError("ver6 출력 폴더의 데이터/설정/초기 가중치/코드가 다릅니다. 새 OUTPUT_DIR를 지정하세요.")
    if not run_path.exists() and any(output_dir.glob("*.ckpt")):
        raise ValueError("run.json 없는 기존 checkpoint가 있습니다. 새 OUTPUT_DIR를 지정하세요.")
    write_json(run_path, {"signature": signature, "config": config, "provenance": provenance})
    manifest_path = output_dir / "mission3_ver6_model.ckpt"
    report_path = output_dir / "mission3_ver6_metrics.json"
    if manifest_path.is_file() and report_path.is_file():
        saved = load_manifest(manifest_path, base_dir=None if config["bundle_base"] else base_dir)
        if saved["signature"] != signature:
            raise ValueError("Final ver6 signature mismatch")
        print("완료된 ver6 재사용: 추가 학습·보정 탐색을 건너뜁니다.", flush=True)
        return manifest_path, json.loads(report_path.read_text(encoding="utf-8"))

    write_json(output_dir / "splits.json", {
        "fit": [paths[i].name for i in fit], "stop": [paths[i].name for i in stop],
        "calibration": [paths[i].name for i in cal], "init_fold": config["init_fold"],
        "source": "Exact stored ver4 fold; no resplitting after warm-start",
    })
    print(f"ver4 재사용 | 새 모델 fit={len(fit):,}, stop={len(stop):,}, calibration={len(cal):,}", flush=True)
    print(f"새 비교: 동결 특징 재사용 → head 후보 비교 → 선택 후보만 최대 {config['finetune_epochs']} epoch "
          f"미세조정 (상위 층={config['finetune_top_layers']})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir / base_manifest["tokenizer"], use_fast=True, local_files_only=True)
    neural_config = {**config, "max_length": base_manifest["config"]["max_length"],
                     "stride": base_manifest["config"]["stride"]}
    documents = neural.tokenize_dialogues(texts, tokenizer, neural_config["max_length"], neural_config["stride"])
    candidate_path, done_path = output_dir / "candidate_best.ckpt", output_dir / "candidate_done.json"
    feature_report = None
    if candidate_path.is_file() and done_path.is_file():
        record = json.loads(done_path.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError("Candidate completion signature mismatch")
        history = record["training"]
        feature_report = record.get("feature_cache")
        print("완료된 추가 모델 재사용", flush=True)
    else:
        # Only original fold fit/stop features are cached. Calibration targets
        # and predictions are not passed to candidate training or head selection.
        indices = np.concatenate([fit, stop])
        feature_directory = config["feature_cache_dir"] or str(output_dir / "frozen_features")
        store = _LazyFeatureStore(len(indices), feature_directory, lambda: features.build_feature_store(
            [documents[i] for i in indices], tokenizer, init_checkpoint, feature_directory, neural_config,
            checkpoint_sha256=provenance["initial_checkpoint_sha256"],
        ))
        try:
            history = neural.train_model(
                [documents[i] for i in fit], targets[fit], [documents[i] for i in stop], targets[stop],
                tokenizer, neural_config, candidate_path, seed=int(config["seed"]) + 5003,
                init_checkpoint_path=init_checkpoint, run_signature=signature, feature_store=store,
                fit_indices=np.arange(len(fit)), stop_indices=np.arange(len(fit), len(indices)),
                stop_groups=groups[stop],
            )
        finally:
            store.close()
        feature_report = store.report()
        write_json(done_path, {"signature": signature, "training": history, "feature_cache": feature_report})
    _release()
    candidate_cache = output_dir / "candidate_calibration.npz"
    candidate_digest = _sha256(candidate_path)
    if candidate_cache.is_file():
        with np.load(candidate_cache, allow_pickle=False) as cache:
            if (cache["signature"].item() != signature or cache["filenames"].tolist() != [paths[i].name for i in cal]
                    or cache["weights_sha256"].item() != candidate_digest):
                raise ValueError("Candidate calibration cache mismatch")
            candidate_probs = np.array(cache["probabilities"])
    else:
        candidate_probs = neural.predict_checkpoint(candidate_path, [documents[i] for i in cal], tokenizer, neural_config)
        temporary = candidate_cache.with_name(candidate_cache.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, probabilities=candidate_probs,
                                filenames=np.asarray([paths[i].name for i in cal]), signature=np.asarray(signature),
                                weights_sha256=np.asarray(candidate_digest))
        temporary.replace(candidate_cache)
    _release()
    # RAW ver4 predictions enter cross-fit; its full-calibration thresholds never do.
    calibrator, report = calibration.calibrate_crossfit(
        targets[cal], base_probs, sparse_probs, candidate_probs, groups[cal],
        seed=int(base_manifest["config"]["seed"]) + 2,
        n_splits=int(base_manifest["config"]["calibration_folds"]),
        min_gain=float(config["min_calibration_gain"]),
        min_improved_folds=config["min_calibration_improved_folds"],
    )
    report.update({
        "signature": signature, "base_signature": base_manifest["signature"],
        "training_samples": len(paths), "candidate_fit_samples": len(fit), "candidate_stop_samples": len(stop),
        "candidate_training": history, "calibrator": calibrator, "config": config,
        "feature_cache": feature_report,
        "versions": _versions(), "validation_used_for_training_or_selection": False,
        "new_models_trained": 1, "head_trials_compared": len(history.get("head_trials", [])),
        "base_models_retrained": 0,
    })
    if config["bundle_base"]:
        _export_base(base_dir, output_dir, base_manifest, provenance["base_artifact_sha256"])
    else:
        print("저장 공간 절약: 기존 ver4를 복사하지 않고 현재 폴더에서 재사용합니다.", flush=True)
    _export_code(output_dir)
    manifest = {
        "format": "mission3_ver6", "version": 1, "signature": signature,
        "target_symptoms": list(TARGET_SYMPTOMS), "portable": config["bundle_base"],
        "base_manifest": ("base/mission3_ver4_model.ckpt" if config["bundle_base"] else
                          str(base_dir / "mission3_ver4_model.ckpt")),
        "base_signature": base_manifest["signature"], "candidate": "candidate_best.ckpt",
        "config": neural_config, "calibrator": calibrator, "versions": _versions(),
        "selected_name": report["selected_name"],
    }
    write_json(report_path, report)
    _atomic_torch_save(manifest_path, manifest)
    print(f"보정 교차평가: ver4={report['baseline_crossfit_macro_f1']:.6f}, "
          f"ver6 후보={report['candidate_crossfit_macro_f1']:.6f}, 선택={report['selected_name']}", flush=True)
    print("이 점수는 Training 내부 선택용이며 공식 Validation 점수가 아닙니다.", flush=True)
    return manifest_path, report


def load_manifest(checkpoint_path, base_dir=None):
    """Validate a portable bundle or an explicit external-base experiment.

    ``base_dir`` relocates an external ver4 dependency without rewriting the
    saved manifest. Portable manifests always retain their internal base.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if manifest.get("format") != "mission3_ver6" or manifest.get("version") != 1:
        raise ValueError("mission3_ver6_model.ckpt를 지정하세요.")
    if manifest.get("target_symptoms") != list(TARGET_SYMPTOMS):
        raise ValueError("ver6 target label order mismatch")
    root = checkpoint_path.parent
    candidate = (root / manifest["candidate"]).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(f"ver6 결과 폴더 전체를 함께 보관하세요: {manifest['candidate']}")
    portable = manifest.get("portable", True)
    if not isinstance(portable, bool):
        raise ValueError("Invalid ver6 portable flag")
    if portable:
        if base_dir is not None:
            raise ValueError("--base_dir is only supported for external-base ver6 experiments")
        base_path = (root / manifest["base_manifest"]).resolve()
        if not base_path.is_relative_to(root) or not base_path.is_file():
            raise FileNotFoundError(f"ver6 결과 폴더 전체를 함께 보관하세요: {manifest['base_manifest']}")
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
        raise ValueError("ver6 references a different ver4 base")
    return manifest


def _predict_base_checkpoint(path, documents, tokenizer, config):
    # Keep archived ver4 source bytes intact for its data/code fingerprint.
    # Its old support check can count emulated BF16 on T4; use native precision
    # during this synchronous inference call and restore the original helper.
    original = base_neural._amp
    base_neural._amp = features.amp_context
    try:
        return base_pipeline.predict_checkpoint(path, documents, tokenizer, config)
    finally:
        base_neural._amp = original


def predict_pipeline(checkpoint_path, label_dir, output, device=None, batch_size=None, cache_dir=None, base_dir=None):
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = load_manifest(checkpoint_path, base_dir=base_dir)
    root = checkpoint_path.parent
    base_manifest = base_pipeline.load_manifest(root / manifest["base_manifest"])
    _check_versions(base_manifest)
    base_root = (root / manifest["base_manifest"]).parent
    config = dict(manifest["config"])
    config["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is not None:
        config["inference_batch_size"] = int(batch_size)
    paths, texts, _ = load_dataset(Path(label_dir), include_targets=False)
    tokenizer = AutoTokenizer.from_pretrained(base_root / base_manifest["tokenizer"], use_fast=True, local_files_only=True)
    documents = neural.tokenize_dialogues(texts, tokenizer, config["max_length"], config["stride"])
    cache = None
    if cache_dir is not None:
        execution = {"versions": {**_versions(), **{name: importlib.metadata.version(name)
                     for name in ("scipy", "joblib")}}, "cuda_version": torch.version.cuda}
        if config["device"] == "cuda":
            execution.update(gpu=torch.cuda.get_device_name(),
                             capability=list(torch.cuda.get_device_capability()),
                             bf16=features.native_bf16_supported())
        cache = PredictionCache(cache_dir, paths, texts, base_root / base_manifest["tokenizer"],
                                TARGET_SYMPTOMS, execution)
    source = Path(__file__).parent

    def predict(kind, path, settings, sources, compute):
        if cache is None:
            return compute()
        return cache.get(kind, path, settings, [source / name for name in sources], compute)

    alpha = np.asarray(manifest["calibrator"]["neural_weights"])
    beta = np.asarray(manifest["calibrator"]["candidate_weights"])
    base_probs = np.zeros((len(paths), len(TARGET_SYMPTOMS)), dtype=np.float64)
    sparse_probs = np.zeros_like(base_probs)
    candidate_probs = None
    if np.any((1 - beta) * alpha > 0):
        base_config = {**base_manifest["config"], "device": config["device"],
                       "inference_batch_size": config["inference_batch_size"]}
        # Keep ver4's original window batching for its numerical inference path.
        for name in base_manifest["folds"]:
            print("Predict base:", name, flush=True)
            base_probs += predict("ver4-neural", base_root / name, base_config,
                                  ["ver4_runtime/neural.py"],
                                  lambda: _predict_base_checkpoint(base_root / name, documents, tokenizer, base_config))
            _release()
        base_probs /= len(base_manifest["folds"])
    if np.any((1 - beta) * (1 - alpha) > 0):
        sparse_probs = predict("ver4-sparse", base_root / base_manifest["sparse"], config,
                               ["ver4_runtime/pipeline.py"],
                               lambda: base_pipeline.predict_sparse(joblib.load(base_root / base_manifest["sparse"]), texts))
    if np.any(beta > 0):
        print("Predict: candidate_best.ckpt", flush=True)
        candidate_probs = predict("ver6-neural", root / manifest["candidate"], config,
                                  ["neural.py", "ver4_runtime/neural.py"],
                                  lambda: neural.predict_checkpoint(root / manifest["candidate"], documents, tokenizer, config))
        _release()
    probabilities = calibration.apply_calibrator(base_probs, sparse_probs, candidate_probs, manifest["calibrator"])
    predictions = write_predictions(paths, probabilities, manifest["calibrator"]["thresholds"], Path(output))
    print(f"{len(paths):,} predictions saved: {output}", flush=True)
    return paths, probabilities, predictions


evaluate_csv = base_pipeline.evaluate_csv
