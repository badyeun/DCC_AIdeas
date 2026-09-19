"""Verify and package the original Training-only ver4 baseline."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import torch
if __package__:
    from .ver4_runtime import pipeline as base_pipeline
    from .ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group
else:
    from ver4_runtime import pipeline as base_pipeline
    from ver4_runtime.data import TARGET_SYMPTOMS, load_dataset, text_group

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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
    # Copy to make the resulting ver9 folder portable without its old sibling.
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
