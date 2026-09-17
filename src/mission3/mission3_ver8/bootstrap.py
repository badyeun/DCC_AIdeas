"""Restore the one-file Mission 3 input bundle using only the standard library.

The uploaded ZIP remains in Drive. Extraction uses a fixed local directory so
epoch resume sees the same initial checkpoint path across Colab sessions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile


BUNDLE_NAME = "mission3_ver5_inputs.zip"
MODEL_PREFIX = "DCC_AIdeas/results/mission3/ver4/"
TRAIN_PREFIX = "DCC_AIdeas/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급/"
VALID_PREFIX = "DCC_AIdeas/DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급/"


def _relative(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name.rstrip("/")):
        raise ValueError(f"Unsafe bundle path: {name}")
    return Path(*path.parts)


def _target(root, name):
    path = root / _relative(name)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Bundle path escapes extraction directory: {name}")
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _files(manifest, archive):
    if manifest.get("format") != "mission3_ver5_inputs_v1":
        raise ValueError("Unsupported Mission 3 input bundle format")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Bundle manifest has no files")
    names = set()
    for row in files:
        name = row["path"]
        _relative(name)
        if name in names or not name.startswith("DCC_AIdeas/"):
            raise ValueError(f"Duplicate or unexpected bundle member: {name}")
        names.add(name)
        if (type(row["size"]) is not int or row["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError(f"Invalid manifest metadata: {name}")
        info = archive.getinfo(name)
        if info.is_dir() or info.file_size != row["size"]:
            raise ValueError(f"Bundle size mismatch: {name}")
    required = {MODEL_PREFIX + name for name in
                ["mission3_ver4_model.ckpt", "splits.json", "calibration_predictions.npz", "sparse.joblib",
                 *[f"fold_{i}.ckpt" for i in range(1, 6)], "tokenizer/tokenizer.json",
                 "tokenizer/tokenizer_config.json", "tokenizer/vocab.txt", "tokenizer/special_tokens_map.json"]}
    if not required <= names:
        raise ValueError(f"Missing ver4 inputs: {sorted(required - names)}")
    for prefix, key in [(TRAIN_PREFIX, "training_count"), (VALID_PREFIX, "validation_count")]:
        count = sum(name.startswith(prefix) and name.endswith(".json") for name in names)
        if count < 1 or type(manifest.get(key)) is not int or count != manifest[key]:
            raise ValueError(f"Bundle JSON count mismatch: {key}")
    infos = archive.infolist()
    if len({info.filename for info in infos}) != len(infos):
        raise ValueError("Duplicate ZIP member")
    allowed_files = names | {"bundle_manifest.json", "README_UPLOAD.txt"}
    for info in infos:
        _relative(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
            raise ValueError("Links and encrypted ZIP entries are not supported")
        if not info.is_dir() and info.filename not in allowed_files:
            raise ValueError(f"Unlisted bundle member: {info.filename}")
    return files


def restore_input_bundle(archive_path, destination):
    """Verify SHA256 while extracting; reuse unchanged files from a verified run."""
    archive_path = Path(archive_path).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".mission3_bundle_complete.json"
    with zipfile.ZipFile(archive_path) as archive:
        manifest_bytes = archive.read("bundle_manifest.json")
        manifest = json.loads(manifest_bytes)
        files = _files(manifest, archive)
        signature = hashlib.sha256(manifest_bytes).hexdigest()
        previous = {}
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("manifest_sha256") == signature:
                previous = saved["files"]
        except (OSError, ValueError, KeyError):
            pass
        pending, state = [], {}
        for row in files:
            target = _target(root, row["path"])
            current = target.stat() if target.is_file() else None
            stamp = [current.st_size, current.st_mtime_ns] if current else None
            if stamp and current.st_size == row["size"] and previous.get(row["path"]) == stamp:
                state[row["path"]] = stamp
            else:
                pending.append(row)
        if not pending:
            print("입력 ZIP: 이전에 검증한 Colab 로컬 파일을 재사용합니다.", flush=True)
            return root / "DCC_AIdeas"
        if shutil.disk_usage(root).free < sum(row["size"] for row in pending):
            raise OSError("Colab 로컬 저장소에 입력 ZIP을 풀 공간이 부족합니다.")
        print(f"입력 ZIP 확인·압축 해제: {len(pending):,}개 파일 (WAV 없음)", flush=True)
        for index, row in enumerate(pending, 1):
            target = _target(root, row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recover completed files from an interrupted extraction without copying them again.
            verified = target.is_file() and target.stat().st_size == row["size"] and _hash(target) == row["sha256"]
            if not verified:
                temporary = None
                try:
                    digest, size = hashlib.sha256(), 0
                    with archive.open(row["path"]) as reader, tempfile.NamedTemporaryFile(
                            dir=target.parent, prefix=".extract_", suffix=".tmp", delete=False) as writer:
                        temporary = Path(writer.name)
                        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                            writer.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if size != row["size"] or digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"ZIP 내용 검증 실패: {row['path']}")
                    temporary.replace(target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            current = target.stat()
            state[row["path"]] = [current.st_size, current.st_mtime_ns]
            if row["size"] > 100 * 1024 * 1024 or index % 2000 == 0 or index == len(pending):
                print(f"압축 해제 {index:,}/{len(pending):,}: {target.name}", flush=True)
        # A marker is published only after every file has passed hash verification.
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"manifest_sha256": signature, "files": state}), encoding="utf-8")
        temporary.replace(marker)
    print(f"준비 완료: Training {manifest['training_count']:,}개 / Validation {manifest['validation_count']:,}개", flush=True)
    return root / "DCC_AIdeas"
