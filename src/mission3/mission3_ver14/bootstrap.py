"""Safely restore label JSONs only; old model weights are never extracted."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

TRAIN_PREFIX = "DCC_AIdeas/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급/"
VALID_PREFIX = "DCC_AIdeas/DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급/"
MARKER_NAME = ".mission3_label_bundle_complete.json"


def _relative(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name.rstrip("/")):
        raise ValueError(f"Unsafe bundle path: {name}")
    return Path(*path.parts)


def _target(root, name):
    target = root / _relative(name)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Bundle path escapes extraction directory: {name}")
    # Existing links must not redirect even to another location inside the root.
    for component in (target, *target.parents):
        if component == root.parent:
            break
        if component.is_symlink():
            raise ValueError(f"Extraction destination contains a symlink: {name}")
    return target


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_files(archive):
    infos = archive.infolist()
    names = set()
    entry_types = {}
    for info in infos:
        # Windows ZipInfo normalizes backslashes in filename; orig_filename
        # retains the stored member spelling that must also pass validation.
        _relative(info.orig_filename)
        _relative(info.filename)
        normalized = info.filename.rstrip("/").casefold()
        if normalized in names:
            raise ValueError(f"Duplicate ZIP member or directory: {info.filename}")
        names.add(normalized)
        entry_types[normalized] = info.is_dir()
        if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
            raise ValueError("Links and encrypted ZIP entries are not supported")
    for name in entry_types:
        if any(parent.as_posix() in entry_types and not entry_types[parent.as_posix()]
               for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise ValueError("ZIP file/directory collision")
    selected = [info for info in infos if not info.is_dir() and info.filename.endswith(".json")
                and info.filename.startswith((TRAIN_PREFIX, VALID_PREFIX))]
    for prefix in (TRAIN_PREFIX, VALID_PREFIX):
        subset = [info for info in selected if info.filename.startswith(prefix)]
        if not subset:
            raise ValueError(f"Input ZIP has no label JSONs in {prefix}")
        if len({PurePosixPath(info.filename).name.casefold() for info in subset}) != len(subset):
            raise ValueError("Duplicate label JSON filenames in input ZIP")
    by_name = {info.filename: info for info in selected}
    if "bundle_manifest.json" in archive.namelist():
        raw = archive.read("bundle_manifest.json")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or manifest.get("format") != "mission3_ver5_inputs_v1":
            raise ValueError("Unsupported Mission 3 input bundle manifest")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("Bundle manifest has no files")
        metadata, listed = {}, set()
        for row in files:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Invalid bundle manifest file entry")
            name = row["path"]
            _relative(name)
            if name.casefold() in listed:
                raise ValueError(f"Duplicate manifest entry: {name}")
            listed.add(name.casefold())
            if name not in by_name:
                continue
            size, digest = row.get("size"), row.get("sha256")
            if (type(size) is not int or size < 0 or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                raise ValueError(f"Invalid manifest metadata: {name}")
            if by_name[name].file_size != size:
                raise ValueError(f"Bundle size mismatch: {name}")
            metadata[name] = {"path": name, "size": size, "sha256": digest}
        if set(metadata) != set(by_name):
            raise ValueError("Bundle manifest does not list every label JSON")
        for prefix, key in ((TRAIN_PREFIX, "training_count"), (VALID_PREFIX, "validation_count")):
            count = sum(name.startswith(prefix) for name in metadata)
            if type(manifest.get(key)) is not int or manifest[key] != count:
                raise ValueError(f"Bundle JSON count mismatch: {key}")
        signature = hashlib.sha256(raw).hexdigest()
    else:
        # Without a manifest, hash only label contents, never the large old models.
        metadata = {}
        for name, info in sorted(by_name.items()):
            digest, size = hashlib.sha256(), 0
            with archive.open(info) as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != info.file_size:
                raise ValueError(f"Bundle size mismatch: {name}")
            metadata[name] = {"path": name, "size": size, "sha256": digest.hexdigest()}
        signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return [metadata[name] for name in sorted(metadata)], signature


def _validate_directory(directory, name):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir() or not any(path.is_file() for path in directory.rglob("*.json")):
        raise ValueError(f"{name} label directory does not contain JSON files: {directory}")
    return directory


def prepare_data(input_zip, work_dir, train_dir="", validation_dir=""):
    """Use two explicit label directories, or verify/extract the existing bundle.

    Reuse requires a completion marker plus fresh content hashes. This detects
    changed files even when their size and modification time are preserved.
    """
    if bool(train_dir) != bool(validation_dir):
        raise ValueError("Set both train_dir and validation_dir, or leave both empty")
    if train_dir:
        train = _validate_directory(train_dir, "Training")
        valid = _validate_directory(validation_dir, "Validation")
        if train == valid or train.is_relative_to(valid) or valid.is_relative_to(train):
            raise ValueError("Training and Validation label directories must be distinct and disjoint")
        return train, valid
    archive_path = Path(input_zip).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Input ZIP is missing: {archive_path}")
    requested_root = Path(work_dir).expanduser()
    if requested_root.is_symlink():
        raise ValueError("Extraction work_dir must not be a symlink")
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER_NAME
    with zipfile.ZipFile(archive_path) as archive:
        files, signature = _selected_files(archive)
        # Resolve every destination before any extraction, including existing links.
        destinations = {row["path"]: _target(root, row["path"]) for row in files}
        expected_paths = set(destinations.values())
        for prefix in (TRAIN_PREFIX, VALID_PREFIX):
            directory = root / prefix
            if directory.exists():
                unexpected = set(directory.rglob("*.json")) - expected_paths
                if unexpected:
                    raise ValueError("Extraction directory contains label JSONs from another input bundle")
        previous = None
        if marker.exists():
            if marker.is_symlink():
                raise ValueError("Completion marker must not be a symlink")
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None
        expected = {row["path"]: {"size": row["size"], "sha256": row["sha256"]} for row in files}
        reuse = (isinstance(previous, dict) and previous.get("signature") == signature
                 and previous.get("files") == expected)
        pending = []
        for row in files:
            target = destinations[row["path"]]
            verified = (target.is_file() and target.stat().st_size == row["size"]
                        and _hash(target) == row["sha256"])
            if not verified:
                pending.append(row)
        if reuse and not pending:
            return root / TRAIN_PREFIX, root / VALID_PREFIX
        if marker.exists():
            marker.unlink()
        if shutil.disk_usage(root).free < sum(row["size"] for row in pending):
            raise OSError("Insufficient local disk space to extract label JSONs")
        for row in pending:
            target = destinations[row["path"]]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest, size = hashlib.sha256(), 0
                with archive.open(row["path"]) as reader, tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=".extract_", suffix=".tmp", delete=False) as writer:
                    temporary = Path(writer.name)
                    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != row["size"] or digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"ZIP content verification failed: {row['path']}")
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        # Do not publish an incomplete extraction as a reusable data set.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                             prefix=".marker_", suffix=".tmp", delete=False) as writer:
                temporary = Path(writer.name)
                json.dump({"signature": signature, "files": expected}, writer, ensure_ascii=False)
            temporary.replace(marker)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return root / TRAIN_PREFIX, root / VALID_PREFIX
