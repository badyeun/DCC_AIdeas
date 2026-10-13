"""Recover only the pinned ver19 Training models; never execute source code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile

import numpy as np

from . import data


REFERENCE_PATH = Path(__file__).with_name("source_reference.json")
REFERENCE_FILE_SHA256 = "efcae59b68909dc1ba94e6ad954cfd3ec6ace8c4ffb2d4f25ae1f5972ea855ef"
CHUNK_SIZE = 4 * 1024 * 1024
MAX_ZIP_MEMBERS = 100000


def _check(budget, stage):
    if budget is not None:
        budget.check(stage)


def _equal(actual, expected, name):
    if actual != expected:
        raise ValueError(f"ver19 source mismatch: {name}")


def _relative(name):
    """Accept one canonical, portable relative path, without normalization."""
    if (not isinstance(name, str) or not name or len(name) > 1024
            or "\\" in name or ":" in name or "\0" in name):
        raise ValueError("Unsafe source path")
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(part.rstrip(" .") != part for part in path.parts)):
        raise ValueError("Unsafe source path")
    return path


def _safe_file(root, relative):
    root = Path(root).resolve()
    path = root.joinpath(*_relative(relative).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Source file escapes its root")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Source symlinks are not permitted")
    return path


def _hash_file(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            _check(budget, "source checksum")
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_reference():
    raw = REFERENCE_PATH.read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), REFERENCE_FILE_SHA256, "pinned reference checksum")
    reference = json.loads(raw.decode("utf-8"))
    _equal(reference["format"], "mission3_ver21_pinned_source_v1", "reference format")
    _equal(reference["version"], "ver19", "source version")
    _equal(reference["labels"], list(data.TARGET_SYMPTOMS), "label order")
    _equal(reference["decision_rule"], "categorical predict/argmax", "decision rule")
    _equal(reference["provenance"]["validation_accessed"], False, "Training-only provenance")
    for name in reference["files"]:
        _relative(name)
    return reference


def _verify_files(root, reference, budget=None):
    verified = {}
    for name, expected in reference["files"].items():
        path = _safe_file(root, name)
        if not path.is_file():
            raise ValueError(f"Required ver19 source file is missing: {name}")
        _equal(path.stat().st_size, expected["size"], f"file size ({name})")
        digest = _hash_file(path, budget)
        _equal(digest, expected["sha256"], f"file checksum ({name})")
        verified[name] = digest
    _verify_manifests(root, reference)
    return verified


def _verify_manifests(root, reference):
    for key, scope, count in (("fit_checkpoint", "fit_only", reference["fit_samples"]),
                              ("full_checkpoint", "all_training", reference["training_samples"])):
        relative = reference[key]
        manifest = json.loads(_safe_file(root, relative).read_text(encoding="utf-8"))
        for field, expected in (("version", "ver19"), ("training_scope", scope),
                                ("fitted_samples", count), ("labels", reference["labels"]),
                                ("decision_rule", reference["decision_rule"]),
                                ("model_id", "klue/roberta-base"),
                                ("resolved_revision", "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"),
                                ("utterance_evidence", True), ("loss_mode", "nausea_ce_gce")):
            _equal(manifest.get(field), expected, f"{key} {field}")
        parent = PurePosixPath(relative).parent
        weight = (parent / _relative(manifest["weights"])).as_posix()
        assets = (parent / _relative(manifest["assets"])).as_posix()
        if weight not in reference["files"] or not any(p.startswith(assets + "/") for p in reference["files"]):
            raise ValueError("Unpinned model weights or assets")


def _zip_inventory(archive):
    """Validate every member before selecting a small, exact-size whitelist."""
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ValueError("ver19 ZIP contains too many members")
    names = {}
    for info in infos:
        if info.orig_filename != info.filename:
            raise ValueError("ver19 ZIP contains a normalized or truncated unsafe path")
        name = info.filename[:-1] if info.is_dir() else info.filename
        _relative(name)
        mode = info.external_attr >> 16
        if (stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
                or bool(info.flag_bits & 1)):
            raise ValueError("ver19 ZIP contains a symlink, special file, or encrypted member")
        if stat.S_ISDIR(mode) and not info.is_dir():
            raise ValueError("ver19 ZIP directory type mismatch")
        folded = name.casefold()
        if folded in names:
            raise ValueError("ver19 ZIP contains duplicate paths")
        names[folded] = (name, info)
    for name, info in names.values():
        for parent in PurePosixPath(name).parents:
            if str(parent) == ".":
                break
            entry = names.get(parent.as_posix().casefold())
            if entry and not entry[1].is_dir():
                raise ValueError("ver19 ZIP contains a file/directory collision")
    return {name: info for name, info in names.values()}


def _zip_selection(archive, reference):
    inventory = _zip_inventory(archive)
    marker = reference["fit_checkpoint"]
    prefixes = []
    for name in inventory:
        if name == marker:
            prefixes.append("")
        elif name.endswith("/" + marker):
            prefixes.append(name[:-len(marker)])
    if len(prefixes) != 1:
        raise ValueError("ver19 ZIP must contain exactly one matching source run")
    prefix = prefixes[0]
    selected = {}
    for relative, expected in reference["files"].items():
        info = inventory.get(prefix + relative)
        if info is None or info.is_dir():
            raise ValueError(f"ver19 ZIP is missing a required source file: {relative}")
        _equal(info.file_size, expected["size"], f"ZIP member size ({relative})")
        selected[relative] = info
    return selected


def _copy_stream(handle, destination, expected, budget=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with destination.open("xb") as output:
        while True:
            _check(budget, "source copy")
            block = handle.read(min(CHUNK_SIZE, expected["size"] - count + 1))
            if not block:
                break
            count += len(block)
            if count > expected["size"]:
                raise ValueError("Source stream exceeds its pinned size")
            output.write(block)
            digest.update(block)
    _equal(count, expected["size"], "copied file size")
    _equal(digest.hexdigest(), expected["sha256"], "copied file checksum")
    return digest.hexdigest()


def _copy_source(location, destination, reference, budget=None):
    verified = {}
    if location.is_dir():
        for relative, expected in reference["files"].items():
            path = _safe_file(location, relative)
            if not path.is_file():
                raise ValueError(f"Required ver19 source file is missing: {relative}")
            _equal(path.stat().st_size, expected["size"], f"source size ({relative})")
            with path.open("rb") as handle:
                verified[relative] = _copy_stream(handle, _safe_file(destination, relative), expected, budget)
    elif location.is_file() and location.suffix.lower() == ".zip":
        with zipfile.ZipFile(location) as archive:
            selected = _zip_selection(archive, reference)
            for relative, info in selected.items():
                with archive.open(info) as handle:
                    verified[relative] = _copy_stream(handle, _safe_file(destination, relative), reference["files"][relative], budget)
    else:
        raise ValueError("ver19 source must be a run directory or its ZIP archive")
    _verify_manifests(destination, reference)
    return verified


def _directory_runs(root, reference, budget=None):
    """Bounded discovery of run wrappers without traversing data/model files."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return []
    marker = reference["fit_checkpoint"]
    found, pending = [], [(root, 0)]
    while pending:
        directory, depth = pending.pop(0)
        _check(budget, "source discovery")
        if (directory / marker).is_file():
            found.append(directory)
            continue
        if depth >= 4:
            continue
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            if child.is_dir() and not child.is_symlink() and (
                    depth == 0 or child.name.startswith(("ver19", reference["run_id"]))):
                pending.append((child, depth + 1))
    return found


def _candidate_locations(source_location, reference, budget=None, search_roots=None):
    if source_location:
        path = Path(source_location).expanduser()
        if path.is_dir():
            runs = _directory_runs(path, reference, budget)
            if not runs:
                raise FileNotFoundError("The selected directory does not contain the required complete ver19 run")
            return runs
        if path.is_file() and path.suffix.lower() == ".zip":
            return [path]
        raise FileNotFoundError("ver19 source was not found; choose the complete ver19 result directory or ZIP")
    roots = ([Path(p) for p in search_roots] if search_roots is not None else [
        Path("/content/drive/MyDrive/DCC_AIdeas/results/mission3"),
        Path("/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver19"),
        Path("/content/drive/MyDrive"),
        Path.cwd() / "results/mission3",
    ])
    found = []
    for root in roots:
        _check(budget, "source discovery")
        if not root.is_dir():
            continue
        # Search only known project/result wrappers and ver19 named children.
        candidates = [root] if (search_roots is not None or root.name.startswith("ver19") or root.name == "mission3") else [
            root / "DCC_AIdeas/results/mission3", *sorted(root.glob("ver19*")),
        ]
        for path in candidates:
            if path.is_file() and path.suffix.lower() == ".zip":
                found.append(path)
            elif path.is_dir():
                found.extend(_directory_runs(path, reference, budget))
                found.extend(sorted(path.glob("ver19*.zip")))
    return list(dict.fromkeys(path.resolve() for path in found))


def _result(root, reference, verified):
    return {"root": root, "fit_checkpoint": root / reference["fit_checkpoint"],
            "full_checkpoint": root / reference["full_checkpoint"], "reference": reference,
            "source_hashes_verified": verified}


def _prepare_source(source_location, work_dir, reference, reference_hash, budget=None, search_roots=None):
    """Internal injectable helper for synthetic tests; public entry pins reference."""
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    prefix = "source_ver19_" + reference_hash[:16] + "_"
    if not source_location:
        for cached in sorted(work.glob(prefix + "*")):
            if not cached.is_dir() or cached.is_symlink() or cached.name.endswith(".partial"):
                continue
            marker = cached / "source_complete.json"
            if marker.is_file() and not marker.is_symlink():
                try:
                    complete = json.loads(marker.read_text(encoding="utf-8"))
                    _equal(complete["reference_sha256"], reference_hash, "cache reference")
                    verified = _verify_files(cached, reference, budget)
                    _equal(complete["artifact_hashes"], verified, "cache artifact hashes")
                    return _result(cached, reference, verified)
                except (ValueError, KeyError, OSError) as error:
                    if isinstance(error, TimeoutError):
                        raise
                    continue
    candidates = _candidate_locations(source_location, reference, budget, search_roots)
    if not candidates:
        raise FileNotFoundError("Required ver19 result not found. Keep the complete ver19 result folder or ZIP in DCC_AIdeas/results/mission3, or set SETTINGS['source_run'] explicitly.")
    failures = []
    for candidate in candidates:
        temporary = work / (prefix + uuid.uuid4().hex[:12] + ".partial")
        temporary.mkdir()
        try:
            verified = _copy_source(candidate, temporary, reference, budget)
            marker = {"format": "mission3_ver21_source_cache_v1", "reference_sha256": reference_hash,
                      "artifact_hashes": verified, "validation_files_copied": False}
            (temporary / "source_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            finished = temporary.with_suffix("")
            os.replace(temporary, finished)
            return _result(finished, reference, verified)
        except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            if isinstance(error, TimeoutError):
                raise
            failures.append(type(error).__name__ + ": " + str(error))
    raise ValueError("No candidate matched the pinned ver19 source. " + "; ".join(failures[:3]))


def prepare_source(source_location, work_dir, budget=None, search_roots=None):
    """Return a local, hash-verified minimal ver19 bundle from a folder or ZIP.

    Selection MUST use ``fit_checkpoint``. ``full_checkpoint`` is reserved for
    refitting a chosen head after Training-holdout selection is complete.
    Incomplete caches are never reused and Drive originals are never modified.
    """
    reference = _load_reference()
    return _prepare_source(source_location, work_dir, reference, REFERENCE_FILE_SHA256,
                           budget=budget, search_roots=search_roots)


def array_digest(values, dtype):
    values = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256(json.dumps(list(values.shape)).encode("ascii") + b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def dataset_digest(texts, targets):
    """Exactly the ver19 runtime text/order/target digest, without identifiers."""
    if len(texts) != len(targets):
        raise ValueError("Training texts and targets must have equal lengths")
    digest = hashlib.sha256()
    for text, target in zip(texts, targets):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(target, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _indices(values, total, name):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
            or (values < 0).any() or (values >= total).any()
            or len(np.unique(values)) != len(values)):
        raise ValueError(f"Invalid Training {name} indices")
    return values.astype("<i8")


def verify_training(texts, targets, fit, holdout, source):
    """Bind both source scopes and historical predictions to exact Training rows."""
    reference = _load_reference()
    _equal(source["reference"], reference, "prepared source reference")
    root = Path(source["root"])
    targets = data._binary(targets, reference["training_samples"], "targets")
    fit = _indices(fit, len(targets), "fit")
    holdout = _indices(holdout, len(targets), "holdout")
    if len(fit) + len(holdout) != len(targets) or len(np.union1d(fit, holdout)) != len(targets):
        raise ValueError("Training fit and holdout must partition all rows")
    _equal(len(fit), reference["fit_samples"], "fit sample count")
    _equal(len(holdout), reference["holdout_samples"], "holdout sample count")
    _equal(dataset_digest(texts, targets), reference["training_digest"], "Training text/target/order digest")
    hashes = reference["array_hashes"]
    for name, values, dtype in (("fit_indices", fit, "<i8"), ("holdout_indices", holdout, "<i8"),
                                ("training_targets", targets, "i1"), ("holdout_targets", targets[holdout], "i1")):
        _equal(array_digest(values, dtype), hashes[name], name)
    # Recheck small source arrays at use, even after a complete cache was loaded.
    for name in ("training_split.npz", "holdout_predictions.npz", "run_identity.json"):
        _equal(_hash_file(_safe_file(root, name)), reference["files"][name]["sha256"], name + " checksum")
    with np.load(root / "training_split.npz", allow_pickle=False) as saved:
        if not np.array_equal(fit, saved["fit"]) or not np.array_equal(holdout, saved["holdout"]):
            raise ValueError("Stored ver19 Training split does not match")
    with np.load(root / "holdout_predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(holdout, saved["indices"]) or not np.array_equal(targets[holdout], saved["targets"]):
            raise ValueError("Stored ver19 holdout indices or targets do not match")
        predictions = data._binary(saved["predictions"], len(holdout))
        _equal(array_digest(predictions, "i1"), hashes["predictions"], "holdout prediction hash")
        metrics = data.score_predictions(targets[holdout], predictions)
    _equal(metrics, reference["metrics"], "recomputed Training metrics")
    identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    _equal(identity["training_digest"], reference["training_digest"], "source Training identity")
    _verify_manifests(root, reference)
    return {"verified": True, "version": "ver19", "source_run_id": reference["run_id"],
            "training_digest": reference["training_digest"], "fit_samples": len(fit),
            "holdout_samples": len(holdout), "reference_file_sha256": REFERENCE_FILE_SHA256,
            "baseline_metrics": metrics, "validation_accessed": False,
            "selection_source_scope": "fit_only", "refit_source_scope": "all_training"}
