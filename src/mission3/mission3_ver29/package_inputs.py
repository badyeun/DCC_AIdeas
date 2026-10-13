"""Build verified labels and a completed ver28 analysis for a private Dataset.

Original JSON bytes are copied unchanged. Model checkpoints, cached predictions,
and unrelated data are excluded except the explicitly supplied analysis ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from .bootstrap import TRAIN_PREFIX, VALID_PREFIX, _relative, _selected_files


MANIFEST_NAME = "bundle_manifest.json"
DEFAULT_COUNTS = (29_200, 3_640)
REFERENCE_PREFIX = "ver28_reference/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(files: list[dict[str, Any]]) -> tuple[int, int]:
    return tuple(sum(row["path"].startswith(prefix) for row in files)
                 for prefix in (TRAIN_PREFIX, VALID_PREFIX))


def _check_counts(files: list[dict[str, Any]], expected: tuple[int, int]) -> None:
    if _counts(files) != expected:
        raise ValueError(f"Label counts differ: expected {expected}, found {_counts(files)}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _validate_reference(raw: bytes) -> None:
    """Check that the unchanged ZIP identifies one completed screening run."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        for info in archive.infolist():
            _relative(info.orig_filename)
        states = [name for name in names if name.endswith("/run_state.json")]
        if len(states) != 1:
            raise ValueError("Reference analysis must identify one completed ver28 screening run")
        state = json.loads(archive.read(states[0]))
        if (state.get("format") != "mission3_ver28_v1"
                or state.get("run_mode") != "screening" or state.get("status") != "screening_complete"):
            raise ValueError("Reference analysis must be a completed ver28 screening run")
        prefix = states[0].rsplit("/", 1)[0] + "/"
        required = {prefix + name for name in ("split_indices.npz", "split_manifest.json",
                                               "development_predictions.npz", "confirmation_predictions.npz",
                                               "selection.json", "artifact_manifest.json")}
        if not required.issubset(names):
            raise ValueError("Reference analysis is missing screening provenance or predictions")


def verify_package(path: Path, expected_counts: tuple[int, int] = DEFAULT_COUNTS) -> dict[str, Any]:
    """Re-read every label and require its content to match the stored manifest."""
    with zipfile.ZipFile(path) as archive:
        files, signature = _selected_files(archive)
        _check_counts(files, expected_counts)
        manifest = json.loads(archive.read(MANIFEST_NAME))
        reference = manifest.get("reference_analysis")
        allowed = {row["path"] for row in files} | {MANIFEST_NAME}
        if reference is not None:
            if not isinstance(reference, dict) or set(reference) != {"path", "size", "sha256"}:
                raise ValueError("Invalid reference analysis metadata")
            name = reference["path"]
            _relative(name)
            if (not name.startswith(REFERENCE_PREFIX) or not name.endswith(".zip")
                    or len(name.split("/")) != 2):
                raise ValueError("Reference analysis must be one ZIP under ver28_reference")
            raw = archive.read(name)
            if len(raw) != reference["size"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError("Reference analysis content differs from manifest")
            _validate_reference(raw)
            allowed.add(name)
        if set(archive.namelist()) != allowed:
            raise ValueError("Compact input archive must contain only labels, its manifest, and the declared reference")
        if len(manifest["files"]) != len(files):
            raise ValueError("Compact manifest must describe labels only")
        for row in files:
            raw = archive.read(row["path"])
            if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Label content differs from manifest")
        return {
            "training_count": expected_counts[0],
            "validation_count": expected_counts[1],
            "label_file_count": len(files),
            "label_uncompressed_bytes": sum(row["size"] for row in files),
            "manifest_sha256": signature,
            "reference_analysis": reference,
            "verification": "all_label_and_declared_reference_sizes_and_sha256_match",
        }


def package_inputs(source: Path, destination: Path,
                   expected_counts: tuple[int, int] = DEFAULT_COUNTS,
                   *, reference_analysis: Path | None = None) -> dict[str, Any]:
    """Create a new archive atomically; refuse to replace any existing output."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source archive is missing: {source}")
    requested_destination = Path(destination).expanduser()
    if requested_destination.is_symlink():
        raise ValueError("Output archive cannot be a symlink")
    destination = requested_destination.resolve()
    if source == destination or destination.exists():
        raise FileExistsError("Output already exists; choose a new output path")
    if any(type(count) is not int or count < 1 for count in expected_counts) or len(expected_counts) != 2:
        raise ValueError("Expected Training and Validation counts must be positive integers")
    before = source.stat()
    reference = Path(reference_analysis).expanduser().resolve() if reference_analysis is not None else None
    reference_bytes = reference.read_bytes() if reference is not None else None
    reference_metadata = None
    if reference is not None:
        if reference == destination or reference == source:
            raise ValueError("Reference analysis must be separate from the input and output archives")
        _validate_reference(reference_bytes)
        reference_metadata = {"path": REFERENCE_PREFIX + reference.name, "size": len(reference_bytes),
                              "sha256": hashlib.sha256(reference_bytes).hexdigest()}
    temporary: Path | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source) as original:
            files, source_signature = _selected_files(original)
            _check_counts(files, expected_counts)
            manifest = {
                "format": "mission3_ver5_inputs_v1",
                "producer": "mission3_ver29.package_inputs",
                "privacy": "private_dataset_only",
                "content": "original_label_json_bytes_and_declared_ver28_analysis",
                "training_count": expected_counts[0],
                "validation_count": expected_counts[1],
                "files": files,
            }
            if reference_metadata is not None:
                manifest["reference_analysis"] = reference_metadata
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".ver29_inputs_",
                                             suffix=".tmp", delete=False) as writer:
                temporary = Path(writer.name)
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as compact:
                for row in files:
                    raw = original.read(row["path"])
                    if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                        raise ValueError("Source label content differs from manifest")
                    compact.writestr(_zip_info(row["path"]), raw, compresslevel=6)
                if reference_metadata is not None:
                    compact.writestr(_zip_info(reference_metadata["path"]), reference_bytes, compresslevel=6)
                raw_manifest = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode("utf-8")
                compact.writestr(_zip_info(MANIFEST_NAME), raw_manifest, compresslevel=6)
            source_label_compressed_bytes = sum(original.getinfo(row["path"]).compress_size for row in files)
        verification = verify_package(temporary, expected_counts)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source archive changed while packaging")
        compact_bytes = temporary.stat().st_size
        report = {
            "format": "mission3_ver29_input_packaging_report_v1",
            "source_path": str(source),
            "source_archive_bytes": before.st_size,
            "source_manifest_sha256": source_signature,
            "source_label_compressed_bytes": source_label_compressed_bytes,
            "output_path": str(destination),
            "output_archive_bytes": compact_bytes,
            "output_archive_sha256": _sha256(temporary),
            "bytes_saved": before.st_size - compact_bytes,
            "size_fraction_of_source": compact_bytes / before.st_size,
            "original_json_bytes_preserved": True,
            "old_models_included": False,
            "reference_archive_bytes_preserved": reference_metadata is not None,
            "required_dataset_visibility": "private",
            **verification,
        }
        # Path.rename refuses an existing target on Windows. The prior existence
        # checks additionally guard ordinary invocations on other platforms.
        if destination.exists():
            raise FileExistsError("Output appeared during packaging; refusing to overwrite it")
        temporary.rename(destination)
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Packaging report already exists; choose a new path")
    if report_path.resolve() in {args.source.resolve(), args.output.resolve(), args.reference_analysis.resolve()}:
        raise ValueError("Report must be separate from source and output archives")
    report = package_inputs(args.source, args.output, reference_analysis=args.reference_analysis)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_path.parent,
                                         prefix=".ver29_report_", suffix=".tmp", delete=False) as writer:
            temporary = Path(writer.name)
            json.dump(report, writer, ensure_ascii=False, indent=2)
            writer.write("\n")
        if report_path.exists():
            raise FileExistsError("Packaging report appeared during packaging")
        temporary.rename(report_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
