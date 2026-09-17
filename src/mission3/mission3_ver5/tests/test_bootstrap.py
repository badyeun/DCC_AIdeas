"""Exercise the uploaded input bundle with small, real ZIP files."""
import hashlib
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch
import zipfile

import pytest

from mission3_ver5 import bootstrap


def _bundle(tmp_path, *, bad_hash=False, extra=None):
    model_names = [
        "mission3_ver4_model.ckpt", "splits.json", "calibration_predictions.npz", "sparse.joblib",
        *[f"fold_{i}.ckpt" for i in range(1, 6)], "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json", "tokenizer/vocab.txt", "tokenizer/special_tokens_map.json",
    ]
    assets = {bootstrap.MODEL_PREFIX + name: f"dummy asset {name}\n".encode() for name in model_names}
    assets[bootstrap.TRAIN_PREFIX + "train.json"] = json.dumps(
        {"utterances": [{"text": "서울 Training 대화"}], "symptom": ["두통"]}, ensure_ascii=False).encode()
    assets[bootstrap.VALID_PREFIX + "valid.json"] = json.dumps(
        {"utterances": [{"text": "서울 Validation 대화"}], "symptom": []}, ensure_ascii=False).encode()
    manifest = {
        "format": "mission3_ver5_inputs_v1", "training_count": 1, "validation_count": 1,
        "files": [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in assets.items()],
    }
    if bad_hash:
        manifest["files"][0]["sha256"] = "0" * 64
    archive_path = tmp_path / "mission3_ver5_inputs.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, data in assets.items():
            archive.writestr(name, data)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False))
        archive.writestr("README_UPLOAD.txt", "Upload this ZIP without extracting it in Drive.")
        if extra is not None:
            archive.writestr(extra, b"outside")
    return archive_path, assets


def test_restore_exact_inputs_and_reuse_preserves_uploaded_zip(tmp_path, capsys):
    archive_path, assets = _bundle(tmp_path)
    original_archive = archive_path.read_bytes()
    original_stat = archive_path.stat()
    destination = tmp_path / "local_inputs"
    result = bootstrap.restore_input_bundle(archive_path, destination)
    assert result == destination / "DCC_AIdeas"
    assert len(assets) == 15
    for name, expected in assets.items():
        assert (destination / name).read_bytes() == expected
    marker = destination / ".mission3_bundle_complete.json"
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert set(recorded["files"]) == set(assets)
    with zipfile.ZipFile(archive_path) as archive:
        assert recorded["manifest_sha256"] == hashlib.sha256(archive.read("bundle_manifest.json")).hexdigest()
    timestamps = {name: (destination / name).stat().st_mtime_ns for name in assets}
    original_open = zipfile.ZipFile.open

    def manifest_only(archive, name, *args, **kwargs):
        actual_name = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert actual_name == "bundle_manifest.json", "Verified files must not be extracted again"
        return original_open(archive, name, *args, **kwargs)

    with patch.object(zipfile.ZipFile, "open", new=manifest_only):
        assert bootstrap.restore_input_bundle(archive_path, destination) == result
    assert "재사용" in capsys.readouterr().out
    assert {name: (destination / name).stat().st_mtime_ns for name in assets} == timestamps
    assert archive_path.read_bytes() == original_archive
    assert archive_path.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_bad_manifest_hash_leaves_no_completion_marker_or_partial_file(tmp_path):
    archive_path, assets = _bundle(tmp_path, bad_hash=True)
    original = archive_path.read_bytes()
    destination = tmp_path / "invalid_inputs"
    with pytest.raises(ValueError, match="검증 실패"):
        bootstrap.restore_input_bundle(archive_path, destination)
    assert not (destination / ".mission3_bundle_complete.json").exists()
    assert not list(destination.rglob("*.tmp"))
    assert not any(path.is_file() for path in destination.rglob("*"))
    assert not (destination / next(iter(assets))).exists()
    assert archive_path.read_bytes() == original


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "DCC_AIdeas/../../outside.txt",
                                  "DCC_AIdeas\\outside.txt", "C:/outside.txt"])
def test_unsafe_zip_path_is_rejected_before_extraction(tmp_path, name):
    archive_path, _ = _bundle(tmp_path, extra=name)
    destination = tmp_path / "unsafe_inputs"
    with pytest.raises(ValueError, match="Unsafe bundle path"):
        bootstrap.restore_input_bundle(archive_path, destination)
    assert not list(destination.iterdir())
    assert not (tmp_path / "outside.txt").exists()


def test_zip_symlink_is_rejected_before_extraction(tmp_path):
    link = zipfile.ZipInfo("DCC_AIdeas/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive_path, _ = _bundle(tmp_path, extra=link)
    destination = tmp_path / "symlink_inputs"
    with pytest.raises(ValueError, match="Links"):
        bootstrap.restore_input_bundle(archive_path, destination)
    assert not list(destination.iterdir())


def test_existing_destination_symlink_cannot_escape_extraction_root(tmp_path):
    archive_path, _ = _bundle(tmp_path)
    destination, outside = tmp_path / "local_inputs", tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "DCC_AIdeas").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes extraction directory"):
        bootstrap.restore_input_bundle(archive_path, destination)
    assert not list(outside.iterdir())
    assert not (destination / ".mission3_bundle_complete.json").exists()


def test_changed_extracted_file_is_repaired_without_rewriting_other_assets(tmp_path):
    archive_path, assets = _bundle(tmp_path)
    destination = tmp_path / "repair_inputs"
    bootstrap.restore_input_bundle(archive_path, destination)
    timestamps = {name: (destination / name).stat().st_mtime_ns for name in assets}
    name = bootstrap.MODEL_PREFIX + "fold_3.ckpt"
    damaged = destination / name
    # Keep the same size so this covers content verification, not only size checks.
    damaged.write_bytes(b"x" * len(assets[name]))
    os.utime(damaged, ns=(damaged.stat().st_atime_ns, timestamps[name] + 1_000_000_000))
    bootstrap.restore_input_bundle(archive_path, destination)
    assert damaged.read_bytes() == assets[name]
    for other_name, expected in assets.items():
        assert (destination / other_name).read_bytes() == expected
        if other_name != name:
            assert (destination / other_name).stat().st_mtime_ns == timestamps[other_name]
    assert not list(destination.rglob("*.tmp"))
    marker = json.loads((destination / ".mission3_bundle_complete.json").read_text(encoding="utf-8"))
    assert marker["files"][name] == [damaged.stat().st_size, damaged.stat().st_mtime_ns]
