"""Prepare one pinned public encoder; training and inference only use local files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid

MODEL_ID = "klue/roberta-large"
MODEL_REVISION = "28d911204e9022eda172571ca8cc61eaffd942f7"
MANIFEST_NAME = "pretrained_manifest.json"
FORMAT = "mission3_ver8_pretrained_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pretrained(directory) -> dict:
    """Return provenance only after checking a complete, safe local artifact set.

    This function never contacts Hugging Face and does not load model weights.
    File hashes are returned under ``files[name]['sha256']`` for run identities.
    """
    root = Path(directory).expanduser().resolve()
    marker = root / MANIFEST_NAME
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"사전학습 모델 준비가 완료되지 않았습니다: {marker}")
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"잘못된 사전학습 모델 manifest: {marker}") from error
    if (metadata.get("format") != FORMAT or metadata.get("complete") is not True
            or not isinstance(metadata.get("model_id"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("revision", "")))):
        raise ValueError("사전학습 모델 manifest 형식/고정 revision이 올바르지 않습니다.")
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("사전학습 모델 파일 목록이 없습니다.")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required <= files.keys() or not ("model.safetensors" in files
                                           or "model.safetensors.index.json" in files):
        raise ValueError("encoder config/safetensors/fast tokenizer 파일이 빠졌습니다.")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {MANIFEST_NAME}:
        raise ValueError("사전학습 모델의 실제 파일 목록과 manifest가 다릅니다.")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or name == MANIFEST_NAME):
            raise ValueError(f"안전하지 않은 사전학습 파일 경로: {name}")
        path = root / name
        if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root)
                or not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError(f"정상적인 로컬 사전학습 파일이 아닙니다: {name}")
        if (not isinstance(expected, dict) or path.stat().st_size != expected.get("size")
                or _sha256(path) != expected.get("sha256")):
            raise ValueError(f"사전학습 파일 무결성 검사 실패: {name}")
    return metadata


def prepare_pretrained(destination, model_id=MODEL_ID, revision=MODEL_REVISION) -> dict:
    """Download a pinned encoder/tokenizer once and atomically publish local files.

    Only this explicit preparation entry point may use the network. It receives no
    competition texts or labels. A valid completed directory is reused offline.
    A failed preparation leaves the existing destination untouched.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("revision은 변경되지 않는 40자리 Hugging Face commit SHA여야 합니다.")
    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise ValueError("사전학습 모델 저장 경로에 심볼릭 링크를 사용할 수 없습니다.")
    root = raw_destination.resolve()
    if root.exists():
        try:
            metadata = validate_pretrained(root)
        except ValueError:
            # Refuse to replace unrelated directories; our own corrupt bundle may be repaired.
            marker = root / MANIFEST_NAME
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
            if previous.get("format") != FORMAT:
                raise ValueError(f"완료 manifest가 없는 기존 폴더입니다. 비어 있는 새 경로를 사용하세요: {root}")
        else:
            if metadata["model_id"] != model_id or metadata["revision"] != revision:
                raise ValueError("기존 사전학습 모델과 요청한 모델/revision이 다릅니다. 새 경로를 사용하세요.")
            print(f"검증된 사전학습 모델 재사용: {root}", flush=True)
            return metadata
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + root.name + ".preparing-", dir=root.parent))
    backup = None
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        print(f"공개 사전학습 encoder 다운로드: {model_id}@{revision}", flush=True)
        options = {"revision": revision, "local_files_only": False, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **options)
        # Keep the pretrained MLM head for TAPT; classifiers load only the encoder.
        model = AutoModelForMaskedLM.from_pretrained(model_id, use_safetensors=True, **options)
        tokenizer.save_pretrained(staging)
        model.save_pretrained(staging, safe_serialization=True)
        del model, tokenizer
        # Prove that the published form is self-contained and works offline.
        check_model = AutoModelForMaskedLM.from_pretrained(staging, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True)
        check_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        if not check_tokenizer.is_fast:
            raise ValueError("window 생성에 사용할 fast tokenizer가 필요합니다.")
        del check_model, check_tokenizer
        metadata = {
            "format": FORMAT, "complete": True, "model_id": model_id, "revision": revision,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {path.relative_to(staging).as_posix(): {"size": path.stat().st_size, "sha256": _sha256(path)}
                      for path in sorted(staging.rglob("*")) if path.is_file()},
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
        validate_pretrained(staging)
        if root.exists():
            backup = root.with_name("." + root.name + ".previous-" + uuid.uuid4().hex)
            os.replace(root, backup)
        try:
            os.replace(staging, root)
        except BaseException:
            if backup is not None:
                os.replace(backup, root)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        print(f"사전학습 모델 준비 완료: {root}", flush=True)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    prepare_pretrained(args.destination, model_id=args.model_id, revision=args.revision)


if __name__ == "__main__":
    main()
