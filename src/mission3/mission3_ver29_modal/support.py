"""Persist launcher records outside immutable ver29 run manifests."""
from __future__ import annotations

import codecs
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

INPUT_SHA256 = "cfa2f2e0a63a5788848e175f92b685ac7b36ca62b99142664a3b589d2986f7ff"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as writer:
            json.dump(value, writer, ensure_ascii=False, indent=2, allow_nan=False)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def volume_path(volume_dir):
    root = Path(volume_dir).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise FileNotFoundError(
            "Volume 폴더가 없습니다. Files에서 기존 Volume을 연결하고 VOLUME_DIR를 확인하세요.")
    # Never create a missing mount as an ephemeral directory.
    return root.resolve()


def writable_dir(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("저장 경로가 Volume 밖을 가리킵니다.")
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError("결과 저장 경로에 심볼릭 링크를 사용할 수 없습니다.")
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def input_path(root, explicit="", expected_sha=INPUT_SHA256):
    path = Path(explicit).expanduser() if explicit else root / "mission3_ver29_inputs.zip"
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(
            "입력 ZIP이 없습니다. Volume 안에 mission3_ver29_inputs.zip을 업로드하세요. "
            "하위 폴더에 올렸다면 INPUT_ZIP에 정확한 경로를 넣으세요.")
    if file_hash(path) != expected_sha:
        raise ValueError("입력 ZIP 검증 실패. 제공된 mission3_ver29_inputs.zip을 다시 업로드하세요.")
    return path


def check_run_path(root, value, mode, resume=False):
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("SOURCE_RUN과 RESUME_RUN은 실행 폴더의 전체 경로여야 합니다.")
    path = path.resolve()
    if not path.is_relative_to(root) or not (path / "run_state.json").is_file():
        raise ValueError("같은 Volume 안에서 run_state.json이 있는 정확한 실행 폴더를 지정하세요.")
    state = json.loads((path / "run_state.json").read_text(encoding="utf-8"))
    expected_status = "paused" if resume else "screening_complete"
    expected_mode = mode if resume else "screening"
    if state.get("status") != expected_status or state.get("run_mode") != expected_mode:
        raise ValueError(f"선택한 실행은 {expected_mode}/{expected_status} 상태여야 합니다.")
    return str(path)


def prepare(settings, volume_dir, scratch, *, expected_sha=INPUT_SHA256):
    """Preflight before dependency installation; recipe validation stays in core."""
    from mission3_ver29.settings import validate_settings

    root = volume_path(volume_dir)
    archive = input_path(root, settings.get("input_zip", ""), expected_sha)
    mode = settings["run_mode"]
    source = check_run_path(root, settings.get("source_run", ""), mode)
    resume = check_run_path(root, settings.get("resume_run", ""), mode, resume=True)
    if source and resume:
        raise ValueError("SOURCE_RUN과 RESUME_RUN을 동시에 지정하지 마세요.")
    settings = dict(settings)
    settings.update(input_root=str(root), input_zip=str(archive), reference_run="",
                    source_run=source, resume_run=resume,
                    output_dir=str(root / "results/mission3/ver29"),
                    cache_dir=str(root / "cache/huggingface"),
                    work_dir=str(Path(scratch) / "data"), run_validation=mode == "full")
    settings = validate_settings(settings)
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    for relative in ("exports", "results/mission3/ver29", "cache/huggingface", "invocations"):
        writable_dir(root, relative)
    invocation = root / "invocations" / invocation_id
    invocation.mkdir(exist_ok=False)
    write_json(invocation / "settings.json", settings)
    write_json(invocation / "launcher.json", {
        "platform": "modal-notebook", "invocation_id": invocation_id,
        "volume_dir": str(root), "input_sha256": expected_sha,
        "status": "prepared", "submission_ready": False,
    })
    return settings, invocation


def install_command(requirements, constraints, torch_version, executable=None):
    """Use Modal's uv when available; keep the installed CUDA torch unchanged."""
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not match or not (int(match[1]) == 2 and int(match[2]) >= 6):
        raise RuntimeError("CUDA PyTorch 2.6 이상 3 미만이 필요합니다. PyTorch를 자동 교체하지 않습니다.")
    Path(constraints).write_text("torch==" + torch_version + "\n", encoding="utf-8")
    python = executable or sys.executable
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "install", "--python", python]
    elif importlib.util.find_spec("pip") is not None:
        command = [python, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed"]
    else:
        raise RuntimeError("uv와 pip를 찾지 못했습니다. Modal 기본 Python 이미지를 사용하세요.")
    return command + ["--quiet", "-c", str(constraints), "-r", str(requirements)]


def run_logged(command, log_path, deadline, *, cwd=None, env=None, margin=10):
    """Wait in the foreground, mirror UTF-8 output, and clean up the process group."""
    from mission3_ver29.budget import BudgetExceeded, _stop_process

    if deadline - time.monotonic() <= margin:
        raise BudgetExceeded("subprocess startup", deadline_monotonic=deadline)
    if not isinstance(command, (list, tuple)) or not command:
        raise TypeError("command must be a nonempty argument sequence")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # Refuse to truncate logs or silently start a duplicate cell invocation.
    with Path(log_path).open("xb", buffering=0) as writer, Path(log_path).open("rb") as reader:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=writer,
                                   stderr=subprocess.STDOUT, start_new_session=os.name == "posix")

        def drain(final=False):
            chunk = reader.read(256 * 1024)
            while chunk:
                print(decoder.decode(chunk), end="", flush=True)
                chunk = reader.read(256 * 1024)
            if final:
                print(decoder.decode(b"", final=True), end="", flush=True)

        try:
            while process.poll() is None:
                drain()
                if time.monotonic() >= deadline - margin:
                    raise BudgetExceeded("subprocess execution", deadline_monotonic=deadline)
                time.sleep(0.2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            _stop_process(process, min(deadline, time.monotonic() + margin))
            raise
        finally:
            drain(final=True)
            os.fsync(writer.fileno())
    return 0


def collect_result(result_file, output_root, previous_states, failure=None):
    """Attribute only this invocation's output; never bless a killed checkpoint."""
    try:
        result = json.loads(Path(result_file).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Invalid result")
    except (OSError, ValueError):
        result = {}
    output_root = Path(output_root).resolve()
    new_states = set(output_root.glob("*/run_state.json")) - set(previous_states)
    new_dirs = {path.parent.resolve() for path in new_states}
    run_dir = Path(result["run_dir"]).resolve() if result.get("run_dir") else None
    if run_dir not in new_dirs:
        result = {"status": "interrupted", "submission_ready": False}
        if len(new_dirs) == 1:
            result["run_dir"] = str(next(iter(new_dirs)))
    else:
        known = {"paused", "screening_complete", "complete", "failed"}
        if result.get("status") not in known:
            result.update(status="interrupted", submission_ready=False)
        if result.get("status") != "complete":
            result["submission_ready"] = False
        # A successful runtime result already written before a launcher failure
        # remains evidence, but the launcher failure is always reported separately.
    if failure:
        result["launcher_error_type"] = failure
    write_json(result_file, result)
    return result


def export_analysis(run_dir, exports):
    """Export bounded metadata and predictions without changing the run manifest."""
    run_dir = Path(run_dir).resolve()
    exports = Path(exports)
    exports.mkdir(parents=True, exist_ok=True)
    archive = exports / ("ver29_analysis_" + run_dir.name + ".zip")
    temporary = archive.with_suffix(".zip.tmp")
    allowed = {".json", ".csv", ".npz", ".npy", ".md", ".txt", ".py"}
    excluded = {"checkpoints", "checkpoint", "resume_checkpoint", "best_model", "model",
                "tokenizer", "cache", "data"}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(run_dir.rglob("*")):
                relative = path.relative_to(run_dir)
                if (not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed
                        or any(part in excluded for part in relative.parts)
                        or not path.resolve().is_relative_to(run_dir)
                        or path.stat().st_size > 20 * 1024 * 1024):
                    continue
                handle.write(path, (Path(run_dir.name) / relative).as_posix())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def execute(settings, invocation, package_parent, env, deadline):
    """Finish launcher records/analysis even when the supervised child fails."""
    invocation = Path(invocation)
    output_root = Path(settings["output_dir"])
    previous = set(output_root.glob("*/run_state.json"))
    result_file = invocation / "result.json"
    write_json(result_file, {"status": "starting", "submission_ready": False})
    failure = None
    try:
        run_logged([sys.executable, "-u", "-m", "mission3_ver29.run_notebook",
                    "--settings", str(invocation / "settings.json"), "--result-file", str(result_file)],
                   invocation / "training.log", deadline, cwd=str(package_parent), env=env)
    except BaseException as error:
        failure = error
    result = collect_result(result_file, output_root, previous,
                            type(failure).__name__ if failure else None)
    if result.get("run_dir"):
        archive = export_analysis(result["run_dir"], invocation.parent.parent / "exports")
        result["analysis_zip"] = str(archive)
    write_json(invocation / "launcher.json", result)
    print("\n실행 상태:", result.get("status"))
    print("결과 폴더:", result.get("run_dir", "생성되지 않음"))
    print("분석 ZIP:", result.get("analysis_zip", "생성되지 않음"))
    print("로그 폴더:", invocation)
    if result.get("status") == "screening_complete":
        print("비교 완료. 분석 ZIP을 확인한 뒤 full을 별도 실행하세요. 아직 제출 모델이 아닙니다.")
    elif result.get("status") == "paused":
        print("중간 저장 완료. 같은 L4와 RUN_MODE를 유지하고 위 결과 폴더를 RESUME_RUN에 넣으세요.")
    elif result.get("status") == "complete" and result.get("submission_ready"):
        print("전체 학습 및 제출 준비 완료. 공식 Validation 지표와 제출 manifest를 확인하세요.")
    else:
        print("완료 상태가 아닙니다. 오류 기록을 확인한 뒤 재개 여부를 판단하세요.")
    print("분석 ZIP에는 모델 가중치가 없습니다. Volume의 전체 결과 폴더를 보관하세요.")
    print("저장된 파일을 확인하고 Stop kernel로 GPU를 종료하세요.")
    if failure:
        if isinstance(failure, KeyboardInterrupt):
            raise failure
        raise RuntimeError("ver29 실행 오류: " + type(failure).__name__ + ". 위 로그 폴더를 확인하세요.") from None
    if result.get("status") not in {"screening_complete", "paused", "complete"}:
        raise RuntimeError("ver29 결과가 완성되지 않았습니다. result.json을 확인하세요.")
    return result
