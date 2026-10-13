"""Launch the bounded ver23 experiment in a fresh interpreter after installation."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback

SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "budget.py", "neural.py", "head.py", "adaptation.py",
    "source_bundle.py", "source_reference.json", "runtime.py", "portability.py", "inference.py",
    "run_notebook.py", "requirements.txt", "README.md",
)


def _failure_report(error, stage):
    """Describe failures without copying sample text, paths, or exception values."""
    category = "unclassified"
    reason = "실행 중 오류가 발생했습니다. 아래 단계와 코드 위치를 포함한 오류 기록을 확인하세요."
    try:
        message = str(error)
    except Exception:
        message = ""
    if stage == "settings_load":
        category = "settings_unreadable"
        reason = "실행 설정을 읽지 못했습니다. 수정된 노트북을 처음 셀부터 모두 실행하세요."
    elif isinstance(error, ImportError):
        category = "dependency_import_failed"
        reason = "실행에 필요한 패키지를 불러오지 못했습니다. 패키지 준비 셀의 설치 결과를 확인하세요."
    elif "Colab에서 GPU 런타임을 선택하세요" in message:
        category = "gpu_unavailable"
        reason = "GPU가 연결되지 않았습니다. Colab 런타임 유형을 GPU로 설정한 뒤 처음부터 실행하세요."
    elif "CUDA out of memory" in message or type(error).__name__ == "OutOfMemoryError":
        category = "out_of_memory"
        reason = "모델 실행에 필요한 메모리가 부족합니다. GPU 종류와 메모리 사용량을 확인하세요."
    elif "Current fit-only encoder does not reproduce the pinned ver19 holdout predictions" in message:
        category = "baseline_prediction_mismatch"
        reason = "현재 환경에서 기존 ver19 내부 평가 예측이 정확히 재현되지 않아 새 후보 학습을 중단했습니다. 오류 기록으로 재현 조건을 확인해야 합니다."
    elif "Input ZIP is missing:" in message:
        category = "input_zip_missing"
        reason = "입력 ZIP을 찾지 못했습니다. Drive 연결과 SETTINGS['input_zip']의 실제 파일 경로를 확인하세요."
    elif any(token in message for token in (
        "Required ver19 result not found", "ver19 source was not found",
        "selected directory does not contain the required complete ver19 run",
    )):
        category = "ver19_source_missing"
        reason = "필요한 ver19 결과를 찾지 못했습니다. 가중치가 포함된 전체 결과 폴더 또는 ZIP의 Drive 경로를 SETTINGS['source_run']에 지정하세요."
    elif any(token in message for token in (
        "ver19 source mismatch:", "No candidate matched the pinned ver19 source",
        "Required ver19 source file is missing:", "ver19 ZIP is missing a required source file:",
    )):
        category = "ver19_source_verification_failed"
        reason = "ver19 결과가 요구된 원본과 일치하지 않거나 필수 파일이 누락됐습니다. 전체 결과 폴더 또는 ZIP을 확인하세요."
    elif type(error).__name__ == "BudgetExceeded":
        category = "time_budget_exhausted"
        reason = "설정된 작업 시간 예산이 끝나 실행을 중단했습니다. 완료 표시가 없는 결과는 제출에 사용하지 마세요."
    elif stage == "source_read":
        category = "runtime_sources_unreadable"
        reason = "노트북에 포함된 실행 코드 일부를 읽지 못했습니다. 수정된 노트북을 처음 셀부터 모두 실행하세요."
    elif stage == "result_write":
        category = "result_record_failed"
        reason = "실행 결과 위치를 기록하지 못했습니다. 오류 기록과 Drive 결과 폴더의 완료 표시를 확인하세요."
    frames = [{"file": Path(frame.f_code.co_filename).name, "line": line,
               "function": frame.f_code.co_name}
              for frame, line in traceback.walk_tb(error.__traceback__)]
    report = {"status": "failed", "stage": stage, "error_type": type(error).__name__,
              "category": category, "reason": reason, "frames": frames[-20:]}
    if (isinstance(error, ImportError) and isinstance(error.name, str)
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", error.name, flags=re.ASCII)):
        report["module_name"] = error.name
    if isinstance(error, OSError) and type(error.errno) is int:
        report["errno"] = error.errno
    return report


def _write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    error_file = root / "notebook_error.json"
    stage = "initialize"
    try:
        _write_report(error_file, {"status": "starting", "stage": stage})
        stage = "settings_load"
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
        stage = "runtime_import"
        name = "mission3_ver23_execution"
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                      submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        runtime = importlib.import_module(name + ".runtime")
        stage = "source_read"
        sources = {name: (root / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
        stage = "runtime_run"
        result = runtime.run(settings, runtime_sources=sources)
        stage = "result_write"
        Path(args.result_file).write_text(json.dumps(str(result)), encoding="utf-8")
        _write_report(error_file, {"status": "complete", "stage": "complete"})
    except Exception as error:
        report = _failure_report(error, stage)
        try:
            _write_report(error_file, report)
        except OSError:
            report["diagnostic_file_saved"] = False
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
