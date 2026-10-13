"""Launch the bounded QLoRA representation comparison in a fresh interpreter."""
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
    "__init__.py", "data.py", "analysis.py", "bootstrap.py", "budget.py", "batching.py", "trainer.py", "runtime.py",
    "run_notebook.py", "build_notebook.py", "requirements.txt", "README.md", "model_specs.json", "baseline.py", "baseline_reference.json",
)


def _failure_report(error, stage):
    """Report diagnostic categories without copying raw text or exception values."""
    category = "unclassified"
    reason = "실행 중 오류가 발생했습니다. 실패 단계와 코드 위치를 포함한 오류 보고서를 확인하세요."
    try:
        message = str(error)
    except Exception:
        message = ""
    if stage == "settings_load":
        category = "settings_unreadable"
        reason = "실행 설정을 읽지 못했습니다. 수정된 노트북을 처음 셀부터 모두 실행하세요."
    elif isinstance(error, ImportError):
        category = "dependency_import_failed"
        reason = "필요한 패키지를 불러오지 못했습니다. 패키지 준비 셀의 설치 결과를 확인하세요."
    elif ("Colab에서 GPU 런타임을 선택하세요" in message
          or "CUDA GPU" in message or "GPU required" in message):
        category = "gpu_unavailable"
        reason = "GPU가 연결되지 않았습니다. Colab 런타임 유형을 GPU로 설정한 뒤 처음부터 실행하세요."
    elif "CUDA out of memory" in message or type(error).__name__ == "OutOfMemoryError":
        category = "out_of_memory"
        reason = "GPU 메모리가 부족합니다. 배치 측정 기록과 GPU 메모리를 확인하고 다른 실행을 종료하세요. 가장 작은 배치도 실패하면 GPU 또는 입력 길이 검토가 필요합니다. 재실행에는 추가 리소스가 듭니다."
    elif "Input ZIP is missing:" in message:
        category = "input_zip_missing"
        reason = "입력 ZIP을 찾지 못했습니다. Drive 연결과 SETTINGS['input_zip']의 실제 경로를 확인하세요."
    elif isinstance(error, ValueError) and "ver26 baseline:" in message:
        category = "baseline_reference_invalid"
        reason = "ver26 기준 결과의 무결성 또는 예측 비교를 확인하지 못했습니다. baseline_run을 지정된 완료 실행 폴더로 맞추고 누락된 원본 결과 파일을 복원하세요. 다른 실행으로 교체하지 말고 runtime.json의 실패 단계를 확인하세요. 완료 표시가 없는 결과는 제출에 사용하지 마세요."
    elif isinstance(error, ValueError) and ("screening-only" in message or "source_run" in message):
        category = "screening_mode_required"
        reason = "ver27은 4,000개 3 epoch 진단 전용입니다. run_mode='pilot', source_run=''을 유지하세요. 전체 fit에서 epoch 확인 전에는 전체 학습을 시작할 수 없습니다."
    elif type(error).__name__ == "BudgetExceeded":
        category = "time_budget_exhausted"
        reason = "설정한 작업 예산으로 현재 실행 단계를 완료하기 어렵거나 시간이 끝나 중단했습니다. 저장된 시간 추정을 확인하세요. 완료 표시가 없는 결과는 제출에 사용하지 마세요."
    elif any(token in message for token in (
        "huggingface.co", "Connection error", "Can't load tokenizer", "Can't load the model",
        "GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError",
    )) or type(error).__name__ in {"GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError"}:
        category = "pretrained_model_unavailable"
        reason = "고정된 공개 모델 또는 토크나이저를 내려받지 못했습니다. Colab 인터넷 연결과 model_specs.json의 모델 공개 상태를 확인하세요."
    elif stage == "source_read":
        category = "runtime_sources_unreadable"
        reason = "노트북에 포함된 실행 코드 일부를 읽지 못했습니다. 수정된 노트북을 처음 셀부터 모두 실행하세요."
    elif stage == "result_write":
        category = "result_record_failed"
        reason = "실행 결과 위치를 기록하지 못했습니다. Drive 결과 폴더의 완료 표시와 오류 보고서를 확인하세요."
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
        if not isinstance(settings, dict):
            raise ValueError("Notebook settings must be a JSON object")
        stage = "runtime_import"
        name = "mission3_ver27_execution"
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
