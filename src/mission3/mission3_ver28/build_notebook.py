"""Build a standalone Kaggle notebook containing the ver28 runtime sources."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def source_files():
    """Include all runtime modules without embedding tests or generated outputs."""
    modules = sorted(path.name for path in HERE.glob("*.py")
                     if not path.name.startswith("test_") and path.name != "conftest.py")
    return modules + ["model_specs.json", "requirements.txt", "README.md"]


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in source_files()}
    required = {"__init__.py", "settings.py", "run_notebook.py", "runtime.py", "budget.py", "trainer.py"}
    if not required.issubset(sources):
        raise FileNotFoundError("Incomplete ver28 runtime source package")
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source):
        cell = {"cell_type": kind, "id": "ver28-" + identifier, "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver28 — KURE + KLUE / Kaggle

    **처음에는 설정을 바꾸지 않고 screening을 실행하세요.**
    Training 내부 fit 20,440개로 KURE-v1과 KLUE를 각각 최대 3 epoch 학습합니다.
    development에서 두 단독 모델과 고정 1:1 평균을 비교한 뒤 선택을 고정하고 confirmation을 확인합니다.
    전체 Training 학습은 자동 시작하지 않습니다. 결과 검토 후 별도 `full` 실행으로 진행합니다.

    1. `mission3_ver28_inputs.zip`을 **Private Dataset**으로 업로드합니다.
    2. 오른쪽 **Add Input**에서 데이터셋을 연결합니다. ZIP이 자동으로 풀려도 됩니다.
    3. **Settings → Accelerator → GPU T4 ×2**, **Internet → On**을 선택합니다.
    4. 아래 첫 코드 셀을 확인하고 **Save Version → Save & Run All**로 한 번 실행합니다.
    5. 실행이 끝나면 저장된 버전의 **Output**에서 `ver28_analysis_*.zip`을 내려받습니다.

    **주의:** 먼저 Run All로 학습한 뒤 Save & Run All을 누르면 처음부터 다시 학습합니다.
    이 노트북은 GPU 한 개에서 두 모델을 순서대로 학습합니다. T4 두 개의 메모리를 합치지 않습니다.
    설치·다운로드·평가·저장을 포함해 실행당 최대 8시간이며, 부족하면 checkpoint를 저장하고 `paused`로 종료합니다.
    시간은 실제 배치 측정 후 안내합니다. Kaggle의 강제 종료나 가용량은 코드가 보장하지 않습니다.

    점수 0.7 달성을 보장하지 않습니다. 기존 ver25~27·친구 모델·원본 데이터는 수정하지 않습니다.
    외부 학습 데이터, 정답 변경, threshold/offset/앙상블 혼합비 검색은 하지 않습니다.
    development와 confirmation에는 이전 실험의 사용 이력이 있습니다.
    """)
    add("code", "settings", """
    # Edit only these fields. Keep defaults for the first run.
    RUN_MODE = "screening"  # "screening" or "full"
    CANDIDATE = "kure"  # Optional second experiment: "qwen_embedding"
    SOURCE_RUN = ""  # Full only: attached completed screening run folder
    RESUME_RUN = ""  # Resume only: attached incomplete run folder of the same phase
    INPUT_ZIP = ""  # Empty: detect the attached ZIP or extracted data directory
    TIME_BUDGET_MINUTES = 480  # Maximum 480; installation time is included

    import time
    if type(TIME_BUDGET_MINUTES) is not int or not 30 <= TIME_BUDGET_MINUTES <= 480:
        raise ValueError("TIME_BUDGET_MINUTES must be an integer between 30 and 480")
    INVOCATION_STARTED = time.monotonic()
    DEADLINE_MONOTONIC = INVOCATION_STARTED + TIME_BUDGET_MINUTES * 60
    PREPARED = False
    DEPENDENCIES_PREPARED = False
    print("실행 모드:", RUN_MODE, "· 후보:", CANDIDATE)
    print("이번 실행의 최대 시간:", TIME_BUDGET_MINUTES, "분")
    """)
    add("code", "restore", """
    # Restore the embedded package. No separate code upload is needed.
    import importlib
    import json
    import os
    from pathlib import Path
    import sys
    import tempfile
    import uuid

    RUNTIME_SOURCES = __SOURCES__
    if "DEADLINE_MONOTONIC" not in globals():
        raise RuntimeError("첫 설정 셀부터 실행하세요.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("이 노트북은 Kaggle용입니다. Kaggle에 업로드한 뒤 실행하세요.")
    if time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("실행 시간이 끝났습니다. 새 실행을 시작하려면 처음 셀부터 실행하세요.")
    scratch_root = Path("/kaggle/temp")
    scratch_root.mkdir(parents=True, exist_ok=True)
    package_parent = Path(tempfile.mkdtemp(prefix="ver28_code_", dir=scratch_root))
    package_dir = package_parent / "mission3_ver28"
    package_dir.mkdir()
    for name, source in RUNTIME_SOURCES.items():
        (package_dir / name).write_text(source, encoding="utf-8")
    # A fresh invocation uses exactly the embedded source revision.
    for name in list(sys.modules):
        if name == "mission3_ver28" or name.startswith("mission3_ver28."):
            del sys.modules[name]
    sys.path.insert(0, str(package_parent))
    from mission3_ver28.settings import default_settings
    from mission3_ver28.budget import run_supervised
    SETTINGS = default_settings()
    SETTINGS.update(run_mode=RUN_MODE, candidate=CANDIDATE, source_run=SOURCE_RUN,
                    resume_run=RESUME_RUN, input_zip=INPUT_ZIP,
                    output_dir="/kaggle/working/results/mission3/ver28",
                    work_dir="/kaggle/temp/mission3_ver28", cache_dir="/kaggle/temp/hf_ver28",
                    time_budget_minutes=TIME_BUDGET_MINUTES,
                    run_validation=(RUN_MODE == "full"),
                    invocation_started_monotonic=INVOCATION_STARTED,
                    deadline_monotonic=DEADLINE_MONOTONIC)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_parent) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["HF_HOME"] = SETTINGS["cache_dir"]
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    PREPARED = True
    print("코드 준비 완료 · 입력 ZIP 또는 자동 압축 해제된 데이터는 학습 전에 확인합니다.")
    """.replace("__SOURCES__", repr(sources)))
    add("code", "dependencies", """
    # Keep Kaggle's compatible CUDA PyTorch. Install only unmet requirements.
    from importlib.metadata import PackageNotFoundError, version
    import re
    if not globals().get("PREPARED"):
        raise RuntimeError("코드 준비가 끝나지 않았습니다. 앞선 셀의 오류를 확인하세요.")
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        raise RuntimeError("Kaggle GPU 이미지의 PyTorch가 없습니다. GPU 설정을 확인하세요.") from None
    match = re.match(r"^(\\d+)\\.(\\d+)", torch_version)
    if not match or not (int(match[1]) == 2 and int(match[2]) >= 6):
        raise RuntimeError("PyTorch 2.6 이상 3 미만의 Kaggle GPU 이미지가 필요합니다. "
                           "CUDA 환경을 보호하기 위해 PyTorch를 자동 교체하지 않습니다.")
    package_deadline = min(DEADLINE_MONOTONIC, time.monotonic() + 25 * 60)
    run_supervised([sys.executable, "-m", "pip", "install", "--quiet",
                    "--upgrade-strategy", "only-if-needed", "-r", str(package_dir / "requirements.txt")],
                   deadline_monotonic=package_deadline, shutdown_margin_s=10,
                   cwd=str(package_parent), env=environment)
    DEPENDENCIES_PREPARED = True
    print("패키지 준비 완료 · 남은 시간:", round((DEADLINE_MONOTONIC - time.monotonic()) / 60, 1), "분")
    print({name: version(name) for name in ("torch", "transformers", "numpy", "scikit-learn")})
    """)
    add("code", "run", """
    # Run the complete selected phase in a fresh Python process.
    import zipfile
    if not globals().get("PREPARED") or not globals().get("DEPENDENCIES_PREPARED"):
        raise RuntimeError("준비가 완료되지 않았습니다. 처음 셀부터 실행하세요.")
    settings_file = package_parent / "settings.json"
    settings_file.write_text(json.dumps(SETTINGS, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    result_file = Path("/kaggle/working/ver28_last_run.json")
    output_root = Path(SETTINGS["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    previous_run_states = set(output_root.glob("*/run_state.json"))
    # Replace stale invocation pointers before launching; existing runs are retained.
    result_file.write_text(json.dumps({"status": "starting", "run_dir": None}), encoding="utf-8")
    failure = None
    try:
        run_supervised([sys.executable, "-u", "-m", "mission3_ver28.run_notebook",
                        "--settings", str(settings_file), "--result-file", str(result_file)],
                       deadline_monotonic=DEADLINE_MONOTONIC, shutdown_margin_s=10,
                       cwd=str(package_parent), env=environment)
    except Exception as error:
        failure = type(error).__name__
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {"status": "interrupted", "submission_ready": False}
        failure = failure or "ResultFileUnavailable"
    if failure and not result.get("run_dir"):
        new_run_states = set(output_root.glob("*/run_state.json")) - previous_run_states
        if len(new_run_states) == 1:
            # A hard kill does not prove that the resumable checkpoint is intact.
            result.update(run_dir=str(next(iter(new_run_states)).parent),
                          status="interrupted", submission_ready=False)
            result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("실행 상태:", result.get("status", "확인 필요"))
    print("이번 실행 시간:", round((time.monotonic() - INVOCATION_STARTED) / 60, 1), "분")
    if result.get("run_dir"):
        run_dir = Path(result["run_dir"])
        print("결과 폴더:", run_dir)
        selection_file = run_dir / "selection.json"
        if selection_file.is_file():
            selection = json.loads(selection_file.read_text(encoding="utf-8"))
            for name, scores in selection.get("scores", {}).items():
                print("Development", name, "Macro F1 =", round(scores["macro_f1"], 6))
            print("고정한 선택:", selection.get("selected_recipe"))
        validation_file = run_dir / "validation_metrics.json"
        if validation_file.is_file():
            validation = json.loads(validation_file.read_text(encoding="utf-8"))
            print("공식 Validation Macro F1 =", round(validation["macro_f1"], 6))
        archive = Path("/kaggle/working") / ("ver28_analysis_" + run_dir.name + ".zip")
        allowed = {".json", ".csv", ".npz", ".npy", ".md", ".txt", ".py"}
        excluded = {"checkpoints", "checkpoint", "resume_checkpoint", "best_model", "model", "tokenizer", "cache", "data"}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(run_dir.rglob("*")):
                relative = path.relative_to(run_dir)
                if (path.is_file() and path.suffix.lower() in allowed
                        and not any(part in excluded for part in relative.parts)
                        and path.stat().st_size <= 20 * 1024 * 1024):
                    handle.write(path, Path(run_dir.name) / relative)
        print("분석용 ZIP:", archive)
        print("분석 ZIP에는 큰 가중치와 이어 학습 checkpoint가 없습니다. 실행 폴더 전체도 보관하세요.")
    if result.get("status") == "screening_complete":
        print("후보 비교 완료. 분석 ZIP을 확인한 뒤 full 학습을 별도 실행하세요. 아직 제출 모델이 아닙니다.")
    elif result.get("status") == "paused":
        print("시간 제한으로 일시 중단했습니다. 저장된 Output을 Add Input으로 연결하고 RESUME_RUN에 실행 폴더를 넣으세요.")
    elif result.get("status") == "complete" and result.get("submission_ready"):
        print("전체 학습 및 제출 준비 완료. 최종 지표와 제출 manifest를 확인하세요.")
    elif result.get("status") == "interrupted":
        print("프로세스가 강제로 끝났습니다. checkpoint와 manifest 검증 전에는 재개 가능 여부를 확정할 수 없습니다.")
    else:
        print("완료 상태가 아닙니다. run_state.json과 오류 기록을 확인하세요.")
    if result.get("error_type"):
        print("오류 종류:", result["error_type"])
    if failure:
        raise RuntimeError("ver28 실행 오류: " + failure + ". 저장된 상태와 로그를 확인하세요.") from None
    """)
    add("markdown", "results", """
    ## 결과와 다음 실행

    출력 위치는 `/kaggle/working/results/mission3/ver28/실행시각_고유번호/`입니다.
    저장된 버전의 Output에서 `ver28_analysis_*.zip`을 내려받아 분석에 사용합니다.
    분석 ZIP은 작은 지표·예측·코드 모음이며 가중치의 백업이나 재개용 파일이 아닙니다.

    | 상태 | 의미 | 다음 행동 |
    |---|---|---|
    | `screening_complete` | 후보 비교와 선택 완료 | 결과 검토 후 별도 full 실행 |
    | `paused` | 예산에 맞춰 중간 저장 | 이전 Output 연결 후 같은 모드로 재개 |
    | `complete` + `submission_ready=true` | 전체 학습·제출 준비 완료 | 공식 Validation 지표와 제출 파일 확인 |
    | `failed` | 오류 | 오류 기록 확인; 완료 모델로 사용하지 않기 |
    | `interrupted` | 강제 종료로 저장 완전성 미확인 | 로그·checkpoint 검증 후 재개 여부 판단 |

    **이어 실행:** 새 노트북에 이전 저장 버전의 Output을 Add Input으로 연결합니다.
    `RESUME_RUN`에 그 안의 `run_state.json`이 있는 실행 폴더를 지정합니다.
    `RUN_MODE`와 `CANDIDATE`는 이전과 같게 유지합니다. 원본 데이터셋도 연결해야 합니다.
    입력 폴더는 읽기 전용이며, 재개 결과는 새 실행 폴더에 저장합니다.

    **전체 학습:** 결과를 검토한 뒤 완료된 screening Output을 새 노트북에 연결합니다.
    `RUN_MODE="full"`, `SOURCE_RUN="/kaggle/input/.../results/mission3/ver28/실행폴더"`,
    `RESUME_RUN=""`로 바꾸고 Save Version → Save & Run All을 실행합니다.
    development에서 고정한 모델 구성과 epoch로 전체 Training을 학습하며,
    공식 Validation 정답은 학습에 사용하지 않습니다. 실행당 8시간 제한과 재개 기능은 동일합니다.

    Save & Run All은 새 실행입니다. 이미 실행 중인 버전에 반복해서 누르지 마세요.
    대화형 세션을 켰다면 별도로 종료하고, 버전 저장 완료와 Output 파일을 확인하세요.
    """)
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def build(output=None):
    target = Path(output) if output else HERE.parent / "mission3_ver28.ipynb"
    notebook = make_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
