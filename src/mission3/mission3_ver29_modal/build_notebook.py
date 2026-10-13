"""Build a standalone Modal adapter without changing the pinned ver29 sources."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

from ..mission3_ver29.build_notebook import HERE as CORE, source_files

HERE = Path(__file__).resolve().parent


def make_notebook():
    sources = {name: (CORE / name).read_text(encoding="utf-8") for name in source_files()}
    support = (HERE / "support.py").read_text(encoding="utf-8")
    cells = []

    def add(kind, identifier, source):
        cell = {"cell_type": kind, "id": "ver29-modal-" + identifier, "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            compile("".join(cell["source"]), identifier, "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver29 — KLUE-ASL + Qwen3-Embedding / Modal L4

    **기존 `mission3-ver29-data` Volume과 L4 ×1을 이 노트북에 연결한 뒤 시작하세요.**
    GPU와 Volume 설정은 새 노트북에서 다시 확인합니다. 기존 입력 ZIP을 그대로 사용합니다.

    1. Files에서 `/mnt/mission3-ver29-data`에 `mission3_ver29_inputs.zip`을 업로드합니다.
       이미 있다면 다시 올리지 않아도 됩니다. 압축은 직접 풀지 않습니다.
    2. 처음에는 아래 기본값 `screening`을 유지하고 **Run All을 한 번** 누릅니다.
    3. 두 후보를 순서대로 학습합니다. L4의 계산 정밀도와 빠른 배치 크기는 자동 선택합니다.
       예상 시간은 실제 배치 속도 측정 후 출력합니다.
    4. 종료 후 Volume의 `exports/ver29_analysis_*.zip`을 내려받습니다.
       전체 모델과 재개용 checkpoint는 `results` 폴더에 남겨 두세요.
    5. 저장된 파일을 확인한 뒤 **Stop kernel**로 GPU를 종료합니다.

    Training 내부 fit 20,440개로 KLUE-ASL과 Qwen3-Embedding-0.6B를 각각 최대 3 epoch 학습합니다.
    고정된 기존 ver28, 새 단독 모델 둘, 새 모델 1:1 평균, 네 모델 동일 평균을 development에서 비교하고
    선택을 고정한 뒤 confirmation을 확인합니다. **전체 Training 학습은 자동 시작하지 않습니다.**
    데이터 분할·학습 방식·선택 정책은 기존 ver29와 같습니다. F1 0.7 초과 여부는 실제 결과로 확인합니다.

    한 번에 노트북 하나만 학습하세요. 이미 실행 중이면 Run All을 다시 누르지 마세요.
    시간 예산은 설치부터 저장까지 최대 480분입니다. 부족하면 `paused`로 저장하고 나중에 재개합니다.
    무료 크레딧 소진이나 Modal 강제 종료까지 코드가 보장하지는 않습니다.
    첫 실행은 새 screening으로 시작하세요. Kaggle T4의 FP16 중간 상태를 L4 BF16로 바로 재개하지 않습니다.
    """)
    add("code", "settings", """
    # Edit only these fields. Keep defaults for the first L4 run.
    VOLUME_DIR = "/mnt/mission3-ver29-data"
    RUN_MODE = "screening"  # "screening" or "full"
    SOURCE_RUN = ""  # Full only: exact completed screening folder on the Volume
    RESUME_RUN = ""  # Resume only: exact paused folder from the same phase and GPU precision
    INPUT_ZIP = ""  # Empty: VOLUME_DIR/mission3_ver29_inputs.zip
    TIME_BUDGET_MINUTES = 480  # Installation, downloads, evaluation and saving are included

    import time
    if type(TIME_BUDGET_MINUTES) is not int or not 30 <= TIME_BUDGET_MINUTES <= 480:
        raise ValueError("TIME_BUDGET_MINUTES must be an integer between 30 and 480")
    INVOCATION_STARTED = time.monotonic()
    DEADLINE_MONOTONIC = INVOCATION_STARTED + TIME_BUDGET_MINUTES * 60
    PREPARED = False
    DEPENDENCIES_PREPARED = False
    LAUNCHED = False
    print("실행 모드:", RUN_MODE, "· 최대", TIME_BUDGET_MINUTES, "분")
    print("영구 저장 Volume:", VOLUME_DIR)
    """)
    add("code", "restore", """
    # Restore exact ver29 sources and verify the supplied input before installing packages.
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    import tempfile
    import uuid

    RUNTIME_SOURCES = __SOURCES__
    SUPPORT_SOURCE = __SUPPORT__
    if "DEADLINE_MONOTONIC" not in globals():
        raise RuntimeError("첫 설정 셀부터 실행하세요.")
    if time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("이번 실행 시간이 끝났습니다. 첫 설정 셀부터 다시 시작하세요.")
    if os.name != "posix" or not Path("/mnt").is_dir():
        raise RuntimeError("Modal 노트북에서 L4와 Volume을 연결한 뒤 실행하세요.")
    package_parent = Path(tempfile.mkdtemp(prefix="mission3_ver29_"))
    package_dir = package_parent / "mission3_ver29"
    package_dir.mkdir()
    for name, source in RUNTIME_SOURCES.items():
        (package_dir / name).write_text(source, encoding="utf-8")
    support_file = package_parent / "modal_support.py"
    support_file.write_text(SUPPORT_SOURCE, encoding="utf-8")
    for name in list(sys.modules):
        if name == "mission3_ver29" or name.startswith("mission3_ver29."):
            del sys.modules[name]
    sys.path.insert(0, str(package_parent))
    support_spec = importlib.util.spec_from_file_location("modal_support_" + uuid.uuid4().hex, support_file)
    modal_support = importlib.util.module_from_spec(support_spec)
    support_spec.loader.exec_module(modal_support)
    from mission3_ver29.settings import default_settings
    SETTINGS = default_settings()
    SETTINGS.update(run_mode=RUN_MODE, source_run=SOURCE_RUN, resume_run=RESUME_RUN,
                    input_zip=INPUT_ZIP, time_budget_minutes=TIME_BUDGET_MINUTES,
                    invocation_started_monotonic=INVOCATION_STARTED,
                    deadline_monotonic=DEADLINE_MONOTONIC)
    SETTINGS, INVOCATION_DIR = modal_support.prepare(SETTINGS, VOLUME_DIR, package_parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_parent) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["HF_HOME"] = SETTINGS["cache_dir"]
    environment["HF_HUB_CACHE"] = str(Path(SETTINGS["cache_dir"]) / "hub")
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    modal_support.run_logged([sys.executable, "-u", "-c",
        "import torch; "
        "assert torch.cuda.is_available(), 'L4 GPU is not connected'; "
        "print('GPU:', torch.cuda.get_device_name(0)); "
        "print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); "
        "print('GPU memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"],
        INVOCATION_DIR / "gpu_check.log", min(DEADLINE_MONOTONIC, time.monotonic() + 120),
        cwd=str(package_parent), env=environment)
    PREPARED = True
    print("입력 ZIP 검증 완료:", SETTINGS["input_zip"])
    print("결과 저장 위치:", SETTINGS["output_dir"])
    print("이번 호출 로그:", INVOCATION_DIR)
    """.replace("__SOURCES__", repr(sources)).replace("__SUPPORT__", repr(support)))
    add("code", "dependencies", """
    # Preserve Modal's installed CUDA PyTorch and use uv or the existing pip.
    from importlib.metadata import version
    if not globals().get("PREPARED"):
        raise RuntimeError("앞선 코드 준비 셀의 오류부터 확인하세요.")
    if globals().get("DEPENDENCIES_PREPARED"):
        raise RuntimeError("패키지 준비가 이미 끝났습니다. 다음 학습 셀을 실행하세요.")
    torch_before = version("torch")
    command = modal_support.install_command(package_dir / "requirements.txt",
                                             package_parent / "constraints.txt", torch_before)
    modal_support.run_logged(command, INVOCATION_DIR / "install.log",
                             min(DEADLINE_MONOTONIC, time.monotonic() + 25 * 60),
                             cwd=str(package_parent), env=environment)
    installed = {name: version(name) for name in ("torch", "transformers", "numpy", "scikit-learn")}
    if installed["torch"] != torch_before:
        raise RuntimeError("CUDA PyTorch 버전이 바뀌었습니다. 학습을 시작하지 않습니다.")
    modal_support.write_json(INVOCATION_DIR / "installed_packages.json", installed)
    DEPENDENCIES_PREPARED = True
    print("패키지 준비 완료:", installed)
    print("남은 실행 시간:", round((DEADLINE_MONOTONIC - time.monotonic()) / 60, 1), "분")
    """)
    add("code", "run", """
    # Keep this cell running until the supervised child finishes and exports its results.
    if not globals().get("PREPARED") or not globals().get("DEPENDENCIES_PREPARED"):
        raise RuntimeError("처음 셀부터 준비를 완료하세요.")
    if globals().get("LAUNCHED"):
        raise RuntimeError("이 호출은 이미 시작했습니다. 새 실행은 설정 셀부터 시작하세요.")
    LAUNCHED = True
    RESULT = modal_support.execute(SETTINGS, INVOCATION_DIR, package_parent, environment,
                                    DEADLINE_MONOTONIC)
    print("이번 실행 시간:", round((time.monotonic() - INVOCATION_STARTED) / 60, 1), "분")
    """)
    add("markdown", "results", """
    ## 결과와 다음 실행

    Files에서 Volume 아래 `exports/ver29_analysis_*.zip`을 내려받습니다.
    작은 분석 ZIP에는 큰 가중치와 optimizer checkpoint가 없습니다.
    `results/mission3/ver29/실행번호/` 전체 폴더를 삭제하지 마세요.
    학습 로그는 별도의 `invocations/호출번호/training.log`에 저장됩니다.

    - **`screening_complete`**: 후보 비교 완료. ZIP 결과를 검토한 뒤 full을 결정합니다.
    - **`paused`**: 중간 저장 후 종료. 같은 L4와 `RUN_MODE`를 유지하고 `RESUME_RUN`에
      출력된 실행 폴더의 정확한 전체 경로를 넣습니다. `SOURCE_RUN`은 비웁니다.
    - **`complete` + `submission_ready=true`**: full 결과. 공식 Validation 지표와 제출 파일을 확인합니다.
    - **`failed` / `interrupted`**: 오류 또는 강제 종료. 완료 결과로 사용하지 않습니다.
      로그와 저장 상태를 먼저 확인합니다. 강제 종료가 정상적인 `paused`를 뜻하지는 않습니다.

    **결과 검토 후 전체 학습**은 `RUN_MODE="full"`, `RESUME_RUN=""`로 바꾸고
    `SOURCE_RUN`에 완료된 ver29 screening 폴더를 넣은 뒤 첫 셀부터 한 번 실행합니다.
    선택된 1개·2개·4개 모델을 Training 전체 29,200개로 다시 학습합니다.
    full이 paused로 끝나면 다음 실행은 `SOURCE_RUN=""`와 그 full 폴더의 `RESUME_RUN`을 사용합니다.

    새 노트북에 옮길 때도 같은 Volume과 L4를 다시 연결하세요.
    Volume 밖 `/tmp`에는 다시 만들 수 있는 코드와 압축 해제한 데이터만 둡니다.
    입력 ZIP·모델 다운로드 캐시·학습 결과·로그는 Volume에 둡니다.
    Modal은 Volume을 주기적으로 저장하고 정상 종료 때 최종 저장하지만 갑작스러운 종료는 별개입니다.
    완료 출력과 저장된 파일을 확인한 뒤 **Stop kernel**로 GPU 사용을 끝내세요.

    [Modal Notebooks 안내](https://modal.com/docs/guide/notebooks) ·
    [Volume 저장 동작](https://modal.com/docs/guide/volumes#background-commits)
    """)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}}}


def build(output=None):
    target = Path(output) if output else HERE.parent / "mission3_ver29_modal.ipynb"
    target.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
