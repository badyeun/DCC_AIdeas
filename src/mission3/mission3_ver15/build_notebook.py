"""Build a self-contained Colab notebook with a shared wall-clock deadline."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "neural.py", "budget.py", "runtime.py",
    "inference.py", "run_notebook.py", "requirements.txt", "README.md",
)


def default_settings():
    return {
        "version": "ver15", "backend": "neural", "model_id": "klue/roberta-base",
        "revision": "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da",
        "label_attention": False, "contrastive_weight": 0.0, "ce_views": 1,
        "input_zip": "/content/drive/MyDrive/mission3_ver5_inputs.zip",
        "output_dir": "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver15",
        "work_dir": "/content/mission3_ver15", "train_dir": "", "validation_dir": "",
        "run_validation": True, "seed": 42, "holdout_size": 0.2,
        "max_length": 512, "overlap": 64, "batch_size": 4,
        "encoder_window_batch": 4, "gradient_accumulation": 4,
        "max_epochs": 4, "patience": 2, "learning_rate": 2e-5,
        "head_learning_rate": 1e-4, "refit_full_training": True,
        "allow_cpu": False, "log_every": 250,
        "time_budget_minutes": 180, "finalization_reserve_seconds": 1200,
        "budget_safety_factor": 1.25, "resume_run": "",
    }


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver15-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver15 — 여러 증상의 재현율 개선 · 실행 예산 3시간

    **이 노트북 한 개와 기존 `mission3_ver5_inputs.zip`으로 실행합니다.**
    Colab에서 GPU 런타임을 선택하고 **런타임 → 모두 실행**을 누르세요.
    Drive 연결 요청에는 ZIP 바로가기가 있는 **공용 계정**을 선택합니다.

    ver14의 전체 Macro F1은 0.63118이었고, 개선의 대부분은 오심 학습 가중치에서 나왔습니다.
    ver15는 오심 가중치 3을 유지하면서 다른 증상도 학습 중 놓치는 비용을 높여 봅니다.
    반복 읽기·Attention·대조학습을 제외하고 **최대 2개 후보와 최종 재학습 1회**에 집중합니다.
    최종 선택은 Training 내부 Macro F1으로만 결정합니다. 공식 Validation은 최종 모델에 한 번 사용합니다.

    **3시간은 설치·데이터 준비·학습·최종 평가를 포함한 작업 시간 예산입니다.**
    Drive 로그인에 걸리는 사람의 대기 시간은 제외합니다. 느린 GPU에서는 두 번째 후보나 전체 Training
    재학습을 생략하고 완료된 후보로 평가할 수 있습니다. 정상 중단 경로에서 예산이 소진되면
    `budget_exhausted`로 기록합니다. 강제 종료나 Colab 회수 시에는 마지막 기록만 남을 수 있습니다.
    최종 예측·평가가 없는 미완료 결과를 성공으로 표시하지 않습니다.
    모든 GPU에서 전체 재학습 완료나 F1 상승을 보장하지 않습니다.

    **기본 설정은 결과를 Drive에 저장하고 표시한 뒤 GPU 런타임을 자동 반환합니다.**
    실행 실패 또는 3시간 제한에 도달해도 반환을 요청합니다. Drive 파일은 남고 `/content`의 임시 파일은
    사라질 수 있으므로 결과는 Drive에서 확인하세요. 디버깅하려면 `AUTO_RELEASE_GPU`를 끌 수 있습니다.
    """)
    add("markdown", "experiments", """
    ## 비교할 학습 가중치

    가중치는 **학습 중 정답이 양성인 증상의 손실에만** 적용합니다. 추론은 증상별 두 class의 `argmax`입니다.
    사후 threshold 조정, 확률 이동, 원본 정답 수정, 외부 데이터 추가는 없습니다.

    | 증상 | recall_moderate | recall_broader |
    | --- | ---: | ---: |
    | 고열 | 1.4 | 1.75 |
    | 구토 | 1.8 | 2.25 |
    | 두통 | 2.0 | 2.5 |
    | 복통 | 1.2 | 1.25 |
    | 어지러움 | 1.3 | 1.5 |
    | 열상 | 1.0 | 1.0 |
    | 오심 | 3.0 | 3.0 |
    | 전신쇠약 | 1.5 | 1.75 |
    | 호흡곤란 | 1.3 | 1.5 |

    두 후보 모두 공개 초기 모델에서 새로 학습하며 모델·분할·seed·학습률·batch 조건을 맞춥니다.
    전체 Macro F1을 우선하여 고르므로 개별 증상의 정밀도 또는 F1이 떨어질 수도 있습니다.
    검증 결과에는 증상별 정밀도·재현율·F1을 함께 저장합니다.

    실행 중 측정한 속도와 남은 시간으로 다음 후보·epoch·최종 재학습 가능 여부를 판단합니다.
    가능한 경우 후보를 3epoch까지 학습하는 것을 우선하며, 느린 환경에서는 전체 Training 재학습을 생략할 수 있습니다.
    마지막 평가·저장을 위해 20분을 남겨 두고, 같은 L4급 환경에서는 약 140분을 예상합니다.
    다운로드·Drive 속도와 제공된 GPU에 따라 실행 범위와 완료 시간은 달라질 수 있습니다.
    """)
    add("code", "settings", """
    # @title 실행 설정
    SETTINGS = __SETTINGS__
    AUTO_RELEASE_GPU = True # @param {type:"boolean"}
    # Keep train_dir and validation_dir empty to use the existing input ZIP.
    # Resume only a compatible run; set resume_run to its complete Drive directory.
    # Each explicit rerun starts a NEW time budget and uses additional Colab resources.
    """.replace("__SETTINGS__", repr(default_settings())))
    add("code", "drive", """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    """)
    add("code", "budget", """
    # @title 3시간 작업 예산 시작 (이후 패키지 설치 시간도 포함)
    import time
    import threading
    BUDGET_STARTED_MONOTONIC = time.monotonic()
    DEADLINE_MONOTONIC = BUDGET_STARTED_MONOTONIC + float(SETTINGS["time_budget_minutes"]) * 60
    SETTINGS["deadline_monotonic"] = DEADLINE_MONOTONIC
    previous_timer = globals().get("RELEASE_TIMER")
    if previous_timer is not None:
        previous_timer.cancel()
    RELEASE_TIMER = None
    RELEASE_LOCK = threading.Lock()
    RELEASE_REQUESTED = False

    def release_gpu():
        global RELEASE_REQUESTED
        if not AUTO_RELEASE_GPU:
            return
        with RELEASE_LOCK:
            if RELEASE_REQUESTED:
                return
            RELEASE_REQUESTED = True
            if RELEASE_TIMER is not None:
                RELEASE_TIMER.cancel()
        try:
            from google.colab import runtime as colab_runtime
            print("Colab GPU 런타임 반환을 요청합니다. 저장된 결과는 Drive에서 확인하세요.", flush=True)
            colab_runtime.unassign()
        except Exception as error:
            print("자동 반환 요청 실패:", type(error).__name__,
                  "· Colab 메뉴 '런타임 → 연결 해제 및 런타임 삭제'를 직접 실행하세요.", flush=True)

    if AUTO_RELEASE_GPU:
        RELEASE_TIMER = threading.Timer(max(0, DEADLINE_MONOTONIC - time.monotonic()), release_gpu)
        RELEASE_TIMER.daemon = True
        RELEASE_TIMER.start()
    print("작업 시간 예산:", SETTINGS["time_budget_minutes"], "분")
    """)
    add("code", "restore", """
    # @title 실행 코드 준비 (추가 파일 업로드 불필요)
    import importlib.util
    import sys
    import tempfile
    from pathlib import Path
    RUNTIME_SOURCES = __SOURCES__
    try:
        runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver15_"))
        for name, source in RUNTIME_SOURCES.items():
            (runtime_dir / name).write_text(source, encoding="utf-8")
        budget_spec = importlib.util.spec_from_file_location("mission3_ver15_budget_supervisor", runtime_dir / "budget.py")
        budget_module = importlib.util.module_from_spec(budget_spec)
        sys.modules[budget_spec.name] = budget_module
        budget_spec.loader.exec_module(budget_module)
        print("실행 코드 준비 완료")
    except BaseException:
        release_gpu()
        raise
    """.replace("__SOURCES__", repr(sources)), form=True)
    add("code", "dependencies", """
    # @title 기존 프로젝트와 같은 버전의 패키지 준비 (최대 20분, 전체 예산에 포함)
    package_deadline = min(DEADLINE_MONOTONIC, time.monotonic() + 20 * 60)
    try:
        budget_module.run_supervised(
            [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(runtime_dir / "requirements.txt")],
            deadline_monotonic=package_deadline, shutdown_margin_s=10,
        )
        print("패키지 준비 완료 · 남은 작업 시간:", round(max(0, DEADLINE_MONOTONIC - time.monotonic()) / 60, 1), "분")
    except BaseException:
        release_gpu()
        raise
    """)
    add("code", "run", """
    # @title 후보 학습 → 최종 평가 · 결과 표시 · GPU 반환
    import csv
    import json
    try:
        settings_file = runtime_dir / "notebook_settings.json"
        result_file = runtime_dir / "notebook_result.json"
        settings_file.write_text(json.dumps(SETTINGS, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        # A fresh interpreter uses installed versions; the deadline includes all work.
        budget_module.run_supervised(
            [sys.executable, "-u", str(runtime_dir / "run_notebook.py"),
             "--settings", str(settings_file), "--result_file", str(result_file)],
            deadline_monotonic=DEADLINE_MONOTONIC, shutdown_margin_s=10,
        )
        result_dir = Path(json.loads(result_file.read_text(encoding="utf-8")))
        print("결과 폴더:", result_dir)
        summary = result_dir / "candidate_summary.csv"
        if summary.is_file():
            print("Training 내부 비교:")
            with summary.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    print(row["candidate"], "Macro F1 =", round(float(row["macro_f1"]), 6),
                          "오심 F1 =", round(float(row["nausea_f1"]), 6),
                          "선택 epoch =", row["selected_epochs"], "종료 이유 =", row["stop_reason"])
        for filename in ["training_metrics.json", "validation_metrics.json"]:
            path = result_dir / filename
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8-sig"))
                print(filename, "Macro F1 =", round(report["macro_f1"], 6))
                for symptom, values in report["per_symptom"].items():
                    print(symptom, "F1 =", round(values["f1"], 4),
                          "P =", round(values["precision"], 4), "R =", round(values["recall"], 4))
        manifest_path = result_dir / "mission3_ver15_model.ckpt"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            print("저장 모델 학습 범위:", manifest["training_scope"],
                  "· 학습 표본:", manifest["fitted_samples"], "· epoch:", manifest["selected_epochs"])
            if manifest.get("refit_skipped_reason"):
                print("전체 재학습 생략 이유:", manifest["refit_skipped_reason"])
        status_path = result_dir / "runtime.json"
        status = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8-sig"))
            print("실행 상태:", status.get("status", "기록 확인 필요"))
            if status.get("skipped_candidates"):
                print("생략한 후보:", status["skipped_candidates"])
        print("이번 노트북 소요 시간:", round((time.monotonic() - BUDGET_STARTED_MONOTONIC) / 60, 1), "분")
        print("선택 근거:", result_dir / "final_selection.json")
        print("학습 범위가 기록된 모델 정보:", manifest_path)
        complete = ((result_dir / "run_complete.json").is_file()
                    and status.get("status") in {"complete", "complete_budget_limited"})
        if complete:
            print("최종 작업 완료 · 예측:", result_dir / "mission3.csv")
        else:
            print("미완료 실행:", status.get("status", "상태 기록 없음"),
                  "· 저장된 후보 결과와 로그는 결과 폴더에 남아 있습니다. 최종 제출 결과로 사용하지 마세요.")
    finally:
        release_gpu()
    """)
    add("markdown", "resume-submission", """
    ## 결과 확인 · 중단 후 재개

    기본 실행은 `ver15/실행시각_고유번호/`에 저장됩니다. 이전 버전의 노트북과 결과는 유지합니다.
    `candidate_summary.csv`, `final_selection.json`, `training_metrics.json`,
    `validation_metrics.json`, `runtime.json`을 확인하세요.
    모델 `.ckpt`의 `training_scope`가 전체 Training 학습인지 내부 fit 학습인지를 알려 줍니다.
    시간 때문에 전체 재학습을 생략한 경우에도 실제 완료 범위를 명시합니다.
    `budget_exhausted` 또는 최종 예측·평가 파일 누락은 완료된 제출 결과가 아닙니다.
    강제 종료·Colab 회수 시 마지막 `runtime.json` 상태가 최종 종료 이유를 반영하지 못할 수 있습니다.
    `run_complete.json`은 최종 완료 표시이므로 이 파일이 없는 실행을 완료로 간주하지 마세요.

    재개하려면 `resume_run`에 이전 실행 폴더의 전체 Drive 경로를 넣고 다시 실행합니다.
    재개 가능 여부는 코드·설정·데이터 일치 여부로 확인하며, 완료된 결과만 재사용합니다.
    이미 `complete` 또는 `complete_budget_limited`인 실행을 재개하면 저장된 결과를 검증하고 반환하며,
    추가 학습하지 않습니다. 새 실험을 하려면 `resume_run`을 비우세요.
    중단된 epoch 중간부터 이어 학습하지 않습니다. **재실행에는 새 시간 예산과 추가 Colab 리소스가 듭니다.**

    모델 `.ckpt`는 가중치·assets의 상대 경로를 담은 manifest입니다.
    **실행 폴더 전체와 실행 로그가 있는 노트북을 보관**하세요. 제출용 추론 코드는 `submission/inference.py`입니다.
    예측 CSV의 헤더는 `label file name,symptom`입니다.
    기본 설정은 결과 표시 후 또는 제한 시간에 Colab `runtime.unassign()`으로 GPU 반환을 요청합니다.
    `AUTO_RELEASE_GPU`를 끄거나 자동 반환이 실패하면 **런타임 → 연결 해제 및 런타임 삭제**를 직접 실행하세요.
    자동 반환 기능의 실제 서비스 동작은 Colab에서 확인해야 합니다.
    """)
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "mission3_ver15.ipynb"}, "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver15.ipynb"
    target.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
