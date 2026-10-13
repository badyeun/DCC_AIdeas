"""Build a self-contained Colab notebook with a shared wall-clock deadline."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "neural.py", "budget.py", "runtime.py",
    "inference.py", "reference.py", "training_reference.json", "run_notebook.py", "requirements.txt", "README.md",
)


def default_settings():
    return {
        "version": "ver20", "backend": "neural", "model_id": "klue/roberta-large",
        "revision": "28d911204e9022eda172571ca8cc61eaffd942f7",
        "trainable_encoder_layers": 8, "expected_encoder_layers": 24,
        "label_attention": False, "contrastive_weight": 0.0, "ce_views": 1,
        "input_zip": "/content/drive/MyDrive/mission3_ver5_inputs.zip",
        "output_dir": "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver20",
        "work_dir": "/content/mission3_ver20", "train_dir": "", "validation_dir": "",
        "run_validation": True, "seed": 42, "holdout_size": 0.2,
        "max_length": 512, "overlap": 64, "batch_size": 4,
        "encoder_window_batch": 4, "gradient_accumulation": 4,
        "max_epochs": 3, "scheduler_horizon_epochs": 3, "patience": 2, "learning_rate": 2e-5,
        "head_learning_rate": 1e-4, "refit_full_training": True,
        "allow_cpu": False, "log_every": 250,
        "time_budget_minutes": 180, "finalization_reserve_seconds": 1500,
        "budget_safety_factor": 1.25, "pilot_optimizer_steps": 10, "resume_run": "",
    }


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver20-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver20 — 더 큰 한국어 문맥 모델 · 실행 예산 3시간

    **이 노트북 하나와 기존 `mission3_ver5_inputs.zip`으로 실행합니다.**
    Colab에서 GPU 런타임을 선택하고 **런타임 → 모두 실행**을 누르세요.
    Drive 연결 요청에는 입력 ZIP이 있는 **공용 계정**을 선택합니다.

    이번에는 **KLUE-RoBERTa-large의 뒤 8개 층과 9개 증상 분류기를 학습**합니다.
    앞 16개 층과 임베딩은 고정해 학습 비용을 줄입니다. 기존의 단순 가중 CE로 후보 하나에 집중합니다.
    기존 결과 파일을 다시 올릴 필요 없이, 같은 Training 내부 평가의 저장된 기준 점수와 비교합니다.
    Large 모델의 실제 F1과 GPU 실행 시간은 아직 측정하지 않았습니다. 큰 폭의 상승을 보장하지 않습니다.
    원래 정답과 모든 샘플을 유지하며, 모델 입력은 `utterances[].text`뿐입니다.
    threshold 탐색·사후 보정·외부 학습 데이터 추가 없이 증상별 두 class의 `argmax`로 예측합니다.

    **이 노트북 한 번의 작업 예산은 설치·데이터 준비·학습·최종 평가를 포함해 180분입니다.**
    Drive 로그인 대기 시간은 제외합니다. 첫 학습에서 실제 GPU 속도를 재고 추가 epoch와 전체 Training
    재학습이 가능한지 계산합니다. 느린 GPU에서는 추가 epoch 또는 전체 재학습이 생략될 수 있습니다.
    완료된 모델만 최종 결과에 사용하며, 실제 학습 범위를 결과에 명시합니다.
    작업 제한 시간과 모든 학습 단계의 완료는 다릅니다. 최종 완료 표시가 없는 실행은 미완료입니다.
    결과 표시 후, 오류 발생 또는 시간 제한에 도달하면 GPU 런타임의 자동 반환을 요청합니다.
    결과는 Drive에 남지만 `/content`의 임시 파일은 사라질 수 있습니다.

    **이번에는 ver20 하나만 실행하면 됩니다. 이전 버전의 결과 파일은 필요하지 않습니다.**
    """)
    add("markdown", "experiments", """

    ## 이번에 학습하는 후보 하나

    `large_partial_ce`: 공개 `klue/roberta-large`에서 새로 시작합니다.
    모델의 24개 인코더 층 중 앞 16개 층과 임베딩은 고정하고, 뒤 8개 층과 분류기를 학습합니다.
    입력 대화 전체를 512-token 창과 64-token 겹침으로 읽고 증상별 두 class를 예측합니다.
    기존 ver15의 단순 분류 구조와 가중 cross entropy를 사용합니다.
    GCE, 발화별 보조 분기, label attention, 대조학습, threshold 조정은 사용하지 않습니다.

    모델은 [공식 저장소의 고정 revision](https://huggingface.co/klue/roberta-large/commit/28d911204e9022eda172571ca8cc61eaffd942f7)을 사용합니다.
    뒤 8개 층 학습은 시간과 모델 적응력 사이의 실험 설정이며 최적값으로 입증된 것은 아닙니다.
    원래 정답과 모든 발화·샘플을 유지하며, 이전에 학습된 모델을 가져오지 않습니다.

    양성 학습 가중치는 ver15의 `recall_moderate`와 같습니다.
    `[고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란]` 순서로
    `[1.4, 1.8, 2.0, 1.2, 1.3, 1.0, 3.0, 1.5, 1.3]`을 사용합니다.
    최대 3epoch와 3epoch 기준 학습률 감소를 사용합니다. 첫 1회 갱신으로 준비한 뒤 10회 갱신의 속도를 측정합니다.
    준비와 측정 과정도 첫 epoch의 실제 학습에 포함되며, 같은 갱신을 다시 수행하지 않습니다.
    기존 기준 모델을 다시 학습하지 않고 이번 후보 하나만 실행합니다. 이후 epoch는 전체 재학습 시간을 함께 고려합니다.
    Training 내부에서 ver19보다 전체 F1과 오심 F1이 모두 높은 epoch 중 전체 F1이 가장 높은 것을 선택합니다.
    그런 epoch가 없으면 전체 F1이 가장 높은 결과를 진단용으로 저장합니다.
    동시 개선을 통과하고 시간이 허용되면 전체 Training으로 새로 학습합니다.
    기존 Training 데이터·정답·분할의 해시가 맞을 때만 내장된 ver19 기준과 비교합니다.
    전체 F1과 오심 F1이 **모두** 높아졌는지, 오심 누락·오탐이 어떻게 바뀌었는지 별도로 표시합니다.
    동시 개선 조건을 충족하지 못하면 전체 재학습을 생략해 시간을 절약합니다.
    이때 내부 fit 학습 결과와 실험용 예측은 보관하지만 새 추천 제출 모델로 표시하지 않습니다.
    이전 제출 모델을 자동으로 교체하지 않습니다.
    반복 사용한 Training holdout 비교는 탐색 결과이며 새 독립 평가셋의 검증으로 해석하면 안 됩니다.
    공식 Validation은 최종 모델을 정한 뒤 한 번 평가하며 학습·epoch 선택에는 사용하지 않습니다.
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
        runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver20_"))
        for name, source in RUNTIME_SOURCES.items():
            (runtime_dir / name).write_text(source, encoding="utf-8", newline="\\n")
        budget_spec = importlib.util.spec_from_file_location("mission3_ver20_budget_supervisor", runtime_dir / "budget.py")
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
        pilot_path = result_dir / "pilot_report.json"
        if pilot_path.is_file():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8-sig"))
            print("초기 GPU 속도 측정:", pilot.get("device", "확인 필요"),
                  "· 갱신 횟수:", pilot.get("observed_optimizer_steps"),
                  "· epoch 예상 분:", round(pilot["estimated_epoch_seconds"] / 60, 1),
                  "· 계획:", pilot.get("decision"))
        comparison_path = result_dir / "training_reference_comparison.json"
        if comparison_path.is_file():
            comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
            if comparison["available"]:
                nausea = comparison["per_symptom"]["오심"]["paired"]
                print("동일 Training 내부 평가 기준:", comparison["reference_version"],
                      "· 전체 F1 변화:", round(comparison["macro_f1_delta"], 6),
                      "· 오심 F1 변화:", round(comparison["nausea_f1_delta"], 6))
                print("오심 누락 회복:", nausea["fn_to_tp"], "· 새 누락:", nausea["tp_to_fn"],
                      "· 오탐 제거:", nausea["fp_to_tn"], "· 새 오탐:", nausea["tn_to_fp"])
                if comparison["both_macro_and_nausea_improved"]:
                    print("Training 내부 평가에서 전체 F1과 오심 F1이 모두 개선됐습니다.")
                else:
                    print("전체 F1과 오심 F1의 동시 개선 조건을 충족하지 못했습니다:", comparison["status"])
                print("이전 모델과 결과는 유지됩니다. 새 실험 결과가 자동으로 기존 제출 모델을 교체하지 않습니다.")
            else:
                print("기존 Training 결과 비교 불가:", comparison.get("reason", "기록 확인 필요"))
        for filename in ["training_metrics.json", "validation_metrics.json"]:
            path = result_dir / filename
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8-sig"))
                print(filename, "Macro F1 =", round(report["macro_f1"], 6))
                for symptom, values in report["per_symptom"].items():
                    print(symptom, "F1 =", round(values["f1"], 4),
                          "P =", round(values["precision"], 4), "R =", round(values["recall"], 4))
        manifest_path = result_dir / "mission3_ver20_model.ckpt"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            print("선택된 후보:", manifest.get("selected_candidate", "모델 정보 확인"))
            print("저장 모델 학습 범위:", manifest["training_scope"],
                  "· 학습 표본:", manifest["fitted_samples"], "· epoch:", manifest["selected_epochs"])
            print("모델:", manifest.get("model_id", SETTINGS["model_id"]),
                  "· 학습 인코더 층:", SETTINGS["trainable_encoder_layers"], "/", SETTINGS["expected_encoder_layers"])
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
                    and status.get("status") in {"complete", "complete_budget_limited", "complete_not_improved"})
        if complete:
            if status.get("status") == "complete_not_improved":
                print("실험 완료 · 동시 개선 미달로 전체 재학습을 생략했습니다.")
                print("새 추천 제출 모델이 아닙니다. 기존 제출 결과를 유지하고 비교 보고서를 확인하세요.")
                print("내부 fit 학습 모델의 실험용 예측:", result_dir / "mission3.csv")
            else:
                print("최종 작업 완료 · 예측:", result_dir / "mission3.csv")
        else:
            print("미완료 실행:", status.get("status", "상태 기록 없음"),
                  "· 저장된 후보 결과와 로그는 결과 폴더에 남아 있습니다. 최종 제출 결과로 사용하지 마세요.")
    finally:
        release_gpu()
    """)
    add("markdown", "resume-submission", """
    ## 결과 확인 · 중단 후 재개

    기본 실행은 `ver20/실행시각_고유번호/`에 저장됩니다. 이전 버전의 노트북과 결과는 유지합니다.
    `pilot_report.json`, `candidate_summary.csv`, `training_reference_comparison.json`, `final_selection.json`,
    `training_metrics.json`, `validation_metrics.json`, `runtime.json`을 확인하세요.
    모델 `.ckpt`의 `training_scope`가 전체 Training 학습인지 내부 fit 학습인지를 알려 줍니다.
    시간 때문에 전체 재학습을 생략한 경우에도 실제 완료 범위를 명시합니다.
    `complete_not_improved`는 동시 개선 미달로 전체 재학습을 생략한 실험 완료 상태입니다.
    이 상태의 예측은 내부 fit 학습 모델의 실험 결과이며 새 추천 제출 모델이 아닙니다.
    `budget_exhausted` 또는 최종 예측·평가 파일 누락은 완료된 제출 결과가 아닙니다.
    강제 종료·Colab 회수 시 마지막 `runtime.json` 상태가 최종 종료 이유를 반영하지 못할 수 있습니다.
    `run_complete.json`은 최종 완료 표시이므로 이 파일이 없는 실행을 완료로 간주하지 마세요.

    재개하려면 `resume_run`에 이전 실행 폴더의 전체 Drive 경로를 넣고 다시 실행합니다.
    재개 가능 여부는 코드·설정·데이터 일치 여부로 확인하며, 완료된 결과만 재사용합니다.
    이미 `complete`, `complete_budget_limited`, `complete_not_improved`인 실행을 재개하면 저장된 결과를 검증하고 반환하며,
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
            "colab": {"name": "mission3_ver20.ipynb"}, "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver20.ipynb"
    target.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
