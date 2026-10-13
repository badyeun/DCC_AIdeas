"""Build the standalone ver14 Colab notebook without changing earlier versions."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "neural.py", "audit.py", "runtime.py",
    "runtime_suite.py", "inference.py", "run_notebook.py", "requirements.txt", "README.md",
)


def default_settings():
    return {
        "version": "ver14", "backend": "neural", "model_id": "klue/roberta-base",
        "revision": "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da",
        "label_attention": False, "contrastive_weight": 0.0,
        "input_zip": "/content/drive/MyDrive/mission3_ver5_inputs.zip",
        "output_dir": "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver14",
        "work_dir": "/content/mission3_ver14", "train_dir": "", "validation_dir": "",
        "run_validation": True, "seed": 42, "holdout_size": 0.2,
        "max_length": 512, "overlap": 64, "batch_size": 4,
        "encoder_window_batch": 4, "gradient_accumulation": 4,
        "max_epochs": 4, "patience": 2, "learning_rate": 2e-5,
        "head_learning_rate": 1e-4, "refit_full_training": True,
        "allow_cpu": False, "log_every": 250,
        "nausea_weight_candidates": [1.0, 2.0, 3.0],
        "stage3_batch_size": 8, "stage3_gradient_accumulation": 2,
        "resume_run": "",
    }


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver14-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver14 — 오심 오류 분석 → 학습 가중치 → attention·대조학습 비교

    **이 노트북 한 개와 기존 `mission3_ver5_inputs.zip`으로 실행합니다.**
    Colab의 GPU 런타임을 선택하고 **런타임 → 모두 실행**을 누르세요.
    Drive 연결 요청에는 ZIP 바로가기가 있는 **공용 계정**을 선택합니다.

    Training 전사와 정답을 먼저 점검한 뒤, 같은 Training 내부 분할에서 6개 후보를 비교합니다.
    최종 선택은 Training 지표로만 결정하고, 선택된 구성 하나를 전체 Training으로 새로 학습합니다.
    공식 Validation은 마지막에 한 번 평가합니다. 예측은 증상별 두 class의 `argmax`입니다.

    **오심 F1과 전체 Macro F1의 개선은 실제 실행 결과로 확인해야 합니다.**
    사후 threshold 조정이나 원본 정답 수정은 하지 않습니다.
    GPU 학습 6회와 최종 재학습 1회이므로 수 시간이 필요합니다.
    비슷한 GPU에서 약 6–10시간을 예상하지만 GPU·조기 종료에 따라 달라집니다. 고정 시간 제한은 없습니다.
    """)
    add("markdown", "evidence", """
    ## 원본 검토에서 확인한 문제

    기존 오류 검토 표 80행, 중복을 제외한 76개 통화를 원본과 대조했습니다.
    아래는 개인정보와 통화 식별자를 제외한 증상 내용의 요약입니다.

    | 원본에서 관찰한 증상 내용 | 제공된 정답과 모델 결과 | 분석 방향 |
    | --- | --- | --- |
    | 메스꺼움과 구토를 함께 말한 통화 | 정답은 구토·어지러움, 모델의 오심은 FP | 표현이 명시되어도 정답과 어긋나는 사례를 검토 |
    | 토할 것 같은 느낌과 어지러움을 말한 통화 | 정답은 어지러움, 모델의 오심은 FP | 구토·오심·어지러움의 표현과 정답 경계를 비교 |
    | 토할 것 같은 느낌을 말한 통화 | 정답에 오심이 있지만 모델은 구토만 예측하여 FN | 유사 표현에서 구토와 오심을 구별하는 학습을 점검 |

    같은 표현이 서로 다른 정답 조합에 나타납니다. 이 관찰만으로 라벨 오류를 확정할 수 없으며,
    **원본 정답은 그대로 유지합니다.** 질문·부정 표현도 단순 존재 여부로 증상을 단정하지 않습니다.

    기존 ver13의 Training holdout 오심 FN 575개 중 567개(98.6%)가 단일 512-token window였습니다.
    ver11에서 맞혔다가 ver13이 놓친 16개도 모두 단일 window였고 최대 417 tokens였습니다.
    정규식으로 찾은 오심 표현 후보는 모두 첫 window 안에 있었습니다.
    따라서 긴 대화 확장보다 정답의 경계·학습 손실·증상별 정밀도와 재현율을 먼저 점검합니다.
    정규식에 포함되지 않은 우회 표현은 별도 사람이 검토해야 합니다.
    """)
    add("markdown", "workflow", """
    ## 1 → 2 → 3 실행 순서

    1. **Training 원본 점검:** 중복·정답 분포·오심/구토 조합·질문/부정 후보를 집계하고 검토 자료를 저장합니다.
       fit/holdout을 구분하며 검토 목록에는 원본 행 위치·정답·후보 표시를 저장합니다. 화면에는 집계만 표시합니다.
    2. **오심 양성 학습 가중치 비교:** attention 없이 교차엔트로피를 사용하고 가중치 1·2·3을 비교합니다.
       1이 이번 실행의 새 기준 모델입니다. 이는 학습 손실의 비용 변경이며 사후 예측 보정이 아닙니다.
    3. **구조와 손실을 분리해 비교:** 2단계에서 고른 가중치를 고정하고 아래 3개를 같은 조건으로 새로 학습합니다.

    | 후보 | Attention | 교차엔트로피 | 대조학습 |
    | --- | --- | --- | --- |
    | attention_off_ce | 끔 | 사용 | 없음 |
    | attention_on_ce | 켬 | 사용 | 없음 |
    | attention_on_supcon | 켬 | 사용 | 0.1 |

    3단계 세 후보는 실제 batch 8, accumulation 2로 맞춥니다.
    세 후보 모두 동일 통화를 dropout으로 두 번 읽고 두 교차엔트로피의 평균을 학습합니다.
    attention_on_supcon에만 대조학습을 더해 두 view 사용 자체와 대조학습 효과를 분리합니다.
    2단계는 batch 4, accumulation 4이며 두 단계 모두 유효 batch는 16입니다.
    대조학습의 비교 표본은 실제 batch 안의 통화이므로 accumulation으로 늘어나지 않습니다.
    encoder·Training 분할·seed·학습률·epoch 선택 규칙도 동일하게 유지합니다.

    최종 후보는 새 기준 모델의 **Macro F1과 오심 F1을 모두 유지하거나 개선한 후보**에서 Macro F1 우선으로 고릅니다.
    조건을 만족하는 개선 후보가 없으면 이번에 학습한 기준 모델을 유지합니다.
    이후 선택된 구성만 공개 초기 가중치에서 전체 Training으로 한 번 다시 학습하고 Validation을 평가합니다.
    """)
    add("code", "settings", """
    # @title 실행 설정
    SETTINGS = __SETTINGS__
    # Keep both explicit label directories empty when using the existing input ZIP.
    # To resume, set resume_run to the complete previously created run directory.
    # Resume checks source, data, and settings fingerprints before reusing completed candidates.
    # GPU memory: adjust each stage's microbatch and accumulation together.
    # SupCon compares only examples in the actual microbatch.
    """.replace("__SETTINGS__", repr(default_settings())))
    add("code", "drive", """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    """)
    add("code", "restore", """
    # @title 실행 코드 준비 (다른 파일 업로드 불필요)
    import tempfile
    from pathlib import Path
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver14_"))
    for name, source in RUNTIME_SOURCES.items():
        (runtime_dir / name).write_text(source, encoding="utf-8")
    print("실행 코드 준비 완료")
    """.replace("__SOURCES__", repr(sources)), form=True)
    add("code", "dependencies", """
    # @title 기존 프로젝트와 같은 버전의 패키지 준비
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "-r", str(runtime_dir / "requirements.txt")])
    """)
    add("code", "run", """
    # @title 1. 원본 점검 → 2. 가중치 비교 → 3. Attention·대조학습 비교 → 최종 학습·평가
    import json
    settings_file = runtime_dir / "notebook_settings.json"
    result_file = runtime_dir / "notebook_result.json"
    settings_file.write_text(json.dumps(SETTINGS, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # A fresh interpreter uses the installed versions and streams progress to this cell.
    subprocess.check_call([sys.executable, "-u", str(runtime_dir / "run_notebook.py"),
                           "--settings", str(settings_file), "--result_file", str(result_file)])
    result_dir = Path(json.loads(result_file.read_text(encoding="utf-8")))
    """)
    add("code", "results", """
    # @title 비교 결과와 최종 점수
    import csv
    import json
    summary = result_dir / "candidate_summary.csv"
    if summary.is_file():
        print("Training 내부 비교:")
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                print(row["candidate"], "Macro F1 =", round(float(row["macro_f1"]), 6),
                      "오심 F1 =", round(float(row["nausea_f1"]), 6),
                      "P =", round(float(row["nausea_precision"]), 6),
                      "R =", round(float(row["nausea_recall"]), 6),
                      "FP/FN =", row["nausea_fp"], "/", row["nausea_fn"])
    for filename in ["training_metrics.json", "validation_metrics.json"]:
        path = result_dir / filename
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            nausea = report["per_symptom"]["오심"]
            print(filename, "Macro F1 =", round(report["macro_f1"], 6),
                  "오심 F1 =", round(nausea["f1"], 6),
                  "P =", round(nausea["precision"], 6), "R =", round(nausea["recall"], 6))
    elapsed = result_dir / "runtime.json"
    if elapsed.is_file():
        report = json.loads(elapsed.read_text(encoding="utf-8-sig"))
        if "seconds" in report:
            print("이번 실행 소요 시간:", round(report["seconds"] / 3600, 2), "시간")
    print("결과 폴더:", result_dir)
    print("후보 비교:", result_dir / "candidate_summary.csv")
    print("최종 선택 근거:", result_dir / "final_selection.json")
    print("최종 예측:", result_dir / "mission3.csv")
    """)
    add("markdown", "resume-submission", """
    ## 중단 후 재개와 제출

    기본 실행은 `ver14/실행시각_고유번호/`에 저장됩니다. 기존 ver10–ver13 결과는 유지합니다.
    중단되면 설정의 `resume_run`에 그 실행 폴더의 전체 Drive 경로를 넣고 다시 실행하세요.
    코드·Training 데이터·설정의 fingerprint가 같을 때 완료된 후보 결과를 재사용합니다.
    다른 설정의 실험은 `resume_run`을 비우고 새로 실행합니다.

    `experiment_plan.json`, `candidate_summary.csv`, `final_selection.json`,
    `training_metrics.json`, `validation_metrics.json`, `runtime.json`을 함께 확인하세요.
    `.ckpt`는 manifest이므로 **같은 실행 폴더의 실제 가중치·assets·submission 코드 전체**를 보관합니다.
    최종 가중치와 assets는 `final_model/시도번호/`에 있고 루트 `.ckpt`가 그 상대 경로를 참조합니다.
    중단된 후보의 epoch 중간 상태는 재개하지 않으며 그 후보는 새 시도에서 처음부터 학습합니다.
    실행 로그가 포함된 이 노트북도 저장해야 합니다. 제출 추론 명령:

    ```bash
    python submission/inference.py --audio_dir /path/audio --label_dir /path/json --ckpt_path mission3_ver14_model.ckpt --output mission3.csv
    ```

    추론은 저장된 로컬 모델로 실행하고 CSV 헤더는 `label file name,symptom`입니다.
    검토 CSV는 실제 통화 전사를 포함하므로 오류 분석용으로 관리하세요.
    실제 결과 PPT는 GPU 학습과 마지막 평가를 마친 뒤 작성합니다.
    """)
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "mission3_ver14.ipynb"}, "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver14.ipynb"
    target.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
