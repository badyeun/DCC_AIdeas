"""Build the standalone Colab medical encoder comparison notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "budget.py", "trainer.py", "runtime.py",
    "inference.py", "run_notebook.py", "requirements.txt", "README.md", "model_specs.json",
)


def default_settings():
    return {'version': 'ver24',
     'input_zip': '/content/drive/MyDrive/mission3_ver5_inputs.zip',
     'output_dir': '/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver24',
     'work_dir': '/content/mission3_ver24',
     'train_dir': '',
     'validation_dir': '',
     'run_validation': True,
     'seed': 42,
     'holdout_size': 0.2,
     'batch_size': 8,
     'max_length': 512,
     'gradient_accumulation': 2,
     'max_epochs': 3,
     'learning_rate': 2e-05,
     'weight_decay': 0.01,
     'dropout': 0.1,
     'warmup_ratio': 0.1,
     'pilot_optimizer_steps': 10,
     'allow_cpu': False,
     'time_budget_minutes': 180,
     'finalization_reserve_seconds': 1200,
     'budget_safety_factor': 1.35,
     'log_every': 100}


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver24-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add('markdown', 'intro', """
# Mission 3 ver24 — 의료 사전학습 모델과 일반 한국어 모델 비교

Colab GPU 런타임에서 **런타임 → 모두 실행**을 누르세요.
Drive 연결 시 입력 ZIP이 있는 **공용 계정**을 선택하세요.
필요한 업로드 파일은 이 노트북과 기존 `mission3_ver5_inputs.zip`입니다.
공개 사전학습 모델은 실행 중 내려받습니다. 이전 ver19~23 모델 파일은 필요하지 않습니다.
마지막 학습 셀만 실행하지 마세요. 준비 셀이 실패하거나 런타임을 초기화했다면 처음부터 실행하세요.

이번에는 **한국어·영어 의료 문장으로 사전학습된 bi-KM-BERT**와
**일반 한국어 KLUE-RoBERTa-base**를 같은 데이터 분할·손실함수·예산 조건에서 비교합니다.
원본 Training 데이터와 9개 증상 정답을 유지하며, 의심스러운 정답도 자동 삭제·수정하지 않습니다.
입력은 대화의 `utterances[].text`만 사용하고 각 대화의 앞부분을 최대 512토큰으로 처리합니다.
질문·부정·시간 표현을 별도 정답으로 만들지 않습니다. 외부 학습 데이터를 추가하지 않습니다.
의료 지식이 실제 오심 분류에 도움이 되는지는 이번 결과로 확인해야 합니다.
이 실험은 친구의 11개 모델 앙상블을 재현하거나 직접 비교하는 실행이 아닙니다.

**180분 예산에는 설치·모델 다운로드·데이터 준비·학습·최종 평가·저장이 모두 포함됩니다.**
Drive 로그인 대기 시간은 제외합니다. 두 모델의 짧은 속도 측정 후 공통 epoch 수를 정합니다.
전체 Training 재학습까지 끝낼 시간이 부족하면 미완료로 종료하며 완료 결과로 표시하지 않습니다.
느린 GPU나 네트워크에서는 모든 단계의 완료를 보장할 수 없습니다.
결과 표시 후와 오류·시간 제한 시 GPU 런타임 반환을 요청하며 결과는 Drive에 저장합니다.
F1 상승 및 실제 Colab 학습 속도는 아직 측정하지 않았습니다.
    """)

    add('markdown', 'experiments', """
## 비교 조건과 모델 선택

| 후보 | 공개 사전학습 모델 | 동일하게 적용하는 분류 학습 |
|---|---|---|
| 일반 한국어 대조군 | `klue/roberta-base` | mean pooling + dropout + 9개 증상 판단층 |
| 의료 사전학습 후보 | `silber2258/bi-KM-BERT` | 동일한 분류 구조와 학습 설정 |

손실은 `BCEWithLogitsLoss`이며 각 증상의 양성 가중치는
`clip(sqrt(음성 수 / 양성 수), 1, 5)`로 학습에 쓰는 데이터에서만 계산합니다.
샘플 삭제·재라벨링·학습 정답 마스킹·증상별 손실함수 변경은 적용하지 않습니다.
두 모델은 같은 seed 42의 Training 내부 분할에서 학습하고 평가합니다.
정확히 같은 원문 대화는 한쪽 분할에만 들어갑니다. 공식 Validation은 분할·후보 선택에 사용하지 않습니다.
의료 모델의 tokenizer는 의료 모델과 함께 공개된 tokenizer를 사용합니다.
두 모델 모두 최대 512토큰, 학습률 0.00002, batch 8 × gradient accumulation 2를 사용합니다.
속도 측정 후 가능한 공통 학습 횟수를 최대 3 epoch 안에서 정하고, 각 후보의 내부 평가가 가장 좋은 epoch 수·예측·지표를 저장합니다.
비교용 fit 가중치는 보관하지 않고, 최종 전체 Training 재학습 가중치만 저장해 Drive 저장량을 줄입니다.
학습률 스케줄은 fit과 전체 Training 재학습 모두 이 공통 epoch 수를 기준으로 하며, 선택 epoch가 더 적어도 압축하지 않습니다.
같은 512토큰이어도 두 tokenizer가 포함하는 실제 원문 길이는 다를 수 있습니다.

**의료 후보의 Macro F1이 대조군보다 높고 오심 F1도 낮아지지 않을 때만 의료 후보를 선택합니다.**
그 외에는 이 실행에서 학습한 일반 한국어 대조군을 선택합니다.
이는 기존 최고 성능 앙상블보다 좋은 모델이라는 뜻이 아닙니다.
선택한 모델 종류와 epoch 수로 **전체 Training을 사전학습 가중치에서 새로 학습**한 뒤
최종 모델 하나를 저장하고 공식 Validation을 한 번 평가합니다.
예측은 모든 증상에 동일한 `logit >= 0` (sigmoid 확률 0.5) 규칙을 고정 적용합니다.
threshold 탐색·클래스별 기준 변경·사후 점수 보정·앙상블 비율 탐색을 하지 않습니다.
반복 사용한 Training holdout 성능은 탐색 결과이며 독립된 새 데이터의 성능을 보장하지 않습니다.
    """)

    add('code', 'settings', """
# @title 실행 설정
SETTINGS = __SETTINGS__
AUTO_RELEASE_GPU = True # @param {type:"boolean"}
RUNTIME_PREPARED = None
DEPENDENCIES_PREPARED = None
# Each rerun starts a new budget and consumes additional Colab resources.
    """.replace("__SETTINGS__", repr(default_settings())))

    add('code', 'drive', """
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
    """)

    add('code', 'budget', """
# @title 3시간 작업 예산 시작 (이후 패키지 설치 시간도 포함)
import time
import threading
RUNTIME_PREPARED = None
DEPENDENCIES_PREPARED = None
if not isinstance(globals().get("SETTINGS"), dict) or "AUTO_RELEASE_GPU" not in globals():
    raise RuntimeError("실행 설정이 없습니다. 런타임 → 모두 실행으로 처음 셀부터 실행하세요.")
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

    add('code', 'restore', """
# @title 실행 코드 준비 (코드 파일 추가 업로드 불필요)
import importlib.util
import sys
import tempfile
from pathlib import Path
RUNTIME_SOURCES = __SOURCES__
RUNTIME_PREPARED = None
DEPENDENCIES_PREPARED = None
if ("DEADLINE_MONOTONIC" not in globals() or "time" not in globals()
        or not callable(globals().get("release_gpu"))):
    raise RuntimeError("작업 예산 준비가 완료되지 않았습니다. 런타임 → 모두 실행으로 처음 셀부터 실행하세요.")
try:
    if time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("이번 작업의 시간 예산이 끝났습니다. 다시 실행하려면 처음 셀부터 모두 실행하세요.")
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver24_"))
    for name, source in RUNTIME_SOURCES.items():
        (runtime_dir / name).write_bytes(source.encode("utf-8"))
    budget_spec = importlib.util.spec_from_file_location("mission3_ver24_budget_supervisor", runtime_dir / "budget.py")
    budget_module = importlib.util.module_from_spec(budget_spec)
    sys.modules[budget_spec.name] = budget_module
    budget_spec.loader.exec_module(budget_module)
    RUNTIME_PREPARED = (str(runtime_dir), DEADLINE_MONOTONIC)
    print("실행 코드 준비 완료")
except BaseException:
    cleanup = globals().get("release_gpu")
    if callable(cleanup):
        try:
            cleanup()
        except Exception as cleanup_error:
            print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· 위의 최초 오류를 확인하세요.")
    raise
    """.replace("__SOURCES__", repr(sources)), form=True)

    add('code', 'dependencies', """
# @title 기존 프로젝트와 같은 버전의 패키지 준비 (최대 20분, 전체 예산에 포함)
DEPENDENCIES_PREPARED = None
required = ("runtime_dir", "budget_module", "DEADLINE_MONOTONIC", "time", "sys")
if (any(name not in globals() for name in required)
        or not callable(globals().get("release_gpu"))):
    raise RuntimeError("실행 코드 준비가 완료되지 않았습니다. 런타임 → 모두 실행으로 처음 셀부터 실행하세요.")
if (globals().get("RUNTIME_PREPARED") != (str(runtime_dir), DEADLINE_MONOTONIC)
        or not (runtime_dir / "requirements.txt").is_file()):
    raise RuntimeError("현재 실행의 코드 준비가 미완료입니다. 앞선 셀의 첫 오류를 확인하고 처음부터 모두 실행하세요.")
try:
    if time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("이번 작업의 시간 예산이 끝났습니다. 다시 실행하려면 처음 셀부터 모두 실행하세요.")
    package_deadline = min(DEADLINE_MONOTONIC, time.monotonic() + 20 * 60)
    budget_module.run_supervised(
        [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(runtime_dir / "requirements.txt")],
        deadline_monotonic=package_deadline, shutdown_margin_s=10,
    )
    DEPENDENCIES_PREPARED = (str(runtime_dir), DEADLINE_MONOTONIC)
    print("패키지 준비 완료 · 남은 작업 시간:", round(max(0, DEADLINE_MONOTONIC - time.monotonic()) / 60, 1), "분")
except BaseException:
    cleanup = globals().get("release_gpu")
    if callable(cleanup):
        try:
            cleanup()
        except Exception as cleanup_error:
            print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· 위의 최초 오류를 확인하세요.")
    raise
    """)

    add('code', 'run', """
# @title 속도 측정 → 두 모델 비교 → 전체 Training 재학습 → 최종 평가 · GPU 반환
import csv
import json
import subprocess
required = ("runtime_dir", "SETTINGS", "budget_module", "DEADLINE_MONOTONIC",
            "BUDGET_STARTED_MONOTONIC", "time", "Path", "sys")
if (any(name not in globals() for name in required)
        or not callable(globals().get("release_gpu"))):
    raise RuntimeError("준비 셀이 완료되지 않았거나 런타임이 초기화됐습니다. 마지막 셀만 실행하지 말고 런타임 → 모두 실행으로 처음부터 실행하세요. 앞선 셀이 실패했다면 그 셀의 첫 오류를 확인하세요.")
prepared_context = (str(runtime_dir), DEADLINE_MONOTONIC)
if (globals().get("RUNTIME_PREPARED") != prepared_context
        or globals().get("DEPENDENCIES_PREPARED") != prepared_context
        or not (runtime_dir / "run_notebook.py").is_file()):
    raise RuntimeError("현재 실행의 코드 또는 패키지 준비가 미완료입니다. 앞선 셀의 첫 오류를 확인하고 처음부터 모두 실행하세요.")
error_file = runtime_dir / "notebook_error.json"
try:
    error_file.write_text(json.dumps({"status": "starting"}), encoding="utf-8")
    if time.monotonic() >= DEADLINE_MONOTONIC:
        raise budget_module.BudgetExceeded("notebook launch", deadline_monotonic=DEADLINE_MONOTONIC)
    settings_file = runtime_dir / "notebook_settings.json"
    result_file = runtime_dir / "notebook_result.json"
    settings_file.write_text(json.dumps(SETTINGS, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    budget_module.run_supervised(
        [sys.executable, "-u", str(runtime_dir / "run_notebook.py"),
         "--settings", str(settings_file), "--result_file", str(result_file)],
        deadline_monotonic=DEADLINE_MONOTONIC, shutdown_margin_s=10,
    )
    result_dir = Path(json.loads(result_file.read_text(encoding="utf-8")))
    print("결과 폴더:", result_dir)
    summary = result_dir / "candidate_summary.csv"
    if summary.is_file():
        print("동일 Training 내부 분할 비교:")
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                print(row["candidate"], "Macro F1 =", round(float(row["macro_f1"]), 6),
                      "오심 F1 =", round(float(row["nausea_f1"]), 6),
                      "선택 epoch =", row["selected_epochs"])
    for filename in ["training_metrics.json", "validation_metrics.json"]:
        path = result_dir / filename
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            print(filename, "Macro F1 =", round(report["macro_f1"], 6))
            for symptom, values in report["per_symptom"].items():
                print(symptom, "F1 =", round(values["f1"], 4),
                      "P =", round(values["precision"], 4), "R =", round(values["recall"], 4))
    manifest_path = result_dir / "mission3_ver24_model.ckpt"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        print("저장된 후보:", manifest.get("selected_candidate", "모델 정보 확인"))
        print("저장 모델 학습 범위:", manifest.get("training_scope", "모델 정보 확인"),
              "· 학습 표본:", manifest.get("fitted_samples", "모델 정보 확인"),
              "· 학습 epoch:", manifest.get("selected_epochs", "모델 정보 확인"))
    status_path = result_dir / "runtime.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
    print("실행 상태:", status.get("status", "기록 확인 필요"))
    print("이번 노트북 소요 시간:", round((time.monotonic() - BUDGET_STARTED_MONOTONIC) / 60, 1), "분")
    print("선택 근거:", result_dir / "final_selection.json")
    complete = ((result_dir / "run_complete.json").is_file() and manifest_path.is_file()
                and (result_dir / "final_model").is_dir()
                and ((result_dir / "mission3.csv").is_file() or not SETTINGS["run_validation"])
                and status.get("status") == "complete")
    if complete:
        if SETTINGS["run_validation"]:
            print("최종 작업 완료 · 전체 Training 재학습과 저장 검증을 마친 예측:", result_dir / "mission3.csv")
        else:
            print("모델 검증 실행 완료 · Validation 예측과 제출 CSV는 만들지 않았습니다.")
            print("제출용 실행은 run_validation=True로 설정해야 합니다.")
        print("이전 최고 모델과의 성능 비교는 validation_metrics.json으로 따로 확인하세요.")
    else:
        print("미완료 실행:", status.get("status", "상태 기록 없음"),
              "· 최종 제출 결과로 사용하지 마세요. 결과 폴더의 로그를 확인하세요.")
except Exception as worker_error:
    failure = {"status": "failed", "stage": "notebook_run", "error_type": type(worker_error).__name__,
               "category": "notebook_execution_failed",
               "reason": "실행 또는 결과 확인 중 오류가 발생했습니다. 실패 단계와 코드 위치를 확인하세요.", "frames": []}
    if isinstance(worker_error, subprocess.CalledProcessError):
        failure.update(stage="worker_process", category="worker_process_failed", exit_code=worker_error.returncode,
                       reason="실행 프로그램이 상세 오류를 저장하기 전에 종료됐습니다. 종료 코드와 준비 상태 확인이 필요합니다.")
        try:
            recorded = json.loads(error_file.read_text(encoding="utf-8"))
            if isinstance(recorded, dict) and recorded.get("status") == "failed":
                failure.update(recorded)
        except (OSError, ValueError):
            pass
    elif type(worker_error).__name__ == "BudgetExceeded":
        failure.update(category="time_budget_exhausted",
                       reason="3시간 작업 예산이 끝나 실행을 중단했습니다. 완료 표시가 없는 결과는 제출에 사용하지 마세요.")
    print("ver24 내부 오류 · 실패 단계:", failure["stage"], "· 오류 종류:", failure["error_type"], flush=True)
    print("확인 내용:", failure["reason"], flush=True)
    if failure.get("module_name"):
        print("불러오지 못한 모듈:", failure["module_name"], flush=True)
    for frame in failure.get("frames", [])[-6:]:
        print("코드 위치:", frame["file"], "·", frame["function"], "·", frame["line"], flush=True)
    saved_error = None
    try:
        from datetime import datetime, timezone
        import uuid
        error_dir = Path(SETTINGS["output_dir"]) / "_diagnostics"
        error_dir.mkdir(parents=True, exist_ok=True)
        saved_error = error_dir / ("ver24_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + "_" + uuid.uuid4().hex[:8] + ".json")
        saved_error.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print("오류 보고서:", saved_error, flush=True)
    except OSError as save_error:
        saved_error = None
        print("Drive 오류 보고서 저장 실패:", type(save_error).__name__, "· 위의 내부 오류 내용을 복사하세요.", flush=True)
    raise RuntimeError("ver24 실행 실패 [" + failure["stage"] + " / " + failure["error_type"] + "]: "
                       + failure["reason"] + (" 오류 보고서: " + str(saved_error) if saved_error else "")) from None
finally:
    cleanup = globals().get("release_gpu")
    if callable(cleanup):
        try:
            cleanup()
        except Exception as cleanup_error:
            print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· Colab에서 직접 연결을 해제하세요.")
    """)

    add('markdown', 'results', """
## 결과 확인

결과는 `DCC_AIdeas/results/mission3/ver24/실행시각_고유번호/`에 저장됩니다.
- `candidate_summary.csv`: 같은 내부 holdout에서 두 후보를 비교한 결과
- `final_selection.json`: 모델과 epoch 선택 근거
- `training_metrics.json`: 선택 후보의 Training 내부 holdout 점수
- `paired_error_changes.json`: 의료 모델에서 오심 누락이 회복되거나 새로 생긴 건수 등
- `training_review_candidates.csv`: 원래 오심 정답과 두 모델의 판단이 많이 다른 Training holdout 검토 후보 최대 200건
- `validation_metrics.json`, `mission3.csv`: 전체 Training 재학습 후 공식 Validation 결과
- `runtime.json`, `run_complete.json`: 실행 상태와 완료 표시
- `mission3_ver24_model.ckpt`, `final_model/`: 최종 모델과 재사용에 필요한 파일

**`run_complete.json`과 최종 CSV가 없는 실행은 제출용 완료 결과가 아닙니다.**
모델 manifest만 단독 보관하지 말고 **실행 폴더 전체를 보관**하세요.
검토 후보에는 원문 대신 Training 행 번호와 확률을 저장합니다. 정답 오류 확정 목록이 아니며 학습 제외에 사용하지 않습니다.
실패 시 GPU 반환 전에 실패 단계·오류 종류·코드 위치를 표시하고
`ver24/_diagnostics/ver24_error_*.json`에 같은 오류 보고서를 저장합니다.
이 보고서를 전달하면 원인을 확인할 수 있습니다. 원문 대화나 개별 데이터 파일 이름을 출력하지 않습니다.
중단된 학습의 자동 재개는 지원하지 않습니다. 재실행은 처음부터 시작하며 새 Colab 리소스를 소비합니다.
자동 반환이 실패하면 **런타임 → 연결 해제 및 런타임 삭제**를 직접 실행하세요.
의료 모델의 효과와 GPU 시간, 실제 Colab 자동 반환은 실행 결과로 확인해야 합니다.
bi-KM-BERT 모델 카드에서 가중치의 명시적 라이선스는 확인되지 않았으며, 출처·revision과 함께 `model_specs.json`에 기록합니다.
    """)
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "mission3_ver24.ipynb"}, "accelerator": "GPU",
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver24.ipynb"
    notebook = make_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
