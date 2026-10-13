"""Build the standalone Colab Qwen QLoRA comparison notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "analysis.py", "bootstrap.py", "budget.py", "batching.py", "trainer.py", "runtime.py",
    "inference.py", "run_notebook.py", "build_notebook.py", "requirements.txt", "README.md", "model_specs.json",
)


def default_settings():
    return {'version': 'ver25',
     'run_mode': 'pilot',
     'pilot_samples': 4000,
     'source_run': '',
     'input_zip': '/content/drive/MyDrive/mission3_ver5_inputs.zip',
     'output_dir': '/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver25',
     'work_dir': '/content/mission3_ver25',
     'train_dir': '',
     'validation_dir': '',
     'run_validation': True,
     'seed': 42,
     'fit_fraction': 0.7,
     'development_fraction': 0.15,
     'confirmation_fraction': 0.15,
     'qwen_batch_size': 1,
     'qwen_gradient_accumulation': 16,
     'qwen_max_length': 1024,
     'control_batch_size': 8,
     'control_gradient_accumulation': 2,
     'control_max_length': 512,
     'max_epochs': 1,
     'benchmark_examples': 64,
     'benchmark_warmup_examples': 16,
     'effective_batch_size': 16,
     'warmup_ratio': 0.05,
     'lora_lr': 0.0001,
     'head_lr': 0.0003,
     'control_lr': 2e-05,
     'weight_decay': 0.01,
     'lora_r': 16,
     'lora_alpha': 32,
     'lora_dropout': 0.05,
     'gradient_clip': 1.0,
     'min_macro_gain': 0.01,
     'allow_cpu': False,
     'time_budget_minutes': 180,
     'finalization_reserve_seconds': 1500,
     'budget_safety_factor': 1.35,
     'log_every': 100}


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver25-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add('markdown', 'intro', """
# Mission 3 ver25 — 4,000개 예비 학습 후 결과 저장·종료

Colab GPU 런타임에서 **런타임 → 모두 실행**을 누르세요.
Drive 연결 시 입력 ZIP이 있는 계정을 선택하세요. 별도 코드 업로드 없이 이 노트북과
기존 `mission3_ver5_inputs.zip`으로 실행합니다. 공개 사전학습 가중치는 실행 중 내려받습니다.
준비 셀이 실패하거나 런타임을 초기화했다면 마지막 셀만 실행하지 말고 처음부터 실행하세요.

**Qwen3-4B-Instruct-2507의 내부 LoRA adapter와 9개 증상 분류층**을 실제 Training 정답으로 학습합니다.
새로 학습한 KLUE 대조군, Qwen 단독, 고정 1:1 확률 평균을 같은 분할에서 비교합니다.
자유 문장 생성이나 외부 LLM 정답 생성은 하지 않습니다. 외부 학습 데이터도 추가하지 않습니다.
기본 실행은 Training fit 분할에서 중복 그룹과 증상 비율을 고려한 **약 4,000개로 두 모델을 1 epoch 학습**합니다.
중복 그룹을 나눌 수 없는 경우 실제 개수가 조금 달라질 수 있으며 선택 내역을 저장합니다.
고정 development에서 비교한 뒤 결과를 저장하고 종료합니다. **전체 학습은 자동으로 이어서 실행하지 않습니다.**
원본 Training 표본과 정답을 유지합니다. 9개 밖의 라벨만 제외하며 해당 통화는 남깁니다.
모델 특징은 배열 순서의 `utterances[].text`뿐이며 줄바꿈으로 발화 경계를 유지합니다.

**기본 180분 작업 예산에는 설치·다운로드·준비·속도 측정·학습·평가·저장이 모두 포함됩니다.**
Drive 로그인 대기 시간은 제외합니다. `time_budget_minutes`로 실행 전에 직접 늘릴 수 있고 자동 연장하지 않습니다.
**시간 제한 없이 실행하려면 `SETTINGS["time_budget_minutes"] = None`으로 설정하세요.**
이때 예상 시간에 따른 시작 보류, 설치·실행 시간 제한, 시간 만료 GPU 반환 타이머를 끕니다.
완료·오류 시 결과 기록과 GPU 반환은 유지하며, Colab 자체의 세션 종료는 막을 수 없습니다.
동일한 대표 표본으로 배치 후보들의 속도와 메모리를 측정한 뒤 현재 GPU에서 가장 빠른 성공 후보를 선택합니다.
시간 제한을 켠 경우 마지막 25분은 저장·재로딩·오류 보고·패키지 검증에 예약합니다.
예산이 부족하면 학습 시작을 보류하거나 미완료 상태로 저장합니다. 각 실행은 추가 GPU 리소스를 소비합니다.
**변경된 자동 배치 선택과 예비 학습의 GPU 실행 시간·F1 개선은 아직 확인되지 않았습니다.**
    """)

    add('markdown', 'experiments', """
## 고정 실험 조건

| 항목 | Qwen 후보 | 새 KLUE 대조군 |
|---|---|---|
| 시작 가중치 | Qwen3-4B-Instruct-2507 | klue/roberta-base |
| 학습 | NF4 4-bit, double quantization, LoRA + 분류층 | 전체 encoder + 분류층 |
| 입력 길이 | 1,024, 앞/뒤 보존 + 공통 판정 접미문 | 특수 토큰 포함 앞부분 512 |
| 자동 측정 batch 후보 | 1 / 2 / 4 / 8 | 4 / 8 / 16 |
| 유효 batch | 16 (누적 횟수 자동 조정) | 16 (누적 횟수 자동 조정) |
| 학습률 | LoRA 1e-4 / head 3e-4 | 2e-5 |

9개 독립 logits에서 weighted BCE를 직접 계산합니다.
`pos_weight = clip(sqrt(음성 수 / max(양성 수, 1)), 1, 5)`는 실제 예비 학습 subset만으로 계산하고,
최종 refit에서는 전체 Training으로 다시 계산합니다. 판정은 항상 `logit >= 0`입니다.
LoRA r=16, alpha=32, dropout=0.05, attention/MLP projection 전체를 학습합니다.
양자화 원본은 고정합니다. bf16 미지원 GPU의 fp16 경로는 pilot에서 먼저 시험합니다.

정규화한 텍스트 중복 그룹을 보존하여 Training을 새 **70% fit / 15% development / 15% confirmation**으로 나눕니다.
라벨 비율은 가능한 한 유지하며 실제 분할과 데이터 digest를 저장합니다.
과거 전체 Training 모델은 이 비교에 사용하지 않습니다. 새로운 외부 검증셋이라는 뜻도 아닙니다.
기본 예비 실행은 1 epoch이며 같은 subset으로 학습한 두 모델을 고정 development에서 비교합니다.
소규모 1 epoch 결과만으로 전체 데이터 성능이나 수렴 여부를 단정하지 않습니다.
Qwen 또는 사전에 정한 1:1 확률 평균이 대조군보다 0.01 이상 높으면 추진 후보로 삼습니다.
0.01은 자원 배분 기준이며 통계적 유의성이나 목표 점수 보장이 아닙니다.
**Macro F1이 우선이며 오심 소폭 하락만으로 후보를 자동 탈락시키지 않습니다.**

**예비 실행에서는 confirmation을 평가하지 않고 공식 Validation을 모델 평가에 사용하지 않습니다.**
결과를 검토한 뒤 별도 full 실행을 시작하면 예비 실행에서 동결한 선택을 confirmation에서 한 번 비교합니다.
이 점수로 후보를 다시 고르지 않으며 선택 recipe를 공개 초기 가중치에서 전체 Training으로 새로 학습합니다.
9개 F1/AP, 오심 P/R/TP/FP/FN, 공통 오답·새 FP/FN, 통화 단위 bootstrap을 저장합니다.
공식 Validation은 후보 선택에 사용하지 않고 전체 Training refit 이후 한 번 확인합니다.
이미 반복 관찰된 공식 Validation 점수를 완전히 독립된 일반화 평가라고 부르지 않습니다.
threshold/offset/calibration/혼합비 검색, 재라벨링, 삭제, 정답 마스킹은 하지 않습니다.
첫 실행은 **weighted BCE만** 사용합니다. ASL 재학습과 3-fold OOF는 후속 실험이며 자동 실행하지 않습니다.
친구 11개 모델 앙상블은 보존할 별도 baseline이며 이번 새 KLUE 대조군과 다릅니다.
    """)

    add('code', 'settings', """
# @title 실행 설정
SETTINGS = __SETTINGS__
# First run: keep run_mode='pilot'. Later: set run_mode='full', source_run to
# the completed pilot run folder. Set time_budget_minutes=None for no time limit,
# or use a positive integer from its time estimate for a bounded run.
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
# @title 설정한 작업 예산 시작 (이후 패키지 설치 시간도 포함)
import time
import threading
import math
RUNTIME_PREPARED = None
DEPENDENCIES_PREPARED = None
if not isinstance(globals().get("SETTINGS"), dict) or "AUTO_RELEASE_GPU" not in globals():
    raise RuntimeError("실행 설정이 없습니다. 런타임 → 모두 실행으로 처음 셀부터 실행하세요.")
configured_minutes = SETTINGS["time_budget_minutes"]
if configured_minutes is not None and (
        isinstance(configured_minutes, bool) or not isinstance(configured_minutes, (int, float))
        or not math.isfinite(configured_minutes) or configured_minutes <= 0):
    raise ValueError("time_budget_minutes must be a finite positive number or None")
BUDGET_STARTED_MONOTONIC = time.monotonic()
DEADLINE_MONOTONIC = (None if configured_minutes is None else
                      BUDGET_STARTED_MONOTONIC + float(configured_minutes) * 60)
SETTINGS["deadline_monotonic"] = DEADLINE_MONOTONIC
SETTINGS["invocation_started_monotonic"] = BUDGET_STARTED_MONOTONIC
previous_timer = globals().get("RELEASE_TIMER")
if previous_timer is not None:
    previous_timer.cancel()
RELEASE_TIMER = None
RELEASE_LOCK = threading.Lock()
RELEASE_REQUESTED = False

def save_setup_failure(error, stage):
    # Never record exception text: it can contain input text or private paths.
    import json
    import traceback
    import uuid
    from datetime import datetime, timezone
    from pathlib import Path
    frames = [{"file": Path(frame.f_code.co_filename).name, "line": line,
               "function": frame.f_code.co_name}
              for frame, line in traceback.walk_tb(error.__traceback__)]
    report = {"status": "failed", "stage": stage, "error_type": type(error).__name__,
              "category": "preparation_failed", "frames": frames[-20:]}
    print("준비 실패 단계:", stage, "· 오류 종류:", type(error).__name__, flush=True)
    try:
        folder = Path(SETTINGS["output_dir"]) / "_diagnostics"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("ver25_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                         + "_" + uuid.uuid4().hex[:8] + ".json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("오류 보고서:", path, flush=True)
    except OSError as save_error:
        print("Drive 오류 보고서 저장 실패:", type(save_error).__name__, flush=True)

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

if AUTO_RELEASE_GPU and DEADLINE_MONOTONIC is not None:
    RELEASE_TIMER = threading.Timer(max(0, DEADLINE_MONOTONIC - time.monotonic()), release_gpu)
    RELEASE_TIMER.daemon = True
    RELEASE_TIMER.start()
if DEADLINE_MONOTONIC is None:
    print("작업 시간 제한: 없음 · 완료 또는 오류 시 결과 기록 후 GPU 반환")
else:
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
    if DEADLINE_MONOTONIC is not None and time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("이번 작업의 시간 예산이 끝났습니다. 다시 실행하려면 처음 셀부터 모두 실행하세요.")
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver25_"))
    for name, source in RUNTIME_SOURCES.items():
        (runtime_dir / name).write_bytes(source.encode("utf-8"))
    budget_spec = importlib.util.spec_from_file_location("mission3_ver25_budget_supervisor", runtime_dir / "budget.py")
    budget_module = importlib.util.module_from_spec(budget_spec)
    sys.modules[budget_spec.name] = budget_module
    budget_spec.loader.exec_module(budget_module)
    RUNTIME_PREPARED = (str(runtime_dir), DEADLINE_MONOTONIC)
    print("실행 코드 준비 완료")
except BaseException as setup_error:
    reporter = globals().get("save_setup_failure")
    if callable(reporter):
        reporter(setup_error, "source_restore")
    cleanup = globals().get("release_gpu")
    if callable(cleanup):
        try:
            cleanup()
        except Exception as cleanup_error:
            print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· 위의 최초 오류를 확인하세요.")
    raise RuntimeError("실행 코드 준비 실패: " + type(setup_error).__name__ + ". Drive 오류 보고서를 확인하세요.") from None
    """.replace("__SOURCES__", repr(sources)), form=True)

    add('code', 'dependencies', """
# @title ver25 전용 QLoRA 패키지 준비 (시간 제한 사용 시 최대 20분, 전체 예산에 포함)
DEPENDENCIES_PREPARED = None
required = ("runtime_dir", "budget_module", "DEADLINE_MONOTONIC", "time", "sys")
if (any(name not in globals() for name in required)
        or not callable(globals().get("release_gpu"))):
    raise RuntimeError("실행 코드 준비가 완료되지 않았습니다. 런타임 → 모두 실행으로 처음 셀부터 실행하세요.")
if (globals().get("RUNTIME_PREPARED") != (str(runtime_dir), DEADLINE_MONOTONIC)
        or not (runtime_dir / "requirements.txt").is_file()):
    raise RuntimeError("현재 실행의 코드 준비가 미완료입니다. 앞선 셀의 첫 오류를 확인하고 처음부터 모두 실행하세요.")
try:
    if DEADLINE_MONOTONIC is not None and time.monotonic() >= DEADLINE_MONOTONIC:
        raise RuntimeError("이번 작업의 시간 예산이 끝났습니다. 다시 실행하려면 처음 셀부터 모두 실행하세요.")
    package_deadline = (None if DEADLINE_MONOTONIC is None else
                        min(DEADLINE_MONOTONIC, time.monotonic() + 20 * 60))
    budget_module.run_supervised(
        [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(runtime_dir / "requirements.txt")],
        deadline_monotonic=package_deadline, shutdown_margin_s=10,
    )
    DEPENDENCIES_PREPARED = (str(runtime_dir), DEADLINE_MONOTONIC)
    if DEADLINE_MONOTONIC is None:
        print("패키지 준비 완료 · 작업 시간 제한 없음")
    else:
        print("패키지 준비 완료 · 남은 작업 시간:", round(max(0, DEADLINE_MONOTONIC - time.monotonic()) / 60, 1), "분")
except BaseException as setup_error:
    reporter = globals().get("save_setup_failure")
    if callable(reporter):
        reporter(setup_error, "dependency_install")
    cleanup = globals().get("release_gpu")
    if callable(cleanup):
        try:
            cleanup()
        except Exception as cleanup_error:
            print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· 위의 최초 오류를 확인하세요.")
    raise RuntimeError("패키지 준비 실패: " + type(setup_error).__name__ + ". Drive 오류 보고서를 확인하세요.") from None
    """)

    add('code', 'run', """
# @title 실행 모드에 따라 학습 → 결과 저장 → 종료 · GPU 반환
import csv
import json
import math
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
    if DEADLINE_MONOTONIC is not None and time.monotonic() >= DEADLINE_MONOTONIC:
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
        print("동일 Training development 분할 비교:")
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                print(row["candidate"], "Macro F1 =", round(float(row["macro_f1"]), 6),
                      "오심 F1 =", round(float(row["nausea_f1"]), 6),
                      "선택 epoch =", row.get("selected_epochs", "기록 확인"))
    for filename in ["training_metrics.json", "validation_metrics.json"]:
        path = result_dir / filename
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            print(filename, "Macro F1 =", round(report["macro_f1"], 6))
            for symptom, values in report["per_symptom"].items():
                print(symptom, "F1 =", round(values["f1"], 4),
                      "P =", round(values["precision"], 4), "R =", round(values["recall"], 4))
    manifest_path = result_dir / "mission3_ver25_model.ckpt"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        selection = manifest.get("selection", {})
        print("저장된 후보:", selection.get("selected_candidate", "모델 정보 확인"))
        print("저장 모델 학습 범위:", manifest.get("training_scope", "모델 정보 확인"),
              "· 학습 표본:", manifest.get("fitted_samples", "모델 정보 확인"),
              "· 학습 epoch:", selection.get("selected_epochs", "모델 정보 확인"))
    status_path = result_dir / "runtime.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
    print("실행 상태:", status.get("status", "기록 확인 필요"))
    print("이번 노트북 소요 시간:", round((time.monotonic() - BUDGET_STARTED_MONOTONIC) / 60, 1), "분")
    print("선택 근거:", result_dir / "final_selection.json")
    components = manifest.get("components", [])
    components_present = bool(components) and all(
        isinstance(item, dict) and isinstance(item.get("model_directory"), str)
        and (result_dir / item["model_directory"]).is_dir() for item in components)
    complete = ((result_dir / "run_complete.json").is_file() and manifest_path.is_file()
                and manifest.get("training_scope") == "all_training" and components_present
                and ((result_dir / "mission3.csv").is_file() or not SETTINGS["run_validation"])
                and status.get("status") in {"complete", "complete_not_improved"})
    pilot_path = result_dir / "pilot_complete.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8-sig")) if pilot_path.is_file() else {}
    pilot_complete = (SETTINGS["run_mode"] == "pilot" and status.get("status") == "pilot_complete"
                      and pilot.get("status") == "pilot_complete"
                      and pilot.get("training_scope") == "training_subset"
                      and pilot.get("submission_ready") is False
                      and all((result_dir / name).is_file() for name in
                              ("candidate_summary.csv", "final_selection.json", "full_run_estimate.json"))
                      and not manifest_path.exists() and not (result_dir / "run_complete.json").exists())
    if pilot_complete:
        fitted_samples = pilot.get("fitted_samples", SETTINGS["pilot_samples"])
        print("예비 학습 완료 ·", fitted_samples, "개 학습 결과를 저장하고 종료했습니다.")
        print("전체 Training 학습과 confirmation·공식 Validation 평가는 실행하지 않았습니다.")
        print("후속 전체 학습 예상 시간:", result_dir / "full_run_estimate.json")
        estimate = json.loads((result_dir / "full_run_estimate.json").read_text(encoding="utf-8-sig"))
        recommended_minutes = estimate.get("recommended_budget_minutes") if isinstance(estimate, dict) else None
        if (isinstance(recommended_minutes, (int, float)) and not isinstance(recommended_minutes, bool)
                and math.isfinite(recommended_minutes) and recommended_minutes > 0):
            print("예상 권장 전체 실행 예산:", recommended_minutes, "분 (약",
                  round(recommended_minutes / 60, 1), "시간) · 실측 기반 추정이며 보장 시간은 아닙니다.")
        print("결과를 검토한 뒤 SETTINGS의 run_mode='full', source_run=위 결과 폴더,")
        print("time_budget_minutes=None(시간 제한 없음) 또는 예상 시간을 반영한 예산으로 설정하고 처음부터 모두 실행하세요.")
        print("예비 학습 결과는 최종 제출 모델이 아닙니다.")
    elif complete and SETTINGS["run_mode"] == "full":
        if SETTINGS["run_validation"]:
            print("최종 작업 완료 · 전체 Training 재학습과 저장 검증을 마친 예측:", result_dir / "mission3.csv")
        else:
            print("모델 검증 실행 완료 · Validation 예측과 제출 CSV는 만들지 않았습니다.")
            print("제출용 실행은 run_validation=True로 설정해야 합니다.")
        print("이전 최고 모델과의 성능 비교는 validation_metrics.json으로 따로 확인하세요.")
        if status.get("status") == "complete_not_improved":
            print("새 표현 후보가 대조군보다 개선되지 않음 · 기존 baseline보다 개선한 결과로 표시하지 않습니다.")
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
                       reason="설정한 작업 예산이 끝나 실행을 중단했습니다. 완료 표시가 없는 결과는 제출에 사용하지 마세요.")
    print("ver25 내부 오류 · 실패 단계:", failure["stage"], "· 오류 종류:", failure["error_type"], flush=True)
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
        saved_error = error_dir / ("ver25_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + "_" + uuid.uuid4().hex[:8] + ".json")
        saved_error.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print("오류 보고서:", saved_error, flush=True)
    except OSError as save_error:
        saved_error = None
        print("Drive 오류 보고서 저장 실패:", type(save_error).__name__, "· 위의 내부 오류 내용을 복사하세요.", flush=True)
    raise RuntimeError("ver25 실행 실패 [" + failure["stage"] + " / " + failure["error_type"] + "]: "
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
## 결과와 완료 상태

결과는 `DCC_AIdeas/results/mission3/ver25/실행시각_고유번호/`에 저장합니다.
- `model_specs.json`, split/data digests: 정확한 모델 revision과 학습 데이터 출처
- `candidate_summary.csv`, `final_selection.json`: development 선택과 고정 recipe
- development 예측 및 지표: 9개 F1/AP, 오심 지표, 공통 오답 변화와 불확실성
- 배치 속도 측정 기록: 후보별 처리량·GPU 메모리·선택 결과
- `runtime.json`: 실행 상태, 실측 시간, 예산 판단
- `pilot_complete.json`: 예비 학습 저장 검증 완료, `submission_ready=false`
- `full_run_estimate.json`: 이후 전체 학습에 필요한 시간의 실측 기반 추정

**기본 pilot는 결과 저장 후 종료합니다. confirmation·공식 Validation·전체 Training refit은 실행하지 않습니다.**
작은 subset의 development 점수는 확장 여부를 판단하는 자료이며 전체 학습의 최종 성능이 아닙니다.
예비 체크포인트와 고정 모델 revision을 포함하므로 실행 폴더 전체를 보관하세요.

## 결과를 보고 전체 학습을 따로 실행하기

1. 예비 결과의 F1·오심 지표와 `full_run_estimate.json`을 확인합니다.
2. 위의 **실행 설정 셀 하나에서** `run_mode`를 `'full'`로 바꿉니다.
3. `source_run`에는 `pilot_complete.json`이 있는 완료된 예비 실행 폴더의 절대 경로를 넣습니다.
4. 시간 제한 없이 끝까지 실행하려면 `time_budget_minutes=None`으로 설정합니다. 시간 제한을 원하면 예상 시간과 준비·저장 여유를 반영해 분 단위 정수로 지정합니다. 기본값은 180분입니다.
5. GPU 런타임에서 **처음부터 모두 실행**합니다. 원래 예비 결과는 보존하고 별도 결과 폴더를 만듭니다.

full 실행은 원본 데이터·분할·고정 학습 조건과 완료된 예비 실행의 무결성을 확인합니다.
같은 모델 revision과 동결한 후보·epoch를 사용하고 현재 GPU의 배치를 다시 측정합니다.
confirmation은 동결한 선택을 한 번 확인하는 용도이며 후보를 다시 고르지 않습니다.
작은 subset에서 이어서 학습하는 대신 선택 recipe만 공개 초기 가중치에서 전체 Training으로 새로 학습합니다.
공식 Validation은 전체 refit 후에만 `run_validation=True`일 때 평가합니다.

full 실행에서만 `mission3_ver25_model.ckpt`, 최종 모델 폴더, `submission/`을 만듭니다.
`validation_metrics.json`, `mission3.csv`는 공식 Validation 평가를 켠 full 실행 결과입니다.
`run_complete.json`은 전체 Training refit과 최종 저장·오프라인 재로딩 검증을 통과했을 때만 만듭니다.

**fit-only / 미완료 결과는 전체 Training 최종 제출 모델이 아닙니다.**
`complete_not_improved`는 새 표현 후보가 대조군보다 개선되지 않은 실행의 전체 Training refit과 저장 검증을 마친 상태입니다.
기존 친구 baseline보다 개선했다는 뜻이 아닙니다. 최종 완료 표시는 전체 refit과 저장 검증이 끝났을 때만 만듭니다.
manifest 파일 하나만 보관하지 말고 실행 폴더 전체를 보관하세요.
NF4 Qwen 및 Qwen 포함 앙상블은 저장된 로컬 가중치로 추론할 때도 CUDA GPU가 필요합니다.
KLUE 단독 모델은 CPU 추론도 지원합니다.
실패 시 원문이나 개별 데이터 파일명을 출력하지 않고 실패 단계·오류 종류·코드 위치를
`ver25/_diagnostics/`에 저장한 뒤 GPU 반환을 요청합니다.
중단된 학습의 자동 재개는 지원하지 않습니다. 자동 반환 실패 시 Colab의 연결 해제/런타임 삭제를 직접 실행하세요.

Qwen은 Apache-2.0, KLUE는 CC-BY-SA-4.0이며 출처와 조건을 패키지에 보존합니다.
의존성은 ver25 전용 `requirements.txt`를 사용합니다. 노트북 의존성 설치는 Colab에서 실행합니다.
    """)
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "mission3_ver25.ipynb"}, "accelerator": "GPU",
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver25.ipynb"
    notebook = make_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
