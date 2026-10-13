"""Build the standalone Colab Qwen QLoRA comparison notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "analysis.py", "bootstrap.py", "budget.py", "batching.py", "trainer.py", "runtime.py",
    "run_notebook.py", "build_notebook.py", "requirements.txt", "README.md", "model_specs.json", "baseline.py", "baseline_reference.json",
)


def default_settings():
    return {'version': 'ver27',
     'run_mode': 'pilot',
     'pilot_samples': 4000,
     'source_run': '',
     'baseline_run': '/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver26/20261011T133609Z_f22b4e55',
     'input_zip': '/content/drive/MyDrive/mission3_ver5_inputs.zip',
     'output_dir': '/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver27',
     'work_dir': '/content/mission3_ver27',
     'train_dir': '',
     'validation_dir': '',
     'run_validation': False,
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
     'max_epochs': 3,
     'benchmark_examples': 64,
     'benchmark_warmup_examples': 16,
     'effective_batch_size': 16,
     'warmup_ratio': 0.05,
     'lora_lr': 3e-05,
     'head_lr': 9e-05,
     'control_lr': 2e-05,
     'weight_decay': 0.01,
     'lora_r': 16,
     'lora_alpha': 32,
     'lora_dropout': 0.05,
     'gradient_clip': 1.0,
     'min_macro_gain': 0.01,
     'allow_cpu': False,
     'time_budget_minutes': 480,
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
        cell = {"cell_type": kind, "id": f"ver27-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add('markdown', 'intro', """
# Mission 3 ver27 — Qwen 학습률 0.3배 대조 실험

이 노트북을 Colab에 업로드하고 **GPU 런타임 → 모두 실행**을 누르세요.
기존 `mission3_ver5_inputs.zip`이 있는 계정으로 Drive를 연결합니다.
기본 설정을 그대로 실행하면 같은 4,000개 표본으로 Qwen과 새 KLUE 대조군을 각각 3 epoch 학습하고
각 epoch의 development 지표·예측·loss·학습률을 저장한 뒤 GPU 반환을 요청합니다.

**ver26과 같은 4,000개·3 epoch에서 Qwen 학습률만 0.3배로 낮춰 비교합니다.**
LoRA 1e-4 → 3e-5, head 3e-4 → 9e-5이며 다른 학습 조건은 유지합니다.
`baseline_run`에는 완료된 ver26 기준 실행이 지정되어 있습니다. 기준 폴더를 지우지 마세요.
모델 다운로드 전에 기준 파일·분할·subset·정답·설정을 검사합니다. 누락 또는 불일치하면 즉시 실패합니다.
점수 개선이나 Macro F1 0.7 달성을 보장하는 노트북이 아닙니다.
`ver25`·`ver26` 결과는 건드리지 않고 `results/mission3/ver27/실행시각_고유번호/`에 새로 저장합니다.
**최종 제출 모델을 만들지 않으며 `run_mode='full'`은 거절합니다.**
작은 subset의 최고 epoch를 전체 데이터의 최적 epoch로 간주하지 않습니다.
유망한 경우 다음 실험에서 fit 전체 20,440개로 epoch를 확인한 뒤 최종 학습 조건을 정해야 합니다.

기본값 `time_budget_minutes=480`은 설치·준비·평가·저장을 포함한 **실행당 최대 8시간**입니다.
기본 `AUTO_RELEASE_GPU=True`를 유지하면 완료·오류·시간 만료 시 반환을 요청합니다.
마지막 25분은 저장과 검증을 위해 예약합니다. 3 epoch 전체가 예상 예산에 들어올 때만 시작합니다.
Colab 자체의 세션 종료는 막을 수 없습니다.
시간이 모자라다고 1 epoch로 줄여 성공 처리하지 않습니다.
배치 후보를 직접 측정해서 가장 빠른 성공 후보와 기울기 누적을 선택합니다.

기준 ver26은 L4에서 준비·두 모델 3 epoch·저장까지 **2시간 51분 42초** 걸렸습니다.
이번 실행도 약 3시간 수준의 작업량을 예상하지만 기준 검증·GPU·캐시·저장 속도에 따라 달라집니다.
8시간은 예상 소요 시간이 아니라 상한입니다. ver27의 실제 GPU 실행·성능은 아직 미확인입니다.
준비 셀이 실패하면 마지막 셀만 실행하지 말고 처음부터 모두 실행하세요.
    """)

    add('markdown', 'experiments', """
## 고정 실험 조건

| 항목 | Qwen | 새 KLUE 대조군 |
|---|---|---|
| 공개 모델 | Qwen3-4B-Instruct-2507 | klue/roberta-base |
| 학습 | NF4 4-bit + LoRA r16 + 9출력 분류층 | 전체 encoder + 분류층 |
| 입력 길이 | 1,024, 앞/뒤 + 고정 판정 접미문 | 앞부분 512 |
| 배치 측정 후보 | 1 / 2 / 4 / 8 | 4 / 8 / 16 |
| 유효 batch | 16 | 16 |
| learning rate | LoRA 3e-5 / head 9e-5 (ver26의 0.3배) | 2e-5 (유지) |
| 학습 | 같은 4,000개, 3 epoch | 같은 4,000개, 3 epoch |

완료된 ver25와 같은 공개 모델 revision, seed42, 중복 그룹을 보존한 70/15/15 분할과 subset 선택법을 사용합니다.
Training 원본 정답만 사용하고 학습 입력은 `utterances[].text`뿐입니다. 외부 데이터·재라벨링·표본 삭제는 없습니다.
weighted BCE와 `pos_weight=clip(sqrt(음성/양성),1,5)`, 고정 판정 `logit >= 0`를 유지합니다.
각 epoch마다 development 4,380개를 평가합니다. confirmation·공식 Validation은 평가하지 않습니다.
입력 ZIP 준비 과정에서 Validation 파일을 추출·무결성 확인할 수 있으나 모델 평가에는 열지 않습니다.

**ver26과 scheduler의 총 길이 3 epoch를 동일하게 유지하고 Qwen 학습률만 낮춥니다.**
기준 ver26 KLUE 예측을 그대로 사용한 1:1 평균으로 Qwen 변경의 영향을 비교합니다.
새 KLUE와의 평균도 보조 지표로 저장합니다. 이는 저장된 예측 비교이며 과거 KLUE 가중치를 새 제출 모델로 가져오지 않습니다.
같은 실행의 학습 곡선과 기준 실행 대비 변화를 함께 봅니다. 최적 epoch의 일반화 여부는 fit 전체에서 추가 확인해야 합니다.
Development는 과거 실험에도 사용된 셋이므로 완전히 새로운 외부 검증이 아닙니다.
가장 좋은 Qwen/KLUE epoch와 고정 1:1 확률 평균은 진단 참고용으로만 비교합니다.
threshold/offset/calibration/혼합비 검색은 없으며 ASL·대형 모델·전체 학습은 자동 실행하지 않습니다.
    """)

    add('code', 'settings', """
# @title 실행 설정
SETTINGS = __SETTINGS__
# Screening only: keep run_mode='pilot', max_epochs=3, run_validation=False.
# Full training is disabled. The full invocation is limited to 8 hours.
# baseline_run is mandatory and must identify the pinned completed ver26 run.
# Results are diagnostic and are not a submission model.
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
if (isinstance(configured_minutes, bool) or not isinstance(configured_minutes, (int, float))
        or not math.isfinite(configured_minutes) or not 0 < configured_minutes <= 480):
    raise ValueError("time_budget_minutes must be a finite positive number at most 480; unlimited is disabled")
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
        path = folder / ("ver27_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver27_"))
    for name, source in RUNTIME_SOURCES.items():
        (runtime_dir / name).write_bytes(source.encode("utf-8"))
    budget_spec = importlib.util.spec_from_file_location("mission3_ver27_budget_supervisor", runtime_dir / "budget.py")
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
# @title ver27 전용 QLoRA 패키지 준비 (시간 제한 사용 시 최대 20분, 전체 예산에 포함)
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
# @title 3 epoch 학습 곡선 확인 → 저장 → 종료 · GPU 반환
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
    for filename in ["training_metrics.json"]:
        path = result_dir / filename
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            print(filename, "Macro F1 =", round(report["macro_f1"], 6))
            for symptom, values in report["per_symptom"].items():
                print(symptom, "F1 =", round(values["f1"], 4),
                      "P =", round(values["precision"], 4), "R =", round(values["recall"], 4))
    status_path = result_dir / "runtime.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
    print("실행 상태:", status.get("status", "기록 확인 필요"))
    print("이번 노트북 소요 시간:", round((time.monotonic() - BUDGET_STARTED_MONOTONIC) / 60, 1), "분")
    marker_path = result_dir / "screening_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig")) if marker_path.is_file() else {}
    complete = (status.get("status") == marker.get("status") == "screening_complete"
        and marker.get("submission_ready") is False and marker.get("scheduler_horizon_epochs") == 3
        and marker.get("validation_evaluated") is False and marker.get("confirmation_evaluated") is False
        and marker.get("baseline_comparison_completed") is True
        and all((result_dir / name).is_file() for name in (
            "qwen_history.json", "control_history.json", "screening_selection.json", "learning_curve_summary.json",
            "screening_protocol.json", "screening_portable_verification.json",
            "baseline_reference_verified.json", "historical_comparison.json", "baseline_gate.json"))
        and not (result_dir / "run_complete.json").exists())
    if complete:
        print("3 epoch 학습 곡선 확인 완료 ·", marker.get("fitted_samples"), "개 학습 결과 저장 후 종료합니다.")
        for kind in ("qwen", "control"):
            history = json.loads((result_dir / (kind + "_history.json")).read_text(encoding="utf-8"))
            for row in history:
                print(kind, "epoch", row["selected_epochs"], "Macro F1", round(row["macro_f1"], 6),
                      "train loss", round(row["train_mean_loss"], 4),
                      "dev loss", round(row["development_weighted_bce"], 4))
        gate = json.loads((result_dir / "baseline_gate.json").read_text(encoding="utf-8"))
        print("ver26 기준 비교:", gate["status"], "· Macro F1 변화 =", round(gate["macro_f1_delta"], 6))
        print("과거 KLUE를 동일하게 고정한 예측 평균 비교입니다. 공식 Validation 점수가 아닙니다.")
        print("제출 모델이 아닙니다. 유망한 경우에도 더 큰 fit에서 학습 조건을 확인해야 합니다.")
        print("이 노트북에서 full 실행은 지원하지 않으며 confirmation·공식 Validation은 평가하지 않았습니다.")
    else:
        print("미완료 실행:", status.get("status", "상태 기록 없음"), "· 결과 폴더의 로그를 확인하세요.")
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
    print("ver27 내부 오류 · 실패 단계:", failure["stage"], "· 오류 종류:", failure["error_type"], flush=True)
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
        saved_error = error_dir / ("ver27_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + "_" + uuid.uuid4().hex[:8] + ".json")
        saved_error.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print("오류 보고서:", saved_error, flush=True)
    except OSError as save_error:
        saved_error = None
        print("Drive 오류 보고서 저장 실패:", type(save_error).__name__, "· 위의 내부 오류 내용을 복사하세요.", flush=True)
    raise RuntimeError("ver27 실행 실패 [" + failure["stage"] + " / " + failure["error_type"] + "]: "
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

결과 폴더: `MyDrive/DCC_AIdeas/results/mission3/ver27/실행시각_고유번호/`

- `qwen_history.json`, `control_history.json`: epoch별 train loss, development weighted BCE, 9개 F1/AP, 오심 P/R/TP/FP/FN, LR 시작/끝, optimizer update 수
- `learning_curve/`: 매 epoch의 development 예측과 지표
- `learning_curve_summary.json`: epoch1 대비 후속 epoch 개선 여부, 최고 epoch, 정체·하락 표시
- `candidate_summary.csv`, `screening_selection.json`: development 기준 진단 비교
- `screening_protocol.json`: scheduler 길이와 해석 한계, 다음 단계
- `baseline_reference_verified.json`, `baseline_ver26/`: 무결성을 검사한 ver26 기준 자료
- `historical_comparison.json`, `historical_comparison_predictions.npz`: 이전 실행과의 짝지은 비교
- `baseline_gate.json`: 더 큰 fit 확인을 진행할 근거와 한계; 전체 학습은 항상 비허용
- `batch_selection.json`: 실제 배치 속도 측정과 선택
- `screening_portable_verification.json`: 두 모델의 최고 epoch 저장·재로딩 예측 확인
- `screening_complete.json`: 3 epoch 완료 및 저장 검증, `submission_ready=false`
- `source_snapshot/`, 모델 폴더, 토크나이저, `shared_base/`: 재현에 필요한 코드와 가중치

정상 완료 상태는 `runtime.json`의 `status='screening_complete'`입니다.
`run_complete.json`, 제출 CSV, 최종 제출 manifest는 만들지 않습니다.
실행 폴더 전체를 보관하세요. 이 결과를 ver25의 `source_run`으로 넣으면 안 됩니다.
중단된 학습의 자동 재개는 지원하지 않습니다.

결과를 본 뒤 유망한 학습 조건을 **fit 전체 20,440개에서 먼저 확인**하고 최종 epoch를 고정합니다.
그 후 보류한 confirmation을 한 번 확인하고 전체 Training refit을 별도로 구현·실행하는 순서입니다.
작은 subset의 최고 epoch를 그대로 전체 Training에 적용하지 않습니다.
새 KLUE 대비 높은 점수만으로 개선을 주장하지 않습니다. ver26과 같은 KLUE를 고정한 평균의
Macro F1이 0.005 이상 오르고 기술적 bootstrap 하한이 0보다 높을 때만 더 큰 fit 확인 후보로 표시합니다.
Development 재사용과 epoch 선택에 따른 낙관 편향·seed 변동은 이 구간에 포함되지 않습니다.
해당 표시는 전체 학습 허가나 목표 점수 달성 보장이 아닙니다.

Qwen은 Apache-2.0, KLUE는 CC-BY-SA-4.0이며 출처와 조건을 결과에 보존합니다.
의존성 버전은 ver25와 같습니다. 자동 반환 실패 시 Colab에서 연결 해제/런타임 삭제를 직접 실행하세요.
    """)
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "mission3_ver27.ipynb"}, "accelerator": "GPU",
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver27.ipynb"
    notebook = make_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
