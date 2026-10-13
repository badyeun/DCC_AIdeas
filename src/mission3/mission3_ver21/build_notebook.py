"""Build a standalone Colab notebook for frozen ver19 nausea-head experiments."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "budget.py", "neural.py", "head.py",
    "source_bundle.py", "source_reference.json", "runtime.py", "inference.py",
    "run_notebook.py", "requirements.txt", "README.md",
)


def default_settings():
    return {
        "version": "ver21", "input_zip": "/content/drive/MyDrive/mission3_ver5_inputs.zip",
        "output_dir": "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver21",
        "work_dir": "/content/mission3_ver21", "train_dir": "", "validation_dir": "",
        "source_run": "", "run_validation": True, "seed": 42, "holdout_size": 0.2,
        "batch_size": 4, "encoder_window_batch": 4, "max_head_epochs": 80,
        "head_learning_rate": 0.001, "head_weight_decay": 0.0, "positive_weight": 3.0,
        "soft_f1_weight": 0.5, "head_eval_every": 5, "refit_full_training": True,
        "allow_cpu": False, "log_every": 250, "time_budget_minutes": 180,
        "finalization_reserve_seconds": 1500, "budget_safety_factor": 1.25, "resume_run": "",
    }


def make_notebook():
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    for name, source in sources.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    cells = []

    def add(kind, identifier, source, form=False):
        cell = {"cell_type": kind, "id": f"ver21-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if form:
                cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add("markdown", "intro", """
    # Mission 3 ver21 — 기존 문장 특징을 활용한 오심 판단층 비교

    Colab에서 GPU 런타임을 선택하고 **런타임 → 모두 실행**을 누르세요.
    마지막 학습 셀만 실행하지 마세요. 런타임을 다시 연결하거나 초기화했다면 처음부터 모두 실행합니다.
    준비 셀이 실패하면 그 셀의 첫 오류를 확인하고 해결한 뒤 다시 실행하세요.
    Drive 연결 요청에는 기존 입력 데이터와 ver19 결과가 있는 **공용 계정**을 선택합니다.

    **필요한 파일은 이 노트북, 기존 `mission3_ver5_inputs.zip`, 기존 ver19 결과 폴더 또는 ZIP입니다.**
    ver19의 내부 평가용 모델과 전체 Training 모델의 가중치·assets가 모두 필요합니다.
    이전에 업로드한 ver19 결과를 Drive에서 자동으로 찾습니다. 찾지 못하면 오류에 나온 안내에 따라
    `source_run`에 ver19 결과 폴더 또는 ZIP의 전체 Drive 경로를 넣으세요.
    입력 ZIP만으로 실행되는 버전이 아닙니다. 새 공개 모델을 추가로 내려받지 않습니다.

    문장을 읽는 기존 ver19 모델과 중간 특징·발화 근거 계산은 고정하고 **오심의 마지막 판단층만 학습**합니다.
    다른 8개 증상의 출력은 기존 모델 그대로 유지합니다. 문장 특징을 한 번 저장해 두 학습법에 재사용합니다.
    원래 정답과 모든 샘플·발화를 유지하며 입력은 `utterances[].text`뿐입니다.
    threshold 탐색·사후 점수 보정·외부 학습 데이터 추가 없이 증상별 두 class의 `argmax`로 예측합니다.
    F1 상승과 실제 Colab 속도는 아직 측정하지 않았습니다. 큰 폭의 상승을 보장하지 않습니다.

    **작업 예산은 설치·데이터 준비·특징 계산·학습·최종 평가를 포함해 180분입니다.**
    Drive 로그인 대기 시간은 제외합니다. 느린 GPU나 Drive 상태에 따라 새 학습을 생략할 수 있습니다.
    시간 예산과 모든 단계의 완료는 다릅니다. 완료 표시가 없는 실행은 제출 결과로 사용하지 마세요.
    결과 표시 후, 오류 발생 또는 시간 제한에 도달하면 GPU 런타임의 자동 반환을 요청합니다.
    결과는 Drive에 남지만 `/content`의 임시 파일은 사라질 수 있습니다.
    """)
    add("markdown", "experiments", """
    ## 비교와 최종 모델 선택

    | 후보 | 학습 대상 | 학습 목표 |
    |---|---|---|
    | `baseline` | 기존 ver19 그대로 | 새 학습 없음 |
    | `head_ce` | 오심의 마지막 2-class 판단층 | 기존 양성 가중치 3.0의 cross entropy |
    | `head_ce_soft_f1` | 동일한 판단층과 동일한 시작 가중치 | 같은 CE + 0.5 × soft-F1 loss |

    두 후보는 같은 초기 가중치와 seed 42에서 시작합니다. 최대 80번 갱신하고 5번마다 내부 평가합니다.
    각 갱신에서 Training fit 전체의 저장된 특징을 함께 사용해 soft-F1을 계산합니다.
    단순히 작은 batch의 F1을 평균하거나 gradient accumulation으로 대신하지 않습니다.
    soft-F1은 학습 중의 미분 가능한 목적 함수이며, 최종 판정 방식은 그대로 argmax입니다.

    기존 Training 분할을 검증하고 **내부 fit 23,360건으로 학습한 ver19 모델**에서 특징을 추출해
    동일한 holdout 5,840건에서 후보를 비교합니다. 전체 Training 모델을 이 비교에 사용하지 않습니다.
    Training 내부 평가에서 기존 기준보다 **전체 Macro F1과 오심 F1이 모두 높아진 경우만** 새 후보를 채택합니다.
    공식 Validation은 후보와 학습 횟수를 정한 뒤 한 번 평가하며 학습·선택에 사용하지 않습니다.
    반복 사용한 Training holdout의 개선은 탐색 결과이며 새 독립 평가셋의 성능 보장이 아닙니다.

    개선 후보가 있으면 기존 **전체 Training 29,200건으로 학습한 ver19 모델**에서 특징을 다시 계산합니다.
    이 모델의 원래 오심 판단층에서 시작해 선택한 방법·횟수로 새 판단층을 학습합니다.
    fit 모델의 새 판단층을 전체 Training 모델에 그대로 붙이지 않습니다.
    개선에 실패하거나 새 학습을 완료할 시간이 없으면 **기존 전체 Training ver19 모델을 그대로 보존한 결과**를 저장합니다.
    이 경우 새 모델의 성능 향상으로 표시하지 않습니다. 기존 Drive의 ver19 파일은 수정하지 않습니다.
    필수 소스 확인·내부 비교·최종 저장 자체가 시간 내 끝나지 않으면 미완료로 종료합니다.
    """)
    add("code", "settings", """
    # @title 실행 설정
    SETTINGS = __SETTINGS__
    AUTO_RELEASE_GPU = True # @param {type:"boolean"}
    RUNTIME_PREPARED = None
    DEPENDENCIES_PREPARED = None
    # Leave source_run empty to discover the existing ver19 result folder or ZIP.
    # If discovery fails, set source_run to its complete Drive path.
    # Each explicit rerun starts a NEW budget and consumes additional Colab resources.
    """.replace("__SETTINGS__", repr(default_settings())))
    add("code", "drive", """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    """)
    add("code", "budget", """
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
    add("code", "restore", """
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
        runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver21_"))
        for name, source in RUNTIME_SOURCES.items():
            (runtime_dir / name).write_bytes(source.encode("utf-8"))
        budget_spec = importlib.util.spec_from_file_location("mission3_ver21_budget_supervisor", runtime_dir / "budget.py")
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
    add("code", "dependencies", """
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
    add("code", "run", """
    # @title 기존 모델 확인 → 특징 저장 · 후보 비교 → 최종 평가 · GPU 반환
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
        raise RuntimeError("현재 실행의 코드 또는 패키지 준비가 미완료입니다. 앞선 셀의 첫 오류를 확인하고 런타임 → 모두 실행으로 처음부터 실행하세요.")
    try:
        if time.monotonic() >= DEADLINE_MONOTONIC:
            raise RuntimeError("이번 작업의 시간 예산이 끝났습니다. 다시 실행하려면 처음 셀부터 모두 실행하세요.")
        settings_file = runtime_dir / "notebook_settings.json"
        result_file = runtime_dir / "notebook_result.json"
        error_file = runtime_dir / "notebook_error.json"
        error_file.write_text(json.dumps({"status": "starting"}), encoding="utf-8")
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
            print("Training 내부 비교:")
            with summary.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    print(row["candidate"], "Macro F1 =", round(float(row["macro_f1"]), 6),
                          "오심 F1 =", round(float(row["nausea_f1"]), 6),
                          "선택 갱신 횟수 =", row["selected_epochs"], "종료 이유 =", row["stop_reason"])
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
                    print("전체 F1과 오심 F1의 동시 개선 조건을 충족하지 못했습니다.")
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
        manifest_path = result_dir / "mission3_ver21_model.ckpt"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            print("저장된 후보:", manifest.get("selected_candidate", "모델 정보 확인"))
            print("저장 모델 학습 범위:", manifest.get("training_scope", "모델 정보 확인"),
                  "· 학습 표본:", manifest.get("fitted_samples", "모델 정보 확인"),
                  "· 오심 판단층 갱신 횟수:", manifest.get("head_epochs", 0))
            if manifest.get("refit_skipped_reason"):
                print("추가 학습 생략 이유:", manifest["refit_skipped_reason"])
        status_path = result_dir / "runtime.json"
        status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
        print("실행 상태:", status.get("status", "기록 확인 필요"))
        print("이번 노트북 소요 시간:", round((time.monotonic() - BUDGET_STARTED_MONOTONIC) / 60, 1), "분")
        print("선택 근거:", result_dir / "final_selection.json")
        complete = ((result_dir / "run_complete.json").is_file() and (result_dir / "mission3.csv").is_file()
                    and status.get("status") in {"complete", "complete_budget_limited", "complete_not_improved"})
        if complete:
            if status.get("status") == "complete_not_improved":
                print("비교 완료 · 개선 미달로 기존 전체 Training ver19 모델을 그대로 보존했습니다.")
                print("성능 향상 결과가 아닙니다. 최종 예측:", result_dir / "mission3.csv")
            elif status.get("status") == "complete_budget_limited":
                print("예산 제한 · 새 학습을 완료하지 못해 기존 전체 Training ver19 모델을 보존했습니다.")
                print("성능 향상 결과가 아닙니다. 최종 예측:", result_dir / "mission3.csv")
            else:
                print("최종 작업 완료 · 새 오심 판단층을 적용한 예측:", result_dir / "mission3.csv")
            print("기존 Drive의 ver19 원본 파일은 유지됩니다.")
        else:
            print("미완료 실행:", status.get("status", "상태 기록 없음"),
                  "· 최종 제출 결과로 사용하지 마세요. 결과 폴더의 로그를 확인하세요.")
    except subprocess.CalledProcessError as worker_error:
        failure = {"status": "failed", "stage": "worker_process", "error_type": "WorkerProcessError",
                   "exit_code": worker_error.returncode,
                   "reason": "실행 프로그램이 상세 오류를 저장하기 전에 종료됐습니다. 종료 코드와 준비 상태 확인이 필요합니다.",
                   "frames": []}
        try:
            recorded = json.loads(error_file.read_text(encoding="utf-8"))
            if isinstance(recorded, dict) and recorded.get("status") == "failed":
                failure.update(recorded)
        except (OSError, ValueError):
            pass
        print("ver21 내부 오류 · 실패 단계:", failure["stage"], "· 오류 종류:", failure["error_type"], flush=True)
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
            saved_error = error_dir / ("ver21_error_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                      + "_" + uuid.uuid4().hex[:8] + ".json")
            saved_error.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
            print("오류 보고서:", saved_error, flush=True)
        except OSError as save_error:
            saved_error = None
            print("Drive 오류 보고서 저장 실패:", type(save_error).__name__, "· 위의 내부 오류 내용을 복사하세요.", flush=True)
        raise RuntimeError("ver21 실행 실패 [" + failure["stage"] + " / " + failure["error_type"] + "]: "
                           + failure["reason"] + (" 오류 보고서: " + str(saved_error) if saved_error else "")) from None
    finally:
        cleanup = globals().get("release_gpu")
        if callable(cleanup):
            try:
                cleanup()
            except Exception as cleanup_error:
                print("GPU 반환 중 오류:", type(cleanup_error).__name__, "· Colab에서 직접 연결을 해제하세요.")
    """)
    add("markdown", "results", """
    ## 결과 확인 · 다시 실행할 때

    결과는 `DCC_AIdeas/results/mission3/ver21/실행시각_고유번호/`에 저장합니다.
    `candidate_summary.csv`, `training_reference_comparison.json`, `final_selection.json`,
    `training_metrics.json`, `validation_metrics.json`, `runtime.json`을 확인하세요.
    `complete`는 개선 후보를 전체 Training 특징으로 학습한 결과입니다.
    `complete_not_improved`는 개선 미달, `complete_budget_limited`는 시간 제약 때문에
    **기존 전체 Training ver19 모델을 그대로 보존한 결과**입니다. 새 성능 향상으로 해석하지 마세요.
    `run_complete.json`과 최종 CSV가 없는 실행은 완료된 제출 결과가 아닙니다.
    실행이 실패하면 GPU 반환 전에 실패 단계·오류 종류·코드 위치를 표시하고
    `ver21/_diagnostics/ver21_error_*.json`에 같은 내용을 저장합니다.
    원인 분석에는 이 오류 보고서를 보내주세요. 원문 대화와 개별 데이터 파일 이름은 포함하지 않습니다.

    `resume_run`에 이전 실행 폴더의 전체 Drive 경로를 넣으면 코드·설정·데이터 일치를 확인합니다.
    완료된 결과는 검증 후 반환하며, 중단된 학습은 마지막 갱신부터 이어지는 것을 보장하지 않습니다.
    **재실행에는 새 시간 예산과 추가 Colab 리소스가 듭니다.** 새 실험은 `resume_run`을 비우고 실행합니다.
    모델 `.ckpt`는 가중치·assets의 상대 경로를 담은 manifest이므로 **실행 폴더 전체를 보관**하세요.
    제출용 추론 코드는 `submission/inference.py`이며 예측 CSV의 헤더는 `label file name,symptom`입니다.
    원문 대화나 개인정보를 화면과 결과 요약에 출력하지 않습니다.
    자동 반환이 실패하면 **런타임 → 연결 해제 및 런타임 삭제**를 직접 실행하세요.
    자동 반환 기능의 실제 서비스 동작은 Colab에서 확인해야 합니다.
    """)
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "mission3_ver21.ipynb"}, "accelerator": "GPU",
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def build(output=None):
    target = Path(output) if output is not None else HERE.parent / "mission3_ver21.ipynb"
    notebook = make_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build())
