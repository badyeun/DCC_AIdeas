"""Build the standalone ver7 Colab notebook from the checked-in runtime sources."""
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def build():
    cells = []

    def add(kind, identifier, source, **metadata):
        source = textwrap.dedent(source).strip() + "\n"
        cell = {"cell_type": kind, "id": identifier, "metadata": metadata, "source": source.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "ver7-intro", """
    # Mission 3 ver7 — 새로운 KoELECTRA encoder 전체 미세조정

    **기존 `mission3_ver5_inputs.zip`과 이 노트북 하나로 실행합니다.**

    1. Drive 공유 문서함의 `mission3_ver5_inputs.zip`에서 **정리 → 바로가기 추가 → 내 드라이브**를 선택합니다.
    2. 내 드라이브 바로 아래에 같은 이름의 바로가기를 둡니다.
    3. 이 노트북을 Colab에서 열고 **GPU 런타임 → 모두 실행**을 선택합니다.
    4. 바로가기를 만든 계정으로 Drive를 연결합니다. 다른 위치라면 아래 `INPUT_ZIP`만 바꾸세요.

    ZIP의 기존 ver4 모델과 Training·Validation JSON을 그대로 복원합니다. WAV와 별도 DATA 폴더는 필요 없습니다.
    새 encoder `monologg/koelectra-base-v3-discriminator`와 tokenizer는 공개 저장소에서 자동 다운로드합니다.
    고정된 공식 커밋을 사용하며, 다운로드 단계 이후의 학습·추론은 로컬 파일만 읽습니다.
    [모델 원본](https://huggingface.co/monologg/koelectra-base-v3-discriminator/tree/68b30cd259f34a4b5aa8786392612ba2a2617fcc).

    **기존 입력:** `/content/mission3_ver5_inputs/DCC_AIdeas/results/mission3/ver4`  
    **새 결과:** `내 드라이브/DCC_AIdeas/results/mission3/ver7`

    ver7은 공개 사전학습 KoELECTRA에서 시작해 전체 encoder와 새 9-label attention 분류기를 함께 학습합니다.
    기존 ver4의 5개 모델은 비교·혼합용으로 재사용합니다. 새 encoder 학습에는 frozen feature cache가 없습니다.
    기본값은 최대 3 epoch, 개선 없는 2 epoch 후 중단입니다. 모델 선택은 Training 내부에서 끝냅니다.
    최종 선택 결과와 새 후보 단독의 Validation 점수를 마지막에 각각 표시합니다.
    후보 단독 점수는 진단용이며 최종 모델·혼합 비율·임계값을 바꾸지 않습니다.
    """)
    add("code", "ver7-settings", """
    # 1. 기본 경로에서는 값을 바꾸지 않고 모두 실행하면 됩니다.
    INPUT_ZIP = "/content/drive/MyDrive/mission3_ver5_inputs.zip"
    BASE_DIR = None
    TRAIN_LABEL_DIR = None
    VALID_LABEL_DIR = None
    OUTPUT_DIR = None           # Drive/MyDrive/DCC_AIdeas/results/mission3/ver7
    INFERENCE_CACHE_DIR = None  # Drive의 공용 results/mission3/prediction_cache
    PRETRAINED_DIR = None       # Colab 로컬의 고정 경로에 새 encoder 자동 준비
    RUN_VALIDATION = True
    RUN_CANDIDATE_DIAGNOSTIC = True  # 선택 확정 후 후보 단독 Validation 진단도 저장
    TRAIN_CONFIG = {
        "bundle_base": False,
        "max_epochs": 3,
        "patience": 2,
        "learning_rate": 2e-5,
        "head_learning_rate": 1e-4,
        "stop_crossfit_splits": 3,
        "min_calibration_gain": 0.002,
        "min_calibration_improved_folds": 3,
    }
    INFERENCE_BATCH_SIZE = 8
    # GPU 메모리에 맞춰 microbatch를 선택하고 유효 batch 32를 유지합니다.
    # 설정을 바꾸어 새로 실험할 때는 OUTPUT_DIR을 새 폴더로 지정하세요.
    """)
    add("code", "ver7-dependencies", """
    # 2. 기존 저장 모델과 새 KoELECTRA에 필요한 패키지 설치
    import subprocess
    import sys

    REQUIREMENTS = __REQUIREMENTS__
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *REQUIREMENTS], check=True)
    print("환경 준비 완료. 모델 준비·학습·추론은 새 Python 프로세스에서 실행합니다.")
    """.replace("__REQUIREMENTS__", repr((HERE / "requirements.txt").read_text().splitlines())))
    add("code", "ver7-paths", """
    # 3. Drive 연결 → 같은 입력 ZIP 검증·복원 → 경로 연결
    import json
    import os
    import tempfile
    import time
    import uuid
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    preparation_started = time.monotonic()
    START_DIR = Path.cwd().resolve()
    try:
        from google.colab import drive
    except ImportError:
        IN_COLAB = False
    else:
        IN_COLAB = True
        drive.mount("/content/drive", force_remount=False)
    BOOTSTRAP_SOURCE = __BOOTSTRAP__
    bootstrap_namespace = {}
    exec(compile(BOOTSTRAP_SOURCE, "mission3_bootstrap.py", "exec"), bootstrap_namespace)
    bundle_path = Path(INPUT_ZIP).expanduser().resolve() if INPUT_ZIP is not None else None
    BUNDLE_ROOT = None
    if bundle_path is not None:
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"ZIP 바로가기를 찾지 못했습니다: {bundle_path}\\n"
                "공유 ZIP의 '정리 → 바로가기 추가 → 내 드라이브'를 선택하고 INPUT_ZIP을 확인하세요. "
                "바로가기를 만든 계정으로 연결하세요. 연결 후 바로가기를 추가했다면 "
                "drive.mount의 force_remount=True로 한 번 재연결하세요.")
        extract_dir = (Path("/content/mission3_ver5_inputs") if IN_COLAB
                       else Path(tempfile.gettempdir()) / "mission3_ver5_inputs")
        BUNDLE_ROOT = bootstrap_namespace["restore_input_bundle"](bundle_path, extract_dir)
        BASE_DIR = BUNDLE_ROOT / "results/mission3/ver4"
    if BASE_DIR is None:
        raise ValueError("INPUT_ZIP 또는 기존 ver4 폴더인 BASE_DIR를 지정하세요.")
    BASE_DIR = Path(BASE_DIR).expanduser().resolve()
    for name in ("mission3_ver4_model.ckpt", "splits.json", "calibration_predictions.npz",
                 "sparse.joblib", "tokenizer", *[f"fold_{i}.ckpt" for i in range(1, 6)]):
        if not (BASE_DIR / name).exists():
            raise FileNotFoundError(f"기존 ver4 입력 파일이 빠졌습니다: {BASE_DIR / name}")
    data_root = ((BUNDLE_ROOT if BUNDLE_ROOT is not None else BASE_DIR.parents[2])
                 / "DATA/대학부 데이터")
    def labels(explicit, split, name, required=True):
        path = (data_root / split / "2.라벨링데이터" / name if explicit is None
                else Path(explicit).expanduser()).resolve()
        if path.is_dir() and next(path.rglob("*.json"), None) is not None:
            return path
        if required:
            raise FileNotFoundError(f"{split} JSON 경로를 확인하세요: {path}")
        return None
    TRAIN_LABEL_DIR = labels(TRAIN_LABEL_DIR, "Training", "TL_서울_구급")
    VALID_LABEL_DIR = labels(VALID_LABEL_DIR, "Validation", "VL_서울_구급", RUN_VALIDATION)
    result_parent = (Path("/content/drive/MyDrive/DCC_AIdeas/results/mission3") if IN_COLAB
                     else START_DIR / "results/mission3")
    OUTPUT_DIR = (result_parent / "ver7" if OUTPUT_DIR is None else Path(OUTPUT_DIR).expanduser()).resolve()
    INFERENCE_CACHE_DIR = (result_parent / "prediction_cache" if INFERENCE_CACHE_DIR is None
                           else Path(INFERENCE_CACHE_DIR).expanduser()).resolve()
    local_pretrained = ((Path("/content") if IN_COLAB else Path(tempfile.gettempdir()))
                        / "mission3_ver7_pretrained/koelectra_base_v3")
    PRETRAINED_DIR = (local_pretrained if PRETRAINED_DIR is None else Path(PRETRAINED_DIR).expanduser()).resolve()
    if OUTPUT_DIR == BASE_DIR or OUTPUT_DIR.is_relative_to(BASE_DIR) or BASE_DIR.is_relative_to(OUTPUT_DIR):
        raise ValueError("OUTPUT_DIR는 기존 ver4 입력 폴더와 분리된 새 폴더여야 합니다.")
    PREPARATION_MINUTES = (time.monotonic() - preparation_started) / 60
    print("입력 ZIP:", bundle_path)
    print("기존 ver4 모델:", BASE_DIR)
    print("Training JSON:", TRAIN_LABEL_DIR)
    print("최종 평가 JSON:", VALID_LABEL_DIR)
    print("새 KoELECTRA 로컬 경로:", PRETRAINED_DIR)
    print("새 ver7 결과 (Drive):", OUTPUT_DIR)
    print("공유 원시 예측 캐시 (Drive):", INFERENCE_CACHE_DIR)
    print(f"Drive 연결·입력 준비: {PREPARATION_MINUTES:.1f}분")
    """.replace("__BOOTSTRAP__", repr((HERE / "bootstrap.py").read_text(encoding="utf-8"))))

    names = ("__init__.py", "bootstrap.py", "neural.py", "data.py", "pretrained.py", "selection.py",
             "calibration.py", "prediction_cache.py", "pipeline.py", "train.py", "inference.py",
             "requirements.txt", "README.md", "RESEARCH.md")
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in names}
    for path in sorted((HERE / "ver4_runtime").iterdir()):
        if path.is_file():
            sources["ver4_runtime/" + path.name] = path.read_text(encoding="utf-8")
    add("code", "ver7-runtime", """
    # @title 4. 노트북에 포함된 실행 코드 준비 (별도 소스 업로드 불필요)
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver7_runtime_"))
    for name, source in RUNTIME_SOURCES.items():
        path = runtime_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    config_path = runtime_dir / "notebook_config.json"
    config_path.write_text(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_python(arguments, allow_download=False):
        command = [sys.executable, "-u", *map(str, arguments)]
        offline = "0" if allow_download else "1"
        environment = dict(os.environ, HF_HUB_OFFLINE=offline, TRANSFORMERS_OFFLINE=offline,
                           HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=environment)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def run_python(arguments):
        return _run_python(arguments)

    def run_pretrained():
        # 공개 모델 다운로드에만 네트워크 허용. 대회 입력 경로는 전달하지 않습니다.
        return _run_python([runtime_dir / "pretrained.py", "--destination", PRETRAINED_DIR], allow_download=True)

    print("실행 코드:", runtime_dir)
    """.replace("__SOURCES__", repr(sources)), cellView="form")
    add("code", "ver7-pretrained", """
    # 5. 공개 사전학습 encoder 자동 다운로드·검증 (같은 런타임에서는 검증 후 재사용)
    download_started = time.monotonic()
    run_pretrained()
    PRETRAINED_MINUTES = (time.monotonic() - download_started) / 60
    print(f"사전학습 모델 준비: {PRETRAINED_MINUTES:.1f}분")
    """)
    add("markdown", "ver7-training-note", """
    ## 6. 새로운 encoder 학습 → Training 내부 비교·선택

    발화 순서를 유지하며 모든 발화 text를 같은 `[SEP]`로 구분합니다. speaker·증상 정답에 따라 입력을 바꾸지 않습니다.
    긴 대화는 window로 나누고, token 표현에 9개 증상별 attention을 적용합니다.
    **KoELECTRA 전체 encoder와 새 분류기**를 BCE로 함께 학습합니다. frozen feature cache는 생성하지 않습니다.

    - 원래 fold 1의 **fit 21,032개**로 가중치를 학습합니다.
    - **stop 5,288개**의 3-fold threshold 교차평가로 epoch를 고릅니다.
    - 같은 stop에서 원래 ver4 **fold 1 단독**도 비교합니다. 이 비교는 5개 모델 앙상블과 구분합니다.
    - 별도 Training **calibration 2,880개**의 5-fold 교차평가로 혼합과 임계값을 정합니다.
    - 기존 구성보다 Macro F1 **0.002 이상**, **5개 fold 중 3개 이상** 개선될 때 새 구성을 채택합니다.

    5개의 새 encoder를 각각 학습하는 OOF 실험은 아닙니다. stop에서 epoch를 반복 비교한 선택 편향도 남습니다.
    공개 KoELECTRA의 한국어 표현과 [PLM-ICD의 segment/label attention](https://aclanthology.org/2022.clinicalnlp-1.2.pdf)을
    적용한 실험이며, Mission 3에서의 F1 상승이나 총 CU는 실행 결과로 확인해야 합니다.

    **중단된 학습 재개:** 같은 코드·데이터·설정, 같은 GPU 종류·AMP 정밀도·관련 패키지 버전을 사용하세요.
    **완료된 결과 재사용:** 다른 GPU에서도 가능하며 실행 조건이 다른 예측 캐시는 다시 계산합니다.
    새 Colab 세션에서는 입력 ZIP 복원과 공개 encoder 준비를 다시 수행합니다.
    """)
    add("code", "ver7-train", """
    training_started = time.monotonic()
    run_python([
        runtime_dir / "train.py", "--train_label_dir", TRAIN_LABEL_DIR,
        "--base_dir", BASE_DIR, "--pretrained_dir", PRETRAINED_DIR,
        "--output_dir", OUTPUT_DIR, "--config_json", config_path, "--auto_batch",
    ])
    TRAINING_MINUTES = (time.monotonic() - training_started) / 60
    CHECKPOINT_PATH = OUTPUT_DIR / "mission3_ver7_model.ckpt"
    print(f"전체 encoder 학습·내부 비교·보정·저장: {TRAINING_MINUTES:.1f}분")
    print("ver7 최종 모델 구성 파일:", CHECKPOINT_PATH)
    """)
    add("code", "ver7-report", """
    # 7. Training 내부 결과. 공식 Validation은 아직 모델 선택에 사용하지 않았습니다.
    REPORT_PATH = OUTPUT_DIR / "mission3_ver7_metrics.json"
    training_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    candidate_report = training_report.get("candidate_training", {})
    print("KoELECTRA 전체 미세조정 최적 epoch:", candidate_report.get("best_epoch", "상세 기록 확인"))
    if candidate_report.get("best_is_initial_baseline", candidate_report.get("best_epoch") == 0):
        print("공개 encoder와 초기 무작위 분류기 상태가 선택됐습니다. 학습한 epoch가 최종 후보에 반영되지 않았습니다.")
    print("새 후보 stop 교차평가 Macro F1:", candidate_report.get("best_stop_macro_f1", "상세 기록 확인"))
    print("기존 ver4 fold 1 단독 stop Macro F1:",
          training_report.get("baseline_single_fold_stop", {}).get("stop_macro_f1", "상세 기록 확인"))
    print("후보 선택 기준:", candidate_report.get("selection_scope", "Training stop threshold 교차평가"))
    for field, title in (
        ("candidate_standalone_crossfit_macro_f1", "새 후보 단독 calibration 교차평가"),
        ("baseline_crossfit_macro_f1", "기존 ver4 앙상블 calibration 교차평가"),
        ("candidate_crossfit_macro_f1", "혼합 후보 calibration 교차평가"),
        ("candidate_raw_crossfit_macro_f1", "확률 평균 후보"),
        ("candidate_logit_crossfit_macro_f1", "logit 평균 후보"),
        ("selected_name", "최종 선택"), ("selected_blend_space", "선택된 혼합 방식"),
        ("selection_gain", "calibration F1 개선 폭"), ("minimum_gain", "최소 개선 폭 기준"),
        ("candidate_improved_folds", "개선된 calibration fold 수"),
        ("minimum_improved_folds", "최소 개선 fold 수 기준"),
        ("candidate_fold_gains", "calibration fold별 개선 폭"), ("fallback_reason", "선택 조건 판단"),
    ):
        if field in training_report:
            print(title + ":", training_report[field])
    if training_report.get("selected_name") == "ver4_baseline":
        print("Training 내부 채택 기준을 충족하지 못해 최종 출력에는 기존 구성을 유지합니다.")
    selected = training_report.get("calibrator", {})
    labels = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
    print("증상 | 기존 신경망 alpha | 새 후보 beta | 임계값")
    for label, alpha, beta, threshold in zip(labels, selected.get("neural_weights", []),
                                             selected.get("candidate_weights", []), selected.get("thresholds", [])):
        print(f"{label} | {alpha:.2f} | {beta:.2f} | {threshold:.9f}")
    print("계산에는 저장된 보정값의 전체 정밀도를 사용합니다. 전체 기록:", REPORT_PATH)
    """)
    add("markdown", "ver7-evaluation-note", """
    ## 8. 확정된 최종 구성과 후보 단독의 Validation 전체 평가

    기본 서울 Validation **3,640개 전체**를 사용합니다. 예측 CSV를 확정한 뒤 symptom 정답으로 평가합니다.
    `mission3.csv`는 Training에서 선택한 최종 구성, `candidate_only.csv`는 새 encoder 단독 진단입니다.
    후보 단독의 임계값도 Training calibration에서 미리 고정합니다. 여기의 두 점수는 모델 선택에 되먹이지 않습니다.
    진단을 켜면 최종 출력에서 후보를 사용하지 않는 경우에도 후보의 추가 추론 비용이 필요합니다.
    입력·가중치·실행 조건이 같은 원시 예측은 공용 캐시를 재사용합니다.

    이전 전체 Validation 기록: **ver4 0.656648948 / ver5 0.656717758 / ver6 0.656669718**.
    이전 실행과 현재 GPU·batch 등에 따른 수치 차이가 있을 수 있습니다. 아래 차이는 실행 결과 비교용입니다.
    """)
    add("code", "ver7-evaluate", """
    PREDICTION_PATH = None
    CANDIDATE_PREDICTION_PATH = None
    evaluation_started = time.monotonic()
    run_stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    if RUN_VALIDATION and VALID_LABEL_DIR is not None:
        prediction_dir = OUTPUT_DIR / "predictions" / run_stamp
        PREDICTION_PATH = prediction_dir / "mission3.csv"
        if RUN_CANDIDATE_DIAGNOSTIC:
            CANDIDATE_PREDICTION_PATH = prediction_dir / "candidate_only.csv"
        run_python([
            runtime_dir / "inference.py", "--audio_dir", "", "--label_dir", VALID_LABEL_DIR,
            "--ckpt_path", CHECKPOINT_PATH, "--output", PREDICTION_PATH,
            *(["--base_dir", BASE_DIR] if not TRAIN_CONFIG.get("bundle_base", False) else []),
            "--batch_size", INFERENCE_BATCH_SIZE, "--cache_dir", INFERENCE_CACHE_DIR, "--evaluate",
            *(["--candidate_output", CANDIDATE_PREDICTION_PATH] if CANDIDATE_PREDICTION_PATH else []),
        ])
        for title, name in [("최종 선택 구성", "validation_metrics.json"),
                            *([("새 후보 단독 (진단용)", "candidate_validation_metrics.json")]
                              if RUN_CANDIDATE_DIAGNOSTIC else [])]:
            metrics_path = prediction_dir / name
            if not metrics_path.is_file():
                print(title + ": CSV 생성 완료. 정답이 없는 입력이면 점수를 생성하지 않습니다.")
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"{title} — 현재 입력 {metrics['samples']:,}개 Macro F1: {metrics['macro_f1']:.9f}")
            for version, reference in (("ver4", 0.6566489480388843), ("ver5", 0.6567177583831921),
                                       ("ver6", 0.6566697183939796)):
                difference = metrics["macro_f1"] - reference
                print(f"{version} 이전 기록 대비: {difference:+.9f} ({difference * 100:+.5f}%p)")
            print("증상 | Precision | Recall | F1 | 실제 양성 개수")
            for symptom, values in metrics["per_symptom"].items():
                print(f"{symptom} | {values['precision']:.6f} | {values['recall']:.6f} | "
                      f"{values['f1']:.6f} | {values['support']}")
        print("Validation 진단은 저장된 모델·혼합·임계값을 변경하지 않습니다.")
    else:
        print("학습·모델 저장 완료. RUN_VALIDATION=False이므로 최종 평가를 생략했습니다.")
    EVALUATION_MINUTES = (time.monotonic() - evaluation_started) / 60
    timing = {
        "preparation_minutes": PREPARATION_MINUTES,
        "pretrained_preparation_minutes": PRETRAINED_MINUTES,
        "training_and_calibration_minutes": TRAINING_MINUTES,
        "inference_and_evaluation_minutes": EVALUATION_MINUTES,
        "output_dir": str(OUTPUT_DIR),
        "prediction_csv": str(PREDICTION_PATH) if PREDICTION_PATH else None,
        "candidate_prediction_csv": str(CANDIDATE_PREDICTION_PATH) if CANDIDATE_PREDICTION_PATH else None,
        "note": "벽시계 시간이며 CU 측정값이 아닙니다. 패키지 설치 시간은 제외합니다.",
    }
    timing_path = OUTPUT_DIR / ("notebook_runtime_" + run_stamp + ".json")
    timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(f"최종 추론·평가: {EVALUATION_MINUTES:.1f}분")
    print("새 ver7 결과:", OUTPUT_DIR)
    print("최종 모델 구성:", CHECKPOINT_PATH)
    print("최종 예측 CSV:", PREDICTION_PATH)
    print("후보 단독 진단 CSV:", CANDIDATE_PREDICTION_PATH)
    print("실행 시간 기록:", timing_path)
    """)
    add("markdown", "ver7-finish", """
    ## 결과 보관

    **내 드라이브/DCC_AIdeas/results/mission3/ver7**에 다음 파일이 저장됩니다.

    - `mission3_ver7_model.ckpt`: 확정된 모델 구성·혼합·임계값.
    - `candidate_best.ckpt`: 학습한 KoELECTRA와 9-label attention 가중치.
    - `mission3_ver7_metrics.json`: 후보 학습 이력, 원래 fold 1 비교, 내부 교차평가와 채택 판단.
    - `predictions/실행시각/mission3.csv`, `validation_metrics.json`: 최종 예측과 전체·증상별 점수.
    - 같은 폴더의 `candidate_only.csv`, `candidate_validation_metrics.json`: 후보 단독 진단.
    - `notebook_runtime_실행시각.json`: 준비·모델 다운로드·학습·추론의 실제 시간. CU는 Colab에서 확인합니다.
    - `submission/`: 추론 소스, 패키지 목록, README·RESEARCH 문서.

    기본 설정은 기존 5개 모델을 다시 복사하지 않습니다. **원래 입력 ZIP과 ver7 결과를 함께 보관하세요.**
    Colab 로컬의 복원 파일과 공개 사전학습 파일은 새 세션에서 자동으로 다시 준비합니다.
    최종 결과에 포함된 새 후보 가중치로 추론할 때 공개 모델을 다시 다운로드하지 않습니다.

    ```bash
    python submission/inference.py --audio_dir "" --label_dir "JSON 폴더" \\
      --ckpt_path mission3_ver7_model.ckpt --base_dir "ZIP에서 복원한 ver4 폴더" --output mission3.csv
    ```

    `--audio_dir`는 대회 인터페이스 호환용입니다. 실행 로그가 남은 노트북도 보관하세요.
    """)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU",
                             "colab": {"name": "mission3_ver7.ipynb", "provenance": []}}}
    target = HERE.parent / "mission3_ver7.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
