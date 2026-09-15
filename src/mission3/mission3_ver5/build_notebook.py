"""Build the standalone Colab notebook; includes code, not multi-GB weights."""
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

    add("markdown", "ver5-intro", """
    # Mission 3 ver5 — 기존 모델 재사용 + 증상별 단어 attention

    **GPU 런타임을 선택하고 위에서부터 모두 실행하세요.** Drive 연결 창이 나오면 연결합니다.
    실행 코드가 포함되어 있으므로 별도 `.py` 업로드는 필요 없습니다.
    **기존 ver4 결과 폴더 전체와 서울 Training JSON은 Drive에 있어야 합니다.**

    ver4의 최적 모델 5개와 보정용 예측을 재사용하고, 구조를 개선한 모델 **1개만 추가 학습**합니다.
    새 모델은 fold 1 최적 가중치에서 시작하며 **encoder 고정 1 epoch + 전체 미세조정 최대 3 epoch**를 수행합니다.
    다섯 모델을 처음부터 다시 학습하거나 여러 학습률을 장시간 탐색하지 않습니다.

    이전 A100 학습 기록은 22 epoch, 학습+stop 평가 합 630.28분입니다.
    ver5 전체 실행은 **약 2~4시간을 계획용으로 예상**하지만 아직 실측이 아니며,
    GPU·데이터 읽기·약 4GB 재개 파일 저장·기본 모델 복사·최종 추론에 따라 달라집니다.
    로그의 `train_ETA_minutes`는 현재 epoch의 학습 잔여 시간이며 평가·저장 시간은 제외합니다.
    기존 ver4 외에 **약 12GiB 이상의 추가 저장 공간**을 준비하세요.

    성능은 실행 후 확인해야 합니다. Training 내부 교차평가에서 새 구성이 더 좋지 않으면
    **기존 ver4 보정 구성을 선택**합니다. 이 선택이 실제 Test 성능의 향상을 보장하지는 않습니다.
    """)
    add("markdown", "ver5-research", """
    ## 논문 전체 검토와 적용

    `mission3_ver5`의 PDF 3개는 **서로 다른 논문 2편**입니다.
    `1-s2.0-S2405844022009719-main.pdf`와
    `Automatic-symptoms-identification-from-a-massive-volume-of-unstruct_2022_Hel.pdf`는
    12페이지 텍스트와 삽입 이미지가 같은 사본입니다. `2403.06734v1.pdf`는 별도의 13페이지 논문입니다.

    | 연구 | 논문에서 확인한 내용 | ver5에 적용 |
    |---|---|---|
    | [Faris et al., Heliyon 2022](https://doi.org/10.1016/j.heliyon.2022.e09683), pp.4,6–8 | 동결한 AraBERT 특징과 작은 분류기로 아랍어 증상 추천 | 한국어 ver4 encoder를 재사용하고 새 head를 먼저 동결 encoder 위에서 학습 |
    | [CognitiveEMS, 2024 v1](https://arxiv.org/abs/2403.06734), pp.6–7,10 | label마다 서로 다른 token 증거를 보는 attention과 지식 융합 | 9개 증상별 token attention을 기존 global/window head에 추가 |

    이는 논문을 Mission 3에 맞춘 **설계 응용**이며 원 논문의 완전한 재현은 아닙니다.
    기존 ver4는 CLS/평균으로 window를 압축한 뒤 window 사이에서만 증상별 attention을 했습니다.
    ver5는 짧은 대화에서도 증상마다 중요한 token을 직접 선택하고, 전체 문맥도 함께 사용합니다.
    추가 출력은 0으로 초기화하여 기존 모델에서 출발합니다.

    영어 TinyClinicalBERT·외부 의료 지식 그래프·합성 데이터/API, 상위 5개 강제 출력,
    희소 증상 제거, 부정 표현/숫자 삭제, 증상 자동 재라벨링은 적용하지 않습니다.
    두 논문의 대상 언어·라벨 수·평가 지표가 달라 그 점수를 우리 Macro F1과 직접 비교할 수 없습니다.
    전체 검토 근거는 실행 폴더와 제출 코드 폴더에 포함된 `RESEARCH.md`에 남깁니다.
    """)
    add("code", "ver5-settings", """
    # 1. 기존 폴더 구조라면 수정하지 않고 모두 실행하세요.
    BASE_DIR = None          # ver4 전체 결과 폴더. 예: /content/drive/MyDrive/DCC_AIdeas/results/mission3/ver4
    TRAIN_LABEL_DIR = None   # TL_서울_구급 폴더 자동 탐색
    VALID_LABEL_DIR = None   # VL_서울_구급 자동 탐색. 다른 Test JSON 폴더를 직접 지정해도 됩니다.
    OUTPUT_DIR = None        # 기본값: 기존 ver4 폴더와 나란한 results/mission3/ver5
    RUN_VALIDATION = True    # 학습/보정 종료 후 예측. 정답이 있으면 평가도 수행합니다.

    TRAIN_CONFIG = {
        "warmup_epochs": 1,
        "finetune_epochs": 3,
        "patience": 2,
    }
    # 기본은 GPU 메모리에 따라 batch/window를 8/8, 4/4, 2/2로 선택하고 유효 batch 32를 유지합니다.
    # 수동 예: TRAIN_CONFIG.update(batch_size=2, window_batch_size=2, gradient_accumulation_steps=16)
    # 학습 설정을 바꾸는 새 실험은 OUTPUT_DIR를 새 폴더로 지정하세요.
    INFERENCE_BATCH_SIZE = 8
    """)
    add("code", "ver5-dependencies", """
    # 2. 저장 모델과 호환되는 패키지. 이미 호환되는 Colab torch는 유지합니다.
    import subprocess
    import sys

    REQUIREMENTS = __REQUIREMENTS__
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *REQUIREMENTS], check=True)
    print("환경 준비 완료. 학습/추론은 새 Python 프로세스에서 실행하므로 패키지 설치 후 커널 재시작이 필요하지 않습니다.")
    """.replace("__REQUIREMENTS__", repr((HERE / "requirements.txt").read_text().splitlines())))
    add("code", "ver5-paths", """
    # 3. Drive 연결과 경로 확인
    import json
    import os
    import tempfile
    import uuid
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    START_DIR = Path.cwd().resolve()
    try:
        from google.colab import drive
    except ImportError:
        IN_COLAB = False
    else:
        IN_COLAB = True
        drive.mount("/content/drive", force_remount=False)
    roots = ([Path("/content"), Path("/content/drive/MyDrive")] if IN_COLAB else [])
    roots += [START_DIR, *list(START_DIR.parents)[:6]]
    roots = list(dict.fromkeys(roots))

    def locate(explicit, candidates, valid, name, required=True):
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
            if not valid(path):
                raise FileNotFoundError(f"{name} 경로를 확인하세요: {path}")
            return path
        for path in dict.fromkeys(candidates):
            if valid(path):
                return path.resolve()
        if required:
            raise FileNotFoundError(f"{name}을 찾지 못했습니다. 첫 설정 셀에 실제 폴더 경로를 지정하세요.")
        return None

    base_candidates = []
    for root in roots:
        base_candidates.extend([root, root / "results/mission3/ver4", root / "DCC_AIdeas/results/mission3/ver4"])
    BASE_DIR = locate(BASE_DIR, base_candidates, lambda p: (p / "mission3_ver4_model.ckpt").is_file(), "BASE_DIR")
    for name in ("splits.json", "calibration_predictions.npz", "sparse.joblib", "tokenizer"):
        if not (BASE_DIR / name).exists():
            raise FileNotFoundError(f"기존 ver4 전체 결과 묶음이 필요합니다: {BASE_DIR / name}")

    def label_candidates(name, split):
        values = []
        for root in [*roots, *list(BASE_DIR.parents)[:4]]:
            short = "train" if split == "Training" else "validation"
            values.extend([root / name, root / "mission3_data" / short / name])
            for base in [root, root / "DATA", root / "data", root / "DCC_AIdeas/DATA", root / "DCC_AIdeas/data"]:
                for prefix in [base, base / "대학부 데이터"]:
                    values.append(prefix / split / "2.라벨링데이터" / name)
        return values

    has_json = lambda p: p.is_dir() and next(p.rglob("*.json"), None) is not None
    TRAIN_LABEL_DIR = locate(TRAIN_LABEL_DIR, label_candidates("TL_서울_구급", "Training"), has_json, "TRAIN_LABEL_DIR")
    VALID_LABEL_DIR = locate(VALID_LABEL_DIR, label_candidates("VL_서울_구급", "Validation"), has_json, "VALID_LABEL_DIR", required=False)
    OUTPUT_DIR = (BASE_DIR.parent / "ver5" if OUTPUT_DIR is None else Path(OUTPUT_DIR).expanduser()).resolve()
    if OUTPUT_DIR == BASE_DIR or OUTPUT_DIR.is_relative_to(BASE_DIR) or BASE_DIR.is_relative_to(OUTPUT_DIR):
        raise ValueError("OUTPUT_DIR는 기존 ver4와 별도 폴더여야 합니다.")
    print("기존 모델:", BASE_DIR)
    print("Training:", TRAIN_LABEL_DIR)
    print("최종 예측 입력:", VALID_LABEL_DIR)
    print("ver5 결과:", OUTPUT_DIR)
    """)

    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in
               ("__init__.py", "neural.py", "calibration.py", "pipeline.py", "train.py", "inference.py",
                "requirements.txt", "README.md", "RESEARCH.md")}
    for path in sorted((HERE / "ver4_runtime").iterdir()):
        if path.is_file():
            sources["ver4_runtime/" + path.name] = path.read_text(encoding="utf-8")
    add("code", "ver5-runtime", """
    # @title 4. 포함된 ver5 실행 코드 준비 (수정 없이 실행)
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver5_runtime_"))
    for name, source in RUNTIME_SOURCES.items():
        path = runtime_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    config_path = runtime_dir / "notebook_config.json"
    config_path.write_text(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_python(arguments):
        # Stream child output into the notebook; stopping the cell also stops the worker.
        command = [sys.executable, "-u", *map(str, arguments)]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
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

    print("실행 코드:", runtime_dir)
    """.replace("__SOURCES__", repr(sources)), cellView="form")
    add("markdown", "ver5-training-note", """
    ## 5. 추가 학습과 Training 내부 비교

    이 단계에는 **Training만 전달**합니다. 저장된 split·데이터·모델 파일의 일치를 먼저 검사합니다.
    첫 무결성 검사에는 약 6.4GiB의 기존 가중치를 읽는 시간이 필요합니다.
    `phase=warmup`은 encoder 고정 학습, `phase=finetune`은 전체 미세조정입니다.
    `epoch=0 baseline`은 학습 전 stop 평가입니다. 후보가 이를 넘지 못하면 epoch 0이 최적으로 남을 수 있습니다.

    혼합 비율 비교는 이미 계산한 확률로 수행하는 CPU 작업입니다. 기존 5개 모델을 다시 학습하지 않습니다.
    학습 후 결과 폴더를 독립적으로 제출할 수 있도록 `base/`에 기존 가중치를 복사합니다.

    중간에 끊겼다면 같은 설정으로 모두 실행하세요. 마지막 **완료 epoch**에서 복원하며 중단된 epoch는 재실행합니다.
    설정·데이터·코드가 바뀌면 다른 실험으로 취급하므로 새 `OUTPUT_DIR`가 필요합니다.
    """)
    add("code", "ver5-train", """
    run_python([
        runtime_dir / "train.py", "--train_label_dir", TRAIN_LABEL_DIR,
        "--base_dir", BASE_DIR, "--output_dir", OUTPUT_DIR,
        "--config_json", config_path, "--auto_batch",
    ])
    CHECKPOINT_PATH = OUTPUT_DIR / "mission3_ver5_model.ckpt"
    """)
    add("code", "ver5-report", """
    # 6. Training 내부 선택 결과 — 이 점수는 공식 Validation 점수가 아닙니다.
    training_report = json.loads((OUTPUT_DIR / "mission3_ver5_metrics.json").read_text(encoding="utf-8"))
    print("새 모델의 선택 epoch:", training_report["candidate_training"]["best_epoch"])
    print("기존 ver4 교차평가:", training_report["baseline_crossfit_macro_f1"])
    print("ver5 후보 교차평가:", training_report["candidate_crossfit_macro_f1"])
    print("최종 선택:", training_report["selected_name"])
    if training_report["selected_name"] == "ver4_baseline":
        print("추가 모델이 내부 비교에서 개선되지 않아 기존 ver4 보정을 사용합니다.")
    print("증상 | 기존 신경망 alpha | 새 후보 beta | 임계값")
    labels = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
    selected = training_report["calibrator"]
    for name, alpha, beta, threshold in zip(labels, selected["neural_weights"], selected["candidate_weights"], selected["thresholds"]):
        print(f"{name} | {alpha:.2f} | {beta:.2f} | {threshold:.9f}")
    print("계산에는 저장된 보정값의 전체 정밀도를 사용합니다.")
    """)
    add("markdown", "ver5-evaluation-note", """
    ## 7. 고정된 최종 모델의 예측과 평가

    여기부터 처음으로 공식 Validation/Test 입력을 사용합니다. JSON의 전사 text만 모델에 전달하고,
    CSV를 저장한 뒤에만 symptom 정답을 읽어 평가합니다. 이 결과로 모델·보정값을 다시 선택하지 않습니다.
    필요하다고 선택된 모델만 추론하며, 기존 5개와 새 후보가 모두 선택되면 최대 6개 모델의 추론이 필요합니다.

    GPU·batch에 따른 수치 차이가 있을 수 있습니다. 기존 전체 Validation Macro F1은 ver4 **0.656649**이며,
    새 결과와의 실제 차이는 아래 최종 평가에서 확인하세요.
    """)
    add("code", "ver5-evaluate", """
    PREDICTION_PATH = None
    if RUN_VALIDATION and VALID_LABEL_DIR is not None:
        # Preserve previous evaluation outputs when rerunning or changing the input folder.
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
        prediction_dir = OUTPUT_DIR / "predictions" / (stamp + "_" + uuid.uuid4().hex[:6])
        PREDICTION_PATH = prediction_dir / "mission3.csv"
        run_python([
            runtime_dir / "inference.py", "--audio_dir", "", "--label_dir", VALID_LABEL_DIR,
            "--ckpt_path", CHECKPOINT_PATH, "--output", PREDICTION_PATH,
            "--batch_size", INFERENCE_BATCH_SIZE, "--evaluate",
        ])
        metrics_path = prediction_dir / "validation_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"현재 입력 {metrics['samples']:,}개 Macro F1: {metrics['macro_f1']:.6f}")
    else:
        print("학습·모델 저장 완료. RUN_VALIDATION=False 또는 예측 폴더가 없어 최종 추론을 생략했습니다.")
    print("모델 묶음:", OUTPUT_DIR)
    print("manifest:", CHECKPOINT_PATH)
    print("CSV:", PREDICTION_PATH)
    print("제출용 CLI:", OUTPUT_DIR / "submission/inference.py")
    """)
    add("markdown", "ver5-finish", """
    ## 결과 보관

    - `mission3_ver5_model.ckpt`: 선택된 구성과 보정값.
    - `candidate_best.ckpt`: 새 모델의 최적 epoch 가중치.
    - `candidate_best_last.ckpt`: 완료 epoch 재개 상태. 추론에는 필요하지 않습니다.
    - `base/`: 재사용한 기존 모델 5개, tokenizer, TF-IDF 모델.
    - `mission3_ver5_metrics.json`: 후보 학습 이력, 교차평가 비교와 선택 이유.
    - `predictions/실행시각/mission3.csv`: 제출 형식의 예측. 정답이 있으면 같은 폴더에 `validation_metrics.json`.
    - `submission/`: 모델 코드, 추론 CLI, requirements, 논문 검토 문서.

    **학습 로그가 남은 이 노트북도 저장하세요.** 결과 폴더 전체를 보관하면 다른 환경에서도 아래 명령으로 실행할 수 있습니다.

    ```bash
    python submission/inference.py --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \\
      --ckpt_path mission3_ver5_model.ckpt --output mission3.csv
    ```

    추론만 다시 할 때는 학습이 필요하지 않습니다. 대회가 요구한 결과 정리 PPT는 별도 작성 대상입니다.
    """)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU",
                             "colab": {"name": "mission3_ver5.ipynb", "provenance": []}}}
    target = HERE.parent / "mission3_ver5.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
