"""Rebuild the standalone Colab notebook from the reviewed Python source files.

Run: python src/mission3_ver4/build_notebook.py
No third-party packages needed. The notebook embeds all runtime source code.
"""
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME_FILES = ("data.py", "calibration.py", "neural.py", "pipeline.py", "train.py", "inference.py", "evaluate.py")


def build():
    cells = []

    def add(kind, source):
        source = textwrap.dedent(source).strip() + "\n"
        cell = {"cell_type": kind, "id": f"ver4-{len(cells):02d}", "metadata": {}, "source": source.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
    # Mission 3 ver4 — 전체 대화 + 보정된 앙상블

    **이 노트북 하나만 Colab에 업로드하고 GPU 런타임에서 위에서부터 실행하세요.**
    데이터는 `TL_서울_구급`, `VL_서울_구급` 폴더를 포함하도록 압축을 풉니다.
    학습 코드·추론 코드가 아래에 모두 포함되어 있어 다른 `.py` 업로드는 필요 없습니다.

    | 확인된 기존 결과 | Validation Macro F1 |
    |---|---:|
    | all_in_one_ver2 → results/mission3/ver1 | 0.600123 |
    | mission3_ver3 → results/mission3/ver3 | 0.626015 |

    ver4는 개선을 목표로 만든 새 실험입니다. **전체 GPU 학습 전에는 기존 버전보다 높은 점수를 보장할 수 없습니다.**
    기존 384토큰 제한에서 Training의 11.27%가 잘렸습니다. ver4는 512토큰과 겹치는 window로 끝까지 읽습니다.

    - `klue/roberta-large` 5개 fold 모델 + 문자/단어 TF-IDF Logistic Regression.
    - Training의 90%에서 모델 학습·epoch 선택, 나머지 10%에서 **실제 최종 ensemble**의 가중치·임계값 보정.
    - 보정 데이터 자체도 5분할해 보정 성능을 교차 평가합니다. Validation은 마지막 CSV 생성·평가에서만 사용합니다.
    - 모델 입력은 양측 발화의 `text`만 사용하며 0개 증상 출력도 허용합니다.

    A100/L4 같은 GPU 런타임과 충분한 Drive 공간을 준비하세요. 5개 large 가중치만 수 GB이며 학습 시간은 GPU에 따라 달라집니다.
    """)
    add("code", """
    # 1. 필요한 버전을 설치합니다. Colab에 이미 있는 호환 torch는 유지합니다.
    import subprocess
    import sys
    REQUIREMENTS = __REQUIREMENTS__
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS])
    """.replace("__REQUIREMENTS__", repr(HERE.joinpath("requirements.txt").read_text().splitlines())))
    add("code", """
    # 2. 저장 위치. 경로를 직접 지정하려면 None을 실제 폴더 문자열로 바꾸세요.
    import os
    import tempfile
    from pathlib import Path

    START_DIR = Path.cwd().resolve()
    TRAIN_LABEL_DIR = None
    VALID_LABEL_DIR = None
    try:
        from google.colab import drive
    except ImportError:
        IN_COLAB = False
    else:
        IN_COLAB = True
        drive.mount("/content/drive", force_remount=False)

    candidates = [START_DIR, START_DIR / "DCC_AIdeas", *list(START_DIR.parents)[:4]]
    PROJECT_ROOT = next((p for p in candidates if (p / "docs").is_dir() and (p / "src").is_dir()), START_DIR)
    OUTPUT_DIR = (Path("/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver4") if IN_COLAB
                  else PROJECT_ROOT / "results/mission3/ver4")
    CODE_DIR = (Path("/content/mission3_ver4_runtime") if IN_COLAB
                else Path(tempfile.mkdtemp(prefix="mission3_ver4_runtime_")))
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("결과 저장:", OUTPUT_DIR)
    print("실행 코드:", CODE_DIR)
    """)
    add("code", """
    # 3. /content를 먼저 찾고 없을 때 Drive를 찾습니다. Validation 없이 학습도 가능합니다.
    def locate_labels(name, explicit=None, required=True):
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(path)
            return path
        split = "Training" if name.startswith("TL_") else "Validation"
        short = "train" if name.startswith("TL_") else "validation"
        roots = ([Path("/content"), Path("/content/drive/MyDrive")] if IN_COLAB
                 else [PROJECT_ROOT, START_DIR])
        for root in roots:
            candidates = [root / "mission3_data" / short / name, root / name]
            for base in [root, root / "DATA", root / "DCC_AIdeas/DATA", root / "mission3_data"]:
                candidates.extend([base / "대학부 데이터" / split / "2.라벨링데이터" / name,
                                   base / split / "2.라벨링데이터" / name])
            for path in candidates:
                if path.is_dir():
                    return path.resolve()
            if IN_COLAB:
                matches = []
                for current, directories, _ in os.walk(root):
                    directories[:] = [d for d in directories if d not in
                                      {"drive", ".git", ".cache", ".config", "__pycache__", "node_modules", "sample_data"}]
                    if Path(current).name == name:
                        matches.append(Path(current))
                        directories[:] = []
                if len(matches) > 1:
                    raise ValueError(f"{name}가 여러 곳에 있습니다. 2번 셀에서 직접 지정하세요: {matches}")
                if matches:
                    return matches[0].resolve()
        if required:
            raise FileNotFoundError(f"{name} 폴더를 못 찾았습니다. 2번 셀의 TRAIN_LABEL_DIR를 지정하세요.")
        return None

    TRAIN_LABEL_DIR = locate_labels("TL_서울_구급", TRAIN_LABEL_DIR)
    VALID_LABEL_DIR = locate_labels("VL_서울_구급", VALID_LABEL_DIR, required=False)
    print("Training:", TRAIN_LABEL_DIR)
    print("Validation:", VALID_LABEL_DIR)
    """)
    add("markdown", """
    ## 4. 실행·제출 코드를 준비합니다

    아래 `%%writefile` 셀은 노트북에 포함된 코드를 임시 실행 폴더에 저장합니다.
    모델 학습은 뒤의 **6번 셀**에서 시작합니다. 코드 수정 후 노트북을 다시 만들 때는 저장소에서
    `python src/mission3_ver4/build_notebook.py`를 실행할 수 있습니다.
    """)
    add("code", """
    # writefile은 이 임시 폴더에만 기록합니다. 데이터/결과 경로는 절대경로로 보관했습니다.
    os.chdir(CODE_DIR)
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    """)
    for name in RUNTIME_FILES:
        add("code", f"%%writefile {name}\n" + (HERE / name).read_text(encoding="utf-8"))
    for name in ("requirements.txt", "README.md"):
        add("code", f"Path({name!r}).write_text({(HERE / name).read_text(encoding='utf-8')!r}, encoding='utf-8')")
    add("code", """
    # 5. 성능 중심 기본 설정. 먼저 기본값으로 한 번 실행해 비교하세요.
    import json
    import torch
    from pipeline import DEFAULT_CONFIG, train_pipeline, predict_pipeline, evaluate_csv
    from data import TARGET_SYMPTOMS, write_json

    CONFIG = {**DEFAULT_CONFIG,
              "model_name": "klue/roberta-large",
              "n_folds": 5,
              "max_epochs": 6,
              "patience": 2,
              "batch_size": 2,
              "gradient_accumulation_steps": 16,
              "window_batch_size": 2}
    if not torch.cuda.is_available():
        raise RuntimeError("전체 학습에는 GPU가 필요합니다. Colab 런타임 유형을 GPU로 바꾸세요.")
    print("GPU:", torch.cuda.get_device_name(0))
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    """)
    add("markdown", """
    ## 6. 학습 및 Training 내부 보정

    각 fold에서 학습 중 가장 좋은 **stop Macro AP**의 가중치를 보관하고 조기 종료합니다.
    fold 다섯 개를 평균낸 보정 집합 예측과 TF-IDF 예측으로 증상별 가중치·임계값을 정합니다.
    `calibration_fit_macro_f1`은 보정에 사용한 점수이고, `calibration_crossfit_macro_f1`은 보정 집합 내부 교차 평가입니다.
    둘 다 공식 Validation 점수가 아닙니다.

    런타임이 끊겼다면 같은 데이터·설정·코드로 다시 실행하면 완료된 fold를 재사용합니다.
    진행 중이던 fold는 처음부터 재학습합니다. 설정을 바꾼 실험은 2번 셀의 `OUTPUT_DIR`를 새 폴더로 바꾸세요.
    """)
    add("code", """
    CHECKPOINT_PATH, training_report = train_pipeline(TRAIN_LABEL_DIR, OUTPUT_DIR, CONFIG)
    print("Training 내부 보정 교차평가:", training_report["calibration_crossfit_macro_f1"])
    """)
    add("markdown", """
    ## 7. 고정된 모델로 Validation CSV 생성

    학습·보정이 모두 끝난 체크포인트를 다시 읽습니다. 이 단계에서는 Validation의 `text`만 추론 입력으로 사용합니다.
    """)
    add("code", """
    PREDICTION_PATH = OUTPUT_DIR / "mission3.csv"
    if VALID_LABEL_DIR is not None:
        _, _, validation_predictions = predict_pipeline(CHECKPOINT_PATH, VALID_LABEL_DIR, PREDICTION_PATH)
    else:
        print("Validation 폴더가 없어 CSV 생성을 생략했습니다. 경로 지정 후 이 셀부터 실행하세요.")
    """)
    add("markdown", """
    ## 8. 마지막 평가와 제출 파일 확인

    CSV를 먼저 저장한 다음에만 Validation 정답을 읽어 평가합니다. 이 점수에 맞추어 임계값을 다시 조정하지 않습니다.
    학습 로그가 남은 이 노트북도 저장하세요. PDF는 별도 결과 정리 PPT도 요구합니다.
    """)
    add("code", """
    if VALID_LABEL_DIR is not None:
        validation_report = evaluate_csv(VALID_LABEL_DIR, PREDICTION_PATH)
        write_json(OUTPUT_DIR / "validation_metrics.json", validation_report)
        metrics_path = OUTPUT_DIR / "mission3_ver4_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["validation_macro_f1"] = validation_report["macro_f1"]
        metrics["validation"] = validation_report
        write_json(metrics_path, metrics)
        print(f"ver1: 0.600123 | ver3: 0.626015 | ver4: {validation_report['macro_f1']:.6f}")
        for symptom in TARGET_SYMPTOMS:
            print(symptom, validation_report["per_symptom"][symptom])

    print("결과 폴더 전체를 보관하세요:", OUTPUT_DIR)
    print("manifest:", CHECKPOINT_PATH)
    print("실행 코드:", OUTPUT_DIR / "submission")
    """)
    add("markdown", """
    최종 추론 명령은 다음과 같습니다. `.ckpt` manifest 하나만 복사하지 말고
    **fold_*.ckpt, tokenizer/, sparse.joblib**를 같은 결과 폴더에 함께 보관해야 합니다.

    ```bash
    python submission/inference.py --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \\
      --ckpt_path mission3_ver4_model.ckpt --output mission3.csv
    ```

    모델 구현 근거: [KLUE RoBERTa](https://huggingface.co/klue/roberta-large),
    [tokenizer overflow](https://huggingface.co/docs/transformers/en/main_classes/tokenizer),
    [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
    실제 점수 향상 여부는 위 학습·최종 평가 출력으로 확인합니다.
    """)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"},
                             "colab": {"name": "mission3_ver4.ipynb", "provenance": []}, "accelerator": "GPU"}}
    target = HERE.parent / "mission3_ver4.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
