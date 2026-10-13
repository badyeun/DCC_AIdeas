"""Embed reviewed runtime sources in four independent Colab notebooks."""
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
VERSIONS = {
    "ver10": {"title": "TF-IDF + LinearSVC / ComplementNB", "backend": "sparse",
              "model_id": "", "revision": "", "label_attention": False, "contrastive_weight": 0.0},
    "ver11": {"title": "KLUE-RoBERTa + categorical heads", "backend": "neural",
              "model_id": "klue/roberta-base", "revision": "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da", "label_attention": False,
              "contrastive_weight": 0.0},
    "ver12": {"title": "KLUE-RoBERTa + label attention + supervised contrastive learning", "backend": "neural",
              "model_id": "klue/roberta-base", "revision": "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da", "label_attention": True,
              "contrastive_weight": 0.1},
    "ver13": {"title": "KcELECTRA + label attention + supervised contrastive learning", "backend": "neural",
              "model_id": "beomi/KcELECTRA-base", "revision": "4cf9eb1dfeec0a672afe73059882fc998a299ceb", "label_attention": True,
              "contrastive_weight": 0.1},
}


def default_settings(version):
    values = VERSIONS[version]
    return {"version": version, **{k: v for k, v in values.items() if k != "title"},
            "input_zip": "/content/drive/MyDrive/mission3_ver5_inputs.zip",
            "output_dir": f"/content/drive/MyDrive/DCC_AIdeas/results/mission3/{version}",
            "work_dir": "/content/mission3_categorical", "train_dir": "", "validation_dir": "",
            "run_validation": True, "seed": 42, "holdout_size": 0.2,
            "sparse_candidates": ["svm_c1", "svm_c4", "complement_nb"],
            "char_features": 80000, "word_features": 20000,
            "max_length": 512, "overlap": 64, "batch_size": 8 if values["contrastive_weight"] else 4,
            "encoder_window_batch": 4, "gradient_accumulation": 2 if values["contrastive_weight"] else 4,
            "max_epochs": 4, "patience": 2, "learning_rate": 2e-5,
            "head_learning_rate": 1e-4, "refit_full_training": True,
            "allow_cpu": False, "log_every": 250}


def build(version):
    values = VERSIONS[version]
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in
               ["__init__.py", "data.py", "bootstrap.py", "neural.py", "runtime.py",
                "inference.py", "run_notebook.py", "requirements.txt", "README.md"]}
    cells = []

    def add(kind, identifier, source):
        cell = {"cell_type": kind, "id": f"{version}-{identifier}", "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", f"""
    # Mission 3 {version} — {values['title']}

    **이 노트북 한 개와 기존 `mission3_ver5_inputs.zip`으로 실행합니다.**
    공용 계정의 ZIP 바로가기는 그대로 사용합니다. 기존 ver4 가중치와 보정값은 가져오지 않습니다.
    Colab에서 {'CPU 또는 GPU' if values['backend'] == 'sparse' else 'GPU'} 런타임을 연결하고 **런타임 → 모두 실행**을 누르세요.
    Drive 연결 요청에는 ZIP 바로가기가 있는 **공용 계정**을 선택합니다.

    - 학습: 서울 Training의 전사 text만. symptom은 9개 정답으로만 사용합니다.
    - 예측: {'분류기의 `predict`' if values['backend'] == 'sparse' else '각 증상 2-class head의 `argmax`'}.
      임계값 검색·보정, 사후 score 이동, 기존 보정 모델 fallback이 없습니다.
    - 모델/epoch 선택: 중복 전사가 겹치지 않는 Training 내부 holdout.
    - 선택이 끝나면 전체 Training으로 다시 학습하고 공식 Validation을 마지막에 평가합니다.
    - 오심·구토를 포함한 증상별 오탐/미탐 CSV와 분포 분석을 자동 저장합니다.

    **F1 개선은 실행 결과로 확인해야 합니다.** 기존 0.6567 점수는 threshold 보정을 포함합니다.
    새 버전의 Training holdout 지표로 먼저 비교하세요. Validation 점수로 설정을 반복해서 고르지 마세요.
    {"" if values['backend'] == 'sparse' else 'GPU 모델은 holdout 선택 학습과 전체 재학습을 모두 수행하므로 수 시간이 걸릴 수 있습니다. 고정 3시간 제한은 없습니다.'}
    """)
    add("markdown", "hypothesis", {
        "ver10": """
        ## 가설과 비교 기준
        오심은 기존 결과에서 재현율뿐 아니라 정밀도가 특히 낮았습니다. 우선 학습된 분류 경계로
        판단하는 희소 모델을 공정한 기준으로 만듭니다. 문자 n-gram은 구어체·오탈자 표현을,
        단어 n-gram은 인접 표현을 보존합니다. C=1/4 LinearSVC와 ComplementNB를 Training holdout에서 비교합니다.
        ComplementNB는 불균형 텍스트 분류 비교 후보이며, 멘토님의 “co 모델”이라고 확정하지 않습니다.
        """,
        "ver11": """
        ## 가설과 비교 기준
        양성 가중치와 threshold 보정 없이 표준 2-class 교차엔트로피를 학습하는 RoBERTa 기준 모델입니다.
        전체 통화를 겹치는 512-token window로 읽어 마지막 문장을 버리지 않습니다.
        window 평균을 사용해 단일 증상 단어가 등장했다는 이유만으로 최대 점수가 채택되는 현상을 피합니다.
        [KLUE 공식 모델](https://huggingface.co/klue/roberta-base)
        """,
        "ver12": """
        ## 가설과 비교 기준
        ver11과 같은 encoder·분할·학습률에서 증상별 window attention과 대조학습을 함께 추가합니다.
        증상마다 관련 문맥에 가중치를 주고, Training 정답이 같은 증상 있음/없음 표현을 가까이 학습합니다.
        같은 통화를 dropout으로 두 번 읽으며 텍스트나 정답을 새로 만들지 않습니다.
        두 요소를 함께 바꾸므로 이 비교 하나로 각각의 기여를 확정할 수 없습니다.
        오심·구토-only 오탐과 질문/부정 문맥을 Training 오류 CSV에서 확인하세요.
        [Supervised Contrastive Learning 논문](https://arxiv.org/abs/2004.11362)
        """,
        "ver13": """
        ## 가설과 비교 기준
        ver12와 같은 구조·분할·손실을 유지하고 encoder를 KcELECTRA v2022로 교체합니다.
        공개 모델 작성자가 구어체·오탈자 텍스트를 목표로 설명한 모델입니다.
        응급 통화에서 더 높은 F1을 얻을지는 아직 확인되지 않았습니다.
        기존 ver7의 KoELECTRA와는 별개 모델입니다.
        [KcELECTRA 공식 모델 카드](https://huggingface.co/beomi/KcELECTRA-base)
        """,
    }[version])
    add("code", "settings", """
    # @title 실행 설정 (기본값이면 기존 ver9 ZIP 바로가기로 실행)
    SETTINGS = __SETTINGS__
    # 직접 JSON 폴더를 쓸 때만 train_dir / validation_dir을 지정하세요.
    # max_epochs를 줄이면 모델/epoch 선택 학습 시간은 줄지만 성능이 달라질 수 있습니다.
    # refit_full_training=False이면 전체 재학습을 생략하고 80% Training 모델을 사용합니다.
    # GPU 메모리 부족 시 batch_size를 절반으로 줄이고 gradient_accumulation을 두 배로 늘립니다.
    # 대조학습은 실제 batch 안에서만 비교하므로 accumulation은 비교 표본 수를 늘리지 않습니다.
    """.replace("__SETTINGS__", repr(default_settings(version))))
    add("code", "drive", """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    """)
    add("code", "dependencies", """
    # @title 기존 프로젝트와 같은 버전의 패키지 준비
    import subprocess
    import sys
    packages = ["numpy==2.1.3", "scikit-learn==1.6.1", "scipy>=1.11,<2",
                "iterative-stratification==0.1.9", "joblib>=1.3,<2"]
    if SETTINGS["backend"] == "neural":
        packages += ["torch>=2.6,<3", "transformers==4.57.1",
                     "sentencepiece>=0.2,<0.3", "protobuf>=4.25,<7"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])
    """)
    add("code", "restore", """
    # @title 실행 코드 복원 (별도 .py 업로드 불필요)
    import tempfile
    from pathlib import Path
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_categorical_"))
    for name, source in RUNTIME_SOURCES.items():
        (runtime_dir / name).write_text(source, encoding="utf-8")
    print("실행 코드 준비 완료")
    """.replace("__SOURCES__", repr(sources)))
    add("code", "run", """
    # @title Training 분석 → 내부 비교 → 최종 학습 → CSV → 마지막 평가
    import json
    settings_file = runtime_dir / "notebook_settings.json"
    result_file = runtime_dir / "notebook_result.json"
    settings_file.write_text(json.dumps(SETTINGS, ensure_ascii=False), encoding="utf-8")
    # 설치 전 커널에 로드된 라이브러리와 섞이지 않게 새 실행 환경을 사용합니다.
    subprocess.check_call([sys.executable, "-u", str(runtime_dir / "run_notebook.py"),
                           "--settings", str(settings_file), "--result_file", str(result_file)])
    result_dir = Path(json.loads(result_file.read_text(encoding="utf-8")))
    """)
    add("code", "results", """
    import json
    for filename in ["training_metrics.json", "validation_metrics.json", "runtime.json"]:
        path = result_dir / filename
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            if "macro_f1" in report:
                print(filename, "Macro F1 =", round(report["macro_f1"], 6))
            else:
                print("소요 시간:", round(report["seconds"] / 60, 1), "분")
    print("결과 폴더:", result_dir)
    print("예측 CSV:", result_dir / "mission3.csv")
    print("Training 오탐·미탐:", result_dir / "training_errors")
    """)
    add("markdown", "submission", f"""
    ## 결과와 제출
    각 실행은 `{version}/실행시각_고유번호`에 저장되어 기존 결과를 덮어쓰지 않습니다.
    `.ckpt`는 manifest이므로 **같은 실행 폴더 전체**의 가중치와 assets도 함께 보관하세요.
    실행 로그가 포함된 이 노트북을 저장합니다. 제출 추론 명령:

    ```bash
    python submission/inference.py --audio_dir /path/audio --label_dir /path/json --ckpt_path mission3_{version}_model.ckpt --output mission3.csv
    ```

    추론은 다운로드가 필요 없는 저장된 모델로 실행합니다. CSV 헤더는 `label file name,symptom`입니다.
    `training_errors` CSV는 실제 통화 전사를 포함하므로 모델 오류 검토에만 사용하세요.
    질문·부정 표현 집계와 검토 표의 후보 표시는 라벨 오류의 확정 근거가 아닙니다.
    문제에서 요구하는 결과 PPT는 실제 학습·평가를 마친 뒤 작성해야 합니다.
    """)
    notebook = {"cells": cells, "metadata": {
        "colab": {"name": f"mission3_{version}.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    if values["backend"] == "neural":
        notebook["metadata"]["accelerator"] = "GPU"
    target = HERE.parent / f"mission3_{version}.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    for version in VERSIONS:
        print(build(version))
