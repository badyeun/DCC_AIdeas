"""Build a self-contained Colab notebook from the exact ver9 runtime sources."""
import json
from pathlib import Path
import textwrap
HERE = Path(__file__).resolve().parent


def build():
    cells = []
    def add(kind, identifier, source):
        cell = {"cell_type": kind, "id": identifier, "metadata": {},
                "source": (textwrap.dedent(source).strip() + "\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", "ver9-intro", """
    # Mission 3 ver9 — 한국어 DeBERTa-v3 + NB-log-count LR

    **ver8에서 쓰던 `mission3_ver5_inputs.zip`과 이 노트북 한 개로 실행합니다.**

    1. 공유 ZIP에서 **정리 → 바로가기 추가 → 내 드라이브**를 선택합니다.
    2. 내 드라이브 바로 아래에 `mission3_ver5_inputs.zip` 바로가기를 둡니다.
    3. 이 노트북을 Colab에서 열고 **GPU 런타임 → 모두 실행**을 선택합니다.
    4. 바로가기를 만든 계정으로 Drive를 연결합니다.

    결과: `내 드라이브/DCC_AIdeas/results/mission3/ver9`.
    새 모델: [한국어 DeBERTa-v3-base](https://huggingface.co/team-lucid/deberta-v3-base-korean),
    [원논문](https://arxiv.org/abs/2111.09543).
    문자·단어 TF-IDF를 [NB 로그비율](https://aclanthology.org/P12-2018/)로 재가중한 Logistic Regression도 추가합니다.

    - 기존 ver4의 5개 모델은 비교·최종 혼합용으로 재사용합니다.
    - 새로운 encoder 전체를 최대 3 epoch 학습합니다. 시간 여유가 충분하면 두 번째 fold도 학습합니다.
    - 새 후보는 `0.75 × DeBERTa 평균 + 0.25 × NB-LR`로 고정합니다.
    - Training 내부 교차평가에서 `+0.002 F1`, 최소 3개 fold 개선을 모두 만족할 때 후보 혼합을 채택합니다.
    - 공식 Validation은 선택을 끝낸 뒤 평가에만 사용합니다. 실패하면 원래 ver4 보정값을 그대로 유지합니다.

    **설정 셀부터 모든 자동 단계가 하나의 3시간 예산을 공유합니다.** 설치·복원·다운로드도 포함하며,
    최종 추론에 25분을 남깁니다. GPU·네트워크·Drive 속도에 따라 학습 fold/epoch 수가 줄거나 평가가 미완료될 수 있습니다.
    시간 초과 시 자식 프로세스를 종료하고 완료된 파일을 보존합니다. 연결 대기·수동 셀 간 대기 역시 시작 후 예산에서 차감됩니다.
    최종 CSV를 먼저 저장하고 남은 시간에 후보 단독 진단을 수행합니다.
    기본 결과 묶음은 기존 ver4를 포함하므로 원래 결과와 별도로 Drive **약 10~15 GB 이상** 공간을 확보하세요.

    **실제 ver9 F1과 GPU 소요 시간은 실행 전에는 알 수 없습니다.** 마지막 셀에서 측정 결과와 선택 사유를 확인하세요.
    """)
    add("code", "ver9-settings", """
    import time
    RUN_STARTED = time.time()  # 이후 셀마다 초기화하지 않는 공통 시작 시각
    INPUT_ZIP = "/content/drive/MyDrive/mission3_ver5_inputs.zip"
    OUTPUT_DIR = "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver9"
    RUN_VALIDATION = True
    RUN_CANDIDATE_DIAGNOSTIC = True
    TRAIN_CONFIG = {
        "max_models": 2, "max_epochs": 3, "patience": 2,
        "learning_rate": 2e-5, "head_learning_rate": 1e-4,
        "final_reserve_seconds": 1500, "calibration_reserve_seconds": 600,
        "min_calibration_gain": 0.002, "min_calibration_improved_folds": 3,
        "keep_completed_resume_states": False,
    }
    INFERENCE_BATCH_SIZE = 4
    # 다른 실험 설정으로 바꾸면 OUTPUT_DIR도 새 폴더로 바꾸세요.
    """)
    add("code", "ver9-drive", """
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False, timeout_ms=120000)
    print(f"설정 이후 {(time.time() - RUN_STARTED) / 60:.1f}분 경과")
    """)
    names = ["__init__.py", "assets.py", "budget.py", "bootstrap.py", "calibration.py", "data.py",
             "model.py", "neural.py", "pipeline.py", "prediction_cache.py", "pretrained.py", "selection.py",
             "sparse.py", "train.py", "inference.py", "run.py", "requirements.txt", "README.md", "RESEARCH.md"]
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in names}
    for path in sorted((HERE / "ver4_runtime").iterdir()):
        if path.is_file():
            sources["ver4_runtime/" + path.name] = path.read_text(encoding="utf-8")
    add("code", "ver9-runtime", """
    # @title 노트북에 포함된 실행 코드 복원 (별도 소스 업로드 불필요)
    import importlib.util
    import json
    import sys
    import tempfile
    from pathlib import Path
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver9_runtime_"))
    for name, source in RUNTIME_SOURCES.items():
        destination = runtime_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
    # 패키지 이름을 고유하게 만들어 이전 ver8 또는 재실행 import와 섞이지 않습니다.
    package_name = runtime_dir.name
    spec = importlib.util.spec_from_file_location(package_name, runtime_dir / "__init__.py",
                                                  submodule_search_locations=[str(runtime_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    import importlib
    supervisor = importlib.import_module(package_name + ".run")
    SETTINGS = {
        "input_zip": INPUT_ZIP, "output_dir": OUTPUT_DIR,
        "run_validation": RUN_VALIDATION,
        "run_candidate_diagnostic": RUN_CANDIDATE_DIAGNOSTIC,
        "train_config": TRAIN_CONFIG, "inference_batch_size": INFERENCE_BATCH_SIZE,
    }
    print("실행 코드 준비 완료:", runtime_dir)
    """.replace("__SOURCES__", repr(sources)))
    add("code", "ver9-run", """
    # 설치부터 마지막 평가까지 단 한 번 호출합니다. 각 단계는 남은 시간만 사용합니다.
    result_dir = supervisor.run(SETTINGS, start_epoch=RUN_STARTED)
    """)
    add("code", "ver9-results", """
    # 이미 확정된 결과를 표시하며 이 값으로 모델을 재선택하지 않습니다.
    result_dir = Path(OUTPUT_DIR)
    for name in ("training_metrics.json", "validation_metrics.json", "candidate_validation_metrics.json", "runtime.json"):
        value = supervisor.LAST_REPORT.get(name)
        if value is None:
            print(name, "— 미생성 (시간 부족/실행 로그 확인)")
            continue
        if "macro_f1" in value:
            print(f"{name}: Macro F1 = {value['macro_f1']:.6f}")
            for label, metrics in value.get("per_symptom", {}).items():
                print(f"  {label}: F1={metrics['f1']:.4f}")
        elif name == "training_metrics.json":
            print("최종 선택:", value["selected_name"])
            print("채택/미채택 사유:", value.get("calibration", {}).get("fallback_reason", value.get("reason")))
            print("학습된 후보:", value.get("candidate_models", []))
        else:
            print(f"전체 경과: {value['elapsed_seconds'] / 60:.1f}분 / 180분")
            print("단계별 분:", {k: round(v / 60, 2) for k, v in value['stages_seconds'].items()})
            print("중단 단계:", value["failures"])
    print("제출 실행 코드:", result_dir / "submission")
    print("최종 manifest:", result_dir / "mission3_ver9_model.ckpt")
    print("최종 CSV:", result_dir / "inference.csv")
    """)
    add("markdown", "ver9-submission", """
    ## 결과 보관·제출

    실행을 마친 노트북을 출력과 함께 저장하세요. `ver9` 결과 폴더 전체에 실제 `.ckpt` 가중치와 tokenizer가 있습니다.
    `mission3_ver9_model.ckpt`만 복사하면 추론할 수 없습니다.
    제출 실행 코드는 `ver9/submission/`에 있습니다.

    ```bash
    python submission/inference.py --audio_dir /path/audio --label_dir /path/json --ckpt_path mission3_ver9_model.ckpt --output inference.csv
    ```

    CSV 헤더는 `label file name,symptom`입니다. 원래 ver4 폴더의 절대 경로는 필요 없습니다.
    상세 모델 설명·논문·문제 조건은 `submission/README.md`, `submission/RESEARCH.md`를 참고하세요.
    문제에서 요구하는 실행 로그가 포함된 학습 노트북과 결과 PPT는 실제 결과를 확인한 뒤 함께 준비하세요.
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "mission3_ver9.ipynb"},
                  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    target = HERE.parent / "mission3_ver9.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)
    return notebook


if __name__ == "__main__":
    build()
