"""Build the standalone ver6 Colab notebook from the checked-in runtime sources."""
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

    add("markdown", "ver6-intro", """
    # Mission 3 ver6 — 손실 함수와 증상별 token attention 비교

    **기존 `mission3_ver5_inputs.zip` 그대로 사용합니다. 새 ZIP이나 ver5 결과는 필요 없습니다.**

    1. Google Drive **공유 문서함**의 `mission3_ver5_inputs.zip`을 우클릭합니다.
    2. **정리 → 바로가기 추가 → 내 드라이브**를 선택합니다. 내 드라이브 바로 아래에 추가하세요.
    3. 이 노트북을 Colab에서 열고 GPU 런타임을 선택한 뒤 **모두 실행**합니다.
    4. Drive 연결은 바로가기를 만든 계정으로 합니다. 다른 위치라면 첫 셀의 `INPUT_ZIP`만 수정하세요.

    ZIP에는 기존 ver4 모델 5개, 보조 파일, Training 29,200개와 Validation 3,640개의 JSON이 있습니다.
    공유 문서함의 별도 `DATA` 폴더를 연결하거나 WAV 30GB를 읽을 필요는 없습니다.
    실행 코드가 이 노트북에 포함되어 있어 별도 `.py` 업로드도 필요 없습니다.

    **`/content/mission3_ver5_inputs/DCC_AIdeas/results/mission3/ver4`는 압축을 푼 입력 모델입니다.**
    **새 ver6 결과는 `내 드라이브/DCC_AIdeas/results/mission3/ver6`에 저장됩니다.**
    ver4라고 표시된 입력 폴더를 ver6으로 이름을 바꾸지 마세요.

    ver5에서는 추가 학습 뒤 점수가 낮아져 학습 전 가중치가 선택됐습니다.
    ver6은 동일한 encoder 특징을 한 번 계산하여 재사용하고, 작은 분류기 실험 6개를 비교합니다.
    A–C는 필요한 CLS·평균 요약을 메모리에 재사용하여 매 epoch마다 전체 token 특징을 읽는 비용도 줄입니다.
    선택된 분류기 하나만 encoder 상위 2개 층과 최대 1 epoch 미세조정합니다.
    기존 모델을 6번 복제하거나 encoder를 6번 처음부터 학습하지 않습니다.
    단, 원래 Training fit·stop 26,320개 대화의 token 특징을 한 번 추출하고 저장하는 비용은 필요합니다.
    별도 calibration 2,880개는 이 분류기 학습용 특징 캐시에서 제외합니다.

    특징 캐시는 Colab 로컬 저장소에 두어 Drive 용량을 절약합니다. 같은 런타임에서는 재사용하며,
    새 런타임에서는 ZIP을 다시 풉니다. 아직 완료하지 않은 분류기 실험이 있으면 특징도 다시 계산합니다.
    분류기 비교가 끝나 미세조정만 재개하거나 완료 결과를 읽을 때는 특징 추출을 생략합니다.
    실제 특징 캐시 필요 용량은 추출 전에 계산합니다.
    기존 모델 5개도 결과 폴더에 중복 저장하지 않습니다. **입력 ZIP과 바로가기는 계속 보관하세요.**

    목표는 Macro F1 개선이지만, **큰 성능 향상은 실행 전 보장할 수 없습니다.**
    Training 내부에서 충분한 개선이 확인되지 않으면 기존 ver4 보정을 유지합니다.
    공식 Validation은 최종 구성을 확정한 후 한 번 평가하며 모델 선택에 사용하지 않습니다.
    """)
    add("markdown", "ver6-experiments", """
    ## 실험 내용과 비교 원칙

    | 실험 | 분류기와 학습 목표 | 확인할 점 |
    |---|---|---|
    | A | 기존 ver4 분류기 + Asymmetric Loss | 쉬운 음성 예제의 영향 완화 |
    | B | 기존 ver4 분류기 + Distribution-Balanced Loss | Training fit의 빈도·동시 출현을 반영 |
    | C | 기존 ver4 분류기 + ASL·soft Macro F1 | 학습 목표를 공식 F1에 더 가깝게 구성 |
    | D | 직접적인 label-wise token attention + weighted BCE | 각 증상의 예측이 해당 token 표현에 직접 의존 |
    | E | D + 분류기 R-Drop | 같은 특징을 두 dropout 경로로 예측해 일관성 학습 |
    | F | E + 작은 label dependency 모듈 | fit 부분에서만 증상 간 관계 학습 |

    D–F는 기존 logit에 0으로 시작하는 작은 출력을 더하는 ver5 구조와 다릅니다.
    각 증상이 token을 직접 선택하고 window를 합쳐 최종 logit을 만듭니다.
    동결 encoder 위의 R-Drop은 작은 분류기에서만 두 번 계산합니다.
    최종 encoder 미세조정에는 기본적으로 두 번의 전체 순전파를 요구하는 R-Drop을 적용하지 않습니다.
    soft Macro F1 보조항도 작은 분류기 단계에만 사용하며, 작은 encoder microbatch의 통계에는 적용하지 않습니다.

    원래 ver4 fold 1의 **fit 21,032개**만 가중치·학습 통계 계산에 사용합니다.
    **stop 5,288개**에서는 대화 중복 그룹을 분리한 3-fold threshold 교차평가 Macro F1으로
    실험·epoch를 비교하고, 별도 **calibration 2,880개**에서는 5-fold 교차평가로 최종 혼합과 임계값을 비교합니다.
    이는 threshold를 평가할 대화의 정답으로 직접 맞추지 않도록 한 비교입니다.
    **5개 encoder를 각각 다시 학습해 얻는 5-fold 모델 OOF 결과는 아닙니다.**
    같은 stop을 여러 후보에 사용하므로 이 점수를 새로운 데이터의 성능으로 해석하지 않습니다.

    논문 원문을 검토한 적용 범위와 단순화는 포함된 `RESEARCH.md`에 기록합니다.
    [PLM-ICD](https://aclanthology.org/2022.clinicalnlp-1.2/),
    [LAAT](https://arxiv.org/abs/2007.06351),
    [ASL](https://arxiv.org/abs/2009.14119),
    [DB-Loss](https://arxiv.org/abs/2007.09654),
    [SigmoidF1](https://arxiv.org/abs/2108.10566),
    [R-Drop](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)의
    아이디어를 한국어 9개 증상 문제에 맞추어 적용하며 논문의 결과를 그대로 재현한다고 주장하지 않습니다.
    외부 데이터·의료 사전·LLM 생성 데이터는 추가하지 않습니다.
    샘플의 label에 따라 입력 전처리를 바꾸지 않습니다. 손실의 정답·학습 통계 사용은 입력 전처리와 구분합니다.
    """)
    add("code", "ver6-settings", """
    # 1. 공유 ZIP의 바로가기를 내 드라이브 바로 아래에 두었다면 그대로 실행하세요.
    INPUT_ZIP = "/content/drive/MyDrive/mission3_ver5_inputs.zip"
    BASE_DIR = None             # ZIP 대신 기존 ver4 폴더를 쓰려면 INPUT_ZIP=None과 함께 지정
    TRAIN_LABEL_DIR = None      # ZIP에 포함된 Training JSON 자동 사용
    VALID_LABEL_DIR = None      # ZIP에 포함된 Validation JSON 자동 사용
    OUTPUT_DIR = None           # 기본: 내 드라이브/DCC_AIdeas/results/mission3/ver6
    INFERENCE_CACHE_DIR = None  # 기본: 같은 results/mission3/prediction_cache (Drive)
    FROZEN_FEATURE_CACHE_DIR = None  # 기본: /content/mission3_ver6_features (Colab 로컬)
    RUN_VALIDATION = True       # 최종 모델을 확정한 뒤 Validation 전체를 평가

    TRAIN_CONFIG = {
        "bundle_base": False,  # 기존 5개 모델 중복 저장 생략. 원래 ZIP을 보관하세요.
        "profiles": ["A", "B", "C", "D", "E", "F"],
        "head_epochs": 4,
        "head_batch_size": 32,
        "head_learning_rate": 3e-4,
        "head_patience": 2,
        "stop_crossfit_splits": 3,
        "finetune_epochs": 1,
        "finetune_top_layers": 2,
        "learning_rate": 2e-6,
        "finetune_head_learning_rate": 5e-5,
        "finetune_rdrop": False,
        "min_calibration_gain": 0.002,
        "min_calibration_improved_folds": 3,
    }
    INFERENCE_BATCH_SIZE = 8
    # CUDA 메모리에 맞춰 encoder batch/window 크기를 자동 선택하고 유효 batch 32를 유지합니다.
    # 설정을 바꾸어 새 실험을 만들 때는 OUTPUT_DIR을 새 폴더로 지정하세요.
    """)
    add("code", "ver6-dependencies", """
    # 2. 기존 ver4 저장 모델과 호환되는 패키지 설치
    import subprocess
    import sys

    REQUIREMENTS = __REQUIREMENTS__
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *REQUIREMENTS], check=True)
    print("환경 준비 완료. 학습과 추론은 새 Python 프로세스에서 실행합니다.")
    """.replace("__REQUIREMENTS__", repr((HERE / "requirements.txt").read_text().splitlines())))
    add("code", "ver6-paths", """
    # 3. Drive 연결 → 기존 ZIP 검증·자동 압축 해제 → 입력과 새 결과 경로 확인
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
    roots = ([Path("/content"), Path("/content/drive/MyDrive")] if IN_COLAB else [])
    roots += [START_DIR, *list(START_DIR.parents)[:6]]
    roots = list(dict.fromkeys(roots))

    BOOTSTRAP_SOURCE = __BOOTSTRAP__
    bootstrap_namespace = {}
    exec(compile(BOOTSTRAP_SOURCE, "mission3_bootstrap.py", "exec"), bootstrap_namespace)
    bundle_path = Path(INPUT_ZIP).expanduser().resolve() if INPUT_ZIP is not None else None
    if bundle_path is None and BASE_DIR is None:
        candidates = [root / relative for root in roots
                      for relative in ("mission3_ver5_inputs.zip", "DCC_AIdeas/mission3_ver5_inputs.zip",
                                       "outputs/upload/mission3_ver5_inputs.zip",
                                       "DCC_AIdeas/outputs/upload/mission3_ver5_inputs.zip")]
        bundle_path = next((path for path in candidates if path.is_file()), None)
    BUNDLE_ROOT = None
    if bundle_path is not None:
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"ZIP 바로가기를 찾지 못했습니다: {bundle_path}\\n"
                "Drive 공유 문서함의 ZIP에서 '정리 → 바로가기 추가 → 내 드라이브'를 선택하세요. "
                "바로가기를 만든 계정으로 Drive를 연결하고 첫 셀의 INPUT_ZIP 경로를 확인하세요. "
                "바로가기 추가 전에 이미 연결했다면 drive.mount의 force_remount=True로 한 번 재연결하세요.")
        # 고정 경로를 유지해야 이전 실행의 체크포인트에서 재개할 수 있습니다.
        extract_dir = (Path("/content/mission3_ver5_inputs") if IN_COLAB
                       else Path(tempfile.gettempdir()) / "mission3_ver5_inputs")
        BUNDLE_ROOT = bootstrap_namespace["restore_input_bundle"](bundle_path, extract_dir)
        BASE_DIR = BUNDLE_ROOT / "results/mission3/ver4"

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
            raise FileNotFoundError(
                f"{name}을 찾지 못했습니다. 기존 mission3_ver5_inputs.zip의 바로가기를 내 드라이브에 추가하고 "
                "첫 셀의 INPUT_ZIP을 확인하세요.")
        return None

    base_candidates = [path for root in roots for path in
                       (root, root / "results/mission3/ver4", root / "DCC_AIdeas/results/mission3/ver4")]
    BASE_DIR = locate(BASE_DIR, base_candidates, lambda p: (p / "mission3_ver4_model.ckpt").is_file(), "BASE_DIR")
    for name in ("splits.json", "calibration_predictions.npz", "sparse.joblib", "tokenizer",
                 *[f"fold_{i}.ckpt" for i in range(1, 6)]):
        if not (BASE_DIR / name).exists():
            raise FileNotFoundError(f"기존 ver4 입력 파일이 빠졌습니다: {BASE_DIR / name}")

    def label_candidates(name, split):
        values = ([BUNDLE_ROOT / "DATA/대학부 데이터" / split / "2.라벨링데이터" / name]
                  if BUNDLE_ROOT is not None else [])
        for root in [*roots, *list(BASE_DIR.parents)[:4]]:
            values.append(root / name)
            for base in (root, root / "DATA", root / "DCC_AIdeas/DATA"):
                values.extend([base / split / "2.라벨링데이터" / name,
                               base / "대학부 데이터" / split / "2.라벨링데이터" / name])
        return values

    has_json = lambda p: p.is_dir() and next(p.rglob("*.json"), None) is not None
    TRAIN_LABEL_DIR = locate(TRAIN_LABEL_DIR, label_candidates("TL_서울_구급", "Training"), has_json, "TRAIN_LABEL_DIR")
    VALID_LABEL_DIR = locate(VALID_LABEL_DIR, label_candidates("VL_서울_구급", "Validation"),
                             has_json, "VALID_LABEL_DIR", required=RUN_VALIDATION)
    result_parent = Path("/content/drive/MyDrive/DCC_AIdeas/results/mission3") if IN_COLAB else BASE_DIR.parent
    if not IN_COLAB and BUNDLE_ROOT is not None:
        result_parent = START_DIR / "results/mission3"
    OUTPUT_DIR = (result_parent / "ver6" if OUTPUT_DIR is None else Path(OUTPUT_DIR).expanduser()).resolve()
    INFERENCE_CACHE_DIR = (result_parent / "prediction_cache" if INFERENCE_CACHE_DIR is None
                           else Path(INFERENCE_CACHE_DIR).expanduser()).resolve()
    feature_default = Path("/content/mission3_ver6_features") if IN_COLAB else Path(tempfile.gettempdir()) / "mission3_ver6_features"
    FROZEN_FEATURE_CACHE_DIR = (feature_default if FROZEN_FEATURE_CACHE_DIR is None
                                else Path(FROZEN_FEATURE_CACHE_DIR).expanduser()).resolve()
    if OUTPUT_DIR == BASE_DIR or OUTPUT_DIR.is_relative_to(BASE_DIR) or BASE_DIR.is_relative_to(OUTPUT_DIR):
        raise ValueError("OUTPUT_DIR는 기존 ver4 입력 폴더와 분리된 새 폴더여야 합니다.")
    TRAIN_CONFIG["feature_cache_dir"] = str(FROZEN_FEATURE_CACHE_DIR)
    PREPARATION_MINUTES = (time.monotonic() - preparation_started) / 60
    print("입력 ZIP:", bundle_path)
    print("기존 ver4 입력 모델:", BASE_DIR)
    print("Training JSON:", TRAIN_LABEL_DIR)
    print("최종 평가 JSON:", VALID_LABEL_DIR)
    print("새 ver6 결과 저장 위치:", OUTPUT_DIR)
    print("동결 특징 캐시 (로컬):", FROZEN_FEATURE_CACHE_DIR)
    print("공유 원시 예측 캐시 (Drive):", INFERENCE_CACHE_DIR)
    print(f"Drive 연결·입력 준비 시간: {PREPARATION_MINUTES:.1f}분")
    """.replace("__BOOTSTRAP__", repr((HERE / "bootstrap.py").read_text(encoding="utf-8"))))

    names = ("__init__.py", "bootstrap.py", "neural.py", "losses.py", "selection.py", "features.py",
             "calibration.py", "prediction_cache.py", "pipeline.py", "train.py", "inference.py",
             "requirements.txt", "README.md", "RESEARCH.md")
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in names}
    for path in sorted((HERE / "ver4_runtime").iterdir()):
        if path.is_file():
            sources["ver4_runtime/" + path.name] = path.read_text(encoding="utf-8")
    add("code", "ver6-runtime", """
    # @title 4. 노트북에 포함된 ver6 실행 코드 준비 (별도 소스 업로드 불필요)
    RUNTIME_SOURCES = __SOURCES__
    runtime_dir = Path(tempfile.mkdtemp(prefix="mission3_ver6_runtime_"))
    for name, source in RUNTIME_SOURCES.items():
        path = runtime_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    config_path = runtime_dir / "notebook_config.json"
    config_path.write_text(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_python(arguments):
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
    print("새 결과가 저장될 위치:", OUTPUT_DIR)
    """.replace("__SOURCES__", repr(sources)), cellView="form")
    add("markdown", "ver6-training-note", """
    ## 5. 특징 1회 추출 → 분류기 A–F 비교 → 선택 후보 미세조정 → 최종 보정

    이 단계는 **Training만 사용**합니다. 기존 split·입력 JSON·모델 가중치·보정 캐시를 먼저 검증합니다.
    초기 모델 검사에는 약 6.3GiB의 가중치를 읽는 시간이 들고, token 특징 추출에는 GPU 순전파가 필요합니다.
    첫 실행은 여섯 분류기의 학습 시간만으로 전체 소요 시간을 추정할 수 없습니다.
    이미 유효한 특징 캐시가 있는 같은 런타임에서는 추출을 생략합니다.

    각 분류기는 최대 4 epoch 학습하며 stop Macro F1로 비교합니다. 선택된 분류기는 상위 2개 층을
    최대 1 epoch 미세조정하고, 성능이 나빠지면 미세조정 전 가중치를 유지합니다.
    **ver6의 최종 `best_epoch=0`은 선택된 분류기를 유지하고 encoder 미세조정만 채택하지 않았다는 뜻입니다.**
    분류기 학습이 최종 가중치에 반영됐는지는 A–F 이력의 최적 epoch에서 따로 확인합니다.
    분류기는 이미 학습된 epoch가 선택될 수도, 초기 epoch 0이 유지될 수도 있습니다.
    최종 확률 평균·logit 평균과 증상별 임계값은 별도 Training calibration에서 CPU로 비교합니다.
    아주 작은 차이만 있는 후보가 선택되는 것을 줄이도록 최종 calibration 교차평가 F1이 기존 대비
    **0.002 이상 개선**되고 **5개 fold 중 3개 이상에서 개선**될 때만 새 구성을 채택합니다.
    이 기준도 공식 Validation 또는 Test 성능의 개선을 보장하지는 않습니다.
    정확한 기준과 판단 결과는 `mission3_ver6_metrics.json`에 저장됩니다.

    중단 후에는 동일한 설정·데이터·코드로 다시 실행하세요. 완료한 단계/epoch를 재사용하고,
    중단된 작업은 해당 단계의 저장 상태에 따라 다시 실행합니다. 새 런타임에서 분류기 비교가 남아 있으면
    특징 캐시를 다시 만듭니다. 분류기 비교가 끝났다면 미세조정 재개를 위해 특징을 다시 만들지 않습니다.
    **진행 중 학습을 재개할 때는 같은 GPU 종류·AMP 정밀도·관련 런타임 패키지 버전을 사용하세요.**
    서로 다른 정밀도로 계산한 stop 점수를 섞지 않도록 실행 조건을 검사합니다.
    최종 결과까지 완료된 실행은 다른 GPU에서도 재사용할 수 있으며, 조건이 다른 예측 캐시는 다시 계산합니다.
    **다른 설정으로 실험할 때는 첫 셀의 `OUTPUT_DIR`을 새 폴더로 바꾸세요.**
    """)
    add("code", "ver6-train", """
    training_started = time.monotonic()
    run_python([
        runtime_dir / "train.py", "--train_label_dir", TRAIN_LABEL_DIR,
        "--base_dir", BASE_DIR, "--output_dir", OUTPUT_DIR,
        "--config_json", config_path, "--auto_batch",
    ])
    TRAINING_MINUTES = (time.monotonic() - training_started) / 60
    CHECKPOINT_PATH = OUTPUT_DIR / "mission3_ver6_model.ckpt"
    print(f"특징 추출·분류기 비교·미세조정·보정·저장: {TRAINING_MINUTES:.1f}분")
    print("ver6 최종 모델 구성 파일:", CHECKPOINT_PATH)
    """)
    add("code", "ver6-report", """
    # 6. Training 내부 선택 결과 — 공식 Validation 점수와 구분합니다.
    REPORT_PATH = OUTPUT_DIR / "mission3_ver6_metrics.json"
    training_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    candidate_report = training_report.get("candidate_training", {})
    trials = candidate_report.get("head_trials", [])
    if isinstance(trials, dict):
        trials = list(trials.values())
    print("분류기 실험 | 최적 epoch | stop 교차평가 Macro F1 | stop AP")
    for trial in trials:
        print(f"{trial.get('profile', '?')} | {trial.get('best_epoch', '?')} | "
              f"{trial.get('best_stop_macro_f1', '기록 확인')} | {trial.get('best_stop_ap', '기록 확인')}")
    print("선택된 분류기:", candidate_report.get("selected_profile", "상세 기록 확인"))
    print("encoder 미세조정 최적 epoch:", candidate_report.get("best_epoch", "상세 기록 확인"))
    if candidate_report.get("best_epoch") == 0:
        print("encoder 미세조정 전의 선택된 분류기 가중치를 유지합니다. 분류기 학습 반영 여부는 위 최적 epoch를 확인하세요.")
    print("후보 선택 기준:", candidate_report.get("selection_scope", "원래 fold 1 stop의 threshold 교차평가"))
    for field, title in (
        ("baseline_crossfit_macro_f1", "기존 ver4 calibration 교차평가"),
        ("candidate_crossfit_macro_f1", "새 후보 calibration 교차평가"),
        ("candidate_raw_crossfit_macro_f1", "확률 평균 후보"),
        ("candidate_logit_crossfit_macro_f1", "logit 평균 후보"),
        ("selected_name", "최종 선택"),
        ("selected_blend_space", "선택된 혼합 방식"),
        ("selection_gain", "calibration F1 개선 폭"),
        ("minimum_gain", "최소 개선 폭 기준"),
        ("candidate_improved_folds", "F1이 개선된 calibration fold 수"),
        ("minimum_improved_folds", "최소 개선 fold 수 기준"),
        ("candidate_fold_gains", "calibration fold별 개선 폭"),
        ("fallback_reason", "선택 조건 판단"),
    ):
        if field in training_report:
            print(title + ":", training_report[field])
    if training_report.get("selected_name") == "ver4_baseline":
        print("Training 내부의 개선 기준을 충족하지 못해 기존 ver4 보정을 유지합니다. ver6 실행 자체는 완료됐습니다.")
    selected = training_report.get("calibrator", {})
    labels = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
    print("증상 | 기존 신경망 alpha | 새 후보 beta | 임계값")
    for label, alpha, beta, threshold in zip(labels, selected.get("neural_weights", []),
                                             selected.get("candidate_weights", []), selected.get("thresholds", [])):
        print(f"{label} | {alpha:.2f} | {beta:.2f} | {threshold:.9f}")
    print("실제 계산에는 저장된 보정값 전체 정밀도를 사용합니다.")
    print("전체 실험 기록:", REPORT_PATH)
    """)
    add("markdown", "ver6-evaluation-note", """
    ## 7. 확정된 모델로 Validation 전체 예측·평가

    여기서 처음으로 공식 Validation 전사 text를 모델에 전달합니다.
    먼저 예측 CSV를 확정한 뒤 symptom 정답을 읽어 평가합니다. 이 결과로 모델이나 임계값을 다시 선택하지 않습니다.
    기본 서울 Validation은 **3,640개 전체**이며 증상별 precision·recall·F1·support를 함께 표시합니다.
    필요하다고 선택된 모델만 추론하고, 입력·가중치·실행 조건이 같은 원시 예측은 공유 캐시에서 재사용합니다.
    cache가 없거나 조건이 다르면 첫 실행에서 기존 5개 모델과 새 후보의 추론 비용이 추가될 수 있습니다.

    비교 기준: 기존 전체 Validation Macro F1은 **ver4 0.656648948 / ver5 0.656717758**입니다.
    두 값은 이전 실행 기록이며 현재 GPU·batch 등에 따른 수치 차이가 있을 수 있습니다.
    아래 차이는 참고용 결과 비교이며 학습·보정에 사용하지 않습니다.
    """)
    add("code", "ver6-evaluate", """
    PREDICTION_PATH = None
    evaluation_started = time.monotonic()
    run_stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    if RUN_VALIDATION and VALID_LABEL_DIR is not None:
        prediction_dir = OUTPUT_DIR / "predictions" / run_stamp
        PREDICTION_PATH = prediction_dir / "mission3.csv"
        run_python([
            runtime_dir / "inference.py", "--audio_dir", "", "--label_dir", VALID_LABEL_DIR,
            "--ckpt_path", CHECKPOINT_PATH, "--output", PREDICTION_PATH,
            *(["--base_dir", BASE_DIR] if not TRAIN_CONFIG.get("bundle_base", True) else []),
            "--batch_size", INFERENCE_BATCH_SIZE, "--cache_dir", INFERENCE_CACHE_DIR, "--evaluate",
        ])
        metrics_path = prediction_dir / "validation_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"현재 입력 {metrics['samples']:,}개 Macro F1: {metrics['macro_f1']:.9f}")
            for version, reference in (("ver4", 0.6566489480388843), ("ver5", 0.6567177583831921)):
                difference = metrics["macro_f1"] - reference
                print(f"{version} 이전 기록 대비: {difference:+.9f} ({difference * 100:+.5f}%p)")
            print("증상 | Precision | Recall | F1 | 실제 양성 개수")
            for symptom, values in metrics["per_symptom"].items():
                print(f"{symptom} | {values['precision']:.6f} | {values['recall']:.6f} | "
                      f"{values['f1']:.6f} | {values['support']}")
        else:
            print("CSV 생성 완료. 정답이 없는 입력이면 평가 점수는 생성하지 않습니다.")
    else:
        print("학습·모델 저장 완료. RUN_VALIDATION=False이므로 최종 평가를 생략했습니다.")
    EVALUATION_MINUTES = (time.monotonic() - evaluation_started) / 60
    timing = {
        "preparation_minutes": PREPARATION_MINUTES,
        "training_and_calibration_minutes": TRAINING_MINUTES,
        "inference_and_evaluation_minutes": EVALUATION_MINUTES,
        "output_dir": str(OUTPUT_DIR),
        "prediction_csv": str(PREDICTION_PATH) if PREDICTION_PATH else None,
        "note": "벽시계 시간이며 CU 측정값이 아닙니다. 패키지 설치 시간은 제외합니다.",
    }
    timing_path = OUTPUT_DIR / ("notebook_runtime_" + run_stamp + ".json")
    timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(f"최종 추론·평가: {EVALUATION_MINUTES:.1f}분")
    print("새 ver6 결과 폴더:", OUTPUT_DIR)
    print("ver6 최종 모델 구성:", CHECKPOINT_PATH)
    print("예측 CSV:", PREDICTION_PATH)
    print("실행 시간 기록:", timing_path)
    print("추론 실행 코드:", OUTPUT_DIR / "submission/inference.py")
    """)
    add("markdown", "ver6-finish", """
    ## 결과 위치와 다음 실행

    **새 결과: `내 드라이브/DCC_AIdeas/results/mission3/ver6`**

    - `mission3_ver6_model.ckpt`: 최종 선택된 모델 구성·혼합 비율·임계값.
    - `candidate_best.ckpt`: 선택된 분류기와 encoder 가중치.
    - `mission3_ver6_metrics.json`: A–F 실험, 미세조정, 내부 교차평가와 최종 선택 기록.
    - `predictions/실행시각/mission3.csv`: 제출 형식 예측. 정답이 있으면 같은 폴더에 `validation_metrics.json`.
    - `notebook_runtime_실행시각.json`: 입력 준비·학습·최종 추론 단계의 실제 시간. CU는 Colab에서 별도 확인합니다.
    - `submission/`: 추론 코드, 패키지 목록, README와 논문 검토 문서.

    `/content/mission3_ver5_inputs/.../ver4`는 기존 입력 모델이고 `/content/mission3_ver6_features`는
    재계산 가능한 특징 캐시입니다. 새 학습 결과 폴더와 혼동하지 마세요.
    기존 5개 모델을 Drive에 중복 저장하지 않으므로 **입력 ZIP과 ver6 결과를 모두 보관해야 합니다.**
    같은 런타임에서는 검증된 압축 해제 파일·특징을 재사용합니다.
    새 런타임에서는 입력 ZIP을 다시 풉니다. 미완료 분류기 실험이 있을 때만 특징을 다시 계산하며,
    이미 저장된 분류기·미세조정 상태와 완료 결과는 검증 후 재사용합니다.
    진행 중 학습 재개에는 같은 GPU 종류·AMP 정밀도·런타임 버전이 필요합니다. 완료된 최종 결과는 다른 GPU에서도 사용할 수 있습니다.

    다른 환경에서 추론할 때는 ZIP을 푼 ver4 폴더를 함께 지정하세요.

    ```bash
    python submission/inference.py --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \\
      --ckpt_path mission3_ver6_model.ckpt --base_dir "복원한 ver4 폴더" --output mission3.csv
    ```

    `--audio_dir`는 대회 인터페이스 호환용이고 WAV는 읽지 않습니다.
    최종 내부 선택이 `ver4_baseline`이면 새 실험의 개선 기준을 충족하지 못해 기존 구성을 유지했다는 뜻입니다.
    실행 로그가 남은 이 노트북도 보관하세요. 실제 F1 개선 여부는 마지막 전체 Validation 결과로 확인합니다.
    """)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}, "accelerator": "GPU",
                             "colab": {"name": "mission3_ver6.ipynb", "provenance": []}}}
    target = HERE.parent / "mission3_ver6.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
