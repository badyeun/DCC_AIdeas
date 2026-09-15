# 2026 데이터+AI 혁신 챌린지 - Mission 1: 신고자 성별 분류

## 프로젝트 구조와 파일 역할

```text
src/mission1/
├── common.py
├── train_cnn.py
├── train_mfcc_rf.py
├── inference.py
├── mission1.md
└── experiment_log.md
```

| 파일 | 역할 | 직접 실행 여부 |
| --- | --- | --- |
| `common.py` | 학습과 추론에서 공통으로 사용하는 코드입니다. WAV 파일 로드, 신고자 발화 구간 추출, Mel-spectrogram·MFCC 특징 생성, CNN 모델 정의를 담당합니다. | 아니요 |
| `train_cnn.py` | `Mel-spectrogram → CNN` 베이스라인 학습 파일입니다. 학습이 끝나면 모델 가중치와 전처리 설정을 `.pt` 체크포인트로 저장합니다. | 예 |
| `train_mfcc_rf.py` | `MFCC → RandomForest` 베이스라인 학습 파일입니다. 학습이 끝나면 RandomForest 모델과 전처리 설정을 `.joblib` 체크포인트로 저장합니다. | 예 |
| `inference.py` | 제출용 추론 파일입니다. `.pt` 체크포인트는 CNN으로, `.joblib` 체크포인트는 RandomForest로 자동 인식하여 제출 CSV를 만듭니다. | 예 |
| `mission1.md` | 현재 문서입니다. 규정, 파일 역할, 실행 방법 및 실험 절차를 설명합니다. | 아니요 |
| `experiment_log.md` | 모델별 Validation Accuracy, 주요 설정, 관찰 사항을 남기는 실험 기록 파일입니다. | 아니요 |

저장되는 체크포인트와 생성되는 제출 CSV는 Git에서 제외됩니다.

```text
saved_models/
├── mission1_model.pt
└── mission1_mfcc_rf.joblib

outputs/
├── mission1_cnn.csv
└── mission1_mfcc_rf.csv
```

## 규정 준수

- 학습에는 서울 Training 데이터만 사용합니다. Validation 데이터는 학습에 넣지 않습니다.
- 학습 시 `gender`는 정답 레이블로만 사용합니다.
- 학습 및 추론 시 라벨 JSON에서는 신고자 구간 추출을 위해 `startAt`, `endAt`, `speaker`만 사용합니다. `speaker=1` 구간만 이어 붙인 뒤 각 모델에 맞는 음성 특징을 생성합니다.
- `text`, `symptom` 등 허용되지 않은 라벨 필드는 읽지 않습니다.

## 전체 실행 흐름

두 베이스라인은 **동일한 입력 데이터와 신고자 구간 전처리**를 사용합니다. 따라서 모델 차이에 따른 성능을 비교할 수 있습니다.

```text
학습 데이터: WAV + JSON
        │
        ├─ train_cnn.py ───────→ Mel-spectrogram + CNN → .pt 체크포인트
        │
        └─ train_mfcc_rf.py ──→ MFCC + RandomForest    → .joblib 체크포인트
                                                          │
                                     inference.py ───────┘
                                           │
                                           └→ 제출 CSV: audio file name,gender
```

`common.py`는 JSON에서 `speaker=1`인 신고자의 `startAt`~`endAt` 구간을 찾아 WAV에서 잘라낸 뒤 이어 붙입니다. 이후 CNN에는 Mel-spectrogram을, RandomForest에는 MFCC와 그 변화량의 통계 특징을 전달합니다. `Speaker`처럼 대문자로 기록된 필드도 인식합니다.

CNN은 300프레임보다 짧은 입력을 padding하고, 긴 입력은 150프레임 간격의 sliding window로 모두 처리합니다. 학습에서는 통화마다 창 하나를 무작위로 골라 긴 통화가 과도하게 학습에 반영되는 것을 막고, 추론에서는 모든 창의 성별 확률을 평균내어 최종 결과를 정합니다.

## 사전 준비

저장소 최상위 경로에서 필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

데이터는 다음처럼 WAV 폴더와 대응 JSON 폴더가 같은 하위 구조를 갖는다고 가정합니다.

```text
data/
└── train/
    ├── wav/
    │   └── example.wav
    └── json/
        └── example.json
```

## 학습

### Mel-spectrogram + CNN

```bash
python src/mission1/train_cnn.py \
  --data_root /path/to/mission1 \
  --ckpt_path saved_models/mission1/mission1_cnn.pt
```

`startAt`과 `endAt`의 단위는 기본적으로 자동 판별합니다. 데이터 확인 후 필요하면 학습 시 `--timestamp_unit milliseconds` 또는 `seconds`를 명시합니다. 해당 설정은 체크포인트에 저장되어 추론에 그대로 적용됩니다.

### MFCC + RandomForest

동일한 신고자 구간과 동일한 학습 데이터를 사용해, 두 베이스라인의 Validation Accuracy를 비교합니다.

```bash
python src/mission1/train_mfcc_rf.py \
  --audio_dir data/train/wav \
  --label_dir data/train/json \
  --ckpt_path saved_models/mission1_mfcc_rf.joblib
```

RandomForest는 MFCC 20개, 1차 변화량, 2차 변화량 각각의 평균·표준편차·25%·75% 분위수(240개)에 더해, **F0(기본주파수) 통계 6개**를 특징으로 사용합니다. 총 246차원입니다.

MFCC는 DCT 과정에서 스펙트럼 포락선만 남기고 F0를 상당 부분 버리는데, 성별을 가르는 가장 강한 단서가 F0입니다(성인 남성 약 85~180Hz, 여성 약 165~255Hz). 그래서 `librosa.yin`으로 F0를 따로 추출해 log2 스케일의 평균·중앙값·표준편차·25%·75% 분위수와 유성음 비율을 덧붙입니다.

F0 추출은 시간이 걸리므로, 기여도를 비교하려면 `--no_f0`로 끈 버전을 함께 학습해 Accuracy를 비교합니다.

```bash
python src/mission1/train_mfcc_rf.py \
  --audio_dir data/train/wav \
  --label_dir data/train/json \
  --no_f0 \
  --ckpt_path saved_models/mission1_mfcc_rf_nof0.joblib
```

> `train_mfcc_rf.py`가 출력하는 `[참고] 학습 데이터 재현 정확도`는 **성능 지표가 아닙니다.** 학습에 쓴 데이터를 그대로 채점한 값이고, `max_depth` 제한이 없으면 거의 항상 1.0에 가깝게 나옵니다. 실제 성능은 반드시 별도 검증셋으로 측정하세요.

## Validation 성능 비교

Validation 데이터에 대해서는 각 모델로 별도 CSV를 생성한 뒤, 정답 `gender`와 비교하여 Accuracy를 기록합니다. **Validation 데이터는 어떤 경우에도 학습 명령의 입력으로 사용하지 않습니다.**

```bash
# CNN 결과
python src/mission1/inference.py \
  --audio_dir data/validation/wav \
  --label_dir data/validation/json \
  --ckpt_path saved_models/mission1_model.pt \
  --output outputs/mission1_cnn.csv

# MFCC-랜덤 포레스트 결과
python src/mission1/inference.py \
  --audio_dir data/validation/wav \
  --label_dir data/validation/json \
  --ckpt_path saved_models/mission1_mfcc_rf.joblib \
  --output outputs/mission1_mfcc_rf.csv
```

두 결과의 Accuracy와 학습 시간, 실패 사례를 `experiment_log.md`에 기록한 뒤 더 좋은 모델을 최종 제출 모델로 선택합니다.

## 추론 및 제출 CSV 생성

주최 측 규격에 맞는 한 번의 실행 명령입니다.

```bash
python src/mission1/inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path {checkpoint_file} \
  --output ./outputs/mission1.csv
```

생성되는 CSV 컬럼은 정확히 `audio file name,gender`이며, 성별 값은 `M` 또는 `F`입니다.

### 실패 허용 동작

평가 폴더에 어떤 파일이 들어올지 알 수 없으므로 `inference.py`는 개별 파일의 실패로 중단되지 않습니다.

- 대응 JSON은 ①동일한 하위 경로 → ②파일명(stem) 일치 순으로 찾습니다. 원천/라벨 폴더의 하위 구조가 달라도 매칭됩니다.
- JSON 파싱 실패, 신고자 발화 없음, 오디오 로드 실패 등이 발생하면 해당 파일만 학습 데이터의 다수 클래스(`fallback_gender`, 체크포인트에 저장됨)로 채우고 진행합니다.
- 확장자가 `.WAV`인 파일도 인식합니다.
- 실패 내역은 stderr에 남기며, **CSV 행 수는 항상 입력 WAV 개수와 같습니다.**

이 동작이 없으면 파일 하나만 잘못되어도 CSV가 생성되지 않아 0점이 됩니다.

CNN 체크포인트(`.pt`)와 MFCC-랜덤 포레스트 체크포인트(`.joblib`) 모두 같은 `inference.py` 명령으로 실행할 수 있습니다.

최종 제출 시에는 성능이 더 좋은 모델의 체크포인트 경로만 `--ckpt_path`에 지정하면 됩니다. 주최 측 실행 환경에서는 요구된 네 개의 인자를 포함한 `inference.py` 실행 한 번으로 CSV가 생성됩니다.
