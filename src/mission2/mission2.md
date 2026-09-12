# Mission 2 — 신고자 / 119대원 분류

각 통화의 JSON `utterances`에서 `startAt`~`endAt` 구간을 잘라, 신고자(`1`)와
119대원(`0`)을 구분합니다. 이 구현은 text와 발화 순서를 사용하지 않습니다.

## 학습

```bash
python src/mission2/train_mfcc_rf.py \
  --data_root /path/to/data_root \
  --ckpt_path saved_models/mission2_mfcc_rf.joblib
```

`data_root` 아래의 각 part에서 `대학부 데이터/Training`과 `Validation`의 서울
데이터를 자동으로 찾습니다. 검증 데이터는 평가에만 사용하며 체크포인트 학습에는
넣지 않습니다.

## 제출 CSV 생성

```bash
python src/mission2/inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path saved_models/mission2_mfcc_rf.joblib \
  --output ./outputs/mission2.csv
```

출력 컬럼은 정확히 `audio file name,startAt,endAt,speaker`입니다. 입력 JSON의 모든
발화 경계마다 한 행을 생성하고 `speaker`는 `0` 또는 `1`입니다.

## 특징

MFCC·Δ·Δ²의 분포 통계에 RMS, zero-crossing rate, spectral centroid/bandwidth/rolloff,
발화 길이를 더해 RandomForest로 분류합니다. 통화 환경 및 발화 스타일 차이를
반영하는 가벼운 기준선이며, 이후 Log-Mel CNN과 비교할 기준으로 사용합니다.
