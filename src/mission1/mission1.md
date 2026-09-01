# 2026 데이터+AI 혁신 챌린지 - Mission 1: 신고자 성별 분류

## 규정 준수

- 학습에는 서울 Training 데이터만 사용합니다. Validation 데이터는 학습에 넣지 않습니다.
- 학습 시 `gender`는 정답 레이블로만 사용합니다.
- 학습 및 추론 시 라벨 JSON에서는 신고자 구간 추출을 위해 `startAt`, `endAt`, `speaker`만 사용합니다. `speaker=1` 구간만 이어 붙여 Mel-spectrogram을 생성합니다.
- `text`, `symptom` 등 허용되지 않은 라벨 필드는 읽지 않습니다.

## 학습

```bash
python src/mission1/train.py \
  --audio_dir data/train/wav \
  --label_dir data/train/json \
  --ckpt_path saved_models/mission1_model.pt
```

`startAt`과 `endAt`의 단위는 기본적으로 자동 판별합니다. 데이터 확인 후 필요하면 학습 시 `--timestamp_unit milliseconds` 또는 `seconds`를 명시합니다. 해당 설정은 체크포인트에 저장되어 추론에 그대로 적용됩니다.

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
