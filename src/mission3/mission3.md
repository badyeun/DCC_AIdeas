# Mission 3 — 환자 증상 인식

전사 텍스트만 사용해 통화별 9개 증상(`고열`, `구토`, `두통`, `복통`, `어지러움`, `열상`, `오심`, `전신쇠약`, `호흡곤란`)을 멀티라벨로 예측하는 baseline입니다. 학습은 서울 Training만, 검증은 서울 Validation만 사용합니다.

## 학습과 검증

```bash
python src/mission3/train_tfidf_lr.py \
  --data_root /path/to/data_root \
  --ckpt_path saved_models/mission3_tfidf_lr.joblib
```

단어 1–2 gram과 문자 2–5 gram TF-IDF를 합쳐 각 증상별 Logistic Regression을 학습합니다. Vocabulary와 IDF는 Training text에서만 학습하며, Validation Macro F1을 이용해 증상별 결정 임계값을 고릅니다.

## 제출 CSV 생성

```bash
python src/mission3/inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path saved_models/mission3_tfidf_lr.joblib \
  --output ./outputs/mission3.csv
```

Mission 3에서는 `audio_dir`를 읽지 않으며 공통 제출 실행 규격을 맞추기 위해서만 받습니다. 출력 컬럼은 정확히 `label file name,symptom`입니다. `symptom`은 `['두통', '복통']` 같은 string이며, 증상이 없으면 `[]`를 기록합니다.
