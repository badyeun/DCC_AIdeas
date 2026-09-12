# Mission 1 실험 기록

> 모델 체크포인트와 원본 데이터는 Git에 포함하지 않습니다. 아래 기록은 같은 코드와
> 데이터 구성에서 실험을 재현하고, 후속 실험과 비교하기 위한 기준선입니다.

## EXP-001 — Log-Mel + SimpleCNN baseline

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission1/train_cnn.py` |
| 실행 환경 | CUDA |
| 학습 데이터 | 서울 Training WAV–JSON pair 29,200개 |
| 검증 데이터 | 서울 Validation WAV–JSON pair 3,640개 |
| 데이터 무결성 | Training/Validation 모두 WAV–JSON 1:1 매칭, JSON 없는 WAV 0개 |
| 학습 설정 | epoch 10, batch size 16, Adam, learning rate 0.001, seed 42 |
| 입력 특징 | 신고자(`speaker=1`) 발화 구간의 Log-Mel spectrogram |
| 윈도우 처리 | 최대 300 frames, hop 150 frames; 학습은 통화별 무작위 window 1개, 검증은 모든 window logits 평균 |
| 다수 클래스 기준선 | `F`, Accuracy 0.5323 |
| 최적 epoch | 9 |
| 최종 선택 지표 | Validation Accuracy 0.9396 / Macro F1 0.9393 |
| 체크포인트 | `/content/drive/MyDrive/saved_models/mission1/mission1_cnn.pt` |

### Epoch별 결과

| Epoch | Train loss | Train accuracy | Validation accuracy | Validation macro F1 | Best checkpoint |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 0.5242 | 0.7888 | 0.8720 | 0.8678 | ✓ |
| 2 | 0.2938 | 0.8834 | 0.9137 | 0.9130 | ✓ |
| 3 | 0.2776 | 0.8912 | 0.9091 | 0.9079 | |
| 4 | 0.2619 | 0.8987 | 0.9146 | 0.9135 | ✓ |
| 5 | 0.2493 | 0.9051 | 0.9192 | 0.9181 | ✓ |
| 6 | 0.2414 | 0.9076 | 0.9212 | 0.9200 | ✓ |
| 7 | 0.2351 | 0.9098 | 0.9346 | 0.9340 | ✓ |
| 8 | 0.2345 | 0.9108 | 0.9283 | 0.9276 | |
| 9 | 0.2263 | 0.9136 | **0.9396** | **0.9393** | ✓ |
| 10 | 0.2145 | 0.9207 | 0.9382 | 0.9376 | |

### 최적 체크포인트 검증 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9289 | 0.9411 | 0.9350 | 1,681 |
| F | 0.9489 | 0.9382 | 0.9435 | 1,959 |
| **Macro avg** | **0.9389** | **0.9397** | **0.9393** | **3,640** |
| Accuracy |  |  | **0.9396** | **3,640** |

### 해석 및 다음 비교 기준

- 다수 클래스만 예측하는 기준선 대비 Validation Accuracy가 **40.73%p** 높습니다.
- `M`과 `F`의 F1-score가 각각 0.9350, 0.9435로 큰 편차 없이 동작했습니다.
- epoch 9가 최고 성능이었으므로, 최종 평가는 epoch 10 가중치가 아닌 epoch 9에 저장된 체크포인트를 사용했습니다.
- 후속 모델은 동일한 Validation split에서 **Accuracy 0.9396 및 Macro F1 0.9393**을 넘는지를 기준으로 비교합니다.

### 재현 명령

아래는 이번 실행과 같은 기본 하이퍼파라미터를 사용하는 명령입니다. `data_root`와 `ckpt_path`는 실행 환경에 맞게 바꿉니다.

```bash
python src/mission1/train_cnn.py \
  --data_root /content/drive/MyDrive/mission1 \
  --ckpt_path /content/drive/MyDrive/saved_models/mission1/mission1_cnn.pt \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --seed 42
```

## EXP-002 — MFCC + RandomForest (F0 포함)

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission1/train_mfcc_rf.py` |
| 실행 환경 | Google Colab (`/content/drive/MyDrive/mission1`) |
| 학습 데이터 | 서울 Training WAV–JSON pair 29,200개 |
| 검증 데이터 | 서울 Validation WAV–JSON pair 3,640개 |
| 데이터 무결성 | Training/Validation 모두 WAV–JSON 1:1 매칭, JSON 없는 WAV 0개 |
| 입력 특징 | 신고자(`speaker=1`) 발화 구간의 MFCC·ΔMFCC·Δ²MFCC 통계 + F0 통계 |
| 특징 차원 | 246 (`F0 사용: True`) |
| 특징 추출 시간 | Train 5시간 42분 46초 / Validation 38분 59초 |
| 다수 클래스 기준선 | `F`, Accuracy 0.5323 |
| Validation Accuracy | **0.9423** |
| Validation Macro F1 | **0.9419** |
| 체크포인트 | `/content/drive/MyDrive/saved_models/mission1/mission1_mfcc_rf.joblib` |

### Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9428 | 0.9316 | 0.9372 | 1,681 |
| F | 0.9419 | 0.9515 | 0.9467 | 1,959 |
| **Macro avg** | **0.9423** | **0.9415** | **0.9419** | **3,640** |
| Accuracy |  |  | **0.9423** | **3,640** |

### CNN baseline과 비교

| 모델 | Validation Accuracy | Macro F1 | 비고 |
| --- | ---: | ---: | --- |
| Log-Mel + SimpleCNN (EXP-001) | 0.9396 | 0.9393 | epoch 9 checkpoint |
| **MFCC + RandomForest + F0 (EXP-002)** | **0.9423** | **0.9419** | 현재 최고 성능 |

- CNN baseline보다 Accuracy **+0.27%p**, Macro F1 **+0.26%p** 높았습니다.
- 학습 데이터 재현 정확도 1.0000은 학습에 사용한 데이터를 다시 예측한 참고 값이며, 모델 선택은 Validation 지표를 기준으로 합니다.
- 현재 실행 중인 `--no_f0` 결과와 비교해 F0 특징의 실제 기여도를 확정합니다.

### 재현 명령

```bash
python src/mission1/train_mfcc_rf.py \
  --data_root /content/drive/MyDrive/mission1 \
  --ckpt_path /content/drive/MyDrive/saved_models/mission1/mission1_mfcc_rf.joblib
```

## EXP-003 — MFCC + RandomForest (F0 제외)

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission1/train_mfcc_rf.py --no_f0` |
| 실행 환경 | Google Colab (`/content/drive/MyDrive/mission1`) |
| 학습 / 검증 데이터 | EXP-002와 동일 (29,200 / 3,640 pairs) |
| 데이터 무결성 | Training/Validation 모두 WAV–JSON 1:1 매칭, JSON 없는 WAV 0개 |
| 입력 특징 | 신고자(`speaker=1`) 발화 구간의 MFCC·ΔMFCC·Δ²MFCC 통계 |
| 특징 차원 | 240 (`F0 사용: False`) |
| 특징 추출 시간 | Train 44분 09초 / Validation 5분 29초 |
| Validation Accuracy | 0.9365 |
| Validation Macro F1 | 0.9361 |
| 체크포인트 | `/content/drive/MyDrive/saved_models/mission1/mission1_mfcc_rf_no_f0.joblib` |

### Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9394 | 0.9221 | 0.9307 | 1,681 |
| F | 0.9342 | 0.9490 | 0.9415 | 1,959 |
| **Macro avg** | **0.9368** | **0.9355** | **0.9361** | **3,640** |
| Accuracy |  |  | **0.9365** | **3,640** |

### F0 ablation 비교

| 모델 | 특징 차원 | Train 특징 추출 | Validation Accuracy | Macro F1 | F0 포함 대비 |
| --- | ---: | ---: | ---: | ---: | --- |
| MFCC + RandomForest + F0 (EXP-002) | 246 | 5시간 42분 46초 | **0.9423** | **0.9419** | 기준 |
| MFCC + RandomForest, F0 제외 (EXP-003) | 240 | **44분 09초** | 0.9365 | 0.9361 | Accuracy −0.58%p / Macro F1 −0.58%p |

- F0 통계 6개를 추가하면 검증 성능이 유의미하게 개선되며, 특히 `M` recall이 0.9221에서 0.9316으로 올랐습니다.
- 반면 F0 추출 비용 때문에 Train 특징 추출 시간은 약 **7.8배** 길어집니다. 성능 우선 최종 제출 모델은 **EXP-002 (F0 포함)**를 사용합니다.
- 빠른 반복 실험이나 디버깅에는 EXP-003을 사용하고, 최종 평가·제출 전에는 EXP-002 체크포인트로 전환합니다.

### 재현 명령

```bash
python src/mission1/train_mfcc_rf.py \
  --data_root /content/drive/MyDrive/mission1 \
  --no_f0 \
  --ckpt_path /content/drive/MyDrive/saved_models/mission1/mission1_mfcc_rf_no_f0.joblib
```

## 다음 실험 계획 — WavLM 기반 성별 분류

현재 최종 기준선은 **EXP-002 (MFCC + F0 + RandomForest, Validation Macro F1 0.9419)**입니다.
WavLM 실험은 이 수치를 동일한 Validation split에서 넘어서는지로 판단합니다.

| 순서 | 실험 | 목적 | 유지 / 변경 사항 | 성공 기준 |
| ---: | --- | --- | --- | --- |
| 1 | 데이터·평가 고정 | 공정 비교 | 기존 Training 29,200 / Validation 3,640 pair, `speaker=1` 구간, Accuracy·Macro F1 유지 | 기준선 재현 가능 |
| 2 | WavLM Base+ feature probe | 사전학습 표현의 단독 효용을 저비용으로 확인 | `microsoft/wavlm-base-plus` 동결, raw waveform → pooling → 작은 분류기 | Macro F1 0.9419 근접 또는 초과 |
| 3 | head-only fine-tuning | 과적합 위험 없이 task 적응 | encoder 동결, classification head만 학습; 3개 seed로 반복 | 평균 Macro F1 기준선 초과 |
| 4 | partial fine-tuning | 화자/성별 단서에 표현을 적응 | 마지막 2~4 transformer layer와 head만 해제, encoder는 더 작은 learning rate | EXP-002 대비 유의미한 개선 |
| 5 | augmentation ablation | 실제 통화 환경 강건성 확인 | noise/gain/time masking을 하나씩 추가·제거 | Validation 성능 유지 또는 개선 |
| 6 | soft-voting ensemble | 상호 보완성 검증 | WavLM 확률과 MFCC+F0 RF 확률을 validation에서 가중 평균 | 단일 최고 모델 초과 시 채택 |

### 구현 원칙

- 입력은 `speaker=1` 발화 구간만 이어 붙인 raw waveform이며, **16 kHz mono**로 통일합니다.
- 통화가 길면 학습·검증 모두 동일한 sliding window 규칙을 사용합니다. 검증에서는 모든 window의 **확률 또는 logits 평균**으로 통화 단위 예측을 만듭니다.
- 긴 발화를 무조건 첫 구간만 자르지 않습니다. 성별 단서는 통화 전체에 분포할 수 있으므로, window 평균이 안전합니다.
- feature probe 단계에서는 WavLM encoder를 동결하고 embedding을 캐시합니다. 이 단계가 기준선보다 현저히 낮다면 expensive fine-tuning을 진행하지 않습니다.
- fine-tuning은 class-weight 없이 먼저 시작하고, 최적 checkpoint는 Accuracy뿐 아니라 **Macro F1**도 함께 저장합니다. 최종 선택은 Macro F1 우선, Accuracy를 보조 지표로 사용합니다.
- seed 42 한 번의 최고값만으로 모델을 고르지 않습니다. 유망한 설정은 최소 3개 seed의 평균·표준편차를 기록합니다.

### 모델 선택 기준

1. **WavLM Base+를 Wav2Vec2보다 먼저 실행**한다. WavLM은 화자 관련 과제를 포함한 범용 음성 처리용으로 사전학습되었고, Base+는 대규모 다도메인 음성으로 사전학습되었다.
2. WavLM이 EXP-002보다 낮거나 비슷하면, F0 RF를 최종 단일 모델로 유지한다. 대형 모델이라는 이유만으로 교체하지 않는다.
3. WavLM과 F0 RF의 오류가 다르면 soft-voting을 평가한다. validation에서만 가중치(예: 0.0~1.0)를 정하고, 개선이 없으면 앙상블을 채택하지 않는다.
4. Wav2Vec2는 WavLM 결과가 기준선에 못 미치거나, 두 모델의 앙상블 다양성을 확인할 필요가 있을 때 **후순위 ablation**으로 실행한다.
