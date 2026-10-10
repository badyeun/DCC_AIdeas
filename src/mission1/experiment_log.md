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

## EXP-004 — Frozen WavLM Base+ embedding + classification head (1 chunk)

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `train_wavlm.py` |
| 실행 환경 | Google Colab, CPU (embedding cache 재사용 후 head 학습) |
| encoder | `microsoft/wavlm-base-plus`, 동결 |
| 입력 | 신고자(`speaker=1`) raw waveform, 16 kHz mono |
| 통화 처리 | 8초 chunk 1개 — 긴 통화는 중앙 8초; WavLM hidden state masked mean pooling |
| embedding | 768차원, Train 29,200 / Validation 3,640개 cache 재사용 |
| head | `Linear(768→256) → ReLU → Dropout(0.2) → Linear(256→2)` |
| 학습 설정 | AdamW, epoch 15, batch size 256, learning rate 0.001, seed 42 |
| 최적 epoch | 9 |
| Validation Accuracy | **0.9566** |
| Validation Macro F1 | **0.9563** |
| checkpoint | `/content/drive/MyDrive/saved_models/mission1/mission1_wavlm_1chunk.pt` |
| scaler | `/content/drive/MyDrive/saved_models/mission1/mission1_wavlm_1chunk.scaler.joblib` |

### Validation 상세

`GENDER_TO_INDEX` 기준 `class 0 = M`, `class 1 = F`입니다.

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9607 | 0.9447 | 0.9526 | 1,681 |
| F | 0.9532 | 0.9668 | 0.9600 | 1,959 |
| **Macro avg** | **0.9569** | **0.9557** | **0.9563** | **3,640** |
| Accuracy |  |  | **0.9566** | **3,640** |

### 기존 모델과 비교

| 모델 | Validation Accuracy | Macro F1 | 최고 epoch / 비고 |
| --- | ---: | ---: | --- |
| Log-Mel + SimpleCNN (EXP-001) | 0.9396 | 0.9393 | epoch 9 |
| MFCC + RandomForest + F0 (EXP-002) | 0.9423 | 0.9419 | F0 포함 |
| **Frozen WavLM Base+ + head (EXP-004)** | **0.9566** | **0.9563** | epoch 9, 중앙 8초 1개 |

- WavLM은 기존 최고 모델(EXP-002)보다 Accuracy **+1.43%p**, Macro F1 **+1.44%p** 높았습니다.
- 입력을 통화 전체가 아닌 중앙 8초 한 구간으로 제한했음에도 기준선을 명확히 넘었습니다. encoder fine-tuning 없이도 WavLM 표현이 성별 분류에 유효함을 확인했습니다.
- Train accuracy(0.9614)와 Validation accuracy(0.9566)의 차이가 작아, 현 설정에서 심한 head 과적합 징후는 보이지 않습니다.

### 다음 우선순위

1. **seed 재현:** 같은 embedding cache로 `seed 52`, `seed 62`를 추가 실행하고, Macro F1 평균·표준편차로 EXP-004의 안정성을 확인한다. cache가 있으므로 encoder 재추출은 필요 없다.
2. **3-chunk 비교:** GPU 런타임에서 `--chunks_per_call 3`을 실행해 앞·중간·뒤 구간 평균이 1-chunk보다 개선되는지 비교한다. CPU에서 1-chunk cache를 만드는 데 약 17시간 38분이 걸렸으므로, CPU 재추출은 우선순위가 낮다.
3. **partial fine-tuning:** 재현성과 3-chunk 결과를 확인한 뒤 마지막 2~4개 WavLM layer만 해제하는 실험을 한다. 이때 1-chunk frozen 결과 0.9563을 새 기준으로 사용한다.
4. **제출 파이프라인:** 최종 모델로 확정되면 `inference.py`에 WavLM encoder·pooling·scaler·head 로딩을 추가해 주최 측의 단일 실행 규격을 만족시킨다.

## EXP-005 — WavLM + MFCC/F0 RandomForest late fusion

| 항목 | 결과 |
| --- | --- |
| 구성 | EXP-004 WavLM 확률 + EXP-002 MFCC+F0 RandomForest 확률의 가중 평균 |
| 검증 데이터 | Validation 3,640개, RF/WavLM 샘플 순서 일치 확인 |
| 확률 차원 | RF `(3640, 2)`, WavLM `(3640, 2)` |
| 가중치 탐색 | WavLM weight 0.00~1.00, 0.05 간격 |
| 최적 WavLM / RF weight | **0.45 / 0.55** |
| Validation Accuracy | **0.9635** |
| Validation Macro F1 | **0.9632** |

### 최적 fusion Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9697 | 0.9506 | 0.9600 | 1,681 |
| F | 0.9583 | 0.9745 | 0.9663 | 1,959 |
| **Macro avg** | **0.9640** | **0.9626** | **0.9632** | **3,640** |
| Accuracy |  |  | **0.9635** | **3,640** |

### 단일 모델·fusion 비교

| 모델 | WavLM / RF weight | Validation Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| MFCC + F0 + RF (EXP-002) | 0.00 / 1.00 | 0.9423 | 0.9419 |
| Frozen WavLM + head (EXP-004) | 1.00 / 0.00 | 0.9566 | 0.9563 |
| **Late fusion (EXP-005)** | **0.45 / 0.55** | **0.9635** | **0.9632** |

- 최적 fusion은 단일 WavLM 대비 Accuracy **+0.69%p**, Macro F1 **+0.69%p** 개선됐고, 현재 최고 성능입니다.
- RF 비중이 0.55로 더 높지만 WavLM을 함께 사용했을 때 성능이 오르므로, MFCC/F0와 WavLM이 서로 다른 오류를 보완합니다.
- 이 가중치는 동일한 Validation split에서 탐색한 값입니다. 최종 제출 모델로 확정하기 전, 다른 seed의 WavLM head에서도 `0.45 / 0.55`가 개선되는지 재현해야 합니다. 재현 단계에서는 validation 결과를 다시 보고 가중치를 변경하지 않습니다.

## EXP-006 — WavLM 3-chunk test-time averaging

| 항목 | 결과 |
| --- | --- |
| encoder / head | EXP-004의 동결 WavLM Base+ encoder 및 1-chunk 학습 head·scaler 재사용 |
| 비교 대상 | Validation 1-chunk embedding과 3-chunk embedding (`(3640, 768)`) |
| 통화 처리 | 1-chunk: 중앙 8초 / 3-chunk: 앞·중간·뒤 8초 embedding 평균 |
| 검증 | 1-chunk·3-chunk label 순서 일치 확인 |
| 재학습 여부 | 없음 — 동일 head/scaler로 embedding만 교체한 test-time averaging 비교 |
| 1-chunk Accuracy / Macro F1 | 0.9566 / 0.9563 |
| 3-chunk Accuracy / Macro F1 | **0.9750 / 0.9748** |

### 3-chunk Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9783 | 0.9673 | 0.9728 | 1,681 |
| F | 0.9722 | 0.9816 | 0.9769 | 1,959 |
| **Macro avg** | **0.9753** | **0.9745** | **0.9748** | **3,640** |
| Accuracy |  |  | **0.9750** | **3,640** |

### 해석

- 3-chunk test-time averaging은 1-chunk WavLM 대비 Accuracy **+1.84%p**, Macro F1 **+1.85%p** 개선됐습니다.
- 기존 최고 late fusion(EXP-005)과 비교해도 Accuracy **+1.15%p**, Macro F1 **+1.16%p** 높습니다. 현재 최고 성능은 **EXP-006**입니다.
- 3-chunk cache는 label을 사용하지 않고 raw waveform의 세 구간을 평균한 표현이므로, 이 평가는 합법적인 추론-time aggregation입니다. 단, 이번 validation에서 3-chunk 방식을 선택했으므로 최종 성능 추정에는 선택 편향 가능성이 있습니다.

### 다음 실험 우선순위

1. **3-chunk head 재학습:** 이미 만든 `train_*_8s_3chunks.npz` cache를 이용해 head와 scaler를 3-chunk train embedding에 맞춰 다시 학습·검증한다. encoder 재추출은 필요 없다.
2. **seed 재현:** 3-chunk 재학습 설정을 seed 42/52/62로 실행해 평균·표준편차를 기록한다. 이번에 선택한 3-chunk 방식은 고정한다.
3. **fusion 재검증:** 3-chunk head가 확정된 뒤에만 RF와의 fusion을 한 번 평가한다. 3-chunk WavLM 자체가 이미 높으므로, 개선이 없으면 단일 WavLM을 유지한다.
4. **제출 구현:** 최종 모델은 반드시 3개 chunk를 같은 규칙으로 만들고 embedding을 평균한 뒤, 3-chunk용 scaler와 head를 적용하도록 `inference.py`에 구현한다.

## EXP-007 — WavLM 3-chunk + MFCC/F0 RandomForest late fusion

| 항목 | 결과 |
| --- | --- |
| 구성 | EXP-006 WavLM 3-chunk 확률 + EXP-002 MFCC+F0 RandomForest 확률의 가중 평균 |
| 검증 데이터 | Validation 3,640개, RF/WavLM sample 순서 일치 확인 |
| 단일 WavLM 3-chunk Macro F1 | 0.9748 |
| 최적 WavLM / RF weight | **0.85 / 0.15** |
| Validation Accuracy | **0.9755** |
| Validation Macro F1 | **0.9754** |

### 최적 fusion Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| M | 0.9789 | 0.9679 | 0.9734 | 1,681 |
| F | 0.9727 | 0.9821 | 0.9774 | 1,959 |
| **Macro avg** | **0.9758** | **0.9750** | **0.9754** | **3,640** |
| Accuracy |  |  | **0.9755** | **3,640** |

### 해석 및 선택

- 최적 fusion은 3-chunk WavLM 단독 대비 Accuracy **+0.05%p**, Macro F1 **+0.06%p** 개선입니다. 검증 3,640개 기준 Accuracy 차이는 약 **2개 샘플**에 해당합니다.
- 여러 weight를 같은 Validation split에서 탐색해 얻은 작은 차이이므로, 아직 fusion을 최종 모델로 확정하지 않습니다.
- **현재 권장 최종 후보는 EXP-006 (3-chunk WavLM 단독)**입니다. EXP-007은 3-chunk head의 seed 재현 뒤에도 고정 weight `WavLM 0.85 / RF 0.15`가 일관되게 이길 때만 채택합니다.

## 최종 모델 선정

> **최종 모델: WavLM-base-plus + 3-chunk inference — Validation Accuracy 97.50%**

추가 실험으로 MFCC+F0 RandomForest late fusion을 적용했을 때 Validation Accuracy 97.55%를 기록했습니다. 다만 fusion weight(WavLM 0.85, RF 0.15)를 Validation 결과를 기준으로 결정했고, 단독 WavLM 3-chunk 대비 향상 폭도 0.05%p(순개선 2개 샘플)로 작았습니다. 따라서 최종 모델은 재현과 제출이 더 단순한 **WavLM-base-plus + 3-chunk inference**로 선정합니다.

## 다음 실험 계획 — WavLM 기반 성별 분류

현재 최종 기준선은 **EXP-006 (WavLM 3-chunk test-time averaging, Validation Macro F1 0.9748)**입니다. EXP-007의 fusion 최고값은 0.9754이지만, validation weight 탐색에 따른 작은 차이여서 재현 전까지는 기준선으로 승격하지 않습니다.
후속 WavLM 실험은 이 수치를 동일한 Validation split에서 넘어서는지로 판단합니다.

| 순서 | 실험 | 목적 | 유지 / 변경 사항 | 성공 기준 |
| ---: | --- | --- | --- | --- |
| 1 | 데이터·평가 고정 | 공정 비교 | 기존 Training 29,200 / Validation 3,640 pair, `speaker=1` 구간, Accuracy·Macro F1 유지 | 기준선 재현 가능 |
| 2 | WavLM Base+ feature probe | **완료 (EXP-004)** | `microsoft/wavlm-base-plus` 동결, raw waveform → pooling → 작은 분류기 | Macro F1 **0.9563** |
| 3 | 3-chunk head 재학습·재현성 확인 | embedding 분포에 맞춘 head 학습과 seed 의존성 확인 | encoder 동결, 3-chunk cache의 head/scaler 학습; seed 3개로 반복 | 평균 Macro F1 0.9748에 근접 또는 초과 |
| 4 | partial fine-tuning | 화자/성별 단서에 표현을 적응 | 마지막 2~4 transformer layer와 head만 해제, encoder는 더 작은 learning rate | EXP-004 대비 유의미한 개선 |
| 5 | augmentation ablation | 실제 통화 환경 강건성 확인 | noise/gain/time masking을 하나씩 추가·제거 | Validation 성능 유지 또는 개선 |
| 6 | soft-voting ensemble | EXP-007 완료, 재현 대기 | WavLM 0.85 + MFCC+F0 RF 0.15 확률 가중 평균 | 다른 seed에서도 단일 WavLM을 일관되게 초과 시 채택 |

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

## EXP-008 — WavLM Attention 3-seed ensemble

| 항목 | 결과 |
| --- | --- |
| encoder | `microsoft/wavlm-base-plus`, frozen |
| 입력 | 신고자(`speaker=1`) 구간, 16 kHz mono |
| 통화 처리 | 최대 8초, 짧은 음성은 1 chunk / 긴 음성은 앞·중간·뒤 3 chunks |
| chunk 특징 | WavLM masked mean pooling, 768차원 |
| 통합 방식 | chunk attention pooling + MLP classifier |
| ensemble | seed 42, 123, 2026의 확률 평균 |
| Validation Accuracy | **0.977198** |
| Validation Macro F1 | **0.977067** |
| 오류 | **83 / 3,640** |

- 3-chunk embedding cache는 학습 29,200개 및 Validation 3,640개에 대해 생성했다.
- 각 cache의 filename, label, sample index, chunk 수를 검증했다.
- 기존 mean pooling 방식보다 attention pooling 3-seed ensemble이 개선되어 이후 피처 결합의 기준선으로 사용한다.

## EXP-009 — Attention + F0

| 항목 | 결과 |
| --- | --- |
| 기본 표현 | EXP-008의 WavLM 3-chunk attention 표현 |
| 추가 특징 | F0 기반 6차원 특징 |
| 결합 | attention pooled embedding과 F0를 classifier 입력에서 결합 |
| 내부 holdout Accuracy / Macro F1 | 0.977397 / 0.977293 |
| 공식 Validation Accuracy / Macro F1 | 0.975824 / 0.975678 |
| 공식 Validation 오류 | 88 |

- F0 모델 단독은 EXP-008보다 낮았지만, 일부 다른 오답을 맞혀 앙상블 후보로 유지한다.
- 학습용 F0 cache와 Validation F0 cache는 모두 6차원이며 filename·label·sample index 순서를 검증했다.

## EXP-010 — WavLM frame statistics + F0

### 특징 설계

기존 768차원 embedding 평균만 사용하는 방식과 달리, 각 chunk의 WavLM 마지막 hidden state에서 frame-level 통계를 계산했다.

```text
mean: 768차원
std : 768차원
합계: 1,536차원
```

표준편차는 학습·추론에서 다음과 같이 동일하게 계산했다.

```python
torch.sqrt(frames.var(dim=0, correction=0) + 1e-5)
```

| 모델 | 공식 Validation Accuracy | Macro F1 | 오류 |
| --- | ---: | ---: | ---: |
| Stats Attention | 0.976099 | 0.975960 | 87 |
| Stats + F0 Attention | 0.976374 | 0.976241 | 86 |
| WavLM Attention 3-seed (EXP-008) | 0.977198 | 0.977067 | 83 |

- Validation statistics cache는 19개 shard, 총 3,640개를 생성했다.
- 원본 WavLM cache와 filename, label, chunk 수, sample index가 같은지 검증했다.
- Stats 계열 단독 성능은 기준 attention보다 낮지만 오답 보완성이 있으므로 앙상블에 사용한다.

### 오답 보완성

EXP-008의 83개 오답과 비교한 결과다.

| 보조 모델 | 기존 Attention 오답을 맞힌 수 | 기존 Attention 정답을 틀린 수 |
| --- | ---: | ---: |
| Stats Attention | 23 | 27 |
| Stats + F0 | 15 | 18 |

## EXP-011 — Attention+F0와 Stats+F0 ensemble

내부 holdout에서 후보 조합을 비교한 뒤, 가장 좋은 `Attention+F0`와 `Stats+F0`의 50:50 확률 평균을 선택했다.

```text
final_probs = 0.5 * attention_f0_probs + 0.5 * stats_f0_probs
```

| 조합 | 공식 Validation Accuracy | Macro F1 | 오류 |
| --- | ---: | ---: | ---: |
| Attention 3-seed + Attention F0 | 0.978022 | 0.977891 | 80 |
| **Attention F0 + Stats F0** | **0.978571** | **0.978446** | **78** |

### 현재 최고 성능

> **공식 Validation Accuracy: 97.8571% (3,562 / 3,640 정답, 오류 78개)**

- 목표 98.00%까지 정답 6개가 더 필요하며, 오류는 72개 이하여야 한다.
- 최종 최고 조합은 이전 최고 조합 대비 기존 오답 12개를 맞혔고, 기존 정답 10개를 새로 틀려 순 2개를 개선했다.

### 다음 실험 원칙

1. 현 `Attention F0 + Stats F0` 50:50을 고정 기준선으로 유지한다.
2. 공식 Validation의 78개 오답을 confidence, 성별, 발화 길이, F0 분포별로 분석한다.
3. ensemble weight와 M/F threshold는 공식 Validation이 아니라 내부 holdout 또는 OOF 예측으로만 탐색한다.
4. Attention+F0와 Stats+F0의 multi-seed ensemble을 우선 평가한다.
5. 최종 제출 전, 실제 WAV부터 CSV까지 실행하는 inference가 신고자 구간 추출, 3-chunk, F0, statistics, scaler, 50:50 ensemble을 동일하게 재현하는지 확인한다.
