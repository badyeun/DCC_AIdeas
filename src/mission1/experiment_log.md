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
