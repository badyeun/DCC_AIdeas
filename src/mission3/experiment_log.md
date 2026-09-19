# Mission 3 실험 기록

> Mission 1의 실험 기록과 같은 형식으로 실행 설정, 학습 이력, Validation 상세,
> 해석 및 재현 방법을 정리합니다. 아래 수치는 저장된 결과 JSON과 CSV에 근거합니다.
> 코드의 기본 설정과 실제 저장된 실행 설정을 구분하며, 남아 있지 않은 실행 기록은 임의로 채우지 않습니다.

## 기록 범위와 버전 대응

버전은 **실행 코드의 ver1~ver9**를 기준으로 구분합니다. 기존 [ver4 실행 안내](mission3_ver4/README.md)의 결과 대응표는 `mission3_all_in_one_ver2.ipynb`의 결과를 `results/mission3/ver1`에 연결합니다. 따라서 이 폴더의 점수는 EXP-002에 기록하고, ver1과 ver2의 독립된 두 번의 실행 결과로 중복 집계하지 않습니다. ver5는 Colab의 `ver5_efficient` 결과를 로컬 `results/mission3/ver5/`에 보관한 실행입니다.

| 실험 | 실행 버전 | 코드 / 노트북 | 확인된 결과 |
| --- | --- | --- | --- |
| EXP-001 | ver1 | `train.py`, `mission3_experiment.ipynb` | 모듈형 baseline 구현; 별도 실행 점수·출력 미보관 |
| EXP-002 | ver2 | `mission3_all_in_one_ver2.ipynb` | `results/mission3/ver1/`의 모델·CSV·metrics |
| EXP-003 | ver3 | `mission3_ver3.ipynb` | `results/mission3/ver3/`의 모델·CSV·15 epoch 이력 |
| EXP-004 | ver4 | `mission3_ver4.ipynb`, `mission3_ver4/*.py` | `results/mission3/ver4/`의 5-fold 이력·보정·Validation 결과 |
| EXP-005 | ver5 | `mission3_ver5.ipynb`, `mission3_ver5/*.py` | `results/mission3/ver5/`의 추가 학습 2 epoch·후보 보정·Validation 결과 |
| EXP-006 | ver6 | `mission3_ver6.ipynb`, `mission3_ver6/*.py` | `results/mission3/ver6/`의 A–F 비교·D 미세조정·보정 fallback·Validation 결과 |
| EXP-007 | ver7 | `mission3_ver7.ipynb`, `mission3_ver7/*.py` | `results/mission3/ver7/`의 KoELECTRA 전체 학습 3 epoch·보정 fallback·최종/후보 단독 Validation 결과 |
| EXP-008 | ver8 | `mission3_ver8.ipynb`, `mission3_ver8/*.py` | `results/mission3/ver8/`의 RoBERTa-large 후보 B 3 epoch 기록·보정 fallback·최종/후보 단독 CSV; 로컬 후보 체크포인트는 최종 기록과 불일치 |
| EXP-009 | ver9 | `mission3_ver9.ipynb`, `mission3_ver9/*.py` | `results/mission3/ver9/`의 DeBERTa 2-fold·각 3 epoch 학습·NB-LR·보정 fallback·최종/후보 단독 Validation 결과 |

- 입력은 양측 대화의 전사 `text`이며, 대상은 `고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란` 9개입니다. 각 증상의 유무를 독립적으로 예측합니다.
- 평가 지표는 **9개 증상의 Macro F1**입니다. 한 통화에 여러 증상이 있을 수 있으므로 아래 표의 증상별 Support 합계는 통화 수와 다릅니다.
- `Training 내부 holdout`, `stopping Macro AP`, `calibration cross-fit`, `공식 Validation`은 서로 다른 평가 범위입니다. AP는 임계값을 정하기 전 예측 순위를 평가하는 지표이며 F1과 직접 비교하지 않습니다.
- 2026-09-15에 저장된 ver1·ver3·ver4 CSV를 로컬 Validation JSON 3,640개와 다시 대조했습니다. 세 CSV 모두 파일명이 1:1 대응하고 중복·누락·추가 행이 없으며, 재계산한 증상별 F1과 Macro F1은 저장된 JSON과 일치했습니다. Precision·Recall·Support도 같은 CSV와 정답으로 계산했습니다. 모델 재학습이나 추론 재실행 결과는 아닙니다.
- ver5 CSV도 같은 Validation JSON 3,640개와 다시 대조했습니다. 파일명 중복·누락·추가 행이 없으며, 재계산한 Macro F1·증상별 F1·Support는 저장된 JSON과 일치했습니다. ver4와 비교해 예측이 달라진 통화는 496개, 증상별 이진 판정은 540개입니다.
- 현재 ver1~ver5 실험 노트북에는 셀 출력이 저장되어 있지 않습니다. 구체적인 CPU/GPU 기종, 저장되지 않은 학습 시간·손실은 확인할 수 없습니다. ver3·ver4·ver5의 학습 이력은 별도 JSON에 남아 있으며, ver5의 평가 폴더에는 `20260915_171404` 시각 문자열이 있습니다. 이는 전체 실행의 시작·종료 시각 기록은 아닙니다.
- 2026-09-17에 ver7의 최종·후보 단독 CSV 각각을 같은 Validation JSON 3,640개와 대조했습니다. 두 CSV 모두 파일명 중복·누락·추가 행이 없고, 재계산한 Precision·Recall·F1·Support가 저장된 JSON과 부동소수점 반올림 오차 범위에서 일치했습니다. ver7 노트북에는 셀 출력이 없으므로 실제 실행 설정·장치·학습 이력·시간은 결과 JSON을 근거로 기록합니다. 이번 검증은 저장된 CSV 재평가이며 GPU 재학습·재추론은 수행하지 않았습니다.
- 2026-09-17에 ver8의 최종·후보 단독 CSV도 Validation JSON 3,640개와 대조해 저장된 모든 증상별 지표와 Macro F1을 확인했습니다. 다만 로컬 `trials/B/best.ckpt`는 epoch 0이며 완료 기록의 epoch 3·SHA256과 다릅니다. CSV 재평가 성공과 저장된 가중치의 재현 가능성을 구분하며, 상세 내용은 EXP-008의 보관 상태에 기록합니다.
- 2026-09-19에 ver9 최종·후보 단독 CSV를 같은 Validation JSON 3,640개와 대조해 저장된 지표를 확인했습니다. 최종 선택은 기존 ver4이며 새 후보 비중은 0입니다. manifest가 참조하는 17개 파일의 SHA256과 실행 코드 24개의 해시도 일치했습니다. GPU 재학습·재추론은 수행하지 않았습니다.

## EXP-001 — ver1: 문자 TF-IDF + Logistic Regression 모듈형 baseline

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/train.py`, `common.py`, `inference.py`, `evaluate.py` |
| 실험 노트북 | `src/mission3/mission3_experiment.ipynb` |
| 실행 환경 | scikit-learn 기반 CPU 구현; 실제 실행 장치 기록 없음 |
| 학습 / 검증 데이터 | 서울 Training JSON 학습, Validation JSON 별도 평가 구조; ver1 단독 실행 개수 기록 없음 |
| 입력 특징 | 전체 발화 text를 줄바꿈으로 연결한 문자 TF-IDF |
| 특징 설정 — 코드 기본값 | `analyzer=char_wb`, 2–5그램, `max_features=100000`, `min_df=2`, `sublinear_tf=True`, float32 |
| 분류기 설정 — 코드 기본값 | 9개 One-vs-Rest Logistic Regression; `C=2.0`, L2, liblinear, `class_weight=balanced`, `max_iter=1000`, `n_jobs=-1` |
| 데이터 분할 — 코드 기본값 | Training 내부 80/20 무작위 분할, seed 42 |
| 임계값 선택 | Training 내부 holdout에서 증상별 F1 최대화; 0.10~0.90, 간격 0.05 |
| 최종 학습 | 내부 분할 확인 후 Training 전체로 TF-IDF와 분류기를 다시 학습 |
| 최종 선택 지표 | ver1 단독 실행 수치 미보관; 동일 계열 baseline의 실제 수치는 EXP-002 참고 |
| 체크포인트 기본 경로 | `saved_models/mission3_model.ckpt` — 코드 기본 경로이며 ver1 실행 산출물 확인 기록은 아님 |

### 학습 기록 및 해석

- `mission3_experiment.ipynb`는 데이터 확인과 `train.py` 호출을 묶은 실험용 파일입니다. 실험 메모와 출력이 비어 있어 ver1의 실행 날짜·점수·증상별 결과를 복원할 수 없습니다.
- Logistic Regression 구현은 신경망의 epoch별 학습 로그를 저장하지 않습니다. 내부 holdout에서 임계값을 정하고, 최종 모델은 전체 Training으로 다시 학습하는 구조입니다.
- 후속 ver2는 이 알고리즘을 Colab 단일 노트북으로 옮겨 데이터 탐색, 학습, 추론, 최종 평가와 결과 저장을 한 파일에서 실행하도록 구성했습니다.

### 재현 명령

이하 셸 명령은 `DCC_AIdeas` 프로젝트 루트 기준입니다. 아래는 **현재 코드 기본값을 명시한 재현용 명령**이며 과거 실행 명령이 보관되어 있다는 뜻은 아닙니다. 새 결과는 `outputs/mission3/`에 저장합니다.

```bash
python src/mission3/train.py \
  --label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --ckpt_path outputs/mission3/ver1/mission3_model.ckpt \
  --test_size 0.2 --seed 42 --max_features 100000 \
  --min_df 2 --c 2.0 --max_iter 1000 --n_jobs -1

python src/mission3/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path outputs/mission3/ver1/mission3_model.ckpt \
  --output outputs/mission3/ver1/mission3.csv

python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction outputs/mission3/ver1/mission3.csv
```

## EXP-002 — ver2: 문자 TF-IDF + Logistic Regression Colab 단일 노트북

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_all_in_one_ver2.ipynb` |
| 결과 보관 위치 | `results/mission3/ver1/` — 실행 코드의 버전명과 폴더명이 다름 |
| 실행 환경 | Colab 실행용 CPU 모델; 실제 장치·패키지 버전·실행 시간 미기록 |
| 학습 데이터 | 서울 Training JSON 29,200개 |
| 검증 데이터 | 서울 Validation JSON 3,640개 |
| Training 내부 분할 | 코드상 80/20, seed 42 → fit 23,360개 / holdout 5,840개 |
| 입력 특징 | 전체 대화 text의 `char_wb` TF-IDF, 2–5그램 |
| 학습 설정 — 노트북 기본값 | `max_features=100000`, `min_df=2`, `sublinear_tf=True`, `C=2.0`, L2, liblinear, balanced, `max_iter=1000` |
| ver1 대비 실행 변경 | `n_jobs=1`; 패키지 확인·Drive 연결·데이터 탐색·CSV 평가·metrics 저장을 단일 노트북으로 통합 |
| 최종 모델 | Training 29,200개 전체로 재학습한 TF-IDF + 증상별 Logistic Regression |
| Training holdout Macro F1 | 0.6009 |
| Validation Macro F1 | **0.6001** |
| 체크포인트 | `results/mission3/ver1/mission3_model.ckpt` (joblib 형식) |
| 결과 근거 | [metrics](../../results/mission3/ver1/mission3_metrics.json), [CSV](../../results/mission3/ver1/mission3.csv) |

### 학습 단계별 결과

| 단계 | 데이터 / 처리 | 저장된 결과 |
| --- | --- | --- |
| 내부 모델 학습·임계값 선택 | Training 80% fit / 20% holdout; 증상별 임계값 탐색 | Holdout Macro F1 0.6009 |
| 최종 재학습 | Training 전체로 vectorizer와 classifier 다시 fit | `mission3_model.ckpt` |
| 최종 평가 | 고정된 모델·임계값으로 Validation 3,640개 예측 | Validation Macro F1 0.6001 |

### Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7003 | 0.6136 | 0.6541 | 678 |
| 구토 | 0.5054 | 0.5251 | 0.5151 | 537 |
| 두통 | 0.4556 | 0.4385 | 0.4469 | 374 |
| 복통 | 0.8204 | 0.7470 | 0.7820 | 838 |
| 어지러움 | 0.6252 | 0.6357 | 0.6304 | 774 |
| 열상 | 0.8785 | 0.8121 | 0.8440 | 463 |
| 오심 | 0.2478 | 0.5291 | 0.3375 | 429 |
| 전신쇠약 | 0.5049 | 0.6359 | 0.5629 | 651 |
| 호흡곤란 | 0.6245 | 0.6321 | 0.6283 | 492 |
| **Macro avg** | **0.5958** | **0.6188** | **0.6001** | **5,236** |

### 증상별 판정 임계값

| Class | Threshold |
| --- | ---: |
| 고열 | 0.60 |
| 구토 | 0.55 |
| 두통 | 0.60 |
| 복통 | 0.55 |
| 어지러움 | 0.55 |
| 열상 | 0.70 |
| 오심 | 0.45 |
| 전신쇠약 | 0.55 |
| 호흡곤란 | 0.60 |

### 해석 및 다음 비교 기준

- 가장 높은 F1은 `열상` **0.8440**, 가장 낮은 F1은 `오심` **0.3375**입니다. 전체 비교 기준은 Validation Macro F1 **0.6001**입니다.
- Holdout 점수 0.6009는 같은 holdout에서 임계값을 선택한 뒤 계산한 값입니다. 최종 Validation 0.6001과 평가 범위가 다릅니다.
- epoch·학습 손실·총 학습 시간은 저장되어 있지 않습니다. ver1 대비 성능 향상을 입증하는 별도의 비교 실행도 없습니다.
- 다음 ver3에서는 문자 빈도 기반 특징을 문맥을 처리하는 KLUE-RoBERTa로 변경했습니다.

### 재현 방법 및 평가 명령

1. [mission3_all_in_one_ver2.ipynb](mission3_all_in_one_ver2.ipynb)를 Colab에 올리고 위에서부터 실행합니다.
2. `TL_서울_구급`, `VL_서울_구급` 폴더를 준비하고, 설정 셀의 `SEED=42`, `TEST_SIZE=0.20`, `N_JOBS=1`을 확인합니다.
3. 기본 출력은 `/content/drive/MyDrive/DCC_AIdeas/mission3_outputs`입니다. 기존 결과와 구분하려면 저장 경로 셀의 `OUTPUT_DIR`를 새 폴더로 지정합니다.
4. 다음 명령은 이미 보관된 CSV의 점수를 다시 계산합니다.

```bash
python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver1/mission3.csv
```

## EXP-003 — ver3: KLUE-RoBERTa-base + 9-label sigmoid head

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver3.ipynb` |
| 실행 환경 | CUDA 필수 구현; 실제 GPU 기종·패키지 버전 미보관 |
| 학습 / 검증 데이터 | 서울 Training JSON 29,200개 / Validation JSON 3,640개 |
| Training 내부 분할 | 코드상 80/20, seed 42 → fit 23,360개 / holdout 5,840개 |
| 모델 | `klue/roberta-base` + 첫 토큰 표현 + dropout + 단일 선형 9-label head |
| 입력 처리 | 전체 대화 text, 최대 384토큰으로 자름; 초과 부분은 입력에서 제외 |
| 체크포인트 확인 설정 | `max_length=384`, `dropout=0.1`, `seed=42`, `best_epoch=3`, `training_samples=29200` |
| 최적화 설정 — 노트북 기본값 | batch 16, inference batch 32, Adam, learning rate `3e-5`, epsilon `1e-7`, 최대 15 epoch |
| 손실·학습 보조 — 코드 | `BCEWithLogitsLoss`, 증상별 `pos_weight=negative/positive`, AMP, gradient clipping 1.0 |
| 최적 epoch 선택 | 각 epoch의 Training holdout에서 증상별 임계값 0.10~0.90을 0.05 간격으로 탐색하고 Macro F1 최대 epoch 선택 |
| 최적 epoch | 3 |
| 최고 Training holdout Macro F1 | 0.6444 |
| Validation Macro F1 | **0.6260** |
| 내부 15 epoch 기록 시간 합계 | 71.28분; 전체 Training 재학습·최종 추론 시간 제외 |
| 최종 모델 학습 | holdout 모델을 폐기하고 사전학습 모델에서 시작해 Training 전체를 3 epoch 재학습 |
| 체크포인트 / tokenizer | `results/mission3/ver3/mission3_ver3_model.ckpt`, `mission3_ver3_tokenizer/` |
| 결과 근거 | [history](../../results/mission3/ver3/mission3_ver3_history.json), [metrics](../../results/mission3/ver3/mission3_ver3_metrics.json), [CSV](../../results/mission3/ver3/mission3.csv) |

### Epoch별 결과

| Epoch | Train loss | Holdout macro F1 | 시간 (분) | 최고 갱신 | 최종 선택 |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 1 | 0.7740 | 0.6400 | 4.75 | ✓ |  |
| 2 | 0.6853 | 0.6411 | 4.76 | ✓ |  |
| 3 | 0.6391 | 0.6444 | 4.75 | ✓ | ✓ |
| 4 | 0.5836 | 0.6400 | 4.76 |  |  |
| 5 | 0.5256 | 0.6367 | 4.75 |  |  |
| 6 | 0.4646 | 0.6254 | 4.75 |  |  |
| 7 | 0.4111 | 0.6227 | 4.75 |  |  |
| 8 | 0.3537 | 0.6161 | 4.75 |  |  |
| 9 | 0.3076 | 0.6164 | 4.75 |  |  |
| 10 | 0.2674 | 0.6146 | 4.75 |  |  |
| 11 | 0.2357 | 0.6104 | 4.75 |  |  |
| 12 | 0.2036 | 0.6079 | 4.75 |  |  |
| 13 | 0.1820 | 0.6003 | 4.75 |  |  |
| 14 | 0.1637 | 0.6147 | 4.74 |  |  |
| 15 | 0.1436 | 0.6015 | 4.75 |  |  |

이 표는 Training 내부 80/20 분할 실험의 15 epoch 전부입니다. 최종 Training 전체 재학습 3 epoch의 손실·시간은 JSON에 저장되지 않았고 노트북 출력도 없어 별도의 수치를 적지 않습니다. 최종 체크포인트는 **내부 epoch 3의 가중치를 그대로 보관한 것이 아니라 전체 Training을 3 epoch 재학습한 모델**입니다.

### 최종 체크포인트 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.4840 | 0.7832 | 0.5983 | 678 |
| 구토 | 0.6411 | 0.5754 | 0.6065 | 537 |
| 두통 | 0.5793 | 0.4786 | 0.5242 | 374 |
| 복통 | 0.9255 | 0.6969 | 0.7951 | 838 |
| 어지러움 | 0.6699 | 0.6292 | 0.6489 | 774 |
| 열상 | 0.9002 | 0.8575 | 0.8783 | 463 |
| 오심 | 0.3293 | 0.5105 | 0.4004 | 429 |
| 전신쇠약 | 0.5310 | 0.5914 | 0.5596 | 651 |
| 호흡곤란 | 0.7543 | 0.5305 | 0.6229 | 492 |
| **Macro avg** | **0.6461** | **0.6281** | **0.6260** | **5,236** |

### 증상별 판정 임계값

| Class | Threshold |
| --- | ---: |
| 고열 | 0.50 |
| 구토 | 0.75 |
| 두통 | 0.75 |
| 복통 | 0.75 |
| 어지러움 | 0.70 |
| 열상 | 0.85 |
| 오심 | 0.65 |
| 전신쇠약 | 0.60 |
| 호흡곤란 | 0.90 |

### TF-IDF baseline과 비교

- EXP-002 대비 Validation Macro F1이 **+2.59%p** 높아졌습니다.
- `열상` F1 **0.8783**이 가장 높고, `오심` F1 **0.4004**가 가장 낮았습니다. 일부 증상은 개선되지 않았으며, `고열`은 0.6541에서 0.5983으로 낮아졌습니다.
- Train loss는 0.7740에서 0.1436으로 계속 낮아졌지만, holdout Macro F1은 epoch 3의 0.6444 이후 최종 epoch 15에서 0.6015로 하락했습니다. 기록상 과적합 징후가 있습니다.
- 최종 재학습으로 모델이 바뀐 뒤에도 내부 holdout에서 정한 임계값을 사용합니다. ver4는 별도 calibration 집합으로 실제 배포 ensemble의 출력에 맞춰 임계값을 선택하도록 변경했습니다.

### 재현 방법 및 평가 명령

1. [mission3_ver3.ipynb](mission3_ver3.ipynb)를 Colab CUDA 런타임에서 실행합니다.
2. 설정 셀에서 `MODEL_NAME="klue/roberta-base"`, `MAX_LENGTH=384`, `BATCH_SIZE=16`, `MAX_EPOCHS=15`, `LEARNING_RATE=3e-5`, `SEED=42`를 확인합니다.
3. Training 내부 15 epoch 비교 → 최적 epoch 선택 → 전체 Training 재학습 → 저장 모델 재로딩 → Validation 추론·평가 순서로 실행합니다. 아래 명령은 저장된 CSV만 재평가합니다.

```bash
python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver3/mission3.csv
```

## EXP-004 — ver4: KLUE-RoBERTa-large 5-fold + 문자·단어 TF-IDF 보정 ensemble

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver4.ipynb`, `src/mission3/mission3_ver4/*.py` |
| 실행 코드 사본 | `results/mission3/ver4/submission/`의 7개 실행 Python 파일은 현재 모듈 파일과 일치 |
| 실행 환경 | `run.json`의 `device=cuda`, `require_cuda=true`; 실제 GPU 기종 미보관 |
| 학습 데이터 | 서울 Training JSON 29,200개 |
| Training 내부 분리 | development 26,320개 / calibration 2,880개; 실제 개수는 `splits.json` 기준 |
| 검증 데이터 | 서울 Validation JSON 3,640개; 학습·epoch 선택·보정에 사용하지 않음 |
| 그룹 분할 | 공백을 정규화한 대화 text의 SHA256로 그룹화 후 다중 레이블 분할 |
| 신경망 | `klue/roberta-large` 5개 fold 모델의 예측 확률 평균 |
| 입력 / pooling | 최대 512토큰 window, 겹침 64토큰, 마지막 window까지 사용; window 내부 mean pooling 후 증상별 attention으로 통화 표현 생성 |
| 문자·단어 모델 | 문자 2–5그램 최대 120,000 + 단어 1–2그램 최대 40,000 TF-IDF; development에서만 fit |
| 최적 epoch | fold 1~5 순서로 **3 / 2 / 3 / 2 / 2** |
| epoch 선택 지표 | 별도 stopping 집합의 Macro AP; 최대 6 epoch, patience 2 |
| 기록된 학습 시간 합계 | 630.28분 (약 10시간 30분); 22개 epoch 기록 합계이며 sparse 학습·보정·최종 추론 등 제외 |
| Calibration fit Macro F1 | 0.6695 — 같은 데이터로 보정·평가 |
| Calibration cross-fit Macro F1 | 0.6597 — Training 내부 교차 보정 평가; 2,880개 통화 / 2,880개 그룹 |
| Validation Macro F1 | **0.6566** |
| 최종 체크포인트 | `results/mission3/ver4/mission3_ver4_model.ckpt` (구성·보정값 manifest) + `fold_1.ckpt`~`fold_5.ckpt` |
| 결과 근거 | [run](../../results/mission3/ver4/run.json), [splits](../../results/mission3/ver4/splits.json), [history](../../results/mission3/ver4/mission3_ver4_history.json), [metrics](../../results/mission3/ver4/mission3_ver4_metrics.json), [Validation](../../results/mission3/ver4/validation_metrics.json) |

### 개선 전 긴 대화 점검 기록

[ver4 실행 안내](mission3_ver4/README.md)에 남아 있는 사전 분석은 Training 29,200개를 저장된 ver3 tokenizer로 측정한 결과입니다. 아래 값은 해당 문서의 기록이며 이번 정리에서 토큰화를 다시 실행한 결과는 아닙니다.

| 점검 항목 | 문서에 기록된 결과 |
| --- | --- |
| 384토큰 초과 대화 | 3,292개 (11.27%) |
| 512토큰 초과 대화 | 788개 (2.70%) |
| 최대 대화 길이 | 특수 토큰 포함 873토큰 |

ver3의 384토큰 제한 뒤에 있는 증상 표현도 처리하기 위해 ver4에 전체 대화 window 처리가 도입됐습니다.

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| seed | 42; fold별 학습 seed는 `42 + fold 번호 × 1009`, 보정 cross-fit seed는 44 (코드 기준) |
| fold / calibration | 신경망 5-fold, `calibration_size=0.1`, calibration 내부 5-fold |
| 통화 batch / window batch | 2 / 2 |
| gradient accumulation / 유효 통화 batch | 16 / 최대 32 (마지막 누적 구간은 더 작을 수 있음) |
| 추론 batch | 8 |
| Optimizer | AdamW; encoder LR `1.5e-5`, head LR `1e-4`, layerwise LR decay 0.95 |
| 정규화 / scheduler | weight decay 0.01, dropout 0.2, warmup ratio 0.1 + linear decay |
| Loss | 통화별 BCE; `pos_weight_power=0.5`, 코드상 양성 가중치를 1~5 범위로 제한 |
| 메모리 / 안정화 | gradient checkpointing, AMP (BF16 지원 시 BF16, 아니면 FP16), gradient clipping 1.0 |
| TF-IDF + LR | `min_df=2`, `C=4.0`, balanced, liblinear, `max_iter=1500`; 문자 analyzer는 `char` |
| 보정 | 증상별 neural weight 후보 0 / 0.25 / 0.5 / 0.75 / 1.0; 혼합 확률의 F1을 최대화하는 임계값 선택 |

설정값은 [run.json](../../results/mission3/ver4/run.json)을 우선하며, optimizer·pooling·scheduler 등의 동작은 실행 코드 사본으로 확인했습니다. 현재 모듈의 `DEFAULT_CONFIG`는 저장된 `run.json.config`와 일치합니다.

| 패키지 | 저장된 버전 |
| --- | --- |
| torch | `2.11.0+cu128` |
| transformers | `4.57.1` |
| numpy | `2.1.3` |
| scikit-learn | `1.6.1` |
| iterative-stratification | `0.1.9` |

### Fold별 데이터와 최적 체크포인트

| Fold | Fit 통화 | Stopping 통화 | 실행 epoch | 최적 epoch | Best stopping macro AP | 시간 (분) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21,032 | 5,288 | 5 | 3 | 0.6829 | 145.65 |
| 2 | 21,053 | 5,267 | 4 | 2 | 0.6879 | 112.16 |
| 3 | 21,062 | 5,258 | 5 | 3 | 0.6914 | 145.22 |
| 4 | 21,095 | 5,225 | 4 | 2 | 0.6890 | 112.73 |
| 5 | 21,038 | 5,282 | 4 | 2 | 0.6869 | 114.52 |

`splits.json`의 파일명 기준으로 각 fold의 fit·stopping·calibration 간 중복이 없고, 공식 Validation 파일명도 포함되지 않습니다. 5개 stopping 집합은 development 26,320개를 한 번씩 나눠 가집니다. Calibration 2,880개는 모든 신경망과 TF-IDF 학습 및 epoch 선택에서 제외됩니다.

### Epoch별 결과

| Fold | Epoch | Train loss | Stopping macro AP | 양성 존재 클래스 | 시간 (분) | 최고 갱신 | 최종 선택 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 1 | 1 | 0.5202 | 0.6782 | 9 | 29.02 | ✓ |  |
| 1 | 2 | 0.4179 | 0.6813 | 9 | 29.04 | ✓ |  |
| 1 | 3 | 0.3802 | 0.6829 | 9 | 29.01 | ✓ | ✓ |
| 1 | 4 | 0.3431 | 0.6735 | 9 | 29.36 |  |  |
| 1 | 5 | 0.3047 | 0.6640 | 9 | 29.21 |  |  |
| 2 | 1 | 0.5162 | 0.6754 | 9 | 29.90 | ✓ |  |
| 2 | 2 | 0.4154 | 0.6879 | 9 | 27.16 | ✓ | ✓ |
| 2 | 3 | 0.3796 | 0.6862 | 9 | 27.19 |  |  |
| 2 | 4 | 0.3428 | 0.6829 | 9 | 27.91 |  |  |
| 3 | 1 | 0.5175 | 0.6851 | 9 | 28.84 | ✓ |  |
| 3 | 2 | 0.4178 | 0.6911 | 9 | 28.96 | ✓ |  |
| 3 | 3 | 0.3828 | 0.6914 | 9 | 28.93 | ✓ | ✓ |
| 3 | 4 | 0.3459 | 0.6856 | 9 | 29.48 |  |  |
| 3 | 5 | 0.3096 | 0.6756 | 9 | 29.02 |  |  |
| 4 | 1 | 0.5184 | 0.6795 | 9 | 28.86 | ✓ |  |
| 4 | 2 | 0.4174 | 0.6890 | 9 | 27.90 | ✓ | ✓ |
| 4 | 3 | 0.3820 | 0.6883 | 9 | 27.75 |  |  |
| 4 | 4 | 0.3457 | 0.6813 | 9 | 28.22 |  |  |
| 5 | 1 | 0.5224 | 0.6764 | 9 | 28.53 | ✓ |  |
| 5 | 2 | 0.4170 | 0.6869 | 9 | 28.67 | ✓ | ✓ |
| 5 | 3 | 0.3826 | 0.6854 | 9 | 28.65 |  |  |
| 5 | 4 | 0.3445 | 0.6785 | 9 | 28.67 |  |  |

22개 epoch 전부를 기록했습니다. Fold별 `fold_N_done.json`의 이력·최적 epoch·최고 AP는 통합 `mission3_ver4_history.json`과 일치합니다. 모든 epoch에서 9개 클래스에 양성 정답이 존재했습니다.

### Training 내부 교차 보정 결과

| Calibration fold | 보정값 학습 통화 | 평가 통화 |
| ---: | ---: | ---: |
| 1 | 2306 | 574 |
| 2 | 2309 | 571 |
| 3 | 2299 | 581 |
| 4 | 2303 | 577 |
| 5 | 2303 | 577 |

각 calibration 평가 통화의 정답은 해당 fold의 혼합 비율·임계값 선택에서 제외합니다. 교차 보정 평가가 끝난 뒤 배포용 보정값은 calibration 전체 2,880개로 결정하며, 기본 신경망·TF-IDF 모델은 다시 학습하지 않습니다.

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7363 | 0.6724 | 0.7029 | 519 |
| 구토 | 0.6698 | 0.4865 | 0.5636 | 446 |
| 두통 | 0.5458 | 0.5138 | 0.5293 | 290 |
| 복통 | 0.8960 | 0.7888 | 0.8390 | 677 |
| 어지러움 | 0.6806 | 0.6967 | 0.6885 | 633 |
| 열상 | 0.8830 | 0.9162 | 0.8993 | 346 |
| 오심 | 0.3154 | 0.5269 | 0.3946 | 334 |
| 전신쇠약 | 0.5376 | 0.6328 | 0.5813 | 531 |
| 호흡곤란 | 0.7442 | 0.7328 | 0.7385 | 393 |

Calibration fit Macro F1은 **0.6695**, cross-fit Macro F1은 **0.6597**입니다. 둘 다 Training 내부 결과이며 아래 공식 Validation 점수와 구분합니다.

### 배포용 증상별 혼합 비율과 임계값

| Class | Neural weight | TF-IDF weight | Threshold |
| --- | ---: | ---: | ---: |
| 고열 | 1.00 | 0.00 | 0.527061 |
| 구토 | 0.75 | 0.25 | 0.585741 |
| 두통 | 0.75 | 0.25 | 0.414659 |
| 복통 | 0.75 | 0.25 | 0.525984 |
| 어지러움 | 1.00 | 0.00 | 0.491587 |
| 열상 | 1.00 | 0.00 | 0.527157 |
| 오심 | 0.75 | 0.25 | 0.411040 |
| 전신쇠약 | 1.00 | 0.00 | 0.472925 |
| 호흡곤란 | 0.75 | 0.25 | 0.585972 |

혼합 확률은 `neural_weight × 5개 신경망 평균 확률 + (1 − neural_weight) × TF-IDF 확률`이며, 이 값이 해당 임계값 이상이면 증상을 출력합니다. 증상 최소 1개를 강제하지 않습니다. 표의 임계값은 읽기 편하도록 반올림했으므로 실제 추론은 체크포인트에 저장된 정밀도를 사용합니다.

### 최종 체크포인트 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7701 | 0.6372 | 0.6973 | 678 |
| 구토 | 0.6945 | 0.5419 | 0.6088 | 537 |
| 두통 | 0.5686 | 0.5321 | 0.5497 | 374 |
| 복통 | 0.8931 | 0.7673 | 0.8254 | 838 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 |
| 오심 | 0.3205 | 0.5431 | 0.4031 | 429 |
| 전신쇠약 | 0.5645 | 0.6252 | 0.5933 | 651 |
| 호흡곤란 | 0.7188 | 0.6443 | 0.6795 | 492 |
| **Macro avg** | **0.6699** | **0.6560** | **0.6566** | **5,236** |

### ver3 및 baseline과 비교

- Validation Macro F1은 ver3 대비 **+3.06%p**, TF-IDF baseline 대비 **+5.65%p**입니다.
- ver3 대비 9개 증상 모두 F1이 올랐습니다. `고열`은 0.5983 → 0.6973, `호흡곤란`은 0.6229 → 0.6795입니다.
- `열상` F1 **0.8936**이 가장 높고, `오심` F1 **0.4031**이 가장 낮습니다. 오심은 recall 0.5431에 비해 precision 0.3205가 낮아 오탐을 줄이는 것이 후속 비교 지점입니다.
- encoder 크기, 긴 대화 처리, ensemble, epoch 선택, 보정 방식이 함께 바뀌었습니다. 이번 기록만으로 각 변경의 개별 기여도를 분리할 수는 없습니다.
- 현재 보관된 ver1~ver4 결과 중 최고 Validation Macro F1은 **ver4의 0.6566**입니다. 기존 ver4 README의 “아직 미측정” 문구보다 실제 저장된 Validation 결과를 우선해 기록했습니다.

### 재현 명령

현재 모듈은 결과 폴더의 실행 코드 사본과 일치하며 기본 설정도 저장된 `run.json`과 같습니다. 아래 학습은 새 폴더에 결과를 생성합니다. `run.json` 자체는 설정이 `config` 아래에 들어 있으므로 `--config` 인자로 그대로 넘기지 않습니다.

```bash
python -m pip install -r src/mission3/mission3_ver4/requirements.txt

python src/mission3/mission3_ver4/train.py \
  --label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --output_dir outputs/mission3/ver4_reproduce

python src/mission3/mission3_ver4/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path outputs/mission3/ver4_reproduce/mission3_ver4_model.ckpt \
  --output outputs/mission3/ver4_reproduce/mission3.csv

python src/mission3/mission3_ver4/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction outputs/mission3/ver4_reproduce/mission3.csv \
  --output outputs/mission3/ver4_reproduce/validation_metrics.json
```

현재 저장된 CSV만 재평가하려면 다음 명령을 사용합니다.

```bash
python src/mission3/mission3_ver4/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver4/mission3.csv
```

### 결과 파일 보관

| 파일 | 역할 |
| --- | --- |
| `mission3_ver4_model.ckpt` | 모델 구성·상대 경로·보정값 manifest; 단독으로 추론 불가 |
| `fold_1.ckpt`~`fold_5.ckpt` | 각 fold의 최적 신경망 가중치 |
| `fold_1_done.json`~`fold_5_done.json` | fold별 완료 이력과 최적 epoch·AP |
| `tokenizer/`, `sparse.joblib` | 추론용 tokenizer와 문자·단어 TF-IDF 모델 |
| `calibration_predictions.npz` | Training calibration의 정답·neural/sparse 확률·파일명 |
| `run.json`, `splits.json` | 실행 설정·signature·실제 분할 파일 목록 |
| `mission3_ver4_history.json`, `mission3_ver4_metrics.json` | 통합 epoch 이력과 보정·평가 지표 |
| `mission3.csv`, `validation_metrics.json` | 최종 Validation 예측과 증상별 성능 |
| `submission/` | 실행 코드·requirements·안내·추론 노트북 사본 |

현재 로컬에는 ver3의 대형 모델과 ver4의 5개 fold 가중치가 보존되어 있습니다. 이 대형 파일들과 ver4 ZIP 5개는 Git 업로드에서 제외되어 있으므로, 다른 환경에서 추론하려면 로컬에 보관된 실제 가중치도 함께 옮겨야 합니다. 저장된 모델을 이용한 추론과 새 학습은 환경·패키지 버전에 따라 구분해 기록합니다.

## EXP-005 — ver5: ver4 기반 증상별 token attention 추가 학습 + 확률·logit 보정 비교

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver5.ipynb`, `src/mission3/mission3_ver5/*.py` |
| 실행 코드 사본 | `results/mission3/ver5/submission/`의 실행 코드·requirements·안내 문서는 현재 ver5 소스와 일치 |
| 결과 보관 위치 | 로컬 `results/mission3/ver5/`; 실행 시 Drive 경로는 `/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver5_efficient` |
| 평가 결과 폴더 | `predictions/20260915_171404_d6d25c/` |
| 실행 환경 | 저장된 설정은 `device=cuda`, `require_cuda=true`; 실제 GPU 기종·총 CU 기록 없음 |
| 학습 데이터 | 서울 Training JSON 29,200개; 기존 ver4 split과 모델을 재사용 |
| Training 내부 분리 | fold 1 fit 21,032개 / stop 5,288개 / calibration 2,880개 |
| 검증 데이터 | 서울 Validation JSON 3,640개 전체; 학습·epoch 선택·보정에 사용하지 않음 |
| 시작 가중치 | ver4 `fold_1.ckpt`의 최적 epoch 3 가중치; ver5의 epoch 0은 이 모델에서 시작한 추가 학습 전 상태 |
| 추가 구조 | 기존 global/window head에 9개 증상별 token attention 출력 추가; 추가 출력 가중치와 bias는 0으로 초기화 |
| 입력 처리 | 양측 전사 text만 사용, 최대 512토큰 window / 겹침 64토큰; 모든 샘플에 동일한 전처리 적용, WAV 미사용 |
| 추가 학습 | encoder 고정 1 epoch + 상위 4개 층과 head 미세조정 1 epoch; 기존 5개 모델은 재학습하지 않음 |
| 완료 / 최적 epoch | **2 epoch 완료 / 최적 epoch 0**; 중단 후 재개한 실행은 아님 |
| 최적 epoch 선택 지표 | 별도 stop 집합의 Macro AP; 최고값 **0.682989** |
| Training 내부 최종 선택 | `candidate_mixture`, `blend_space=logit`; 기존 보정으로 되돌리는 fallback은 발생하지 않음 |
| Calibration cross-fit Macro F1 | 기존 보정 **0.659671** / 선택된 후보 **0.659779** |
| Calibration fit Macro F1 | **0.670898** — 같은 calibration 전체로 보정·평가한 값 |
| Validation Macro F1 | **0.656718** (원본 값 `0.6567177583831921`) |
| 기록된 시간 합계 | 초기 stop 평가 포함 **12.70분**; 추가 학습 2 epoch의 학습·stop 평가 합계는 **11.82분** |
| 모델 보관 방식 | `bundle_base=false`, 최종 manifest의 `portable=false`; 기존 ver4 가중치를 결과 폴더에 중복 복사하지 않음 |
| 결과 근거 | [run](../../results/mission3/ver5/run.json), [완료 이력](../../results/mission3/ver5/candidate_done.json), [metrics](../../results/mission3/ver5/mission3_ver5_metrics.json), [Validation](../../results/mission3/ver5/predictions/20260915_171404_d6d25c/validation_metrics.json), [CSV](../../results/mission3/ver5/predictions/20260915_171404_d6d25c/mission3.csv) |

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| seed / 시작 fold | 42 / 1; calibration cross-fit seed 44, 5-fold |
| epoch / patience | warmup 1, finetune 1, `finetune_top_layers=4`, patience 2 |
| 통화 batch / window batch | 4 / 4 |
| gradient accumulation / 유효 통화 batch | 8 / 최대 32 (마지막 누적 구간은 더 작을 수 있음) |
| 추론 batch | 8 |
| Optimizer / 학습률 | AdamW; encoder `5e-6`, fine-tune head `1e-4`, warmup head `3e-4` |
| 정규화 / scheduler | layerwise LR decay 0.95, weight decay 0.01, dropout 0.2, warmup ratio 0.1 + linear decay |
| Loss / 메모리 | `pos_weight_power=0.5`, gradient checkpointing, AMP |
| 증상별 attention 차원 | 256 |
| 전체 매개변수 | 339,040,018개 |
| warmup 학습 매개변수 | 2,383,634개; encoder 학습 매개변수 0개 |
| fine-tune 학습 매개변수 | 52,768,530개 (전체의 약 15.6%); 그중 encoder 50,384,896개 |
| 패키지 | torch `2.11.0+cu128`, transformers `4.57.1`, numpy `2.1.3`, scikit-learn `1.6.1`, iterative-stratification `0.1.9` |

설정은 [run.json](../../results/mission3/ver5/run.json)의 실제 `config`를 기준으로 기록했습니다. 자동 batch 선택은 GPU 메모리에 영향을 받으므로 batch 4라는 값만으로 GPU 기종을 단정하지 않습니다. 매개변수 갱신 범위를 줄여도 고정된 하위 층의 순전파는 필요하므로 시간·CU 절감률을 84.4%로 해석하지 않습니다.

### Epoch별 결과

| Epoch | 단계 | Train loss | Stop macro AP | 시간 (분) | 학습 매개변수 | 최종 선택 |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 0 | 추가 학습 전 baseline 평가 | — | **0.682989** | 0.88 | — | ✓ |
| 1 | encoder 고정, head 학습 | 0.331341 | 0.679172 | 4.80 | 2,383,634 |  |
| 2 | 상위 4개 층 + head 미세조정 | 0.320017 | 0.674949 | 7.02 | 52,768,530 |  |

- 두 epoch 모두 정상 완료했지만 stop AP가 초기값을 넘지 못해 `best_epoch=0`이 유지됐습니다. `candidate_best.ckpt`의 실제 epoch도 0, `candidate_best_last.ckpt`는 완료 epoch 2로 확인했습니다.
- 최적 체크포인트의 `token_evidence_weights`와 `token_evidence_bias`는 모두 0입니다. 마지막 epoch 체크포인트에는 갱신된 값이 있으므로 학습 자체는 수행됐으며, 그 갱신 결과가 최적 후보로 채택되지 않은 실행입니다.
- train loss는 낮아졌지만 stop AP도 낮아졌습니다. 이번 추가 학습의 일반화 성능 개선은 확인되지 않았으며, 원인을 이 기록만으로 특정하지 않습니다.
- 시간은 저장된 `history[].minutes`의 합계입니다. ZIP 복원, 모델 초기 로딩·무결성 검사, 체크포인트 저장, calibration 추론·보정, 최종 Validation 추론 등은 포함되지 않습니다. 전체 소요 시간이나 총 CU로 사용할 수 없습니다.

### Training 내부 교차 보정 결과

| 비교 구성 | Cross-fit Macro F1 | 선택 |
| --- | ---: | :---: |
| 기존 ver4 확률 혼합 보정 | 0.659671 |  |
| ver5 후보를 포함한 확률 평균 (`raw`) | 0.657675 |  |
| ver5 후보를 포함한 logit 평균 (`logit`) | **0.659779** | ✓ |

calibration 2,880개 통화 / 2,880개 그룹을 5-fold로 나누며, 각 fold의 fit/score 개수는 EXP-004와 같습니다. 혼합 비율과 임계값은 각 fold의 fit 부분에서만 학습하고 제외한 부분에서 평가합니다. 후보 중 최고 점수가 같은 분할의 기존 보정 점수를 엄격히 넘을 때만 후보를 선택합니다.

이번 선택 차이는 **+0.0001077493784649 F1 (약 +0.0108%p)**입니다. `candidate_mixture`가 선택됐지만 개선 폭은 작습니다. 절차 선택에 같은 cross-fit 결과를 사용했으므로 이 수치를 비편향 성능 보장이나 공식 Validation 점수로 해석하지 않습니다. 선택 후 calibration 전체에 다시 맞춘 F1 **0.670898**도 별도 지표입니다.

### 배포용 증상별 혼합 비율과 임계값

| Class | 기존 신경망 비율 α | 후보 비율 β | Threshold |
| --- | ---: | ---: | ---: |
| 고열 | 1.00 | 0.50 | 0.565676614 |
| 구토 | 0.75 | 0.00 | 0.576455745 |
| 두통 | 0.75 | 0.00 | 0.409890797 |
| 복통 | 0.75 | 0.00 | 0.518373384 |
| 어지러움 | 1.00 | 0.00 | 0.491587323 |
| 열상 | 1.00 | 0.00 | 0.527157467 |
| 오심 | 0.75 | 0.50 | 0.385722382 |
| 전신쇠약 | 0.75 | 0.50 | 0.369644029 |
| 호흡곤란 | 0.75 | 0.50 | 0.449450620 |

선택된 `logit` 방식은 먼저 ver4 5개 신경망의 평균 확률, TF-IDF 확률, 후보 확률을 각각 logit으로 변환합니다. 그 후 `z = (1 − β) × [α × logit(p_ver4) + (1 − α) × logit(p_sparse)] + β × logit(p_candidate)`를 계산하고 `sigmoid(z)`에 임계값을 적용합니다. 확률은 logit 변환 전에 `[1e-6, 1 − 1e-6]`로 제한합니다. 표는 반올림한 값이며 실제 추론에는 체크포인트의 전체 정밀도를 사용합니다.

이번 후보는 최적 epoch 0이므로 ver4 fold 1에서 시작한 학습 전 모델입니다. 따라서 이번 결과의 변화를 추가 학습으로 얻은 새로운 증상 표현의 효과로 설명할 수 없습니다. 기존 예측의 혼합 방식·비율·임계값 변경을 함께 고려해야 합니다.

### 최종 체크포인트 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7917 | 0.6165 | 0.6932 | 678 |
| 구토 | 0.6862 | 0.5456 | 0.6079 | 537 |
| 두통 | 0.5731 | 0.5241 | 0.5475 | 374 |
| 복통 | 0.8917 | 0.7661 | 0.8241 | 838 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 |
| 오심 | 0.3105 | 0.6014 | 0.4095 | 429 |
| 전신쇠약 | 0.5101 | 0.7358 | 0.6025 | 651 |
| 호흡곤란 | 0.6387 | 0.7114 | 0.6731 | 492 |
| **Macro avg** | **0.6556** | **0.6793** | **0.6567** | **5,236** |

### ver4와 비교 및 해석

| Class | ver4 F1 | ver5 F1 | 차이 (%p) |
| --- | ---: | ---: | ---: |
| 고열 | 0.697337 | 0.693201 | −0.4136 |
| 구토 | 0.608787 | 0.607884 | −0.0903 |
| 두통 | 0.549724 | 0.547486 | −0.2238 |
| 복통 | 0.825417 | 0.824134 | −0.1284 |
| 어지러움 | 0.659022 | 0.659022 | 0.0000 |
| 열상 | 0.893617 | 0.893617 | 0.0000 |
| 오심 | 0.403114 | 0.409524 | +0.6410 |
| 전신쇠약 | 0.593294 | 0.602516 | +0.9221 |
| 호흡곤란 | 0.679528 | 0.673077 | −0.6451 |
| **Macro avg** | **0.656649** | **0.656718** | **+0.0069** |

- 정확한 Validation Macro F1 차이는 **+0.0000688103443078**입니다. 전체 성능은 ver4와 거의 같으며, 실질적인 향상이나 다른 Test 데이터에서의 개선을 입증한 결과는 아닙니다.
- 2개 증상(오심·전신쇠약)은 개선, 5개는 하락, 2개는 동일했습니다. 두 CSV에서 496개 통화 / 540개 증상 판정이 달라졌으므로 예측 파일이 단순 복사된 결과도 아닙니다.
- 전신쇠약의 recall은 0.6252 → 0.7358로 올랐지만 precision은 0.5645 → 0.5101로 낮아졌습니다. 호흡곤란도 recall 상승과 함께 precision이 낮아져 최종 F1은 하락했습니다. 증상을 더 많이 검출하는 변화와 오탐 증가를 함께 봐야 합니다.
- 이번 2 epoch 추가 학습의 갱신 가중치는 최종 모델에 반영되지 않았습니다. 추가 학습 비용에 상응하는 성능 개선은 확인되지 않았으며, 같은 설정으로 epoch만 늘리면 개선된다는 근거도 없습니다.
- Validation은 마지막 평가에만 사용했습니다. 위 증상별 차이는 결과 해석이며, 이 정답으로 저장된 혼합 비율·임계값을 수정하지 않았습니다.

### 재현 방법 및 평가 명령

1. [mission3_ver5.ipynb](mission3_ver5.ipynb)에는 실행 코드가 포함되어 있습니다. 입력 ZIP을 준비하면 ver4 필수 모델과 Training·Validation JSON을 `/content/mission3_ver5_inputs`에 복원하고 새 결과를 Drive에 저장합니다.
2. 같은 설정의 새 실험은 별도 출력 폴더를 사용합니다. 아래는 저장된 `run.json`의 `config`만 추출해 batch 4 / accumulation 8까지 명시적으로 재사용하는 명령입니다. 실행 환경에 따라 점수·시간이 완전히 같다는 보장은 없습니다.

```bash
python -m pip install -r results/mission3/ver5/submission/requirements.txt

python - <<'PY'
import json
from pathlib import Path

run = json.loads(Path("results/mission3/ver5/run.json").read_text(encoding="utf-8"))
path = Path("outputs/mission3/ver5_reproduce_config.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(run["config"], ensure_ascii=False, indent=2), encoding="utf-8")
PY

python results/mission3/ver5/submission/train.py \
  --train_label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --base_dir results/mission3/ver4 \
  --output_dir outputs/mission3/ver5_reproduce \
  --config_json outputs/mission3/ver5_reproduce_config.json
```

현재 저장된 모델로 추론만 다시 할 때는 다음 명령을 사용합니다. 이번 manifest는 Colab의 기존 ver4 절대 경로를 참조하므로, 로컬에서는 `--base_dir`로 같은 ver4 모델의 현재 위치를 지정해야 합니다. 새 예측은 기존 결과와 다른 폴더에 저장합니다.

```bash
python results/mission3/ver5/submission/inference.py \
  --audio_dir "" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver5/mission3_ver5_model.ckpt \
  --base_dir results/mission3/ver4 \
  --output outputs/mission3/ver5_saved_model/mission3.csv \
  --batch_size 8 --evaluate
```

모델 추론 없이 저장된 CSV만 재평가하려면 다음 명령을 사용합니다.

```bash
python results/mission3/ver5/submission/ver4_runtime/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver5/predictions/20260915_171404_d6d25c/mission3.csv
```

### 결과 파일 보관

| 파일 | 역할 |
| --- | --- |
| `mission3_ver5_model.ckpt` | ver5 구성·후보 경로·기존 ver4 경로·혼합 비율·임계값을 담은 manifest; 단독으로 추론 불가 |
| `candidate_best.ckpt` | 최종 후보 가중치; 이번 실행에서는 추가 학습 전 epoch 0 |
| `candidate_best_last.ckpt` | 완료 epoch 2의 가중치·optimizer·scheduler·RNG 재개 상태; 최종 예측에 사용한 가중치와 다름 |
| `candidate_done.json` | 완료 epoch·최적 epoch·stop AP·전체 학습 이력 |
| `candidate_calibration.npz` | 선택된 후보의 Training calibration 예측 |
| `run.json`, `splits.json` | 실제 설정·signature·원본 가중치 해시·fit/stop/calibration 목록 |
| `mission3_ver5_metrics.json` | 학습 이력·확률/logit 후보 비교·최종 선택·보정값 |
| `predictions/20260915_171404_d6d25c/` | 공식 Validation 3,640개 CSV와 상세 평가 JSON |
| `submission/` | ver5 실행 코드, ver4 공통 실행 코드, requirements, 안내·논문 검토 문서 |

`bundle_base=false`이므로 이 결과 폴더에는 기존 ver4 가중치 사본이 없습니다. 원본 입력 ZIP 또는 동일한 `results/mission3/ver4/` 모델 묶음을 함께 보관해야 합니다. 대용량 체크포인트를 포함한 로컬 보관 사실과 Git 업로드 여부는 별개입니다.

## EXP-006 — ver6: 동결 특징 위 A–F 비교 + 선택 분류기 공동 미세조정

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver6.ipynb`, `mission3_ver6/*.py`; 결과의 `submission/` 24개 파일과 현재 소스 일치 |
| 결과 위치 | `results/mission3/ver6/`; Drive `DCC_AIdeas/results/mission3/ver6` |
| 실행 환경 | NVIDIA L4, BF16; torch `2.11.0+cu128`, CUDA `12.8`, transformers `4.57.1`, numpy `2.1.3`, scikit-learn `1.6.1` |
| 입력 | 기존 `mission3_ver5_inputs.zip` 바로가기; 서울 Training 29,200개 / Validation 3,640개, WAV 미사용 |
| 분할 | 원래 ver4 fold 1 fit 21,032 / stop 5,288 / 별도 calibration 2,880 |
| 시작 encoder | ver4 fold 1 최적 epoch 3; A–F에서 동일한 동결 token 특징 재사용 |
| 특징 캐시 | fit·stop 26,320개만 Colab 로컬 float16 캐시; 정답은 저장하지 않음; A–C는 CLS·mean 요약 재사용 |
| 후보·epoch 선택 | stop의 그룹 분리 3-fold threshold 교차평가 Macro F1 우선, Macro AP는 동점 보조 |
| 실제 학습 | 작은 분류기 18 epoch 합계 + 선택된 D와 encoder 상위 2개 층 미세조정 1 epoch |
| 선택 분류기 | D: 직접 label-token attention + weighted BCE, head 최적 epoch 2 |
| encoder 미세조정 최적 epoch | 0: D의 학습된 분류기를 유지하고 추가 encoder 미세조정만 미채택 |
| 최종 구성 선택 | **ver4_baseline**; 새 후보 가중치 9개 모두 0 |
| Calibration cross-fit | 기존 0.659670866 / 새 혼합 후보 0.662116706 / 최종 선택 0.659670866 |
| Calibration fit | 0.669459793; calibration 전체에 보정하고 같은 표본에서 평가한 값 |
| Validation Macro F1 | **0.656669718** (원본 `0.6566697183939796`) |
| 전체 기록 시간 | 24.67분; 패키지 설치 제외, CU 사용량 미기록 |
| 보관 방식 | `bundle_base=false`, `portable=false`; 원래 ver4 모델과 입력 ZIP도 보관 필요 |
| 근거 | [metrics](../../results/mission3/ver6/mission3_ver6_metrics.json), [run](../../results/mission3/ver6/run.json), [완료 이력](../../results/mission3/ver6/candidate_done.json), [시간](../../results/mission3/ver6/notebook_runtime_20260915_193433_ae85f5.json), [Validation](../../results/mission3/ver6/predictions/20260915_193433_ae85f5/validation_metrics.json), [CSV](../../results/mission3/ver6/predictions/20260915_193433_ae85f5/mission3.csv) |

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| 분류기 후보 | A: 기존 head+ASL / B: 기존 head+DB Loss / C: 기존 head+ASL+0.2 soft Macro F1 / D: 직접 LAAT+weighted BCE / E: D+head R-Drop / F: E+fit 전용 증상 관계 |
| 분류기 학습 | 최대 4 epoch, patience 2, 실제 batch 32, LR `3e-4`, dropout 0.2, attention 차원 256 |
| encoder 미세조정 | 최대 1 epoch, 상위 2층, encoder LR `2e-6`, head LR `5e-5`, layerwise decay 0.95 |
| 미세조정 batch | 통화 4 / window 4 / accumulation 8 / 유효 batch 최대 32 |
| 정규화·메모리 | AdamW, weight decay 0.01, warmup 비율 0.1, gradient checkpointing, BF16 |
| 적용 범위 | R-Drop과 soft-F1 보조항은 작은 분류기 단계에서만 사용; encoder 미세조정에는 미적용 |
| seed / 교차평가 | config seed 42, 모델 학습·stop 분리 seed 5045; 별도 calibration seed 44, 5-fold |
| 최종 채택 기준 | 교차평가 F1이 기존 대비 최소 +0.002이고 5개 fold 중 최소 3개에서 엄격히 개선 |

### 후보별 결과

| 후보 | 최적 head epoch | 완료 epoch | 최고 stop Macro F1 | 최고 epoch의 AP |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 2 | 0.639876008 | 0.682962664 |
| B | 0 | 2 | 0.639876008 | 0.682962664 |
| C | 0 | 2 | 0.639876008 | 0.682962664 |
| D | 2 | 4 | 0.640299886 | 0.683637699 |
| E | 2 | 4 | 0.639015120 | 0.684229815 |
| F | 2 | 4 | 0.638881265 | 0.684258080 |

### 모든 분류기 epoch 이력

| 후보 | Epoch | 단계 | Train loss | Stop Macro F1 | Stop Macro AP | 시간 (분) | 최적 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| A | 0 | 초기 평가 | — | 0.639876008 | 0.682962664 | 6.2641 | ✓ |
| A | 1 | 분류기 학습 | 0.113122 | 0.634763687 | 0.675523638 | 0.2394 |  |
| A | 2 | 분류기 학습 | 0.108234 | 0.631472004 | 0.676271040 | 0.0728 |  |
| B | 0 | 초기 평가 | — | 0.639876008 | 0.682962664 | 0.0078 | ✓ |
| B | 1 | 분류기 학습 | 0.087908 | 0.630218911 | 0.665791945 | 0.0745 |  |
| B | 2 | 분류기 학습 | 0.075750 | 0.627353156 | 0.671758021 | 0.0740 |  |
| C | 0 | 초기 평가 | — | 0.639876008 | 0.682962664 | 0.0078 | ✓ |
| C | 1 | 분류기 학습 | 0.206972 | 0.633923361 | 0.674946175 | 0.0852 |  |
| C | 2 | 분류기 학습 | 0.201632 | 0.628295130 | 0.676055063 | 0.0834 |  |
| D | 0 | 초기 평가 | — | 0.287748057 | 0.158932043 | 0.0705 |  |
| D | 1 | 분류기 학습 | 0.360278 | 0.638842574 | 0.684152062 | 0.3985 |  |
| D | 2 | 분류기 학습 | 0.326457 | 0.640299886 | 0.683637699 | 0.3956 | ✓ |
| D | 3 | 분류기 학습 | 0.321497 | 0.638168869 | 0.680568393 | 0.3953 |  |
| D | 4 | 분류기 학습 | 0.317308 | 0.636489163 | 0.679257729 | 0.3931 |  |
| E | 0 | 초기 평가 | — | 0.287748057 | 0.158932043 | 0.0719 |  |
| E | 1 | 분류기 학습 | 0.361168 | 0.638049411 | 0.684258711 | 0.4828 |  |
| E | 2 | 분류기 학습 | 0.327391 | 0.639015120 | 0.684229815 | 0.4836 | ✓ |
| E | 3 | 분류기 학습 | 0.322517 | 0.636919084 | 0.680867269 | 0.4805 |  |
| E | 4 | 분류기 학습 | 0.318712 | 0.635940373 | 0.679420240 | 0.4817 |  |
| F | 0 | 초기 평가 | — | 0.287416653 | 0.158579773 | 0.0721 |  |
| F | 1 | 분류기 학습 | 0.361088 | 0.638786774 | 0.684267295 | 0.4901 |  |
| F | 2 | 분류기 학습 | 0.327379 | 0.638881265 | 0.684258080 | 0.4896 | ✓ |
| F | 3 | 분류기 학습 | 0.322499 | 0.637115193 | 0.680806529 | 0.4907 |  |
| F | 4 | 분류기 학습 | 0.318690 | 0.636641755 | 0.679390073 | 0.4885 |  |

A–C는 두 번의 학습 후에도 초기 점수를 넘지 못해 epoch 0을 유지했습니다. D–F는 epoch 2에서 최고 점수를 기록했습니다. 후보 선택은 F1 우선이므로, AP만 가장 높은 후보를 고르지 않습니다.

### 선택 D 모델의 encoder 미세조정

| Epoch | 단계 | Train loss | Stop Macro F1 | Stop Macro AP | 시간 (분) | 최적 |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 0 | 학습된 D head의 실제 encoder 평가 | — | 0.640452236 | 0.683639541 | 0.8964 | ✓ |
| 1 | 상위 2층 + D head 학습 | 0.320362 | 0.640056399 | 0.682463256 | 5.6949 |  |

- `candidate_best.ckpt`의 head 5개 텐서는 D의 epoch 2 `best_state`와 정확히 같고, encoder 391개 텐서는 원래 ver4 fold 1과 같음을 결과 분석에서 확인했습니다.
- 마지막 미세조정에서는 상위 encoder 2개 층 32개 텐서와 head 5개가 실제 변경됐습니다. 따라서 학습 미실행이 아니라 학습 결과를 선택하지 않은 경우입니다.
- cached head F1 0.640299886과 실제 encoder 평가 0.640452236은 계산 경로가 다릅니다. 코드가 float16 캐시 반올림의 영향을 피하려고 미세조정 전후를 실제 encoder 실행으로 다시 비교합니다.
- 최종 `best_epoch=0`을 “새 분류기도 학습되지 않았다”로 해석하면 안 됩니다. 다만 최종 ensemble은 별도 보정 선택에서 이 D 후보를 제외했습니다.

### 교차 보정과 fallback

| 구성 | Cross-fit Macro F1 | 최종 선택 |
| --- | ---: | :---: |
| 기존 ver4 | 0.659670866 | ✓ |
| 새 후보 포함 raw 혼합 | 0.655891126 | |
| 새 후보 포함 logit 혼합 | 0.662116706 | |

전체 개선 폭은 **+0.002445840 F1 (+0.2446%p)**로 최소 +0.002 기준은 통과했습니다. 그러나 개선된 fold가 **2/5**라 최소 3개 조건을 통과하지 못했습니다. `fallback_reason=candidate_gain_not_consistent_across_folds`입니다.

| Fold | 후보 logit − 기존 F1 | 차이 (%p) | 개선 |
| ---: | ---: | ---: | :---: |
| 1 | -0.004647376 | -0.4647 |  |
| 2 | +0.006019591 | +0.6020 | ✓ |
| 3 | +0.016046253 | +1.6046 | ✓ |
| 4 | -0.003348824 | -0.3349 |  |
| 5 | -0.001795613 | -0.1796 |  |

저장된 calibration 확률과 실제 Training 대화 그룹으로 CPU 교차 보정을 재계산했고, 점수·fold별 차이·선택 결과가 저장 기록과 정확히 일치했습니다. 이 지표는 Training 내부 선택용이며 공식 Validation 또는 비편향 Test 성능 보장이 아닙니다.

### 최종 혼합 비율과 임계값

| Class | 기존 신경망 α | 새 후보 β | Threshold |
| --- | ---: | ---: | ---: |
| 고열 | 1.00 | 0.00 | 0.527061400 |
| 구토 | 0.75 | 0.00 | 0.585740679 |
| 두통 | 0.75 | 0.00 | 0.414659252 |
| 복통 | 0.75 | 0.00 | 0.525984191 |
| 어지러움 | 1.00 | 0.00 | 0.491587323 |
| 열상 | 1.00 | 0.00 | 0.527157467 |
| 오심 | 0.75 | 0.00 | 0.411039874 |
| 전신쇠약 | 1.00 | 0.00 | 0.472925211 |
| 호흡곤란 | 0.75 | 0.00 | 0.585971753 |

기존 ver4 manifest와 α·threshold 9개씩 모두 정확히 같습니다. 새 후보 β는 전부 0이며, 최종 추론 코드가 후보 모델 실행을 생략합니다. 따라서 아래 Validation CSV는 D 후보 자체의 Validation 평가가 아닙니다.

### 최종 Validation 상세

| Class | Precision | Recall | F1-score | Support | ver4 대비 (%p) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 고열 | 0.7705 | 0.6386 | 0.6984 | 678 | +0.1051 |
| 구토 | 0.6945 | 0.5419 | 0.6088 | 537 | +0.0000 |
| 두통 | 0.5686 | 0.5321 | 0.5497 | 374 | +0.0000 |
| 복통 | 0.8931 | 0.7673 | 0.8254 | 838 | +0.0000 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 | +0.0000 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 | +0.0000 |
| 오심 | 0.3205 | 0.5431 | 0.4031 | 429 | +0.0000 |
| 전신쇠약 | 0.5629 | 0.6252 | 0.5924 | 651 | -0.0864 |
| 호흡곤란 | 0.7188 | 0.6443 | 0.6795 | 492 | +0.0000 |
| **Macro avg** | **0.6697** | **0.6561** | **0.656670** | **5236** | **+0.0021** |

- 실제 Validation JSON 3,640개와 CSV를 대조했습니다. 누락·중복·추가 파일이 없고 재계산 Macro F1이 `0.6566697183939796`로 일치했습니다.
- ver4 대비 **+0.000020770355 F1**, ver5 대비 **-0.000048039989 F1**입니다. 최종 성능은 사실상 같습니다.
- ver4와 3,637개 통화의 예측이 같고 3개만 다릅니다. 고열 정답 1개 추가, 전신쇠약 오답 2개 추가입니다. 새 후보 가중치는 0이므로 이 미세 차이를 새 분류기 학습 효과로 설명할 수 없습니다. 수치 계산·추론 경로 차이가 원인인지는 GPU 재추론 없이 확정하지 않았습니다.
- 오심 F1 0.4031, 두통 0.5497, 전신쇠약 0.5924가 낮습니다. D 자체의 공식 Validation 점수는 이번 산출물에 없으므로 별도 평가 없이 D가 더 좋거나 나쁘다고 단정하지 않습니다.

### 실제 시간과 비용 해석

| 단계 | 시간 (분) |
| --- | ---: |
| Drive 연결·ZIP 준비 | 2.5506 |
| 특징 추출·학습·보정·저장 전체 | 21.3144 |
| 최종 추론·평가 | 0.8049 |
| **전체** | **24.6700** |

학습·보정 전체 21.31분 중 특징 추출은 6.2162분, 특징 추출을 제외한 head 학습·초기 평가 합계는 6.3773분, 선택된 D의 실제 encoder 평가·미세조정은 6.5913분입니다. 나머지는 모델 로딩·무결성 검사·저장·보정 등입니다.
**특징 추출 6.2162분은 A epoch 0의 6.2641분 안에 이미 포함됩니다. 두 값을 중복 합산하지 않습니다.** head 학습 epoch 18개의 시간만 합하면 6.0994분입니다. 패키지 설치와 총 CU는 기록에 포함되지 않습니다.

### 해석 및 다음 실험에 남기는 점

- A–C 손실 교체, E/F 정규화·증상 관계 추가는 이번 설정에서 기존 또는 D보다 좋은 stop F1을 만들지 못했습니다. 해당 방법 전체의 무효를 뜻하지 않습니다.
- D는 학습됐지만 기존 head 대비 cached stop F1 상승이 +0.000423878로 작았습니다. D 하나가 최선이라고 단정할 근거는 부족합니다.
- 모든 후보가 같은 task-tuned encoder를 고정하여 사용했고, 마지막에도 상위 2층만 1 epoch 학습했습니다. 새로운 한국어 encoder나 원본 사전학습 모델과 분류기의 전체 공동 미세조정은 이 실험에서 검증하지 않았습니다.
- 최종 성능 개선은 확인되지 않았습니다. 후속 실험은 발화 경계 보존과 독립적인 한국어 encoder의 공동 미세조정을 비교하되, 새 버전의 F1을 실행 전에 확정하지 않습니다.

### 재현 방법 및 보관

기존 입력 ZIP 바로가기와 [mission3_ver6.ipynb](mission3_ver6.ipynb)로 실행합니다. 진행 중 재개는 같은 GPU 종류·정밀도·설정·코드를 유지하고, 새로운 설정은 새 출력 폴더를 사용합니다.

```bash
python results/mission3/ver6/submission/inference.py \
  --audio_dir "" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver6/mission3_ver6_model.ckpt \
  --base_dir results/mission3/ver4 \
  --output outputs/mission3/ver6_saved_model/mission3.csv \
  --batch_size 8 --evaluate
```

`mission3_ver6_model.ckpt`는 구성 manifest이며 `candidate_best.ckpt`, 원래 ver4 모델·tokenizer·sparse 파일이 필요합니다. `candidate_best_last.ckpt`는 미세조정 epoch 1의 재개 상태이고, `head_trials/profile_A.ckpt`~`profile_F.ckpt`는 각 분류기의 마지막·최적 가중치와 이력입니다. `candidate_calibration.npz`는 D 후보의 별도 Training 보정 예측입니다. 최종 best 가중치 SHA256·cache signature와 실행 이력이 일치함을 결과 분석에서 확인했습니다.

## EXP-007 — ver7: KoELECTRA 전체 encoder + 증상별 attention 공동 학습

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver7.ipynb`, `mission3_ver7/*.py`; 노트북 내장 소스·현재 소스·결과 `submission/` 24개 파일 일치 |
| 결과 위치 | `results/mission3/ver7/`; Drive `DCC_AIdeas/results/mission3/ver7` |
| 실행 기록 식별자 | `20260917_103114_36af68`; 노트북이 추론·평가 직전에 만든 KST 시각·식별자이며 전체 실행 시작 시각은 아님 |
| 실행 환경 | NVIDIA L4 1개, BF16; torch `2.11.0+cu128`, CUDA `12.8`, transformers `4.57.1`, numpy `2.1.3`, scikit-learn `1.6.1`, iterative-stratification `0.1.9` |
| 입력 / 데이터 | 기존 `mission3_ver5_inputs.zip`; 서울 Training 29,200개 / Validation 3,640개; 양측 발화 text, WAV 미사용 |
| 분할 | 원래 ver4 fold 1 fit 21,032 / stop 5,288 / 별도 calibration 2,880; 저장된 파일명 집합 간 중복 없음 |
| 새 모델 | `monologg/koelectra-base-v3-discriminator` 공개 가중치 + 새 9-label token attention; 전체 encoder와 분류기 공동 학습 |
| 공개 모델 revision | `68b30cd259f34a4b5aa8786392612ba2a2617fcc` |
| 학습 / 최적 epoch | 3 epoch 완료, 재개 실행 아님; **epoch 2 선택** |
| 같은 stop 비교 | 새 후보 Macro F1 0.617328996 / 기존 ver4 fold 1 단독 0.639684092 |
| 최종 구성 선택 | **ver4_baseline**; 새 후보 가중치 9개 모두 0 |
| Calibration cross-fit | 기존 0.659670866 / 새 모델 포함 최선 혼합 0.658665925 / 후보 단독 0.631151566 |
| 최종 Validation Macro F1 | **0.656669718** (원본 `0.6566697183939796`); ver6과 모든 예측 동일 |
| 후보 단독 Validation Macro F1 | **0.638109662** (원본 `0.6381096624831213`); 최종 구성보다 **-1.8560%p** |
| 전체 기록 시간 | **45.7879분**; 패키지 설치 제외, CU 사용량 미기록 |
| 보관 방식 | `bundle_base=false`, `portable=false`; 원래 ver4 모델 묶음도 필요 |
| 근거 | [metrics](../../results/mission3/ver7/mission3_ver7_metrics.json), [run](../../results/mission3/ver7/run.json), [완료 이력](../../results/mission3/ver7/candidate_done.json), [시간](../../results/mission3/ver7/notebook_runtime_20260917_103114_36af68.json), [최종 Validation](../../results/mission3/ver7/predictions/20260917_103114_36af68/validation_metrics.json), [후보 단독 Validation](../../results/mission3/ver7/predictions/20260917_103114_36af68/candidate_validation_metrics.json) |

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| 대화 전처리 | 발화 순서를 유지하고 비어 있지 않은 발화 사이에 동일한 `[SEP]` 삽입; speaker·증상 정답으로 입력을 바꾸지 않음 |
| 긴 대화 처리 | 최대 512-token window, overlap 64; 여러 window의 token 표현에 증상별 attention 적용 |
| 학습 범위 | 공개 사전학습 encoder의 전체 층 + 새 attention 분류기; 기존 ver4 가중치로 새 encoder를 초기화하지 않음 |
| 손실 | fit 양성 비율로 가중치를 정한 `BCEWithLogitsLoss`; 양성 가중치 `clip(sqrt(max(음성/양성, 1)), 1, 5)` |
| 최대 epoch / patience | 3 / 2; 실제 epoch 3까지 완료 |
| Learning rate | encoder `2e-5`, head `1e-4`, layerwise decay 0.95 |
| Batch | 통화 8 / window 8 / accumulation 4 / 유효 batch 최대 32; 최종 추론 batch 8 |
| 정규화·메모리 | AdamW, weight decay 0.01, warmup 비율 0.1, dropout 0.1, attention 차원 256, gradient checkpointing, BF16 |
| Seed / epoch 선택 | config seed 42, 학습·stop 교차평가 seed 5045; stop 그룹 분리 3-fold threshold 교차평가 Macro F1 우선, AP는 동점 보조 |
| 최종 구성 선택 | 별도 calibration 5-fold, seed 44; 기존 대비 Macro F1 최소 +0.002 및 최소 3/5 fold의 엄격한 개선 |

새 모델 1개를 학습했고 기존 ver4의 5개 모델은 재학습하지 않았습니다. `init_fold=1`은 같은 fit/stop 분할과 비교 모델을 지정하는 값입니다. frozen feature cache를 사용한 ver6과 달리 새 encoder 전체를 처음부터 공동 미세조정했습니다.

### 모든 epoch 학습 이력

| Epoch | 단계 | Train loss | Stop Macro F1 | Stop Macro AP | 시간 (분) | 최적 |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 0 | 공개 encoder + 초기 분류기 평가 | — | 0.268238398 | 0.158641405 | 0.6693 | |
| 1 | 전체 encoder·분류기 학습 | 0.597958 | 0.603710105 | 0.635610916 | 12.4417 | |
| 2 | 전체 encoder·분류기 학습 | 0.442162 | **0.617328996** | 0.657464140 | 12.3874 | ✓ |
| 3 | 전체 encoder·분류기 학습 | 0.416790 | 0.615625525 | 0.657658409 | 12.4702 | |

- epoch 3에서 train loss와 AP는 개선됐지만 선택 기준인 stop F1이 epoch 2보다 **0.001703470** 낮아져 epoch 2를 유지했습니다. 최대 3 epoch를 모두 수행한 결과이며 조기 종료로 2 epoch만 학습한 것이 아닙니다.
- 같은 stop 5,288개·같은 threshold 교차평가에서 기존 ver4 fold 1은 F1 **0.639684092**, AP **0.682987440**입니다. 새 후보는 F1 기준 **-0.022355096 (-2.2355%p)**로 낮았습니다. 이는 기존 단일 모델과의 비교이며 5개 모델 앙상블 점수와 혼동하지 않습니다.
- 학습된 epoch 2가 후보 가중치로 보관됐지만, 아래 최종 혼합 선택에서는 채택되지 않았습니다. ver5·ver6의 추가 encoder 미세조정에서 epoch 0을 유지한 경우와 구분합니다.

### 교차 보정과 기존 구성 유지

| 구성 | Calibration cross-fit Macro F1 | 기존 대비 (%p) | 최종 선택 |
| --- | ---: | ---: | :---: |
| 기존 ver4 | **0.659670866** | — | ✓ |
| 새 후보 포함 raw 확률 혼합 | 0.658665925 | -0.1005 | |
| 새 후보 포함 logit 혼합 | 0.658258699 | -0.1412 | |
| 새 후보 단독 진단 | 0.631151566 | -2.8519 | |

혼합 후보 중 raw가 더 좋았지만 기존 대비 **-0.001004940 F1**로 최소 **+0.002** 기준을 충족하지 못했습니다. 개선된 fold도 **2/5**로 최소 3개에 미달했습니다. 저장된 주된 탈락 사유는 `fallback_reason=candidate_gain_below_minimum`입니다.

| Fold | raw 혼합 − 기존 F1 | 차이 (%p) | 개선 |
| ---: | ---: | ---: | :---: |
| 1 | -0.002041973 | -0.2042 | |
| 2 | +0.006319020 | +0.6319 | ✓ |
| 3 | +0.000968518 | +0.0969 | ✓ |
| 4 | -0.007830696 | -0.7831 | |
| 5 | -0.001837262 | -0.1837 | |

최종 calibration fit F1은 **0.669459793**, 후보 단독 fit F1은 **0.641960455**입니다. 각각 calibration 전체로 임계값을 맞춘 뒤 같은 표본에서 평가한 값이므로 위 cross-fit이나 공식 Validation과 구분합니다. 이 절의 내부 교차평가 수치는 저장된 JSON 기록이며, 이번 로그 반영 시 보정을 재실행한 값은 아닙니다.

### 최종 혼합 비율과 후보 단독 임계값

| Class | 기존 신경망 α | 새 후보 β | 최종 Threshold | 후보 단독 Threshold |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 1.00 | 0.00 | 0.527061400 | 0.475141674 |
| 구토 | 0.75 | 0.00 | 0.585740679 | 0.351180881 |
| 두통 | 0.75 | 0.00 | 0.414659252 | 0.644668818 |
| 복통 | 0.75 | 0.00 | 0.525984191 | 0.459572002 |
| 어지러움 | 1.00 | 0.00 | 0.491587323 | 0.466917574 |
| 열상 | 1.00 | 0.00 | 0.527157467 | 0.863815069 |
| 오심 | 0.75 | 0.00 | 0.411039874 | 0.303352088 |
| 전신쇠약 | 1.00 | 0.00 | 0.472925211 | 0.431516960 |
| 호흡곤란 | 0.75 | 0.00 | 0.585971753 | 0.651484072 |

최종 α·threshold는 기존 ver4 manifest와 저장된 전체 정밀도에서 정확히 같고 β는 전부 0입니다. `mission3.csv`는 기존 신경망 앙상블과 sparse 모델의 출력을 사용합니다. `candidate_only.csv`는 epoch 2 KoELECTRA의 확률과 Training calibration에서 고정한 후보 전용 임계값을 사용합니다. 후보 단독 진단을 요청했으므로 β가 0이어도 후보 추론은 별도로 수행하는 코드 경로입니다.

### 최종 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7705 | 0.6386 | 0.6984 | 678 |
| 구토 | 0.6945 | 0.5419 | 0.6088 | 537 |
| 두통 | 0.5686 | 0.5321 | 0.5497 | 374 |
| 복통 | 0.8931 | 0.7673 | 0.8254 | 838 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 |
| 오심 | 0.3205 | 0.5431 | 0.4031 | 429 |
| 전신쇠약 | 0.5629 | 0.6252 | 0.5924 | 651 |
| 호흡곤란 | 0.7188 | 0.6443 | 0.6795 | 492 |
| **Macro avg** | **0.6697** | **0.6561** | **0.656670** | **5,236** |

### 새 KoELECTRA 단독 Validation 상세

| Class | Precision | Recall | F1-score | Support | 최종 구성 대비 F1 (%p) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 고열 | 0.7500 | 0.6150 | 0.6759 | 678 | -2.2536 |
| 구토 | 0.5960 | 0.5605 | 0.5777 | 537 | -3.1051 |
| 두통 | 0.6598 | 0.4251 | 0.5171 | 374 | -3.2651 |
| 복통 | 0.8784 | 0.7673 | 0.8191 | 838 | -0.6309 |
| 어지러움 | 0.6082 | 0.6718 | 0.6384 | 774 | -2.0594 |
| 열상 | 0.8755 | 0.8661 | 0.8708 | 463 | -2.2824 |
| 오심 | 0.2918 | 0.6061 | 0.3939 | 429 | -0.9175 |
| 전신쇠약 | 0.5123 | 0.6713 | 0.5811 | 651 | -1.1314 |
| 호흡곤란 | 0.7597 | 0.5976 | 0.6689 | 492 | -1.0586 |
| **Macro avg** | **0.6591** | **0.6423** | **0.638110** | **5,236** | **-1.8560** |

- [최종 CSV](../../results/mission3/ver7/predictions/20260917_103114_36af68/mission3.csv)와 [후보 단독 CSV](../../results/mission3/ver7/predictions/20260917_103114_36af68/candidate_only.csv)를 각각 로컬 Validation JSON 3,640개와 대조했습니다. 파일명 중복·누락·추가 행이 없고 9개 증상의 이진 판정으로 재계산한 지표가 저장된 JSON과 일치합니다. 후보 단독 Macro F1의 마지막 자릿수 차이 약 `1e-16`은 합산 반올림 범위입니다.
- 최종 CSV는 ver6과 **3,640개 모두 동일**합니다. ver4와는 통화 3개·이진 판정 3개, ver5와는 통화 493개·이진 판정 537개가 다릅니다. Macro F1 차이는 ver4 대비 **+0.000020770355**, ver5 대비 **-0.000048039989**, ver6 대비 **0**입니다. 새 후보 β가 0이므로 ver4 대비 미세 차이를 KoELECTRA 학습 효과로 볼 수 없습니다.
- 후보 단독과 최종 구성 사이에는 통화 **1,223개**, 증상별 이진 판정 **1,634개**의 차이가 있습니다. 후보 단독 F1은 9개 증상 모두 최종 구성보다 낮았습니다.
- 후보의 `두통`은 precision 0.6598로 높지만 recall 0.4251로 낮아 F1 격차가 가장 컸습니다. `오심`은 recall 0.6061로 높아졌지만 precision 0.2918로 낮아졌으며, 거짓 양성이 최종 구성의 494개에서 후보 단독 631개로 증가했습니다. 검출량 증가가 F1 개선으로 이어지지 않았습니다.

### 실제 시간과 비용 해석

| 단계 | 시간 (분) |
| --- | ---: |
| Drive 연결·입력 준비 | 2.2095 |
| 공개 사전학습 모델 준비 | 0.5248 |
| 전체 encoder 학습·내부 비교·보정·저장 | 41.5284 |
| 최종·후보 단독 추론 및 평가 | 1.5252 |
| **기록된 단계 합계** | **45.7879** |

학습·보정 41.5284분 안에 후보 학습 단계 38.6434분, 기존 단일 fold stop 비교 1.3482분, calibration 예측·선택 0.5045분 등이 포함됩니다. 실제 학습 epoch 1~3의 기록 합계는 **37.2992분**이며 epoch 0 평가는 별도 0.6693분입니다. 내부 단계 시간은 상위 학습·보정 시간에 이미 포함되므로 중복 합산하지 않습니다. 패키지 설치·계측 구간 밖의 처리와 CU 사용량은 이 합계에 포함되지 않습니다.

### 해석 및 다음 실험에 남기는 점

- 새 encoder 전체를 학습해도 이번 설정에서는 기존 단일 fold의 stop 성능을 넘지 못했고, 혼합도 calibration 성능을 높이지 못했습니다. 최종 CSV가 ver6과 같은 이유는 학습 실패나 출력 누락이 아니라 **Training 내부 선택 기준에 따라 기존 구성을 유지했기 때문**입니다.
- ver7은 ver6에서 검증하지 않았던 독립 공개 encoder·발화 경계·전체 공동 미세조정을 실제로 평가했습니다. 다만 이 요소들을 함께 바꿨으므로 각각의 효과나 KoELECTRA 모델 계열 전체의 우열을 분리해서 단정할 수 없습니다.
- epoch 3에서는 손실이 내려가도 stop F1이 하락했습니다. 이 실행만으로 과적합 원인을 확정하거나 epoch를 더 늘리면 개선된다고 결론 내리지 않습니다.
- 후보 단독 Validation은 기존 5개 모델 앙상블과의 비교입니다. 단일 모델 대 앙상블의 차이를 감안하되, 같은 stop에서 기존 단일 fold에도 뒤졌다는 별도 결과를 함께 봅니다.
- Validation은 최종 구성·임계값을 고정한 뒤 진단에만 사용했습니다. 여러 epoch의 stop 선택과 반복 calibration 연구에 따른 선택 편향은 남으며, 이 실행은 여러 seed 반복이나 새 모델 5-fold 학습 결과가 아닙니다.

### 재현 방법 및 보관

기존 입력 ZIP 바로가기와 [mission3_ver7.ipynb](mission3_ver7.ipynb)를 사용합니다. 노트북이 고정 revision의 공개 모델을 준비하고 전체 encoder를 학습합니다. 새 설정 실험은 별도 `OUTPUT_DIR`를 지정하며, 중단된 학습 재개에는 같은 GPU 종류·정밀도·코드·패키지·데이터·설정을 유지합니다. 이번 실제 학습 batch 8 / accumulation 4를 재현하려면 자동 GPU 설정으로 값이 바뀌지 않는지도 확인합니다.

저장된 가중치로 최종·후보 단독 예측을 새 경로에 생성하는 명령입니다. 프로젝트 루트 기준이며 `--base_dir`는 Colab 절대 경로 대신 현재 ver4 위치를 지정합니다.

```bash
python results/mission3/ver7/submission/inference.py \
  --audio_dir "" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver7/mission3_ver7_model.ckpt \
  --base_dir results/mission3/ver4 \
  --output outputs/mission3/ver7_saved_model/mission3.csv \
  --candidate_output outputs/mission3/ver7_saved_model/candidate_only.csv \
  --batch_size 8 --evaluate
```

모델 추론 없이 저장된 CSV를 재평가하려면 다음 명령을 사용합니다. 후보 단독 평가는 `--prediction`의 파일명을 `candidate_only.csv`로 바꿉니다.

```bash
python results/mission3/ver7/submission/ver4_runtime/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver7/predictions/20260917_103114_36af68/mission3.csv
```

| 파일 | 역할 |
| --- | --- |
| `mission3_ver7_model.ckpt` | 기존 모델 위치·후보·보정값·최종 선택을 담은 manifest; 단독 가중치 파일이 아님 |
| `candidate_best.ckpt`, `tokenizer/` | stop F1로 선택한 epoch 2 KoELECTRA·분류기와 전용 tokenizer |
| `candidate_best_last.ckpt` | 완료 epoch 3 가중치·optimizer·scheduler·RNG 재개 상태; 후보 단독 평가에 사용한 최적 epoch와 다름 |
| `candidate_done.json`, `run.json`, `splits.json` | 완료 이력, 실행 설정·출처·해시, fit/stop/calibration 파일 목록 |
| `baseline_single_fold_stop.npz`, `candidate_calibration.npz` | 기존 단일 fold stop 비교 예측과 새 후보의 Training calibration 예측 |
| `mission3_ver7_metrics.json` | epoch 이력·단일 fold 비교·교차 보정·기존 구성 유지 판단 |
| `predictions/20260917_103114_36af68/` | 최종 및 후보 단독 CSV·각각의 상세 평가 JSON |
| `notebook_runtime_20260917_103114_36af68.json`, `submission/` | 단계별 시간, 이번 실행 코드·패키지 목록 |

최적 후보 파일 SHA256과 tokenizer 4개 파일의 해시를 manifest와 대조했고, 후보 SHA256은 `candidate_done.json`과도 일치했습니다. 노트북 내장 소스·현재 소스·보관된 `submission/` 24개 파일이 일치하며, run·metrics·manifest·완료 이력의 signature도 일치합니다. `bundle_base=false`이므로 **원래 입력 ZIP 또는 동일한 ver4 모델 묶음과 ver7 결과를 함께 보관**해야 합니다. 저장된 후보로 추론할 때 공개 사전학습 모델을 다시 다운로드할 필요는 없습니다.

## EXP-008 — ver8: KLUE-RoBERTa-large + 증상별 token attention 전체 공동 학습

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission3/mission3_ver8.ipynb`, `mission3_ver8/*.py`; 현재 노트북 셀 출력 없음 |
| 결과 위치 / 실행 식별자 | `results/mission3/ver8/`, `20260917_145217_a443ea`; 식별자는 추론·평가 직전 생성되며 전체 실행 시작 시각이 아님 |
| 실행 환경 | NVIDIA L4 1개, BF16; torch `2.11.0+cu128`, CUDA `12.8`, transformers `4.57.1`, numpy `2.1.3`, scikit-learn `1.6.1` |
| 입력 / 데이터 | 서울 Training 29,200개 / Validation 3,640개; 양측 발화 text, WAV 미사용 |
| Training 내부 분할 | 기존 ver4 fold 1 fit 21,032 / stop 5,288 / 별도 calibration 2,880 |
| 새 모델 | 공개 `klue/roberta-large` + 새 증상별 직접 token attention 분류기; 전체 encoder와 분류기 공동 학습 |
| 공개 모델 revision | `28d911204e9022eda172571ca8cc61eaffd942f7` |
| 실제 실험 범위 | **후보 B 하나만 실행**; A/C 비교 및 TAPT 미실행 (`profiles=["B"]`, `tapt=null`) |
| 학습 / 최적 epoch | 완료 JSON상 3 epoch, **epoch 3 선택**, `resumed=true`; 로컬 후보 가중치는 아래 불일치 별도 확인 |
| 같은 stop 비교 | 새 후보 Macro F1 0.637975045 / 기존 ver4 fold 1 단독 0.639684092 |
| 최종 구성 선택 | **ver4_baseline**; 새 후보 가중치 9개 모두 0 |
| Calibration cross-fit | 기존 0.659670866 / 새 모델 포함 최선 혼합 0.657450867 / 후보 단독 0.650080649 |
| 최종 Validation Macro F1 | **0.656669718** (원본 `0.6566697183939796`); ver6·ver7 최종 CSV와 모든 예측 동일 |
| 후보 단독 Validation Macro F1 | **0.645292165** (원본 `0.6452921654524023`); 최종 구성보다 **-1.1378%p** |
| 기록된 단계 시간 합계 | **77.3599분**; 패키지 설치 제외, 재개 전 실행까지 포함한 누적 총시간·CU 사용량은 확인하지 못함 |
| 보관 방식 | `bundle_base=false`, `portable=false`; 원래 ver4 모델 묶음도 필요 |
| 근거 | [metrics](../../results/mission3/ver8/mission3_ver8_metrics.json), [run](../../results/mission3/ver8/run.json), [완료 이력](../../results/mission3/ver8/candidate_done.json), [시간](../../results/mission3/ver8/notebook_runtime_20260917_145217_a443ea.json), [최종 Validation](../../results/mission3/ver8/predictions/20260917_145217_a443ea/validation_metrics.json), [후보 단독 Validation](../../results/mission3/ver8/predictions/20260917_145217_a443ea/candidate_validation_metrics.json) |

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| 대화 처리 | 발화 사이 동일한 `[SEP]`; 최대 512-token window, overlap 64; 증상별 attention으로 token 표현 집계 |
| 초기화 / 학습 범위 | 공개 사전학습 RoBERTa-large와 새 token head에서 시작; 전체 encoder·분류기 공동 미세조정 |
| 손실 | fit 양성 비율에 따른 weighted BCE; 양성 가중치 `clip(sqrt(max(음성/양성, 1)), 1, 5)` |
| 최대 epoch / patience | 3 / 2; 완료 이력에는 epoch 3까지 기록 |
| Learning rate | encoder `1.5e-5`, head `1e-4`, layerwise decay 0.95 |
| Batch | 통화 4 / window 4 / accumulation 8 / 유효 batch 최대 32; 추론 batch 8 |
| 정규화·메모리 | AdamW, weight decay 0.01, warmup 0.1, dropout 0.1, attention 차원 256, gradient checkpointing, BF16 |
| Epoch 선택 | config seed 42, 후보 학습 seed 5045; stop 그룹 분리 3-fold threshold 교차평가 Macro F1 우선, AP 동점 보조 |
| 혼합 채택 기준 | 별도 calibration 5-fold에서 기존 대비 F1 최소 +0.002, 최소 3/5 fold 개선 |

설정에 `tapt_steps=500`이 남아 있지만 후보 C를 실행하지 않았으므로 추가 사전학습을 했다는 뜻이 아닙니다. 새 모델은 1개이며 기존 ver4의 5개 모델은 재학습하지 않았습니다. 기존 calibration에서 두통·오심·전신쇠약의 FP/FN을 각각 20개씩, 총 120행 추출한 검토 템플릿도 저장됐습니다. [오류 분석 기록](../../results/mission3/ver8/error_analysis/summary.json)은 `labels_modified=false`, `reviews_used_for_training=false`이며, 사람이 검토를 완료했다는 근거는 아닙니다.

### 완료 JSON에 기록된 모든 epoch 이력

| Epoch | 단계 | Train loss | Stop Macro F1 | Stop Macro AP | 시간 (분) | 최적 |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 0 | 공개 encoder + 초기 분류기 평가 | — | 0.279664607 | 0.172703717 | 0.9555 | |
| 1 | 전체 encoder·분류기 학습 | 0.495355 | 0.628133874 | 0.677788062 | 20.9101 | |
| 2 | 전체 encoder·분류기 학습 | 0.407814 | 0.636357231 | 0.689226510 | 21.0982 | |
| 3 | 전체 encoder·분류기 학습 | 0.376981 | **0.637975045** | 0.686264468 | 21.0540 | ✓ |

- epoch 3에서 AP는 epoch 2보다 낮지만 우선 선택 지표인 stop F1이 올라 최종 후보로 기록됐습니다. 학습 epoch 1~3의 시간 합계는 **63.0623분**이며 epoch 0 평가는 별도입니다.
- 같은 stop에서 기존 ver4 fold 1보다 F1 **-0.001709047 (-0.1709%p)** 낮았습니다. ver7 후보의 같은 stop F1 0.617328996보다는 높았지만, 기존 단일 fold를 넘지는 못했습니다.
- 이 표는 완료 JSON의 실행 이력입니다. 현재 내려받은 `best.ckpt`가 이력의 최적 epoch와 다르므로, 로컬 가중치로 위 성능을 재현했다고 해석하지 않습니다.

### 교차 보정과 기존 구성 유지

| 구성 | Calibration cross-fit Macro F1 | 기존 대비 (%p) | 최종 선택 |
| --- | ---: | ---: | :---: |
| 기존 ver4 | **0.659670866** | — | ✓ |
| 새 후보 포함 raw 확률 혼합 | 0.657258327 | -0.2413 | |
| 새 후보 포함 logit 혼합 | 0.657450867 | -0.2220 | |
| 새 후보 단독 진단 | 0.650080649 | -0.9590 | |

혼합 후보 중 logit이 더 좋았지만 기존 대비 **-0.002219999 F1**이고 개선 fold도 **1/5**였습니다. 최소 +0.002와 최소 3개 fold 개선 조건을 모두 충족하지 못해 `fallback_reason=candidate_gain_below_minimum`으로 기존 구성을 유지했습니다.

| Fold | logit 혼합 − 기존 F1 | 차이 (%p) | 개선 |
| ---: | ---: | ---: | :---: |
| 1 | -0.007207113 | -0.7207 | |
| 2 | +0.005449562 | +0.5450 | ✓ |
| 3 | -0.000161875 | -0.0162 | |
| 4 | -0.004948980 | -0.4949 | |
| 5 | -0.003482919 | -0.3483 | |

최종 혼합은 기존 ver4의 신경망 비율·임계값을 유지하고 새 후보 β는 전부 0입니다. 최종 calibration fit F1은 0.669459793, 후보 단독 fit F1은 0.660181477이며 같은 calibration 표본으로 보정·평가한 값입니다. 위 내부 점수는 저장된 JSON 기록이며 이번 로그 반영 중 보정을 재실행하지 않았습니다.

### 최종 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7705 | 0.6386 | 0.6984 | 678 |
| 구토 | 0.6945 | 0.5419 | 0.6088 | 537 |
| 두통 | 0.5686 | 0.5321 | 0.5497 | 374 |
| 복통 | 0.8931 | 0.7673 | 0.8254 | 838 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 |
| 오심 | 0.3205 | 0.5431 | 0.4031 | 429 |
| 전신쇠약 | 0.5629 | 0.6252 | 0.5924 | 651 |
| 호흡곤란 | 0.7188 | 0.6443 | 0.6795 | 492 |
| **Macro avg** | **0.6697** | **0.6561** | **0.656670** | **5,236** |

### 새 RoBERTa-large 후보 단독 Validation 상세

| Class | Precision | Recall | F1-score | Support | 최종 구성 대비 F1 (%p) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 고열 | 0.8224 | 0.5737 | 0.6759 | 678 | -2.2453 |
| 구토 | 0.7132 | 0.5233 | 0.6037 | 537 | -0.5135 |
| 두통 | 0.5676 | 0.5053 | 0.5347 | 374 | -1.5070 |
| 복통 | 0.8782 | 0.7661 | 0.8184 | 838 | -0.7062 |
| 어지러움 | 0.6065 | 0.7248 | 0.6604 | 774 | +0.1366 |
| 열상 | 0.8563 | 0.9136 | 0.8840 | 463 | -0.9604 |
| 오심 | 0.3101 | 0.4988 | 0.3825 | 429 | -2.0630 |
| 전신쇠약 | 0.5546 | 0.6160 | 0.5837 | 651 | -0.8734 |
| 호흡곤란 | 0.7299 | 0.6098 | 0.6645 | 492 | -1.5077 |
| **Macro avg** | **0.6710** | **0.6368** | **0.645292** | **5,236** | **-1.1378** |

- 최종·후보 단독 CSV 모두 Validation JSON 3,640개와 파일명이 1:1 대응하며 중복·누락·추가 행이 없습니다. CSV로 다시 계산한 Precision·Recall·F1·Support와 Macro F1이 저장된 평가 JSON과 일치했습니다. GPU 학습·추론은 재실행하지 않았습니다.
- 최종 CSV는 ver6·ver7과 통화 **3,640개 모두 동일**합니다. ver4와는 통화 3개·이진 판정 3개, ver5와는 통화 493개·이진 판정 537개가 다릅니다. 이 미세 차이는 새 모델 학습 효과가 아닙니다.
- 새 후보 단독은 최종 구성과 통화 **817개**, 이진 판정 **956개**가 다릅니다. 어지러움 F1만 소폭 높고 나머지 8개 증상은 낮습니다.
- ver7 후보 단독 0.638109662보다 **+0.7183%p** 높았지만 기존 최종 앙상블보다는 **-1.1378%p** 낮았습니다. 후보 단독의 개선과 최종 제출 구성의 개선을 구분합니다.

### 기록된 실행 시간과 해석

| 단계 | 시간 (분) |
| --- | ---: |
| Drive 연결·입력 준비 | 2.3258 |
| 공개 사전학습 모델 준비 | 0.7409 |
| 후보 학습·내부 비교·보정·저장 | 72.2184 |
| 최종·후보 단독 추론 및 평가 | 2.0748 |
| **기록된 단계 합계** | **77.3599** |

내부 `total_pipeline=72.0288분`과 후보 단계 67.8793분 등은 위 학습·보정 시간에 포함되므로 중복 합산하지 않습니다. `resumed=true`이고 별도 재개 전 실행 전체 시간이 없으므로 77.3599분을 모든 시도의 누적 비용으로 단정하지 않습니다. 패키지 설치와 CU 사용량은 이 기록에 없습니다.

후보 B의 학습 기록과 CSV는 남았지만 기존 구성을 개선하는 혼합으로 선택되지는 않았습니다. A/C 비교, TAPT, 새 모델의 5-fold 앙상블은 이번에 실행하지 않아 효과를 판단할 수 없습니다. 이번 단일 후보 결과만으로 RoBERTa나 attention 계열 전체가 효과 없다고 결론 내리지는 않습니다.

### 로컬 체크포인트 불일치와 보관 상태 — 2026-09-17 확인

| 파일 | 현재 확인한 내용 | 보관 판단 |
| --- | --- | --- |
| `ver8/mission3_ver8_model.ckpt` | 최종 선택·경로·보정값·후보 SHA256을 담은 manifest; `selected_name=ver4_baseline` | 유지; 단독 가중치 파일이 아님 |
| `ver8/trials/B/best.ckpt` | 약 1.35 GB, `format=mission3_ver8_neural`, **epoch 0**; 완료 JSON이 기록한 epoch 3과 다름 | 최종 epoch 3 파일을 다시 확보·검증해야 함 |
| `best_last-002.ckpt` (ver8 폴더 밖) | **4,035,285,914 bytes (약 4.04 GB / 3.76 GiB)**, `format=mission3_ver8_last`, **완료 epoch 1**, `completed=false` | 중간 학습 재개용; 최종 결과 재추론에는 사용하지 않음 |
| `ver8/tokenizer/`, `predictions/`, 각 JSON·NPZ | tokenizer, 완료 기록·평가·예측 캐시 | 유지; CSV 점수는 별도로 검증됨 |
| `ver8/submission/` | 현재 디렉터리는 있으나 실행 코드 파일 없음 | 실행 코드 사본도 다시 확보하거나 현재 소스와 실행 기록의 일치를 별도 확인해야 함 |

후보 체크포인트의 SHA256을 실제 파일 전체에서 계산했습니다.

- manifest·`candidate_done.json`·`trials/B/done.json`의 최종 후보 기대값: `b6a5009dde12b48504cb21baf38b3288143876e569a9f1aaae42eac8b2b13de1`
- 현재 `ver8/trials/B/best.ckpt`의 실제 값: `7bd0c2a08ec35f62ae010b302f7e439158f1b6a49e0edc2b4e32b849f0e58f95`

**현재 로컬 ver8 묶음은 저장된 CSV 결과를 그대로 재추론할 수 있는 완전한 묶음으로 확인되지 않았습니다.** 현재 추론 코드는 후보 혼합 비중이 0이어도 manifest를 읽을 때 후보 해시를 검사하므로 이 파일로는 `ver8 candidate weights hash mismatch`가 발생하는 조건입니다. 이는 코드와 해시 비교로 확인했으며 GPU 추론을 실행해 얻은 오류 로그는 아닙니다. Drive에서 최종 `ver8/trials/B/best.ckpt`를 다시 내려받아 위 기대 해시와 epoch 3을 확인해야 합니다. 다운로드·복사 중 어느 단계에서 오래된 파일이 섞였는지는 현재 자료만으로 확정할 수 없습니다.

`best_last-002.ckpt`는 모델 가중치 외에 optimizer·scheduler·scaler·난수 상태를 포함해 용량이 큽니다. 내부 후보 실행 signature와 설정 fingerprint가 `trials/B/best.ckpt` 및 해당 후보 실행 기록과 대응하고, epoch 0~1 이력도 완료 JSON의 앞부분과 같아 **ver8 후보 B의 중간 재개 체크포인트**로 확인했습니다. 최종 manifest는 이 파일을 참조하지 않습니다. 코드도 `keep_completed_resume_states=false`이면 정상 완료 후 원래 `trials/B/best_last.ckpt`를 제거하도록 되어 있습니다.

따라서 epoch 1부터 복구할 필요가 없고 최종 epoch 3 모델을 정상 확보했다면 `best_last-002.ckpt`는 삭제할 수 있습니다. **현재는 최종 후보 가중치가 불일치하므로 먼저 올바른 최종 파일을 확보한 뒤 삭제하는 편이 안전합니다.** 이 중간 파일을 `best.ckpt`로 이름만 바꿔 대체할 수는 없습니다. 이번 로그 정리에서는 체크포인트를 수정·이동·삭제하지 않았습니다.

## EXP-009 — ver9: 한국어 DeBERTa-v3 2-fold + NB-log-count LR

| 항목 | 결과 |
| --- | --- |
| 구현 / 결과 확인일 | 구현 2026-09-17 / 로컬 결과 확인 2026-09-19 |
| 실행 코드 | `src/mission3/mission3_ver9.ipynb`, `mission3_ver9/*.py`; 현재 노트북 셀 출력 없음 |
| 결과 위치 | `results/mission3/ver9/`; 실제 설정·학습 이력·시간은 저장된 JSON 기준 |
| 기록된 실행 시각 | 2026-09-17 23:51:12.794 ~ 2026-09-18 01:49:02.116 KST; `runtime.json`의 시작 timestamp와 경과 시간 기준 |
| 실행 환경 | NVIDIA L4 1개, BF16; torch `2.11.0+cu128`, CUDA `12.8`, transformers `4.57.1`, numpy `2.1.3`, scikit-learn `1.6.1` |
| 입력 / 데이터 | 서울 Training 29,200개 / Validation 3,640개; 양측 발화 text, WAV 미사용 |
| Training 내부 분할 | 기존 ver4 분할 재사용: calibration 2,880개, 나머지 26,320개; fold 1 fit 21,032 / stop 5,288, fold 2 fit 21,053 / stop 5,267 |
| 새 encoder | `team-lucid/deberta-v3-base-korean`, 고정 revision `47a3b00828c5dc2b8b080c6738ef7a58ebd419d5` |
| 새 후보 | DeBERTa 2개 fold 모델의 확률 평균 0.75 + NB-log-count-ratio Logistic Regression 0.25; 사전 고정 비율 |
| 실제 학습 / 최적 epoch | 두 fold 모두 전체 encoder·분류기를 3 epoch 학습, **각 epoch 3 선택**; `resumed=false`, 시간 부족 중단 없음 |
| 최종 구성 선택 | **ver4_baseline**; 새 후보 가중치 9개 모두 0 |
| Calibration cross-fit | 기존 0.659670866 / 새 후보 포함 최선 혼합 0.659546589 / 후보 단독 0.597860936 |
| 최종 Validation Macro F1 | **0.656765674** (원본 `0.6567656741534235`) |
| 후보 단독 Validation Macro F1 | **0.603570283** (원본 `0.6035702825813318`); 최종 구성보다 **-5.3195%p** |
| 전체 경과 시간 | **117.8220분 (약 1시간 58분)**; 설정 셀부터 설치·입력 복원·학습·최종/후보 평가까지 |
| 실행 상태 | `complete=true`, `status=selection_frozen`, `failures=[]`, 최종 Validation 완료 |
| 근거 | [학습 결과](../../results/mission3/ver9/training_metrics.json), [epoch 이력](../../results/mission3/ver9/candidate_history.json), [실행 설정](../../results/mission3/ver9/run_identity.json), [교차 보정](../../results/mission3/ver9/calibration_comparison.json), [최종 Validation](../../results/mission3/ver9/validation_metrics.json), [후보 단독 Validation](../../results/mission3/ver9/candidate_validation_metrics.json), [시간](../../results/mission3/ver9/runtime.json) |

### 저장된 학습 설정

| 항목 | 설정 |
| --- | --- |
| 대화 처리 / head | 최대 512-token window, overlap 64; CLS + mean pooling, 증상별 window attention |
| 초기화 / 학습 범위 | 공개 사전학습 DeBERTa + 새 분류기; 전체 encoder·분류기 공동 미세조정 |
| 손실 / 최대 epoch | unweighted BCE / 3 epoch, patience 2 |
| Learning rate | encoder `2e-5`, head `1e-4`, layerwise decay 0.95 |
| Batch | 통화 4 / window 4 / accumulation 8 / 유효 batch 최대 32; 후보 추론 batch 4 |
| 정규화·메모리 | weight decay 0.01, warmup 0.1, dropout 0.1, gradient checkpointing, BF16 |
| Seed | config 42; 실행 코드의 fold 학습 seed는 1,051 / 2,060 |
| Epoch 선택 | 각 Training stop의 그룹 분리 3-fold threshold 교차평가 Macro F1 우선, AP 동점 보조 |
| 모델 수 | 최대 2개; 첫 fold 실측 시간과 남은 예산으로 두 번째 실행 여부 판단, 이번 실행은 2개 모두 완료 |
| 희소 모델 | calibration 제외 Training 26,320개; 문자·단어 TF-IDF에 증상별 NB log-count-ratio를 곱한 Logistic Regression |
| 희소 설정 — 실행 코드 기본값 | 문자 최대 80,000 / 단어 최대 20,000 특징, `min_df=2`, `C=1.0`, `max_iter=500`, `alpha=1.0`, `class_weight=None` |
| 혼합 채택 기준 | 별도 calibration 5-fold에서 기존 대비 F1 최소 +0.002, 최소 3/5 fold 개선 |
| 실행 예산 | 설정 셀부터 10,800초; 최종 추론 1,500초 + Training 보정 600초 예약 |

기존 ver4 `splits.json`의 SHA256이 이번 `run_identity.json`과 일치하며 위 분할 개수는 해당 파일로 확인했습니다. 공식 Validation은 epoch·혼합·임계값 선택에 사용하지 않았다는 실행 기록(`official_validation_used_for_selection=false`)이 있습니다. NB-LR도 공통 calibration을 제외하고 학습하며 neural epoch 선택에는 관여하지 않습니다. 기존 ver4의 5개 신경망은 재학습하지 않았습니다. 설계 배경은 [ver9 RESEARCH](mission3_ver9/RESEARCH.md)에 있습니다.

### 모든 epoch 학습 이력

| Fold | Epoch | 단계 | Train loss | Stop Macro F1 | Stop Macro AP | 시간 (분) | 최적 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 1 | 0 | 초기 공개 encoder 평가 | — | 0.269861692 | 0.158024656 | 0.8261 |  |
| 1 | 1 | 전체 encoder·분류기 학습 | 0.440643 | 0.420873017 | 0.346833630 | 16.1847 |  |
| 1 | 2 | 전체 encoder·분류기 학습 | 0.330545 | 0.530754054 | 0.524465262 | 16.2114 |  |
| 1 | 3 | 전체 encoder·분류기 학습 | 0.300209 | 0.548217257 | 0.558071991 | 16.2435 | ✓ |
| 2 | 0 | 초기 공개 encoder 평가 | — | 0.273575504 | 0.164720287 | 0.8132 |  |
| 2 | 1 | 전체 encoder·분류기 학습 | 0.437663 | 0.432816768 | 0.375082343 | 16.2377 |  |
| 2 | 2 | 전체 encoder·분류기 학습 | 0.323816 | 0.540373370 | 0.554906139 | 16.3060 |  |
| 2 | 3 | 전체 encoder·분류기 학습 | 0.294254 | 0.558498197 | 0.576185676 | 16.2485 | ✓ |

학습 epoch 1~3의 합계는 fold 1 **48.6395분**, fold 2 **48.7922분**, 총 **97.4318분**입니다. epoch 0 평가와 보정·저장·최종 추론은 이 합계에 포함하지 않습니다. 두 fold 모두 epoch 3까지 stop F1이 상승해 최종 epoch를 선택했으며, 기록상 예산 중단이나 재개는 없습니다. `candidate_history.json`과 `training_metrics.json`의 이력은 일치합니다. stop F1은 반복적인 epoch 선택에 사용한 내부 지표로, 공식 Validation이나 모델 수준 OOF 점수가 아닙니다.

### 교차 보정과 기존 구성 유지

| 구성 | Calibration cross-fit Macro F1 | 기존 대비 (%p) | 최종 선택 |
| --- | ---: | ---: | :---: |
| 기존 ver4 | **0.659670866** | — | ✓ |
| 새 후보 포함 raw 확률 혼합 | 0.659546589 | -0.0124 | |
| 새 후보 포함 logit 혼합 | 0.659284140 | -0.0387 | |
| 새 후보 단독 진단 | 0.597860936 | -6.1810 | |

혼합 후보 중 raw가 더 좋았지만 기존 대비 **-0.000124277 F1 (-0.0124%p)**이고 개선 fold도 **2/5**였습니다. 최소 +0.002와 최소 3개 fold 개선 조건을 모두 충족하지 못해 `fallback_reason=candidate_gain_below_minimum`으로 기존 구성을 유지했습니다.

| Fold | raw 혼합 − 기존 F1 | 차이 (%p) | 개선 |
| ---: | ---: | ---: | :---: |
| 1 | -0.000565350 | -0.0565 |  |
| 2 | -0.000018807 | -0.0019 |  |
| 3 | +0.004222992 | +0.4223 | ✓ |
| 4 | -0.006989464 | -0.6989 |  |
| 5 | +0.001466004 | +0.1466 | ✓ |

최종 calibration fit F1은 **0.669459793**, 후보 단독 fit F1은 **0.608642912**입니다. 같은 calibration 표본으로 보정·평가한 값이며 위 cross-fit과 구분합니다. 혼합 방식 선택에도 cross-fit 점수를 재사용하므로 선택 후 점수를 독립된 일반화 성능 추정이나 통계적 유의성 검정으로 해석하지 않습니다. 이번 로그 반영에서는 학습·보정을 재실행하지 않았습니다.

### 최종 혼합 비율과 후보 단독 임계값

| Class | 기존 neural α | 새 후보 β | 최종 threshold | 후보 단독 threshold |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 1.00 | 0.00 | 0.527061 | 0.374144 |
| 구토 | 0.75 | 0.00 | 0.585741 | 0.285509 |
| 두통 | 0.75 | 0.00 | 0.414659 | 0.213756 |
| 복통 | 0.75 | 0.00 | 0.525984 | 0.389285 |
| 어지러움 | 1.00 | 0.00 | 0.491587 | 0.338866 |
| 열상 | 1.00 | 0.00 | 0.527157 | 0.362315 |
| 오심 | 0.75 | 0.00 | 0.411040 | 0.206591 |
| 전신쇠약 | 1.00 | 0.00 | 0.472925 | 0.311621 |
| 호흡곤란 | 0.75 | 0.00 | 0.585972 | 0.394687 |

최종 예측은 `α × 기존 ver4 neural + (1−α) × 기존 sparse`에 원래 임계값을 적용합니다. 새 후보 β는 전부 0입니다. 후보 단독은 **DeBERTa 2개 평균 + NB-LR의 고정 혼합**이며, DeBERTa만 평가한 점수가 아닙니다.

### 최종 Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 고열 | 0.7705 | 0.6386 | 0.6984 | 678 |
| 구토 | 0.6945 | 0.5419 | 0.6088 | 537 |
| 두통 | 0.5686 | 0.5321 | 0.5497 | 374 |
| 복통 | 0.8931 | 0.7673 | 0.8254 | 838 |
| 어지러움 | 0.6183 | 0.7054 | 0.6590 | 774 |
| 열상 | 0.8805 | 0.9071 | 0.8936 | 463 |
| 오심 | 0.3205 | 0.5431 | 0.4031 | 429 |
| 전신쇠약 | 0.5645 | 0.6252 | 0.5933 | 651 |
| 호흡곤란 | 0.7188 | 0.6443 | 0.6795 | 492 |
| **Macro avg** | **0.6699** | **0.6561** | **0.656766** | **5,236** |

### 새 후보 단독 Validation 상세

| Class | Precision | Recall | F1-score | Support | 최종 구성 대비 F1 (%p) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 고열 | 0.7102 | 0.5457 | 0.6172 | 678 | -8.1206 |
| 구토 | 0.5723 | 0.5531 | 0.5625 | 537 | -4.6287 |
| 두통 | 0.5364 | 0.3743 | 0.4409 | 374 | -10.8779 |
| 복통 | 0.8821 | 0.7232 | 0.7948 | 838 | -3.0663 |
| 어지러움 | 0.5998 | 0.6680 | 0.6320 | 774 | -2.6993 |
| 열상 | 0.8706 | 0.8575 | 0.8640 | 463 | -2.9634 |
| 오심 | 0.3094 | 0.5431 | 0.3942 | 429 | -0.8867 |
| 전신쇠약 | 0.4813 | 0.5730 | 0.5231 | 651 | -7.0153 |
| 호흡곤란 | 0.6700 | 0.5488 | 0.6034 | 492 | -7.6176 |
| **Macro avg** | **0.6258** | **0.5985** | **0.603570** | **5,236** | **-5.3195** |

- 2026-09-19에 [최종 CSV](../../results/mission3/ver9/inference.csv)·[후보 단독 CSV](../../results/mission3/ver9/candidate_inference.csv)를 로컬 Validation JSON **3,640개**와 대조했습니다. 두 CSV 모두 파일명 중복·누락·추가 행이 없고 지정 9개 증상만 포함하며, 재계산한 모든 Precision·Recall·F1·Support와 Macro F1이 저장된 JSON과 일치했습니다.
- 후보 단독은 최종 구성과 통화 **1,567개**, 이진 판정 **2,195개**가 다르며 9개 증상 F1이 모두 낮습니다. 후보 단독 0.603570은 ver8 후보 단독 0.645292보다 **-4.1722%p** 낮습니다.
- 최종 CSV와 기존 버전의 차이는 다음과 같습니다. 같은 파일명을 맞춰 증상 집합을 비교했으며 행 순서·문자열 표현 차이는 제외했습니다.

| 비교 대상 | 예측이 다른 통화 | 다른 이진 판정 |
| --- | ---: | ---: |
| ver4 | 3 | 3 |
| ver5 | 495 | 539 |
| ver6 | 2 | 2 |
| ver7 | 2 | 2 |
| ver8 | 2 | 2 |

ver6·ver7·ver8 대비 달라진 2개 판정은 모두 **전신쇠약 오탐이 제거된 경우**입니다. 최종 점수는 ver8보다 **+0.0096%p**, 기존 최고였던 ver5보다 **+0.0048%p** 높습니다. 그러나 최종 선택은 기존 ver4이고 새 후보 비중은 0이므로 **새 DeBERTa/NB-LR 학습으로 인한 개선이 아닙니다.** 같은 기존 모델 구성의 재추론 결과에서 소수 판정이 달라진 것이며, 그 차이가 발생한 정확한 수치 연산 원인은 이번 CSV 대조만으로 확정하지 않습니다.

### 실제 실행 시간과 해석

| 단계 | 시간 (분) |
| --- | ---: |
| 패키지 설치 | 0.2449 |
| 입력 복원 | 2.4230 |
| 공개 사전학습 모델 준비 | 0.7628 |
| 후보 학습·보정·최종 선택 저장 | 106.0268 |
| 최종 Validation 추론·평가 | 5.7203 |
| 후보 단독 진단 | 2.6166 |
| **단계 시간 합계** | **117.7945** |
| **전체 경과 시간** | **117.8220** |

전체 경과 **7,069.3224초**로 10,800초 예산 안에 최종·후보 단독 평가까지 완료됐습니다. 단계 합계와 전체 경과의 약 1.65초 차이는 단계 밖 경과 시간입니다. `training_metrics.json`의 내부 학습 처리 시간 105.8436분과 epoch 합계 97.4318분은 위 학습 단계 안에 포함되므로 중복 합산하지 않습니다. GPU CU·요금은 기록되지 않았습니다.

encoder 계열·손실·희소 특징·새 모델 수를 함께 바꾼 실행이므로 개별 변경의 기여를 분리할 수 없습니다. 두 fold의 stop F1은 학습 중 상승했지만 후보 단독과 혼합 모두 기존 구성을 넘지 못했습니다. 이 결과만으로 DeBERTa 계열 전체의 성능을 단정하지 않습니다.

### 결과 보관 및 재현 방법

- `mission3_ver9_model.ckpt`는 실제 신경망 가중치가 아닌 **최종 조합·상대 경로·SHA256을 담은 JSON manifest**입니다. `base/`의 기존 5개 fold 가중치·sparse·tokenizer와 새 `candidate_1.ckpt`, `candidate_2.ckpt`, `nb_ratio_lr.joblib`, `tokenizer/`를 함께 보관합니다.
- 로컬에서 manifest가 참조하는 **17개 파일 전체의 SHA256이 모두 일치**했습니다. `submission/`과 현재 `src/mission3/mission3_ver9/`의 실행 Python 파일 24개도 각각 `run_identity.json`의 코드 해시와 일치합니다. 저장된 모델 파일의 무결성 확인이며, GPU 재학습·재추론 검증은 아닙니다.
- 같은 상위 폴더의 `ver9-20260919T052548Z-1-001.zip`~`005.zip`에도 가중치·결과가 분산 보관돼 있습니다. ZIP 이름의 2026-09-19는 위 실제 실행 시각과 구분합니다. 이번 확인에서는 원본 결과나 ZIP을 수정하지 않았습니다.
- [노트북](mission3_ver9.ipynb)과 기존 `mission3_ver5_inputs.zip`으로 같은 설정의 Colab GPU 실행을 재현할 수 있습니다. 새로운 실험은 `OUTPUT_DIR`를 다른 폴더로 지정합니다. 현재 노트북에는 셀 출력이 없으므로 학습 이력과 환경은 위 결과 JSON을 근거로 합니다.

프로젝트 루트에서 저장된 CSV만 재평가하려면 다음을 실행합니다.

```bash
python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver9/inference.csv

python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver9/candidate_inference.csv
```

저장된 모델의 재추론 명령은 다음과 같습니다. 아래는 재현 방법이며 이번 로그 반영 중 실행한 명령은 아닙니다.

```bash
python results/mission3/ver9/submission/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver9/mission3_ver9_model.ckpt \
  --output outputs/mission3/ver9/inference.csv
```

## ver1~ver9 결과 요약

| 실행 버전 | Training 내부 지표 | Validation Macro F1 | 비교 범위 |
| --- | --- | --- | --- |
| ver1 모듈형 baseline | 별도 실행 기록 없음 | 별도 실행 기록 없음 | 코드 구현 기록 |
| ver2 단일 노트북 (`results/mission3/ver1`) | Holdout F1 0.6009 | 0.6001 | 최초 보관 baseline |
| ver3 RoBERTa-base | Holdout F1 0.6444 | 0.6260 | baseline 대비 +2.59%p |
| ver4 RoBERTa-large ensemble | Calibration cross-fit F1 0.659671 | 0.656649 | ver3 대비 +3.06%p |
| ver5 추가 학습 + logit 혼합 | Calibration cross-fit F1 0.659779 | **0.656718** | ver4 대비 +0.0069%p; 최적 후보는 추가 학습 전 epoch 0 |
| ver6 A–F 비교 + D 미세조정 | 후보 cross-fit F1 0.662117, 최종 선택 0.659671 | 0.656670 | 기존 ver4로 fallback; D head epoch 2, encoder 추가 미세조정 미채택 |
| ver7 KoELECTRA 전체 공동 학습 | 혼합 cross-fit F1 0.658666, 최종 선택 0.659671; 후보 단독 0.631152 | 최종 **0.656670** / 후보 단독 **0.638110** | 기존 ver4로 fallback; 새 후보 최적 epoch 2, 최종 CSV는 ver6과 동일 |
| ver8 RoBERTa-large + token attention (B) | 혼합 cross-fit F1 0.657451, 최종 선택 0.659671; 후보 단독 0.650081 | 최종 **0.656670** / 후보 단독 **0.645292** | 기존 ver4로 fallback; 기록상 최적 epoch 3, 최종 CSV는 ver6·ver7과 동일; 로컬 후보 가중치 불일치 |
| ver9 DeBERTa-v3 2-fold + NB-LR | 혼합 cross-fit F1 0.659547, 최종 선택 0.659671; 후보 단독 0.597861 | 최종 **0.656766** / 후보 단독 **0.603570** | 기존 ver4로 fallback; 두 fold 모두 최적 epoch 3; 최종 점수의 미세 차이는 새 후보 학습 효과가 아님 |

Training 내부 수치는 분할과 선택 절차가 달라 직접적인 성능 순위에 사용하지 않습니다. 위 Validation 비교는 동일한 3,640개 파일에 대한 저장된 예측을 기준으로 하며, 별도의 Test 점수나 반복 seed 실험 결과는 보관되어 있지 않습니다.

ver9의 최종 저장 점수 0.656766이 수치상 가장 높지만 이전 최고 ver5와의 차이는 +0.0048%p에 불과합니다. ver9도 기존 ver4 구성을 선택했고 새 후보 비중은 0이므로 이 미세 차이를 새 모델 학습의 성능 개선으로 해석하지 않습니다. ver5 역시 최적 후보가 추가 학습 전 epoch 0으로, 추가 학습으로 인한 실질적인 개선은 확인되지 않았습니다.

ver6도 큰 성능 향상을 확인하지 못했으며, ver6의 최종 CSV는 새 D 후보가 아닌 기존 ver4 구성을 사용한 결과입니다.

ver7은 새 KoELECTRA 전체를 3 epoch 학습했으나 혼합 채택 기준을 통과하지 못했습니다. 최종 결과는 ver6과 같고, 별도로 평가한 후보 단독도 최종 구성보다 1.8560%p 낮았습니다. **이번 실행에서도 최종 성능 개선은 확인되지 않았습니다.**

ver8은 RoBERTa-large 후보 B를 3 epoch 학습한 기록이 있으며 후보 단독 F1은 ver7보다 높아졌습니다. 그러나 혼합 채택 기준을 통과하지 못해 **최종 예측은 ver6·ver7과 동일하며 성능 개선은 확인되지 않았습니다.** 저장된 CSV의 점수는 검증했지만 현재 로컬 후보 체크포인트가 최종 기록과 달라, 제출·재추론 전에 올바른 가중치를 다시 확보해야 합니다.

ver9는 DeBERTa 2개를 각각 3 epoch 학습하고 NB-LR과 혼합했으나, Training 교차 보정의 개선 폭·개선 fold 수 조건을 모두 통과하지 못했습니다. 후보 단독은 최종 구성보다 5.3195%p 낮았고 **새 후보의 성능 개선은 확인되지 않았습니다.** 설치·입력 복원·학습·최종/후보 평가까지 약 117.82분이 걸렸으며, 보관된 모델 파일의 해시는 모두 일치합니다.
