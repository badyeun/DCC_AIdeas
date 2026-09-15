# Mission 3 실험 기록

> Mission 1의 실험 기록과 같은 형식으로 실행 설정, 학습 이력, Validation 상세,
> 해석 및 재현 방법을 정리합니다. 아래 수치는 저장된 결과 JSON과 CSV에 근거합니다.
> 코드의 기본 설정과 실제 저장된 실행 설정을 구분하며, 남아 있지 않은 실행 기록은 임의로 채우지 않습니다.

## 기록 범위와 버전 대응

버전은 **실행 코드의 ver1~ver4**를 기준으로 구분합니다. 기존 [ver4 실행 안내](mission3_ver4/README.md)의 결과 대응표는 `mission3_all_in_one_ver2.ipynb`의 결과를 `results/mission3/ver1`에 연결합니다. 따라서 이 폴더의 점수는 EXP-002에 기록하고, ver1과 ver2의 독립된 두 번의 실행 결과로 중복 집계하지 않습니다.

| 실험 | 실행 버전 | 코드 / 노트북 | 확인된 결과 |
| --- | --- | --- | --- |
| EXP-001 | ver1 | `train.py`, `mission3_experiment.ipynb` | 모듈형 baseline 구현; 별도 실행 점수·출력 미보관 |
| EXP-002 | ver2 | `mission3_all_in_one_ver2.ipynb` | `results/mission3/ver1/`의 모델·CSV·metrics |
| EXP-003 | ver3 | `mission3_ver3.ipynb` | `results/mission3/ver3/`의 모델·CSV·15 epoch 이력 |
| EXP-004 | ver4 | `mission3_ver4.ipynb`, `mission3_ver4/*.py` | `results/mission3/ver4/`의 5-fold 이력·보정·Validation 결과 |

- 입력은 양측 대화의 전사 `text`이며, 대상은 `고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란` 9개입니다. 각 증상의 유무를 독립적으로 예측합니다.
- 평가 지표는 **9개 증상의 Macro F1**입니다. 한 통화에 여러 증상이 있을 수 있으므로 아래 표의 증상별 Support 합계는 통화 수와 다릅니다.
- `Training 내부 holdout`, `stopping Macro AP`, `calibration cross-fit`, `공식 Validation`은 서로 다른 평가 범위입니다. AP는 임계값을 정하기 전 예측 순위를 평가하는 지표이며 F1과 직접 비교하지 않습니다.
- 2026-09-15에 저장된 ver1·ver3·ver4 CSV를 로컬 Validation JSON 3,640개와 다시 대조했습니다. 세 CSV 모두 파일명이 1:1 대응하고 중복·누락·추가 행이 없으며, 재계산한 증상별 F1과 Macro F1은 저장된 JSON과 일치했습니다. Precision·Recall·Support도 같은 CSV와 정답으로 계산했습니다. 모델 재학습이나 추론 재실행 결과는 아닙니다.
- 현재 ver1~ver4 실험 노트북에는 셀 출력이 저장되어 있지 않습니다. 실행 날짜, 구체적인 CPU/GPU 기종, 저장되지 않은 학습 시간·손실은 확인할 수 없습니다. ver3·ver4의 학습 이력은 별도 JSON에 남아 있습니다.

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

## ver1~ver4 결과 요약

| 실행 버전 | Training 내부 지표 | Validation Macro F1 | 비교 범위 |
| --- | --- | --- | --- |
| ver1 모듈형 baseline | 별도 실행 기록 없음 | 별도 실행 기록 없음 | 코드 구현 기록 |
| ver2 단일 노트북 (`results/mission3/ver1`) | Holdout F1 0.6009 | 0.6001 | 최초 보관 baseline |
| ver3 RoBERTa-base | Holdout F1 0.6444 | 0.6260 | baseline 대비 +2.59%p |
| **ver4 RoBERTa-large ensemble** | Calibration cross-fit F1 0.6597 | **0.6566** | ver3 대비 +3.06%p |

Training 내부 수치는 분할과 선택 절차가 달라 직접적인 성능 순위에 사용하지 않습니다. 위 Validation 비교는 동일한 3,640개 파일에 대한 저장된 예측을 기준으로 하며, 별도의 Test 점수나 반복 seed 실험 결과는 보관되어 있지 않습니다.
