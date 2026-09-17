# Mission 3 ver6

기존 ver4 encoder를 재사용하며 **손실 함수, 직접적인 증상별 token attention, 작은 분류기의 정규화**를 비교한다.
실행 파일은 `src/mission3/mission3_ver6.ipynb` 하나이며 필요한 Python 소스를 노트북에 포함한다.
기존 **`mission3_ver5_inputs.zip`을 그대로 사용**한다. ver5 실행 결과나 새 데이터 ZIP은 필요 없다.

## Colab에서 실행

1. Google Drive 공유 문서함의 `mission3_ver5_inputs.zip`에서 **우클릭 → 정리 → 바로가기 추가 → 내 드라이브**를 선택한다.
2. 내 드라이브 **바로 아래**에 같은 이름의 바로가기를 둔다. 파일을 다시 복사하거나 직접 압축을 풀 필요는 없다.
3. `mission3_ver6.ipynb`를 Colab에서 열고 GPU 런타임을 선택한다.
4. 위에서부터 모두 실행하고 바로가기를 만든 계정으로 Drive를 연결한다.

첫 설정 셀의 기본 경로는 다음과 같다.

```python
INPUT_ZIP = "/content/drive/MyDrive/mission3_ver5_inputs.zip"
BASE_DIR = None
TRAIN_LABEL_DIR = None
VALID_LABEL_DIR = None
OUTPUT_DIR = None
INFERENCE_CACHE_DIR = None
FROZEN_FEATURE_CACHE_DIR = None
RUN_VALIDATION = True
```

바로가기를 하위 폴더에 두었다면 `INPUT_ZIP`만 해당 절대 경로로 바꾼다.
ZIP에 Training·Validation JSON이 모두 있으므로 별도 공유 `DATA` 폴더는 연결하지 않는다.
Drive를 연결한 후 바로가기를 추가해서 보이지 않으면 경로 준비 셀의
`drive.mount("/content/drive", force_remount=True)`로 한 번 재연결한다.
실행 중 계정을 바꾸지 말고 바로가기 대상 파일의 읽기 권한을 유지한다.

## 입력과 새 결과의 위치

| 용도 | 기본 경로 | 보관 방식 |
|---|---|---|
| 입력 ZIP 바로가기 | `/content/drive/MyDrive/mission3_ver5_inputs.zip` | Drive에 유지 |
| 압축 해제한 기존 모델 | `/content/mission3_ver5_inputs/DCC_AIdeas/results/mission3/ver4` | Colab 로컬, 새 세션에서 재복원 |
| Training·Validation JSON | `/content/mission3_ver5_inputs/DCC_AIdeas/DATA/대학부 데이터/` | ZIP에서 자동 연결 |
| **새 ver6 결과** | **`/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver6`** | **Drive에 저장** |
| 동결 encoder 특징 캐시 | `/content/mission3_ver6_features` | Colab 로컬, 같은 런타임에서 재사용 |
| 원시 예측 캐시 | `/content/drive/MyDrive/DCC_AIdeas/results/mission3/prediction_cache` | Drive, 실험 간 조건 일치 시 재사용 |

`results/mission3/ver4`라는 입력 폴더가 보이는 것은 정상이다. 이를 ver6으로 이름을 바꾸지 않는다.
ZIP은 약 6.86GB이며 ver4 필수 파일 13개, Training JSON 29,200개, Validation JSON 3,640개를 포함한다.
WAV는 사용하지 않는다. ZIP 복원 시 manifest·CRC·SHA256을 검사하고, 같은 런타임에서 검증된 파일은 재사용한다.

Colab 세션이 삭제되면 압축 해제 파일과 특징 캐시는 사라진다. 새 세션에서는 ZIP을 다시 복원한다.
특징은 **미완료 분류기 실험이 남았을 때만** 다시 계산한다. 분류기 비교가 끝나 encoder 미세조정만 재개하거나
이미 완료된 후보·최종 결과를 재사용할 때는 동결 특징 추출을 생략한다.
특징 캐시는 원래 Training fit·stop **26,320개**만 포함하며 별도 calibration **2,880개**는 제외한다.
크기는 실제 token 수·hidden 크기로 추정하고 가용 공간을 확인한다. 해당 입력의 token별 encoder 특징을 저장하므로
6개의 작은 분류기만 학습하는 시간 외에 최초 특징 추출·읽기·저장 비용이 존재한다.

## ver5에서 바뀐 점

ver5는 기존 출력에 추가한 attention branch를 짧게 학습한 뒤, stop AP 하락으로 최적 epoch가 0에 머물렀다.
이번에는 ver4를 공통 출발점으로 삼아 아래 실험을 수행한다. 학습 입력의 label에 따라 텍스트 전처리를 바꾸지 않는다.

| 실험 | 분류기 | 손실·정규화 |
|---|---|---|
| A | 기존 ver4 global/window 분류기 | Asymmetric Loss |
| B | 기존 ver4 global/window 분류기 | Distribution-Balanced Loss |
| C | 기존 ver4 global/window 분류기 | ASL + soft Macro F1 |
| D | 직접적인 label-wise token attention | Weighted BCE |
| E | D | Weighted BCE + 분류기의 Bernoulli R-Drop |
| F | E + 작은 label dependency 모듈 | fit에서 구한 증상 관계 활용 |

A–F는 **한 번 추출한 동결 encoder 특징**을 공유한다. 각 실험별로 전체 encoder 체크포인트를 복제하지 않는다.
A–C는 분류기에 필요한 CLS·평균 요약을 메모리에 재사용하여 매 epoch마다 전체 token 특징을 다시 읽지 않는다.
D–F는 증상별 attention을 위해 token별 특징을 사용한다.
D–F에서는 각 증상의 query가 token 표현을 선택하고 window를 모아 분류한다.
최종 출력이 해당 표현에 직접 의존하며, ver5처럼 기존 logit에 0으로 시작하는 branch만 더하는 방식이 아니다.
R-Drop의 두 dropout 경로는 작은 분류기에 적용한다. 최종 encoder 미세조정에서는 기본적으로 R-Drop을 끈다.
soft Macro F1 보조항도 작은 분류기 단계에만 사용하며 encoder 미세조정의 작은 microbatch에는 적용하지 않는다.

선택된 분류기 하나만 encoder 상위 2개 층과 최대 1 epoch 미세조정한다.
이 단계에서 stop F1이 떨어지면 미세조정 전 가중치를 유지한다. encoder를 크게 바꾸거나 더 큰 모델을 내려받지 않는다.
**ver6의 최종 `best_epoch=0`은 선택된 분류기를 유지하고 encoder 미세조정만 채택하지 않았다는 뜻이다.**
분류기 학습이 최종 가중치에 반영됐는지는 `head_trials[].best_epoch`에서 따로 확인한다.
이미 학습한 epoch가 선택될 수도 있고, 해당 분류기의 초기 epoch 0이 유지될 수도 있다.
원 논문의 모델·실험 전체를 그대로 재현하는 것은 아니다. 출처·구현 대응·제외 범위는 [RESEARCH.md](RESEARCH.md)를 확인한다.

## 데이터 분할과 선택 기준

- 원래 ver4 fold 1의 **fit 21,032개**만 학습에 사용한다. class 빈도와 label dependency 통계도 fit에서만 계산한다.
- 원래 **stop 5,288개**의 대화 그룹을 분리한 3-fold threshold 교차평가로 epoch·A–F 후보를 비교한다.
  각 평가 fold의 threshold는 나머지 stop fold에서 맞춘다. Macro F1을 우선하고 AP는 보조로 기록한다.
- 별도 **Training calibration 2,880개**의 5-fold 교차평가로 기존 앙상블·새 후보의 혼합 비율과 임계값을 비교한다.
- 최종 후보는 기존 calibration 교차평가 Macro F1보다 **0.002 이상** 좋고, **5개 fold 중 최소 3개**에서
  F1이 엄격히 개선되어야 선택한다. 기준을 충족하지 못하거나 동점이면 기존 ver4 보정을 유지한다.
- **공식 Validation 3,640개 전체**는 최종 선택 후 예측·평가에만 사용한다. 그 정답으로 재학습·재보정하지 않는다.

이 비교는 고정된 ver4 fold 1과 내부 threshold 교차평가를 사용한다.
**5개 encoder를 각각 새로 학습한 5-fold 모델 OOF 실험이 아니다.** 같은 stop에 여러 후보·epoch를 비교하므로
stop 점수에는 선택 편향이 남는다. calibration도 반복 연구에 계속 활용하면 선택 편향이 생길 수 있다.
새 구성을 고르는 최소 개선 기준은 작은 변화의 과대해석을 줄이는 장치이며, Validation 또는 Test 개선을 보장하지 않는다.

이전 전체 Validation 기록은 ver4 **0.6566489480388843**, ver5 **0.6567177583831921**이다.
ver6의 새 F1은 실제 실행 전에는 알 수 없다. notebook은 마지막에 이 두 기록과 비교하며 차이를 선택 과정에 되먹이지 않는다.

## 기본 학습 설정

| 설정 | 기본값 |
|---|---|
| `profiles` | `A, B, C, D, E, F` |
| `head_epochs`, `head_patience` | 4, 2 |
| `head_batch_size`, `head_learning_rate` | 32, `3e-4` |
| `stop_crossfit_splits` | 3 |
| `finetune_epochs`, `finetune_top_layers` | 1, 2 |
| `learning_rate`, `finetune_head_learning_rate` | `2e-6`, `5e-5` |
| `finetune_rdrop` | `False` |
| `min_calibration_gain`, `min_calibration_improved_folds` | 0.002, 3 |
| notebook `bundle_base` | `False` |

CLI의 `--auto_batch`는 GPU 메모리에 따라 encoder batch/window 크기를 선택한다.
같은 실험을 재개할 때는 저장된 batch 설정을 유지한다. **미완료 학습 재개에는 같은 GPU 종류·AMP 정밀도와
관련 런타임 패키지 버전이 필요하다.** 다른 실행 조건의 stop 점수가 섞이지 않도록 중간 저장 파일에서 검사한다.
최종 결과까지 완료된 실행은 다른 GPU에서도 재사용할 수 있다. 이때 기존 원시 예측 캐시의 실행 조건이 다르면 다시 추론한다.
학습 설정·코드·데이터를 바꾼다면
새 `OUTPUT_DIR`을 지정한다. 동일한 런타임에서는 호환되는 특징 캐시를 재사용할 수 있다.
총 CU는 아직 실측하지 않았으며, 높은 GPU 사양이 항상 낮은 총 CU를 의미하지는 않는다.

## 저장 결과와 추론

기본 notebook은 기존 5개 모델을 결과 폴더에 다시 복사하지 않는 `bundle_base=False`로 실행한다.
**결과 폴더와 원래 입력 ZIP을 함께 보관해야 한다.** 별도 CLI에서 `bundle_base=True`를 설정하면
기존 모델까지 포함한 출력 묶음을 만들 수 있지만 저장 용량이 증가한다.

- `mission3_ver6_model.ckpt`: 최종 구성·보정값을 저장한 manifest.
- `candidate_best.ckpt`: 선택된 후보의 encoder와 분류기 가중치.
- `mission3_ver6_metrics.json`: A–F 학습 이력·선택, 미세조정, calibration 교차평가와 fallback 이유.
- `predictions/<실행시각>/mission3.csv`: 제출 형식 예측.
- 같은 예측 폴더의 `validation_metrics.json`: 정답이 있을 때 전체·증상별 precision, recall, F1, support.
- `notebook_runtime_<실행시각>.json`: 입력 준비, 학습·보정, 최종 추론·평가의 벽시계 시간. 패키지 설치 시간과 CU는 별도.
- `submission/`: 다른 환경에서 실행할 추론 소스·패키지 목록·문서.

```bash
python submission/inference.py --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \
  --ckpt_path mission3_ver6_model.ckpt --base_dir "ZIP에서 복원한 ver4 폴더" \
  --output mission3.csv
```

`--audio_dir`는 대회 실행 인터페이스와의 호환용이며 음성은 읽지 않는다.
`--cache_dir`로 공유 예측 캐시를 지정할 수 있다. 정답과 보정값은 원시 예측 캐시에 저장하지 않으며,
입력 text·가중치·tokenizer·추론 코드·관련 실행 조건이 같은 경우에만 재사용한다.
`--evaluate`는 CSV 저장 이후 정답이 있는 입력을 평가하며 모델을 변경하지 않는다.

## 노트북 다시 생성

```bash
python src/mission3/mission3_ver6/build_notebook.py
```

노트북은 현재 runtime 소스를 포함한다. runtime 수정 후 재생성해야 Colab에 올린 단일 파일에 변경 내용이 반영된다.
로컬의 작은 CPU 검증은 수치·재개·캐시·추론 구현을 확인하는 용도이고 실제 전체 데이터의 F1·GPU 시간 검증을 대신하지 않는다.
