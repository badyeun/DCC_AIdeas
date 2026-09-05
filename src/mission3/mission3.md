# Mission 3 첫 도전 가이드

이 문서는 AI 모델을 만들어 본 적이 없는 사람이 Colab에서 Mission 3을 학습하고 결과를 저장할 수 있도록 설명합니다.

권장 실행 방식은 [mission3_all_in_one.ipynb](mission3_all_in_one.ipynb) 파일 하나를 Colab에 올린 뒤 위에서부터 순서대로 실행하는 방식입니다.

## 1. 문제를 한 문장으로 이해하기

119 신고 대화의 **전사 텍스트만** 읽고 환자에게 다음 9개 증상이 있는지 각각 예측하는 문제입니다.

`[고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란]`

한 통화에 증상이 0개일 수도, 여러 개일 수도 있기 때문에 **다중 레이블 분류**라고 부릅니다. 예를 들어 `"머리도 아프고 토했어요"`는 `두통`과 `구토`가 동시에 정답일 수 있습니다.

## 2. 반드시 지켜야 할 규칙

- 학습은 제공된 **서울 Training** 데이터만 사용합니다.
- Validation 데이터는 학습과 임계값 조정에 사용하지 않습니다.
- 모델 입력은 JSON의 `utterances[].text`만 사용합니다.
- `speaker`, `startAt`, `endAt`, `gender`, `address` 등을 모델 입력으로 사용하면 안 됩니다.
- `symptom`은 Training 학습 정답(Y) 또는 최종 평가 정답으로만 사용합니다.
- 9개 외의 symptom만 정답에서 제외하고, 해당 통화 샘플 자체는 제거하지 않습니다.
- 외부 데이터와 GPT 같은 상용 AI API를 사용하지 않습니다.
- 평가 점수는 9개 증상별 F1의 평균인 **Macro F1**입니다. Accuracy가 아닙니다.

## 3. 단일 노트북과 제출용 파일의 차이

### Colab 실험용

`mission3_all_in_one.ipynb` 하나에 다음 기능이 모두 들어 있습니다.

- 패키지 설치
- Google Drive 연결
- `/content`의 Training/Validation JSON 폴더 자동 탐색
- 데이터 개수와 증상 분포 확인
- TF-IDF + Logistic Regression 학습
- Training 내부 Macro F1 계산
- Validation CSV 생성과 Macro F1 계산
- 모델, CSV, 점수 JSON의 Google Drive 저장

### 최종 제출용

출제 PDF는 `.ipynb` 외에도 다음 파일을 요구합니다. 단일 노트북을 만들었다고 기존 파일을 삭제하면 안 됩니다.

- `common.py`: JSON 텍스트와 정답 처리
- `train.py`: 제출 가능한 학습 코드
- `inference.py`: 지정된 명령 한 번으로 제출 CSV 생성
- `evaluate.py`: Validation Macro F1 확인
- `requirements.txt`: 필요 패키지
- 학습 완료 모델 `.ckpt`
- 학습 출력 로그가 저장된 `.ipynb`

## 4. Colab 커널 설정

현재 TF-IDF 베이스라인은 GPU를 사용하지 않습니다.

```text
런타임 > 런타임 유형 변경
> 하드웨어 가속기: 없음
> 런타임 모양: 표준 RAM
```

나중에 KLUE-RoBERTa 같은 Transformer를 학습할 때만 GPU 커널을 사용합니다.

## 5. Colab 데이터 준비

Mission 3은 WAV 음성을 사용하지 않습니다. 필요한 파일은 JSON입니다.

| 구분 | JSON 개수 | 대략적인 용량 |
|---|---:|---:|
| Training | 29,200 | 191MB |
| Validation | 3,640 | 24MB |
| 합계 | 32,840 | 215MB |

전체 30GB 데이터를 이미 `/content`에 풀어 두었다면 그대로 사용해도 됩니다. 노트북은 WAV를 읽지 않고 다음 두 폴더만 자동으로 찾습니다.

```text
TL_서울_구급
VL_서울_구급
```

폴더 이름은 바꾸지 않는 것이 좋습니다. 노트북은 Google Drive를 느린 탐색에서 제외하고 `/content`만 검색합니다.

## 6. Step 1 - 단일 노트북 열기

1. Colab에서 `파일 > 노트 업로드`를 누릅니다.
2. `mission3_all_in_one.ipynb`를 선택합니다.
3. 런타임을 CPU로 연결합니다.
4. `런타임 > 모두 실행`을 누릅니다.

처음 Drive 연결 요청이 나오면 Colab Pro를 구매한 Google 계정을 선택하고 권한을 허용합니다.

## 7. Step 2 - 데이터 탐색 결과 확인

자동 탐색 셀의 출력이 다음과 같아야 합니다.

```text
Training JSON 개수: 29200
Validation JSON 개수: 3640
```

`TL_서울_구급 폴더를 찾지 못했습니다`가 나오면 압축이 아직 끝나지 않았거나 폴더 이름이 바뀐 것입니다.

## 8. Step 3 - 학습 실행과 점수 이해

노트북은 다음 순서로 학습합니다.

1. Training을 80%와 20%로 나눉니다.
2. Training 80%로 모델을 학습합니다.
3. Training 내부 20%로 증상별 임계값과 Macro F1을 구합니다.
4. Validation을 넣지 않고 Training 전체로 최종 모델을 다시 학습합니다.
5. Validation의 `text`로만 예측 CSV를 만듭니다.
6. 완성된 예측과 Validation 정답을 비교해 최종 Macro F1을 구합니다.

출력에서 가장 중요한 값은 다음 두 개입니다.

- `Training 내부 holdout Macro F1`: 모델과 설정을 비교할 때 사용
- `Validation Macro F1`: 선택이 끝난 모델의 최종 성능 확인용

Validation 점수를 보며 임계값을 반복해 바꾸면 안 됩니다.

## 9. Step 4 - Drive 영구 저장 확인

노트북은 다음 결과를 Drive에 자동 저장합니다.

```text
MyDrive/DCC_AIdeas/mission3_outputs/
├─ mission3_model.ckpt
├─ mission3.csv
└─ mission3_metrics.json
```

- `mission3_model.ckpt`: 학습된 TF-IDF와 9개 증상 분류기
- `mission3.csv`: `label file name,symptom` 형식의 예측 결과
- `mission3_metrics.json`: 증상별 F1, Macro F1, 임계값

Colab의 `/content` 데이터는 런타임이 삭제되면 사라지지만, Drive에 저장된 세 파일은 남습니다. 노트북도 실행 출력이 보이는 상태로 저장해 학습 로그를 남깁니다.

## 10. 모델이 배우는 방식

1. `TF-IDF`가 대화에 나타난 문자 조각을 숫자 특징으로 바꿉니다. 예를 들어 `"숨이 차요"`의 `"숨이"`, `"이 차"`같은 조각을 기억합니다.
2. Logistic Regression이 각 증상의 확률을 만듭니다.
3. 증상이 9개이므로 `있음/없음` 분류기 9개를 학습합니다.
4. 각 확률이 증상별 임계값보다 크면 해당 증상을 출력합니다.

이 방식은 최첨단 모델은 아니지만 빠르고 이해하기 쉬워서 첫 기준점으로 적합합니다.

## 11. 점수를 올리는 실험 순서

한 번에 여러 설정을 바꾸면 어떤 변경이 좋았는지 알 수 없습니다.

1. 현재 `Training 내부 holdout Macro F1`을 기록합니다.
2. `MAX_FEATURES` 값을 `50000`, `100000`, `200000`으로 하나씩 비교합니다.
3. `ngram_range=(2, 5)`를 `(2, 4)` 또는 `(3, 6)`으로 하나씩 비교합니다.
4. 증상별 F1을 보고 자주 놓치는 증상과 대화 표현을 확인합니다.
5. TF-IDF 베이스라인을 완성한 후 Colab GPU에서 `klue/roberta-base`를 두 번째 모델로 비교합니다.
6. 두 모델이 다른 실수를 한다면 Training 내부 분할에서 확률 앙상블을 실험합니다.

모든 모델 선택과 임계값 조정은 Training 내부 분할로 합니다. Validation은 선택이 끝난 뒤 최종 확인에만 사용합니다.

## 12. 로컬 스크립트 실행법

단일 노트북 외에 로컬이나 최종 제출 환경을 확인하려면 `DCC_AIdeas` 폴더에서 다음을 실행합니다.

### 학습

```bash
python src/mission3/train.py \
  --label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --ckpt_path saved_models/mission3_model.ckpt
```

### 제출 CSV 생성

```bash
python src/mission3/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path saved_models/mission3_model.ckpt \
  --output outputs/mission3.csv
```

Mission 3은 음성을 사용하지 않지만 주최 측 공통 명령 규격을 맞추기 위해 `--audio_dir`도 받습니다.

### Validation Macro F1 확인

```bash
python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction outputs/mission3.csv
```

CSV 헤더는 정확히 `label file name,symptom`이어야 합니다. PDF에는 symptom 칸의 세부 구분자까지 명시되어 있지 않으므로 주최 측 sample submission이 추가로 제공되면 그 형식을 최우선으로 맞춰야 합니다.

## 13. 자주 만나는 문제

- `TL_서울_구급 폴더를 찾지 못했습니다`: `/content`에 압축을 풀었는지와 폴더 이름을 확인합니다.
- `ModuleNotFoundError`: 노트북의 첫 패키지 설치 셀부터 다시 실행합니다.
- 학습 중 메모리 부족: 학습 설정 셀의 `MAX_FEATURES = 100_000`을 `50_000`으로 낮춥니다.
- 학습이 너무 느림: `N_JOBS = 1`은 메모리를 보호하는 설정입니다. RAM 여유가 확인된 뒤에만 늘립니다.
- Drive에 모델이 없음: Drive 연결 셀에서 권한을 허용했는지와 `mission3_outputs` 폴더를 확인합니다.
- 재연결 후 `/content` 데이터가 없음: Colab 런타임이 삭제되면 임시 디스크도 삭제됩니다. JSON 데이터를 다시 올려야 합니다.
- CSV가 Excel에서 한글이 깨짐: UTF-8 BOM으로 저장되므로 일반적으로 바로 열립니다.
- F1이 0인 증상이 있음: 해당 증상이 드물거나 예측이 어려울 수 있으므로 정답 개수, 예측 개수, 임계값을 확인합니다.

## 14. 최종 제출 전 체크리스트

- [ ] 서울 Training JSON만 학습에 사용했다.
- [ ] 모델 입력으로 `utterances[].text`만 사용했다.
- [ ] Validation을 학습과 임계값 조정에 사용하지 않았다.
- [ ] 예측은 지정된 9개 증상만 출력한다.
- [ ] `inference.py` 명령 한 번으로 CSV가 생성된다.
- [ ] CSV 헤더가 `label file name,symptom`이다.
- [ ] `mission3_model.ckpt`를 새 환경에서 다시 불러와 추론해 봤다.
- [ ] 학습 출력 로그가 보이는 `.ipynb`를 저장했다.
- [ ] `common.py`, `train.py`, `inference.py`, `requirements.txt`를 함께 준비했다.
- [ ] 불필요한 코드를 제거하고 주석을 점검했다.
- [ ] 주최 측 sample submission이 있다면 symptom 칸의 세부 형식을 맞췄다.

> 주의: `mission3_all_in_one.ipynb`는 Colab 실험을 편리하게 하는 단일 파일입니다. 실제 제출 전에는 분리된 학습/추론 코드, 모델 가중치, `requirements.txt`, CSV 형식과 새 환경 실행을 모두 다시 검증해야 합니다.
