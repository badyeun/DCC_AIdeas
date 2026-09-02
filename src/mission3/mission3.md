# Mission 3 첫 도전 가이드

이 문서는 AI 모델을 만들어 본 적이 없는 사람이 Mission 3을 처음부터 실행할 수 있도록 설명합니다.

## 1. 문제를 한 문장으로 이해하기

119 신고 대화의 **전사 텍스트만** 읽고, 환자에게 다음 9개 증상이 있는지 각각 예측하는 문제입니다.

`[고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란]`

한 통화에 증상이 없을 수도, 여러 개일 수도 있어 **다중 레이블 분류**라고 부릅니다. 예를 들어 `"머리도 아프고 토했어요"`는 `두통`과 `구토`가 동시에 정답일 수 있습니다.

## 2. 반드시 지켜야 할 규칙

- 학습은 제공된 **서울 Training** 데이터만 사용합니다.
- Validation 데이터는 학습에 사용하지 않습니다.
- 모델 입력으로는 JSON의 `utterances[].text`만 사용합니다. `speaker`, `startAt`, `endAt`, `gender`, `address` 등을 입력으로 쓰면 안 됩니다.
- `symptom`은 Training에서 정답(Y)으로만 사용합니다.
- 9개 외의 증상만 정답에서 제외하고, 해당 통화 자체는 제거하지 않습니다.
- 외부 데이터와 GPT 같은 상용 AI API는 사용하지 않습니다.
- 평가 점수는 9개 증상별 F1을 구한 뒤 평균한 **Macro F1**입니다. 정확도(Accuracy)가 아닙니다.

## 3. 현재 파일의 역할

- `common.py`: JSON에서 텍스트와 정답을 안전하게 꺼냄
- `train.py`: Training 데이터로 모델을 학습하고 `.ckpt`를 저장
- `inference.py`: 저장한 모델로 예측하고 제출용 CSV를 생성
- `evaluate.py`: Validation을 학습하지 않고 완성된 CSV의 Macro F1만 계산
- `mission3_experiment.ipynb`: 데이터 확인부터 학습까지 실행하고 제출용 학습 로그를 남김

## 4. Step 1 - 프로그램 준비

명령어는 반드시 `DCC_AIdeas` 폴더에서 실행합니다. VS Code에서 `DCC_AIdeas`를 열고 **터미널 > 새 터미널**을 누르면 됩니다.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

터미널 앞에 `(.venv)`가 보이면 준비가 된 것입니다.

## 5. Step 2 - 데이터 위치 확인

현재 저장소의 Training 라벨 폴더는 다음과 같습니다.

```text
DATA/
└─ 대학부 데이터/
   ├─ Training/2.라벨링데이터/TL_서울_구급/
   └─ Validation/2.라벨링데이터/VL_서울_구급/
```

JSON 하나를 열어 다음 두 가지만 위치를 확인합니다.

- 입력 X: `utterances` 목록 안의 `text`
- 학습 정답 Y: 파일 아래쪽의 `symptom`

`symptom`을 텍스트 입력에 붙이면 모델이 정답을 미리 보게 되므로 절대 그렇게 하면 안 됩니다.

## 6. Step 3 - 모델이 배우는 방식 이해

1. `TF-IDF`가 대화에 나타난 문자 조각을 숫자 특징으로 바꾸어 줍니다. 예를 들어 `"숨이 차요"`의 `"숨이"`, `"이 차"`같은 조각을 기억합니다.
2. Logistic Regression이 각 증상의 확률을 만듭니다.
3. 증상이 9개이므로 `있음/없음` 분류기 9개를 학습합니다.
4. 각 확률이 기준값보다 크면 해당 증상을 출력합니다.

이 방식은 최첨단 모델은 아니지만, 동작이 빠르고 이해하기 쉬워서 첫 기준점으로 적합합니다.

## 7. Step 4 - 첫 학습 실행

```bash
python src/mission3/train.py \
  --label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --ckpt_path saved_models/mission3_model.ckpt
```

처음에는 Training의 80%로 연습하고 나머지 20%로 내부 점수를 계산합니다. 이 20%는 주최 측 Validation이 아니라 Training 폴더 안에서 임시로 나눈 데이터입니다. 점수 확인 후에는 Training 전체로 최종 모델을 다시 학습해 `mission3_model.ckpt`를 만듭니다.

출력의 `Macro F1`이 첫 기준 점수입니다. 특정 증상의 F1이 0이라면 코드가 고장난 것이라기보다 해당 증상이 드물거나 예측이 어려운지 먼저 확인해야 합니다.

## 8. Step 5 - Validation 예측 CSV 만들기

출제 조건에 맞게 `--audio_dir`도 받지만 Mission 3 코드는 음성을 읽지 않습니다.

```bash
python src/mission3/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path saved_models/mission3_model.ckpt \
  --output outputs/mission3.csv
```

CSV는 다음 형식입니다.

```csv
label file name,symptom
example1.json,"['두통', '구토']"
example2.json,[]
```

PDF에는 symptom 칸의 세부 구분자까지는 명시되어 있지 않으므로, 주최 측 sample submission이 추가로 제공되면 그 형식을 최우선으로 맞춰야 합니다.

## 9. Step 6 - Validation Macro F1 확인

모델 방식과 기준값을 먼저 고정한 다음, Validation은 최종 성능 확인에만 사용합니다.

```bash
python src/mission3/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction outputs/mission3.csv
```

Validation의 `symptom`을 `train.py`에 넘기지 않는 것이 가장 중요합니다.

## 10. Step 7 - 점수를 올리는 실험 순서

한 번에 여러 개를 바꾸면 무엇 때문에 점수가 바뀌었는지 알 수 없습니다. 다음 순서로 하나씩 실험합니다.

1. 현재 결과를 기록합니다: 코드 버전, 설정, 증상별 F1, Macro F1.
2. `--max_features 50000`, `100000`, `200000`을 하나씩 비교합니다.
3. `train.py`의 `ngram_range=(2, 5)`를 `(2, 4)`, `(3, 6)`으로 하나씩 비교합니다.
4. 오분류된 대화에 어떤 표현이 있었는지 읽고, 자주 놓치는 증상을 확인합니다.
5. 베이스라인을 완전히 이해한 후에만 KoBERT 같은 공개 pre-trained 모델을 비교합니다.

실험은 Training 내부 분할 점수로 비교하고, Validation을 보면서 계속 맞추는 방식은 피합니다.

## 11. 자주 만나는 문제

- `ModuleNotFoundError`: 가상환경을 활성화했는지 확인하고 `python -m pip install -r requirements.txt`를 다시 실행합니다.
- `No .json files found`: `--label_dir`의 한글, 공백, Training/Validation 위치를 확인합니다.
- 학습이 느림: `--max_features 50000 --n_jobs 1`로 먼저 실행합니다.
- CSV가 Excel에서 한글이 깨짐: 파일은 UTF-8 BOM으로 저장되므로 일반적으로 바로 열립니다.
- F1이 0인 증상이 있음: 드문 증상의 정답/예측 개수와 임계값을 먼저 확인합니다.

## 12. 최종 제출 전 체크리스트

- [ ] Training에 서울 Training JSON만 사용했다.
- [ ] 모델 입력으로 전사 `text`만 사용했다.
- [ ] 예측은 9개 대상 증상만 출력한다.
- [ ] `inference.py` 명령 한 번으로 CSV가 생성된다.
- [ ] CSV 헤더가 `label file name,symptom`이다.
- [ ] 가상환경을 새로 만들어 `requirements.txt`만으로 실행해 봤다.
- [ ] 학습 로그가 포함된 `.ipynb`를 따로 정리했다.
- [ ] 불필요한 코드를 제거하고 주석을 점검했다.
- [ ] 주최 측 sample submission이 있다면 symptom 칸의 세부 형식을 다시 맞춰다.

> 주의: 현재 파일은 첫 실험을 위한 베이스라인입니다. 실제 제출 전에는 전체 학습, 추론, CSV 형식, 새 환경 실행을 모두 다시 검증해야 합니다.
