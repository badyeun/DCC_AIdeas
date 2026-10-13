# ver28을 Kaggle에서 처음 실행하기

**처음에는 후보 비교만 실행하고, 결과를 확인한 뒤 전체 학습을 따로 실행합니다.** Google Drive 연결이나 Colab 결제는 필요 없습니다. 아래 경로의 두 파일만 준비하면 됩니다.

| 용도 | PC의 파일 |
|---|---|
| 실행할 노트북 | `DCC_AIdeas/src/mission3/mission3_ver28.ipynb` |
| 원본 데이터 | `DCC_AIdeas/outputs/mission3_ver28/mission3_ver28_inputs.zip` |

PC 기준 전체 프로젝트 경로는 `C:\Users\JooSeong\WorkSpace\03_Dev\21_데이터크리에이트캠프\DCC_AIdeas`입니다. 이전 ver25~27 결과나 친구 모델은 업로드하지 않아도 됩니다. 분석 결과와 데이터는 공개하지 말고 Private로 보관합니다.

## 1. 계정 준비

1. [Kaggle](https://www.kaggle.com/)에서 가입하거나 로그인합니다.
2. 계정 설정에서 전화번호 인증을 완료합니다. GPU나 인터넷 옵션이 비활성화되면 화면의 인증 안내를 확인합니다.
3. 무료 GPU 잔여 시간을 확인합니다. 공식 안내는 주당 30시간 수준이지만 할당량은 바뀔 수 있으며, 실제 계정 표시가 우선입니다. 이번 코드는 실행당 최대 8시간입니다. [GPU 사용 안내](https://www.kaggle.com/docs/efficient-gpu-usage)

학생 인증이나 카드 등록을 기다릴 필요 없이 일반 무료 계정으로 시작하는 흐름입니다. 계정 인증·GPU 할당·실제 화면은 아직 대신 실행해 확인한 상태가 아닙니다.

## 2. 입력 데이터를 비공개로 업로드

1. [Datasets](https://www.kaggle.com/datasets) → **New Dataset**을 누릅니다.
2. `mission3_ver28_inputs.zip`을 선택합니다. 이름은 `mission3-ver28-inputs`로 정하면 찾기 쉽습니다.
3. 공개 범위를 **Private**로 유지하고 생성합니다. 데이터에 임의의 공개 라이선스를 붙여 배포하지 않습니다.
4. 업로드 처리가 끝날 때까지 기다립니다.

ZIP이 폴더와 여러 JSON 파일로 보이는 것은 정상입니다. Kaggle은 업로드한 압축을 자동으로 풉니다. ver28은 ZIP이 남아 있는 경우와 자동으로 풀린 경우를 모두 찾도록 구성했습니다. [데이터 업로드 공식 안내](https://www.kaggle.com/docs/datasets)

## 3. 노트북 가져오기

1. [Code](https://www.kaggle.com/code) → **New Notebook**을 누릅니다.
2. 편집기의 **File → Import Notebook**에서 PC의 `mission3_ver28.ipynb`를 선택합니다.
3. 노트북 이름을 `mission3-ver28-screening`으로 정합니다.
4. 오른쪽 **Add Input**에서 `mission3-ver28-inputs`를 검색해 추가합니다.
5. 설정 패널에서 **Accelerator → GPU T4 ×2**, **Internet → On**으로 바꿉니다.
6. 노트북도 **Private**로 유지합니다.

버튼 이름·위치는 화면 업데이트에 따라 조금 다를 수 있습니다. 설정한 GPU 두 개를 이 코드가 동시에 사용하는 것은 아닙니다. 모델 두 개를 GPU 한 개에서 순차 학습해 저장과 재개를 단순하게 유지합니다.

## 4. 첫 실행 설정

처음 코드 셀은 아래 그대로 둡니다. `INPUT_ZIP`이 비어 있어도 연결된 입력을 자동으로 찾습니다.

```python
RUN_MODE = "screening"
CANDIDATE = "kure"
SOURCE_RUN = ""
RESUME_RUN = ""
INPUT_ZIP = ""
TIME_BUDGET_MINUTES = 480
```

실행하면 자동으로 다음을 처리합니다.

- 코드·패키지 준비와 데이터 무결성 확인
- 같은 fit 20,440개에 대한 KURE-v1과 새 KLUE 학습
- 속도와 GPU 메모리를 측정한 배치 선택
- epoch별 development 점수와 단독 / 1:1 앙상블 비교
- 선택을 고정한 뒤 confirmation 확인
- 가중치·지표·예측·설정·재개 상태 저장

이 단계에서는 전체 Training 학습과 공식 Validation 평가를 자동으로 시작하지 않습니다. 이미 전체 Training을 학습한 친구 모델이나 ver25를 내부 검증에 섞지 않습니다. 이번 목표 점수 0.7 달성 여부는 최종 평가로 확인해야 합니다.

## 5. 실제 실행은 한 번만 시작

오른쪽 위 **Save Version → Save & Run All**을 선택해 시작합니다. 이것은 깨끗한 새 세션에서 전체 노트북을 실행하고 결과를 저장하는 방식입니다. [노트북 저장 공식 안내](https://www.kaggle.com/docs/notebooks)

**먼저 Run All로 전체 학습을 한 뒤 Save & Run All을 누르면 학습을 다시 합니다.** 처음부터 Save & Run All 한 번으로 진행하세요. 진행 중 같은 버튼을 반복해 여러 실행을 만들지 마세요. 대화형 세션을 별도로 켰다면 남겨두지 말고 종료합니다.

실행 로그의 `실행 모드`, 데이터 수, GPU, 배치 측정 결과, epoch 진행을 확인합니다. 처리량으로 계산된 시간은 추정값이며 실제 완료 시간과 다를 수 있습니다. Kaggle의 세션 종료·GPU 부족·주간 할당량 초과는 별도 제한입니다.

## 6. 결과 내려받기

실행 상태가 종료로 바뀌면 **저장된 버전의 Output**을 엽니다.

1. `ver28_analysis_실행번호.zip`을 PC에 내려받습니다.
2. 분석할 때 이 ZIP 또는 저장 경로를 알려주세요.
3. 가중치와 재개 checkpoint가 있는 **전체 실행 폴더도 Kaggle에 보관**합니다. 분석 ZIP만 받아 두고 나머지를 삭제하면 이어 학습할 수 없습니다.

작은 분석 ZIP과 큰 모델 파일이 따로 보이는 것은 정상입니다. 전체 실행 폴더는 `results/mission3/ver28/실행시각_고유번호/` 아래에 있습니다.

| `run_state.json` 상태 | 해야 할 일 |
|---|---|
| `screening_complete` | 결과를 분석한 뒤 전체 학습 여부 결정 |
| `paused` | 아래 방법으로 이어 학습 |
| `complete` + `submission_ready=true` | 최종 지표·제출 파일 확인 |
| `failed` | 오류 기록과 로그 확인 |
| `interrupted` | 강제 종료. 재개 전에 checkpoint / manifest 확인 |

`screening_complete`는 점수 0.7 달성이나 제출 준비 완료를 뜻하지 않습니다.

## 7. `paused`라면 이어 학습

1. 같은 ver28 노트북을 새로 가져오고 이름을 `mission3-ver28-screening-resume` 등으로 정합니다.
2. 원본 데이터셋을 연결합니다.
3. **Add Input**에서 본인의 이전 노트북 Output을 찾아 연결합니다. 중단 실행이 저장된 버전을 선택합니다.
4. 입력 파일 트리에서 해당 실행의 `run_state.json`을 찾고 **파일명이 빠진 폴더 경로**를 복사합니다.
5. `RESUME_RUN`에 그 경로를 넣습니다. `RUN_MODE`와 `CANDIDATE`는 이전과 같게 둡니다.

경로 형태 예시는 다음과 같습니다. `...`를 그대로 넣지 말고 연결한 입력의 실제 경로를 복사하세요.

```python
RUN_MODE = "screening"
CANDIDATE = "kure"
SOURCE_RUN = ""
RESUME_RUN = "/kaggle/input/이전출력의실제이름/results/mission3/ver28/실제실행폴더"
```

다시 Save & Run All을 실행합니다. 원래 3 epoch 일정과 진행 상태를 이어가며, 입력에 연결한 이전 결과를 덮어쓰지 않습니다. 중단 직전 아직 checkpoint에 기록되지 않은 부분은 다시 계산될 수 있습니다. Kaggle이 강제로 종료한 경우 먼저 checkpoint와 저장 Output이 있는지 확인해야 합니다.

## 8. 결과 검토 후 전체 Training 학습

screening이 완료되고 결과를 검토한 뒤 진행합니다.

1. 노트북을 새로 가져오고 이름을 `mission3-ver28-full`로 정합니다.
2. 원본 데이터와 **완료된 screening Output**을 Add Input으로 연결합니다.
3. 아래 설정으로 변경합니다. `SOURCE_RUN`에는 `screening_complete` 상태인 실행 폴더를 넣습니다.

```python
RUN_MODE = "full"
CANDIDATE = "kure"
SOURCE_RUN = "/kaggle/input/완료출력의실제이름/results/mission3/ver28/실제실행폴더"
RESUME_RUN = ""
```

4. Save & Run All을 실행합니다. 선택한 모델과 epoch로 원본 Training 29,200개를 학습합니다.
5. 공식 Validation 평가와 저장 모델 재로딩 확인을 마친 뒤 제출용 결과를 저장합니다.
6. 전체 학습도 `paused`이면 같은 full 모드에서 `RESUME_RUN`을 지정해 이어갑니다.

최종 완료 여부는 **`complete`와 `submission_ready=true`**로 확인합니다. 단순히 Kaggle 실행이 끝났거나 모델 파일이 만들어졌다는 이유만으로 제출 준비가 끝난 것은 아닙니다.

## 문제가 생겼을 때

| 상황 | 먼저 확인할 것 |
|---|---|
| GPU 선택이 회색 | 계정 인증 안내, 남은 GPU 시간, 가용 GPU |
| 모델 다운로드 실패 | Internet On과 로그의 다운로드 단계 |
| 입력을 못 찾음 | Add Input 연결과 실제 폴더 구조; 필요하면 `INPUT_ZIP`에 ZIP 또는 데이터 루트 경로 지정 |
| 입력이 여러 개라 모호함 | 이번 데이터만 연결하거나 `INPUT_ZIP`에 정확한 경로 지정 |
| PyTorch 버전 오류 | 최신 Kaggle GPU 이미지인지 확인; 임의로 CUDA PyTorch를 덮어 설치하지 않기 |
| `paused` | 오류와 구분하고 이전 Output을 연결해 재개 |
| `failed` / `interrupted` | 마지막 `오류 종류`와 `error.json`을 분석 ZIP과 함께 확인; 강제 종료를 정상 중단으로 간주하지 않기 |
| 저장 실패 / 용량 부족 | Output 용량 확인, 기존 결과를 백업하기 전에 삭제하지 않기 |

이 문서 작성 시 공식 Kaggle 문서와 로컬 코드를 확인했습니다. 실제 사용자 계정의 버튼 조작, GPU 배정, 다운로드, 학습 속도와 F1은 아직 Kaggle에서 검증하지 않았습니다.
