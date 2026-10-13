# ver29 Modal L4 실행 안내

이미 만든 **`mission3-ver29-data` Volume과 L4 GPU**를 사용합니다. Modal용 노트북을 새로 올린 뒤 같은 Volume을 다시 연결하세요. 기존 Modal 노트북과 Kaggle 결과는 보관해도 됩니다.

## 1. 준비할 파일

- 노트북: `src/mission3/mission3_ver29_modal.ipynb`
- 입력 ZIP: `outputs/mission3_ver29/mission3_ver29_inputs.zip`

입력 ZIP은 Kaggle에 사용한 ver29 파일을 그대로 재사용합니다. 직접 압축을 풀 필요는 없습니다. 이전 ver28 비교 결과도 들어 있으므로 별도의 큰 모델 파일을 올릴 필요가 없습니다.

## 2. 실행 전 한도 확인

계정 설정이 끝났다면 **Settings → Usage & Billing**에서 남은 크레딧과 **Spend limit = $0**만 확인합니다. Starter를 유지하면 됩니다.

`Spend limit`은 크레딧 적용 후 실제 지출 한도입니다. 별도 항목인 `Workspace budget / usage limit`까지 0으로 설정하지 마세요. L4·CPU·메모리가 크레딧을 사용하며, 크레딧이 소진되면 실행이 멈출 수 있습니다. [비용 한도](https://modal.com/docs/guide/budgets)

## 3. Modal용 노트북과 기존 Volume 연결

1. [Modal Notebooks](https://modal.com/notebooks)를 엽니다.
2. `.ipynb` 업로드 기능으로 **`mission3_ver29_modal.ipynb`**를 올립니다.
3. 새 노트북의 **Files / filesystem**에서 기존 **`mission3-ver29-data`** Volume을 연결합니다. 같은 이름의 Volume을 새로 만들 필요는 없습니다.
4. 연결된 폴더가 **`/mnt/mission3-ver29-data`**인지 확인합니다.
5. 사이드바의 **Compute profile / kernel settings**에서 **L4 ×1**을 선택합니다.

노트북을 새로 업로드하면 다른 노트북의 GPU 설정과 Volume 연결이 자동으로 이어진다고 가정하지 마세요. **새 노트북 안에서 둘 다 확인**합니다. 같은 Volume을 연결하더라도 두 노트북에서 동시에 학습을 실행하지 마세요. [공식 노트북 안내](https://modal.com/docs/guide/notebooks)

## 4. 입력 ZIP 올리기

Files에서 **`/mnt/mission3-ver29-data` 폴더를 연 상태로** `mission3_ver29_inputs.zip`을 드래그하거나 업로드 버튼으로 올립니다.

다음 위치에 파일이 보여야 합니다.

```text
/mnt/mission3-ver29-data/mission3_ver29_inputs.zip
```

이미 그 위치에 같은 ZIP이 있다면 다시 올리지 않아도 됩니다. ZIP은 공개 링크로 공유하지 말고 본인 작업 공간에 보관하세요.

## 5. 처음에는 screening 한 번 실행

첫 설정 셀은 아래 기본값을 유지합니다.

```python
VOLUME_DIR = "/mnt/mission3-ver29-data"
RUN_MODE = "screening"
SOURCE_RUN = ""
RESUME_RUN = ""
INPUT_ZIP = ""
TIME_BUDGET_MINUTES = 480
```

Volume을 다른 경로에 연결했다면 `VOLUME_DIR`를 Files에 표시된 실제 경로로 바꿉니다. ZIP 경로를 직접 지정할 때는 `INPUT_ZIP`에 ZIP 파일의 전체 경로를 넣습니다.

1. **Run All**이 보이면 한 번 누릅니다. 없으면 코드 셀을 위에서 아래 순서로 실행합니다.
2. 설치·데이터 확인·모델 다운로드가 끝날 때까지 기다립니다.
3. GPU가 L4로 인식되는지, 결과 경로가 `/mnt/mission3-ver29-data` 아래인지 확인합니다.
4. 배치 속도 측정이 끝나면 두 후보의 학습이 차례로 진행됩니다.

Modal에서는 Kaggle의 **Save Version → Save & Run All** 절차를 사용하지 않습니다. 실행 중에 Run All을 다시 누르지 마세요.

첫 실행은 **새 screening**입니다. Kaggle T4에서 저장한 중간 checkpoint를 L4에서 그대로 재개하는 방식으로 시작하지 마세요. GPU에 따라 사용하는 계산 정밀도가 달라질 수 있어 기존 중간 학습 상태와 호환되지 않을 수 있습니다.

480분은 설치·다운로드·평가·저장을 포함한 **이번 실행의 시간 예산**입니다. 8시간 안에 전부 끝나거나 Modal이 8시간 실행을 보장한다는 뜻은 아닙니다. 코드가 시간 부족을 감지하면 중간 상태를 저장하고 `paused`로 끝낸 뒤 이어서 실행합니다. 무료 크레딧 소진이나 Modal의 강제 종료는 별개이며 마지막 저장 이후 작업은 잃을 수 있습니다. 전체 Training 학습은 자동으로 시작하지 않습니다.

## 6. 결과 확인하고 내려받기

실행 후 다음 폴더를 확인합니다.

| 위치 | 내용 |
|---|---|
| `/mnt/mission3-ver29-data/results/mission3/ver29/실행번호/` | 모델, checkpoint, 지표 등 전체 실행 결과 |
| `/mnt/mission3-ver29-data/exports/ver29_analysis_실행번호.zip` | 분석용 ZIP |
| `/mnt/mission3-ver29-data/invocations/호출번호/` | 이번 호출의 실행 경로 안내와 로그 |

Files의 새로고침 버튼을 누르고 `exports` 폴더의 **`ver29_analysis_*.zip`**을 내려받아 보내주세요. 파일을 선택한 뒤 다운로드 버튼을 사용합니다. 실행 중 문제가 생기면 `invocations/호출번호/` 안의 `training.log`, 설치 문제는 `install.log`를 확인합니다. `result.json`에는 실제 결과 폴더와 상태가 기록됩니다.

| 결과 상태 | 의미와 다음 행동 |
|---|---|
| `screening_complete` | 비교 실험 완료. 분석 ZIP을 확인한 뒤 full 여부 결정 |
| `paused` | 시간 예산에 맞춰 저장 후 종료. 같은 모드로 이어하기 |
| `failed` / `interrupted` | 로그로 원인 확인. 완료 결과로 사용하지 않음 |
| `complete` + `submission_ready=true` | full 학습 및 제출 준비 결과 확인 |

**분석 ZIP에는 큰 모델과 optimizer checkpoint가 없습니다.** 분석 ZIP을 내려받았더라도 `results`의 전체 실행 폴더를 남겨 두세요. 이어 학습과 최종 추론에 필요합니다. F1 0.7 초과 여부는 실제 결과로 확인해야 합니다.

## 7. paused일 때 이어하기

같은 노트북과 Volume을 사용합니다. 첫 셀에서 `RESUME_RUN`만 **직전 실행 폴더의 정확한 경로**로 바꿉니다.

```python
RUN_MODE = "screening"
SOURCE_RUN = ""
RESUME_RUN = "/mnt/mission3-ver29-data/results/mission3/ver29/실제_실행번호"
```

`실제_실행번호`를 실행 출력에 표시된 폴더 이름으로 바꾸세요. 그 폴더 안에 `run_state.json`이 있어야 합니다. `exports`의 분석 ZIP 경로나 `results` 상위 폴더를 넣지 않습니다.

GPU는 이전과 같은 **L4 ×1**을 유지하고 전체 셀을 다시 한 번 실행합니다. 새 노트북을 사용한다면 같은 Volume을 먼저 연결합니다.

**full 학습이 paused였다면** 아래처럼 설정합니다. `RESUME_RUN`은 중단된 **full 실행 폴더**이고, `SOURCE_RUN`은 반드시 비웁니다.

```python
RUN_MODE = "full"
SOURCE_RUN = ""
RESUME_RUN = "/mnt/mission3-ver29-data/results/mission3/ver29/중단된_full_실행번호"
```

`SOURCE_RUN`과 `RESUME_RUN`을 동시에 채우면 실행을 시작하지 않습니다. `failed`나 `interrupted`는 정상 저장된 `paused`와 다르므로 로그를 확인한 뒤 복구 여부를 판단합니다.

## 8. 결과 확인 후 전체 학습

screening 결과를 검토한 뒤 별도로 진행합니다. 완료된 **ver29 screening 실행 폴더**를 `SOURCE_RUN`에 지정합니다.

```python
RUN_MODE = "full"
SOURCE_RUN = "/mnt/mission3-ver29-data/results/mission3/ver29/완료된_screening_실행번호"
RESUME_RUN = ""
```

선택된 구성의 모델을 전체 Training 29,200개로 다시 학습합니다. 조합에 따라 1개·2개·4개 모델을 학습하므로 시간이 달라집니다. full도 시간 예산이 부족하면 저장 후 이어하기를 사용합니다. 공식 Validation은 full 학습 뒤 평가에 사용하며 학습 데이터에 섞지 않습니다.

## 9. 저장과 종료

결과 저장과 다운로드를 확인한 뒤 **Stop kernel**로 GPU 실행을 종료합니다. 완료 후 불필요하게 커널을 켜 두면 크레딧을 더 사용할 수 있습니다.

연결된 Volume은 커널을 다시 시작해도 유지됩니다. 입력 ZIP·모델 다운로드 캐시·결과·로그는 Volume에 보관하며, 다시 만들 수 있는 코드와 압축 해제한 데이터는 임시 `/tmp`에 둡니다. **Volume 밖 임시 파일은 커널 종료 시 사라져도 다음 실행에서 다시 준비됩니다.**

Modal은 Volume 변경 사항을 주기적으로 저장하고 정상 종료 때 최종 저장합니다. 갑작스러운 종료 직전의 쓰기까지 보장하는 것은 아닙니다. [파일 보존](https://modal.com/docs/guide/notebooks#interactive-file-viewer), [Volume 저장 동작](https://modal.com/docs/guide/volumes#background-commits)

별도 Modal SDK를 설치할 필요는 없습니다. `sync` 명령으로 저장을 확인하는 방식도 사용하지 않습니다. 공식적으로 mount 경로의 `sync`를 통한 commit은 Volume v2에만 설명되어 있습니다.

첫 실행은 탭을 유지하고 진행을 확인하세요. Notebooks의 브라우저 종료 후 실행 유지와 최대 연속 실행 시간은 공식 문서에서 확인되지 않았습니다. idle shutdown 10분은 유휴 종료 설정이며, Functions의 24시간 제한도 이 노트북의 실행 보장을 뜻하지 않습니다.
