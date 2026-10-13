# ver29 Kaggle 실행 안내

처음에는 **새 ver29 입력 Dataset 하나와 새 노트북**으로 시작합니다. 기존 ver28 실행은 보관합니다. 이번 입력 ZIP에 원본 JSON과 ver28 비교용 분석 ZIP이 함께 들어 있어 이전 대형 모델 Output을 연결할 필요는 없습니다.

## 1. 파일 준비

- 노트북: `src/mission3/mission3_ver29.ipynb`
- 입력 ZIP: `outputs/mission3_ver29/mission3_ver29_inputs.zip`

ZIP을 직접 압축 해제할 필요는 없습니다. 이전 `mission3-ver28-inputs` 대신 **새 ver29 입력 ZIP**을 업로드하세요.

## 2. Private Dataset 만들기

1. Kaggle에서 **Create → New Dataset**을 선택합니다.
2. `mission3_ver29_inputs.zip`을 업로드합니다.
3. 이름은 `mission3-ver29-inputs`로 지정하고 공개 범위가 **Private**인지 확인합니다.
4. Create 후 업로드·처리가 완료될 때까지 기다립니다.

원본 데이터에는 통화 전사 내용이 있으므로 Public으로 공개하지 마세요. ZIP이 Dataset 안에서 자동으로 풀려 폴더로 보여도 정상입니다.

## 3. 새 노트북 연결

1. **Create → New Notebook**을 선택합니다.
2. **File → Import Notebook**에서 `mission3_ver29.ipynb`를 올립니다.
3. 오른쪽 **Add Input → Your Datasets**에서 `mission3-ver29-inputs`를 연결합니다.
4. **Session options / Settings → Accelerator → GPU T4 ×2**를 선택합니다.
5. **Internet → On**을 확인합니다. 이 코드는 TPU용이 아닙니다.

첫 코드 셀의 기본값을 유지합니다.

```python
RUN_MODE = "screening"
REFERENCE_RUN = ""
SOURCE_RUN = ""
RESUME_RUN = ""
INPUT_ZIP = ""
TIME_BUDGET_MINUTES = 480
```

KLUE-ASL과 Qwen3-Embedding 두 후보가 순서대로 실행됩니다. 새 후보를 고르는 설정은 따로 없습니다. 빈 경로는 연결된 ver29 데이터와 참조 ZIP을 자동 탐색한다는 뜻입니다.

## 4. 한 번 실행

오른쪽 위 **Save Version → Save & Run All**을 한 번 누릅니다. 먼저 일반 Run All을 하고 다시 Save & Run All을 누르면 중복 학습이 됩니다.

`Queued`는 실행 자원 대기, `Running`은 실행 중입니다. 저장된 버전의 실행은 서버에서 진행되므로 브라우저 창을 계속 열어 둘 필요는 없습니다. **Stop / Cancel**은 누르지 마세요. 실제 상태는 저장된 버전의 Logs에서 확인합니다.

이번 코드는 GPU 하나만 사용합니다. T4 ×2라고 한 번에 두 모델을 병렬 학습하지 않으며 메모리를 합치지 않습니다. T4의 FP16을 사용하고 실제 배치 속도를 측정합니다. 첫 학습 전에 모델 다운로드·설치가 있어 준비 시간이 필요합니다.

실행당 최대 8시간이며 학습 완료 시간은 실측 후 판단합니다. 이번에는 기존 ver28의 전체 학습을 반복하지 않고 새 두 모델을 학습하지만, 8시간 안에 모든 비교가 끝난다고 보장하지는 않습니다.

## 5. 결과 내려받기

실행이 끝난 저장 버전에서 **Output**을 엽니다.

- 분석용: `ver29_analysis_실행번호.zip`을 내려받아 보내주세요.
- 학습·재개용: `results/mission3/ver29/실행폴더/` 전체가 포함된 Kaggle Output을 유지하세요.

분석 ZIP에는 큰 모델·optimizer checkpoint가 없습니다. 분석 ZIP만 남기고 전체 Output을 삭제하면 이어 학습이나 전체 학습에 필요한 파일을 잃을 수 있습니다.

| 상태 | 다음 행동 |
|---|---|
| `screening_complete` | 분석 ZIP 확인. 아직 전체 학습이나 제출 완료가 아님 |
| `paused` | 아래 이어하기로 남은 작업 진행 |
| `failed` / `interrupted` | 로그·분석 ZIP 확인 후 원인 해결 |
| `complete` + `submission_ready=true` | full 결과와 제출 자료 확인 |

이번 비교는 기존 ver28, 새 KLUE-ASL, 새 Qwen, 새 두 모델 평균, 기존·새 모델 4개 동일 평균입니다. 동일 development에서 선택을 고정한 후 confirmation을 확인합니다. **점수 0.7을 보장하지 않으며**, 결과를 보고 다음 단계를 결정합니다.

## 6. paused일 때 이어하기

1. 새 노트북에 같은 `mission3_ver29.ipynb`를 Import합니다.
2. ver29 입력 Dataset을 다시 연결합니다.
3. **Add Input → Notebooks**에서 직전 ver29 실행의 **저장된 Output 하나**를 추가합니다.
4. 같은 실행 모드를 유지하고 아래처럼 바꿉니다.

```python
RUN_MODE = "screening"
SOURCE_RUN = ""
RESUME_RUN = "/kaggle/input"
```

5. T4 ×2 / Internet On 확인 후 **Save Version → Save & Run All**을 한 번 누릅니다.

`/kaggle/input` 아래 실행 폴더가 여러 개면 자동 선택을 거부합니다. 그때는 `RESUME_RUN`에 원하는 `run_state.json` 파일이 있는 폴더의 정확한 경로를 넣으세요. 예: `/kaggle/input/이전노트북/results/mission3/ver29/실행폴더`.

full 실행이 중단됐다면 `RUN_MODE="full"`을 유지해 재개합니다. **ver28의 Output을 ver29 재개 입력으로 사용하지 마세요.**

## 7. 결과 확인 후 전체 학습

screening 성공만으로 자동 실행되지 않습니다. 결과를 먼저 검토하세요. 전체 학습을 진행할 때 새 노트북에 ver29 입력 Dataset과 **완료된 ver29 screening Output 하나**를 연결하고 아래처럼 설정합니다.

```python
RUN_MODE = "full"
SOURCE_RUN = "/kaggle/input"
RESUME_RUN = ""
```

선택된 구성에 따라 1개·2개·4개 모델을 전체 Training 29,200개로 다시 학습합니다. 기존 ver28 모델을 포함하는 조합도 해당 구성원은 새로 전체 학습해야 하므로 시간이 더 길어질 수 있습니다. 8시간을 넘으면 저장 후 이어하기를 사용합니다. 공식 Validation 평가는 full 학습 뒤에만 진행하며 정답을 학습이나 모델 선택에 사용하지 않습니다.

## 8. full 완료 후 대회 추론

`complete`와 `submission_ready=true`를 확인한 뒤 **전체 실행 폴더**를 보관합니다. 그 폴더에서 다음 명령을 실행할 수 있습니다.

```bash
python inference.py --audio_dir /path/to/wavs --label_dir /path/to/jsons --ckpt_path ./submission_manifest.json --output ./outputs/mission3.csv
```

실제 데이터 경로로 바꾸세요. `--ckpt_path`에는 완료된 실행 폴더 자체를 넣어도 됩니다. 모델 폴더·토크나이저·`source_snapshot/`도 함께 있어야 하며 분석 ZIP만으로 실행할 수 없습니다. `--audio_dir`는 대회 인자 형식에 맞춰 받지만 읽지 않고 JSON의 전사 텍스트만으로 예측합니다. `symptom` 정답도 읽지 않습니다. CUDA가 있으면 자동 사용하고, 없으면 CPU를 사용합니다.
