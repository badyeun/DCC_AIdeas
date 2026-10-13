# Mission 3 ver28

Kaggle용 KURE-v1 / KLUE 지도 학습 비교와 별도 전체 학습입니다. 처음 사용하는 경우 [Kaggle 실행 안내](../../../docs/mission3_ver28_kaggle_guide.md)를 참고합니다. 목표는 Macro F1 0.7 초과지만, 실제 GPU 학습과 목표 점수 달성은 실행 결과로 확인해야 합니다.

## 실행 파일

- `../mission3_ver28.ipynb`: 모든 실행 코드를 포함한 독립 노트북입니다.
- 프로젝트 루트의 `outputs/mission3_ver28/mission3_ver28_inputs.zip`: 원본 Training / Validation JSON만 모은 입력입니다. 이전 모델 가중치는 필요하지 않습니다.
- `requirements.txt`: ver27에서 사용하던 패키지의 부분집합입니다. 호환되는 Kaggle CUDA PyTorch와 수치 패키지는 유지하고 실제 버전을 기록합니다. PEFT / bitsandbytes는 필요하지 않습니다.

Kaggle의 Private Dataset에 ZIP을 올린 뒤 노트북에 Add Input으로 연결합니다. ZIP 또는 Kaggle이 압축을 푼 디렉터리를 자동으로 찾습니다. GPU T4 ×2와 Internet On을 설정하고 **Save Version → Save & Run All**을 한 번 실행합니다. 이 구현은 GPU 한 개를 사용하며 두 모델을 순서대로 학습합니다.

## 두 실행 단계

`RUN_MODE="screening"`이 기본값입니다. seed42 / 중복 그룹을 유지한 기존 70/15/15 분할에서 fit 20,440개로 KURE와 새 KLUE를 최대 3 epoch 학습합니다. development 4,380개에서 최고 epoch와 KURE 단독 / KLUE 단독 / 고정 1:1 확률 평균을 비교하고 선택을 고정합니다. confirmation 4,380개는 고정한 선택의 추가 확인에 사용하며 재선택에 사용하지 않습니다. 두 검증 분할 모두 과거 실험의 사용 이력이 있습니다. 공식 Validation은 이 단계에서 평가하지 않습니다.

`CANDIDATE="qwen_embedding"`은 별도 후보 실험을 명시적으로 실행할 때만 사용합니다. 기본 KURE 결과를 확인하지 않고 예비 후보까지 자동 실행하지 않습니다.

결과를 검토한 후 **새 실행**에서 `RUN_MODE="full"`로 바꾸고 `SOURCE_RUN`에 완료된 screening 실행 폴더를 지정합니다. 선택한 구성과 epoch로 전체 Training 29,200개를 학습하고 공식 Validation 평가, 저장 모델 재로딩 확인, 제출용 파일 생성을 진행합니다. 새 노트북에 원본 데이터와 이전 screening Output을 모두 Add Input으로 연결해야 합니다. `SOURCE_RUN`은 `run_state.json`이 있는 실행 폴더입니다.

외부 학습 데이터, 정답 수정, 표본 임의 삭제, threshold / offset / 혼합비 검색은 하지 않습니다. 입력은 통화 전사 텍스트이며 원본 9개 증상 정답을 사용합니다. weighted BCE와 고정 판정 0.5를 유지합니다. 전체 Training을 이미 학습한 ver25 / 친구 모델은 내부 검증 앙상블에 포함하지 않습니다.

## 시간과 재개

최대 480분은 설치·모델 다운로드·준비·평가·저장을 포함합니다. 배치는 실제 처리량으로 정하며, 이전 L4 시간을 T4 소요 시간으로 간주하지 않습니다. 일정한 scheduler 길이를 유지하고 남은 시간이 부족하면 checkpoint를 저장해 `paused`로 종료합니다. 시간 때문에 epoch 수를 줄인 뒤 완료로 표시하지 않습니다.

이전 저장 버전의 Output을 새 노트북에 연결하고 `RESUME_RUN`에 중단된 실행 폴더를 넣으면 같은 모드로 이어갑니다. 재개에는 가중치뿐 아니라 optimizer / scheduler / 혼합 정밀도 / 난수 및 진행 상태가 필요합니다. 따라서 분석 ZIP만으로 재개할 수 없습니다. 이전 입력은 읽기 전용으로 유지하고 새 실행 폴더에 결과를 저장합니다.

데이터 추출물·Hugging Face 캐시는 `/kaggle/temp/`에 두고 결과는 `/kaggle/working/results/mission3/ver28/실행시각_고유번호/`에 저장합니다. 임시 캐시는 다음 실행에서 다시 다운로드될 수 있습니다. Kaggle 강제 종료 전에 기록하지 못한 진행은 보존을 보장할 수 없습니다.

## 상태와 보관

| 상태 | 뜻 |
|---|---|
| `screening_complete` | 후보 비교·선택·추가 확인 완료. 최종 제출 모델은 아님 |
| `paused` | 저장 후 시간 제한으로 중단. 같은 모드에서 재개 필요 |
| `complete` + `submission_ready=true` | 전체 학습과 제출 준비 완료 |
| `failed` | 오류. 완료 결과로 사용하지 않음 |
| `interrupted` | 강제 종료. checkpoint와 manifest가 완전히 저장됐는지 확인 필요 |

`run_state.json`을 먼저 확인합니다. 지표·예측·선택 근거·재현 코드와 모델·checkpoint를 실행 폴더에 보관합니다. 노트북 마지막에는 `ver28_analysis_실행번호.zip`도 만듭니다. 이 ZIP은 분석을 위한 작은 자료 모음이며 **큰 가중치·재개 checkpoint를 포함하지 않습니다**. Kaggle Output의 전체 실행 폴더를 지우지 마세요.

## 재생성과 검증

프로젝트 루트에서 노트북을 재생성합니다.

```bash
python src/mission3/mission3_ver28/build_notebook.py
```

빌더는 현재 ver28의 모든 실행 모듈과 설정·README·requirements를 포함하고 Python 소스와 노트북 코드 셀의 구문을 검사합니다. 테스트 파일과 결과 파일은 포함하지 않습니다. GPU·인터넷·Kaggle 계정·실제 F1은 이 로컬 검사로 검증되지 않습니다. 런타임 코드가 바뀌면 노트북도 다시 생성해야 합니다.

필요한 패키지가 있는 환경에서 로컬 테스트를 실행합니다.

```bash
python -m unittest src.mission3.mission3_ver28.test_runtime src.mission3.mission3_ver28.test_trainer src.mission3.mission3_ver28.test_package_inputs -v
```

최종 결과에는 `INFERENCE_README.md`와 `source_snapshot/`도 포함됩니다. 전체 실행 폴더에서 새 데이터의 전사 텍스트만으로 추론할 수 있습니다.

```bash
python -m source_snapshot.inference --run-dir . --label-dir /path/to/label_jsons --output predictions.csv --device cuda --batch-size 1
```

GPU가 없으면 `--device cpu`로 바꿉니다. 추론은 `symptom` 정답을 읽지 않으며 모델과 토크나이저를 로컬 파일에서 불러옵니다.

모델 출처와 revision / 라이선스는 `model_specs.json` 및 각 실행의 snapshot에 기록합니다.
