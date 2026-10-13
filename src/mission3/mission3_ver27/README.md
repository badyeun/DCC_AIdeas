# Mission 3 ver27

**ver26과 같은 4,000개·3 epoch 조건에서 Qwen 학습률만 0.3배로 낮추는 대조 실험입니다.** LoRA LR은 1e-4 → 3e-5, head LR은 3e-4 → 9e-5로 바꾸고 새 KLUE 대조군은 이전 조건을 유지합니다. 완료된 ver26 기준 결과와 반드시 비교합니다. 최종 제출 모델을 만들지 않으며 Macro F1 개선이나 0.7 달성을 보장하지 않습니다.

## 실행

1. `../mission3_ver27.ipynb`를 Colab에 업로드하고 GPU 런타임을 선택합니다.
2. `mission3_ver5_inputs.zip`이 있는 계정으로 Drive를 연결합니다. 필요하면 `input_zip` 경로만 수정합니다.
3. `baseline_run` 기본값은 `/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver26/20261011T133609Z_f22b4e55`입니다. 이 완료 결과 폴더를 보관합니다. 같은 파일을 다른 경로에 보관했다면 경로만 수정할 수 있고 다른 실행으로 교체할 수는 없습니다.
4. `run_mode='pilot'`, `pilot_samples=4000`, `max_epochs=3`, `run_validation=False`, `time_budget_minutes=480`, `AUTO_RELEASE_GPU=True`를 유지하고 **모두 실행**합니다.
5. 결과는 `MyDrive/DCC_AIdeas/results/mission3/ver27/실행시각_고유번호/`에 저장됩니다. ver25·ver26 폴더는 변경하지 않습니다.

코드가 노트북에 모두 포함됩니다. 의존성은 ver25와 같은 고정 버전이며 추가 패키지를 도입하지 않았습니다. 실행당 최대 8시간에는 설치·다운로드·준비·평가·저장이 모두 포함됩니다. 마지막 25분은 저장·검증용으로 예약합니다. `None`과 480분을 넘는 설정은 준비 전에 거절합니다. Colab 자체의 세션 종료는 막을 수 없습니다. 완료·오류·시간 만료 시 GPU 반환을 요청하며 자동 반환이 실패하면 직접 런타임을 삭제합니다. 중단된 학습 자동 재개는 지원하지 않습니다. 3 epoch가 모두 예상 예산에 들어올 때만 시작하며 적은 epoch로 조용히 줄이지 않습니다.

기준 ver26은 L4에서 준비·두 모델 학습·평가·저장까지 **2시간 51분 42초** 걸렸습니다. ver27도 약 3시간 수준의 작업량을 예상하지만 기준 검증, GPU, 캐시와 저장 속도에 따라 달라집니다. 8시간은 예상 소요 시간이 아니라 상한입니다. 이번 코드의 실제 Colab 실행은 아직 확인하지 않았습니다.

기준 폴더의 완료 표시·필수 파일 해시·Training 정답·분할·subset·모델 revision·고정 학습 조건을 검사하고 저장된 점수를 예측에서 다시 계산합니다. 필수 파일 누락이나 불일치는 **공개 모델 다운로드와 학습 전에 실패**합니다. 기준 모델 가중치는 읽거나 복사하지 않으며 작은 비교 자료만 새 결과 폴더에 복사합니다.

## 비교 조건

- 완료된 ver25와 동일한 공개 Qwen revision `cdbee75f17c01a7cc42f958dc650907174af0554`와 KLUE revision을 고정합니다.
- seed42, 중복 그룹을 보존한 fit/development/confirmation 분할과 4,000개 subset 선택법을 재사용합니다. 실제 원본에서는 20,440 / 4,380 / 4,380입니다.
- 입력은 전사 텍스트만, 원본 정답과 weighted BCE를 유지합니다. 외부 학습 데이터, 정답 변경, 표본 제거, threshold/offset/calibration 탐색을 하지 않습니다.
- Qwen NF4 QLoRA r16, LoRA/head LR 3e-5/9e-5, max1024; 새 KLUE LR2e-5, max512입니다. 모든 라벨을 `logit >= 0`으로 판정합니다.
- Qwen batch1/2/4/8, KLUE batch4/8/16을 측정하며 유효 batch16을 유지하는 가장 빠른 성공 후보를 선택합니다. 측정용 가중치는 버립니다.
- 각 모델은 3 epoch를 완료합니다. 매 epoch development를 평가하고 최고 epoch 가중치가 갱신될 때 저장합니다. 모든 epoch의 예측과 지표는 저장합니다.
- 주 비교는 **새 Qwen + ver26 KLUE의 고정 1:1 확률 평균**과 기존 ver26 앙상블입니다. 같은 과거 KLUE 예측을 사용하여 Qwen 변화의 영향을 확인합니다. 새로 학습한 KLUE와의 평균도 보조 지표로 비교합니다. 이는 저장된 예측 비교이며 과거 KLUE 가중치를 새 제출 모델로 가져오지 않습니다.
- ASL, 8B 모델, 전체 Training 학습은 구현·실행하지 않습니다.

**ver26과 총 3 epoch scheduler를 동일하게 유지하고 Qwen 학습률만 낮춥니다.** 현재 GPU에서 자동 선택한 실제 배치가 기준 실행과 다르면 그 차이를 기록합니다. subset 최고 epoch는 전체 데이터의 최적 epoch가 아니며 development도 이미 과거 실험에 노출된 셋입니다.

confirmation과 공식 Validation은 모델 평가에 사용하지 않습니다. 입력 ZIP 준비 과정에서 Validation 파일을 추출하고 무결성을 확인할 수 있습니다. `run_mode='full'`, `run_validation=True`, 다른 epoch 수는 설정 검증에서 거절합니다.

## 산출물과 다음 판단

- `qwen_history.json`, `control_history.json`: train mean loss, development weighted BCE, Macro F1/AP와 9개 증상별 지표, learning rate 시작/끝, 성공한 optimizer update 수, scheduler 길이
- `learning_curve/{qwen,control}/epoch_*`: 모든 epoch의 development 예측과 지표
- `learning_curve_summary.json`: 같은 3 epoch 실행에서 epoch1 대비 후속 epoch 개선, 최고 epoch, 정체·하락 표시
- `candidate_summary.csv`, `screening_selection.json`: 진단 비교와 선택 근거
- `screening_protocol.json`: 해석 한계와 다음 단계
- `baseline_reference_verified.json`, `baseline_ver26/`: 확인한 ver26 기준 정보와 비교 자료 사본
- `historical_comparison.json`, `historical_comparison_predictions.npz`: 동일 development 예측의 이전 실행 대비 비교
- `baseline_gate.json`: 더 큰 fit 확인 후보 여부와 판정 근거, `full_training_authorized=false`
- `screening_portable_verification.json`: 선택된 두 checkpoint를 다시 로드하여 최대 16개 예측을 비교한 결과
- `screening_complete.json`: 3 epoch 및 저장 검증 완료 표시, `submission_ready=false`
- `source_snapshot/`, `shared_base/`, `fit_qwen/`, `fit_control/`: 재현에 필요한 코드와 가중치

실행 폴더 전체를 보관합니다. `screening_complete`는 필수 기준 비교와 저장 검증을 완료했다는 뜻이며 점수 개선이나 제출 성공 표시가 아닙니다. 제출 CSV, 최종 manifest, `run_complete.json`, ver25가 받아들이는 `pilot_complete.json`은 만들지 않습니다. ver27 결과를 ver25 `source_run`으로 사용하지 마세요.

곡선에서 유망한 조건이 확인되면 **fit 전체 20,440개에서 epoch와 recipe를 먼저 확인**해야 합니다. 이후 조건을 동결하고 reserved confirmation을 한 번 확인한 다음 전체 Training refit을 별도 실행하는 절차가 필요합니다. 현재 노트북은 이 단계를 자동으로 시작하지 않습니다.

주 비교의 Macro F1이 0.005 이상 오르고 통화 단위 paired bootstrap의 기술적 95% 하한이 0보다 높으면 `promising_for_larger_fit`로 표시합니다. 그 외에는 `no_clear_gain`입니다. 이 기준은 자원 배분을 위한 진단일 뿐 전체 학습을 자동 승인하지 않습니다. Development 재사용·epoch 선택의 낙관 편향과 seed·환경 변동은 bootstrap 구간에 포함되지 않습니다. 공식 Validation 점수가 0.7을 넘었다는 뜻이 아닙니다.

## 로컬 검증

프로젝트 루트에서 실행합니다.

```bash
python src/mission3/mission3_ver27/build_notebook.py
python -m unittest src.mission3.mission3_ver27.test_screening src.mission3.mission3_ver27.test_baseline -v
```

작은 임의 초기화 모델로 3 epoch 학습·저장·재로딩, 기준 비교 통합, 평가 경계, 노트북 독립 실행 코드 포함을 확인합니다. 기준 helper는 저장 자료의 무결성과 비교 판정을 별도로 검사합니다. 이것은 실제 NF4 GPU 학습·속도·성능·Colab 자동 반환 검증을 대신하지 않습니다.

Qwen은 Apache-2.0, KLUE는 CC-BY-SA-4.0이며 원본 출처·라이선스 문서를 결과에 보존합니다.
