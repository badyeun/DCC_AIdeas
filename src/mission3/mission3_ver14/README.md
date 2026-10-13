# Mission 3 ver14 — 오심 분석과 단계별 학습 비교

`../mission3_ver14.ipynb` 한 개를 Colab에서 열고 기존 `mission3_ver5_inputs.zip`으로 실행합니다.
실행 코드와 requirements가 노트북에 포함되어 다른 Python 파일 업로드가 필요 없습니다.
이 버전은 기존 ver10–ver13 코드와 결과를 수정하지 않습니다.

## 실행

1. ZIP 바로가기가 있는 공용 계정으로 Colab에 로그인하고 GPU 런타임을 선택합니다.
2. `런타임 → 모두 실행`을 선택합니다. ZIP 기본 경로는 `/content/drive/MyDrive/mission3_ver5_inputs.zip`입니다.
3. 결과는 `MyDrive/DCC_AIdeas/results/mission3/ver14/실행시각_고유번호/`에 저장됩니다.
4. Training 후보 비교와 최종 선택 근거를 먼저 확인하고, 마지막 Validation 결과를 기록합니다.

기본 실행은 후보 6개 학습과 최종 구성 1개 전체 Training 재학습을 수행합니다.
비슷한 GPU에서 약 6–10시간을 예상하지만 GPU 성능·메모리·조기 종료에 따라 달라집니다.
고정 시간 제한은 없습니다. **GPU 실행 시간과 F1 개선은 실제 실행 전에는 확인되지 않았습니다.**

## 1. 원본과 정답 점검

기존 오심 검토 표 80행, 중복을 제외한 76개 통화를 원본과 대조했습니다.
개인정보를 제외한 관찰 사례는 다음과 같습니다.

| 증상 내용의 요약 | 제공된 정답과 모델 결과 | 검토 의미 |
| --- | --- | --- |
| 메스꺼움과 구토를 함께 말함 | 정답은 구토·어지러움, 모델 오심은 FP | 명시 표현과 정답 조합이 어긋나는 사례 |
| 토할 것 같은 느낌과 어지러움을 말함 | 정답은 어지러움, 모델 오심은 FP | 구토·오심·어지러움의 경계 |
| 토할 것 같은 느낌을 말함 | 정답에 오심이 포함되지만 모델은 구토만 예측하여 FN | 유사 표현에서 발생한 증상 혼동 |

관찰만으로 라벨 오류를 확정하지 않습니다. 원본 정답을 그대로 유지하며,
질문·부정·간접 증상 표현과 정답 조합을 사람이 검토합니다.
새 실행은 Training 데이터에서 fit/holdout을 구분해 분포·오심/구토 조합·검토 후보를 저장합니다.
1단계 검토 목록은 원본 행 위치·정답·후보 표시를 포함하고 화면에는 집계만 표시합니다.
학습 후 오류 CSV에는 실제 전사가 저장됩니다.

기존 ver13의 오심 FN 575개 중 567개(98.6%)는 단일 512-token window였습니다.
ver11에서 맞혔다가 ver13이 놓친 16개도 두 tokenizer 모두 단일 window였고,
KcELECTRA 기준 최대 417 tokens였습니다. 정규식으로 찾은 오심 표현 후보는 모두 첫 window에 있었습니다.
이 결과는 긴 대화 확장보다 정답의 경계와 학습 손실을 먼저 점검할 근거입니다.
정규식에 없는 우회 표현이나 실제 정답의 정확성은 별도로 검토해야 합니다.

## 2. 오심 양성 학습 가중치 비교

KLUE-RoBERTa-base의 공개 revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`를 고정합니다.
attention 없이 교차엔트로피를 학습하고 오심 양성 class의 가중치 1·2·3을 비교합니다.
가중치 1이 이번 실행의 새 기준 모델입니다. 가중치는 학습 중 오분류 비용에 적용하며,
최종 예측은 두 class logits의 argmax를 그대로 사용합니다.
오심 재현율이 늘어도 오탐이나 다른 증상이 악화될 수 있으므로 오심 P/R/F1과 전체 Macro F1을 함께 비교합니다.

## 3. Attention과 대조학습을 분리해 비교

2단계에서 선택한 가중치를 고정하고 다음 세 후보를 새로 학습합니다.

| 후보 | Attention | 교차엔트로피 | 대조학습 |
| --- | --- | --- | --- |
| attention_off_ce | 끔 | 사용 | 없음 |
| attention_on_ce | 켬 | 사용 | 없음 |
| attention_on_supcon | 켬 | 사용 | 0.1 |

3단계는 세 후보 모두 실제 batch 8, gradient accumulation 2입니다.
세 후보 모두 동일 통화를 dropout으로 두 번 읽고 두 교차엔트로피의 평균을 사용합니다.
`attention_on_supcon`에만 대조학습을 더해 두 view 사용과 대조학습 손실의 효과를 분리합니다.
2단계는 batch 4, accumulation 4이며 유효 batch는 두 단계 모두 16입니다.
대조학습은 실제 batch 안의 표본만 비교하며 accumulation이 비교집합을 늘리지는 않습니다.
encoder·Training 분할·seed·학습률·epoch 선택 규칙을 맞춰 비교합니다.

최종 선택은 이번 실행의 새 기준 모델 대비 **Macro F1과 오심 F1을 모두 유지하거나 개선한 후보**에 한정하고,
그 안에서 Macro F1을 우선합니다. 개선 후보가 조건을 통과하지 못하면 새 기준 모델을 유지합니다.
선택된 구성 하나만 공개 초기 가중치에서 전체 Training으로 다시 학습하고,
공식 Validation은 마지막 예측 CSV를 저장한 뒤 한 번 평가합니다.

## 실행 조건과 재개

- 입력은 양측 발화의 `text`만 사용합니다. symptom은 정답으로만 사용합니다.
- 중복 전사 그룹이 내부 fit/holdout에 섞이지 않게 분할합니다. Validation은 학습·후보 선택에 쓰지 않습니다.
- 추가 corpus·상용 AI API·사후 threshold 검색·calibration·기존 모델 fallback을 사용하지 않습니다.
- 학습 가중치는 미리 정한 1·2·3만 비교합니다. 검토 표현으로 정답을 수정하거나 샘플을 제거하지 않습니다.
- 모든 window를 유지하며 증상별 두 class의 argmax로 0개 또는 여러 증상을 예측합니다.
- dependency 버전 범위는 기존 ver10–ver13 requirements와 같습니다.

중단 후에는 노트북 설정의 `resume_run`에 이미 생성된 실행 폴더의 전체 Drive 경로를 지정합니다.
코드·Training 데이터·설정 fingerprint를 확인한 뒤 완료된 후보 결과를 재사용합니다.
미완료 후보의 epoch 중간 상태는 이어서 학습하지 않습니다. 해당 후보는 새 시도에서 처음부터 학습합니다.
설정이 다른 실험은 `resume_run`을 비우고 새로 실행합니다.
GPU 메모리가 부족하면 해당 단계의 batch를 줄이고 accumulation을 맞춰 조절하세요.
실제 batch 변경은 대조학습 비교 표본에도 영향을 주므로 서로 다른 조건의 실험을 그대로 재개하지 않습니다.

## 결과와 제출

| 파일 | 내용 |
| --- | --- |
| `experiment_plan.json` | Training 기반 단계별 실험 조건 |
| `candidate_summary.csv` | 6개 후보의 오심 P/R/F1·FP/FN과 전체 Macro F1 |
| `final_selection.json` | 최종 선택과 기준 모델 대비 조건 충족 여부 |
| `training_metrics.json` | Training holdout에서 선택한 구성의 지표 |
| `validation_metrics.json` | 선택·최종 재학습 완료 후 공식 Validation 평가 |
| `runtime.json` | 실행 시간과 완료 상태 |
| `mission3.csv` | 최종 argmax 예측, `label file name,symptom` 헤더 |
| `mission3_ver14_model.ckpt`, 가중치·`assets/`·`submission/` | 이동 가능한 최종 추론 묶음 |

`.ckpt`는 manifest입니다. 같은 실행 폴더의 실제 가중치와 assets를 함께 보관해야 합니다.
최종 가중치와 assets는 `실행폴더/final_model/시도번호/` 아래에 저장되며,
실행 폴더 루트의 `mission3_ver14_model.ckpt`는 해당 파일들의 상대 경로를 참조합니다.
`runtime.json`의 seconds는 재개를 포함한 이번 실행 호출의 시간이며 후보별 시간은 `candidate_summary.csv`에 기록됩니다.
실행 로그가 저장된 학습 노트북과 실제 결과 PPT도 준비합니다.

```text
python submission/inference.py --audio_dir /path/audio --label_dir /path/json --ckpt_path mission3_ver14_model.ckpt --output mission3.csv
```

노트북을 다시 생성하려면 다음을 실행합니다. ver14 파일만 생성합니다.

```text
python src/mission3/mission3_ver14/build_notebook.py
```
