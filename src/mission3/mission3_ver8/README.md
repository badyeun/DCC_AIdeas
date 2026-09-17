# Mission 3 ver8

`mission3_ver8.ipynb` 하나와 기존 `mission3_ver5_inputs.zip`으로 Colab에서 실행합니다.
**실제 전체 데이터 GPU 학습·Validation 성능은 아직 측정하지 않았습니다.**
기본 실행은 **B 하나만 최대 3 epoch** 학습하며, A/C 비교와 TAPT는 생략합니다.

## 실행

1. 공유된 `mission3_ver5_inputs.zip`의 바로가기를 내 드라이브 바로 아래에 둡니다.
2. `src/mission3/mission3_ver8.ipynb`를 Colab에서 열고 GPU 런타임을 선택합니다. L4 이상을 권장합니다.
3. **모두 실행**하고 같은 계정으로 Drive를 연결합니다.
4. 결과는 `내 드라이브/DCC_AIdeas/results/mission3/ver8`에 저장됩니다.

기존 ver4 모델·서울 Training/Validation JSON은 ZIP에서 자동 복원합니다. WAV는 사용하지 않습니다.
공개 `klue/roberta-large`의 encoder와 MLM head를 고정 커밋
`28d911204e9022eda172571ca8cc61eaffd942f7`에서 준비합니다. 이 단계 이후 학습·추론은 오프라인입니다.

## 기본 후보 B와 선택 가능한 비교 실험

| 후보 | 분류 구조 | 추가 사전학습 | 가중치 학습 |
| --- | --- | --- | --- |
| A (선택) | ver4의 CLS+mean/window attention 구조 | 없음 | 공개 encoder + 새 분류기 전체 |
| B (기본) | 증상별 직접 token attention | 없음 | A와 같은 공개 encoder + 새 분류기 전체 |
| C (선택) | B와 같은 직접 token attention | fit 전사만 이용한 TAPT | TAPT encoder + B와 같은 초기 분류기 전체 |

A/B/C를 모두 돌리면 세 후보 중 하나만 선택합니다. 세 후보 전체가 최종 모델에 필수인 것은 아닙니다.
B 단독 실행도 같은 fit 데이터, 512-token 입력, 전체 encoder 학습, 최대 3 epoch를 유지합니다.
대신 A 대비 구조 변경 효과와 C가 더 좋았을 가능성은 확인하지 못합니다. 성능 차이는 실행 전에는 알 수 없습니다.

A는 저장된 ver4 fold를 그대로 복제한 것이 아니라 **같은 공개 모델에서 새로 학습하는 구조 비교군**입니다.
A/B/C 모두 발화 사이에 동일한 `[SEP]`를 넣고 512-token window/64 overlap을 사용합니다.
학습 seed·분할·weighted BCE·learning rate·batch·epoch 선택 절차도 같습니다.
B/C는 encoder 로딩 뒤 seed를 다시 고정하여 초기 분류기 가중치가 같습니다.

- fit: 기존 fold 1의 21,032개. 지도학습과 C의 TAPT에만 사용.
- stop: 5,288개. 같은 그룹 분리 3-fold threshold 교차평가 Macro F1로 epoch·후보 선택. AP는 동점 보조.
- calibration: 2,880개. **stop에서 고른 후보 하나만** 기존 앙상블과 혼합·보정.
- Validation: 선택 완료 후 3,640개 전체에서 최종 구성과 선택된 후보 단독을 평가.

후보 선택에 calibration·Validation을 사용하지 않습니다. 같은 stop을 반복 사용하는 선택 편향은 남습니다.
동점은 A → B → C 순서입니다. 혼합은 기존 대비 F1 최소 +0.002 및 최소 3/5 fold 개선 때 채택합니다.
조건 미달이면 기존 ver4를 유지하되, 실행한 후보의 내부 결과와 선택 후보 단독 점수를 함께 남깁니다.
이번 기본 실행은 한 fold의 탐색이며, 새 모델의 5-fold 앙상블이나 다중 seed 검증을 자동 실행하지 않습니다.

## 오답 검토

학습 전에 기존 calibration 확률로 오심·두통·전신쇠약의 FP/FN을 각각 20건씩 무작위 추출합니다.
오류가 부족하면 있는 만큼만 뽑습니다. 최대 120행이며 같은 통화가 다른 증상에서 다시 나올 수 있습니다.

`error_analysis/review_samples_template.csv`를 `review_samples.csv`로 복사해 사람이 검토합니다.
`review_category`와 `review_note`는 비어 있습니다. 검토 복사본은 재실행 때 보존됩니다.
키워드 규칙이나 LLM으로 라벨 오류를 확정하지 않으며, 정답·전사·학습 가중치를 자동 수정하지 않습니다.
학습은 사람의 검토를 기다리지 않고 진행하므로 이 CSV는 다음 실험을 위한 진단 자료입니다.
현재 오류 집계는 보정에 사용한 표본에서 고정 임계값을 적용한 값이며 비편향 성능 추정치가 아닙니다.

## TAPT와 비용

C를 명시적으로 켠 경우에만 공개 RoBERTa-large의 원래 MLM head로 **fit 전사만** 마스킹 복원 학습을 합니다.
정답·stop·calibration·Validation·사람의 오답 검토는 TAPT에 전달하지 않습니다.
기본 500 optimizer step, 길이 256, mask 비율 0.15, LR `2e-5`, 유효 window batch 32입니다.
패딩·특수 토큰은 마스킹 대상에서 제외합니다. 완료 후 encoder만 C의 분류 학습에 넘깁니다.

분류는 후보별 최대 3 epoch, patience 2, encoder LR `1.5e-5`, head LR `1e-4`,
layerwise decay 0.95, BF16 지원 GPU는 BF16, 나머지 CUDA는 FP16을 사용합니다.
T4/L4/A100 급 메모리의 자동 microbatch는 각각 2/4/8이며 유효 통화 batch는 32입니다.

기본 B 단독은 최대 **3 supervised epoch, TAPT 없음**입니다.
3시간 목표에 맞춘 L4 실행의 잠정 예상은 **약 1.5~2.5시간**입니다. 과거 RoBERTa-large 기록의
epoch당 약 29분을 참고했지만 당시 GPU 정보가 없으므로 오차가 있습니다. 실제 시간·CU·F1은 아직
측정하지 않았고, 다운로드·Drive 저장·GPU 속도에 따라 더 걸릴 수 있습니다. 3시간 자동 종료 기능은 없습니다.
B 가중치는 대략 1.4 GB, 실행 중 Adam 재개 상태 약 4 GB 및 저장 임시 공간이 추가로 필요합니다.
기존 ZIP·이전 결과를 제외한 Drive 여유 공간도 필요합니다. 기본값은 완료된 optimizer 재개 파일만 정리하고
최적 가중치·이력·TAPT encoder는 보존합니다. `keep_completed_resume_states=True`로 재개 상태도 보관할 수 있습니다.
전체 비교는 새 OUTPUT_DIR에서 `profiles=["A", "B", "C"]`로 켭니다.
이 경우 최대 9 supervised epoch + 500 TAPT step이며 후보/TAPT encoder 저장 공간은 약 5.5 GB입니다.
이전 A/B/C 기본 설정으로 이미 실행했다면 변경한 설정은 새 OUTPUT_DIR에서 실행해야 합니다.

## 중단 후 재개

같은 노트북을 같은 GPU 종류·정밀도·라이브러리·설정으로 다시 실행합니다.
완료 후보는 해시 검증 후 재사용하고, 미완료 분류 학습은 마지막 완료 epoch부터 이어집니다.
TAPT는 기본 100 step마다 optimizer·scheduler·마스킹 RNG를 저장하며 마지막 저장 step부터 이어집니다.
새 세션에서는 입력 ZIP과 공개 모델을 다시 로컬에 준비하지만 Drive의 학습 결과는 유지됩니다.
설정·코드·데이터 변경은 새 OUTPUT_DIR를 사용해야 합니다. 완료된 전체 결과 재사용은 다른 GPU에서도 가능합니다.

## 주요 결과

| 경로 | 내용 |
| --- | --- |
| `mission3_ver8_model.ckpt` | 최종 구성 manifest; 선택 후보 상대 경로·ver4 위치·혼합·임계값 |
| `trials/B/best.ckpt`, `done.json` | 기본 후보의 최적 가중치와 모든 epoch 이력; A/C를 켜면 각자 폴더에 저장 |
| `trial_progress.json` | 완료 후보 목록; 중간에 끊겨도 비교 가능 |
| `tapt/encoder/`, `tapt_done.json` | C를 켠 경우에만 fit 전사로 적응한 encoder·마스킹 학습 기록 |
| `error_analysis/` | 오답 집계, 검토 template, 검토 방법 |
| `mission3_ver8_metrics.json` | 실행한 후보 내부 결과, 선택, 혼합 결과, 후보 단독 내부 지표 |
| `predictions/<시각>/` | `mission3.csv`, `candidate_only.csv`, 두 상세 평가 JSON |
| `notebook_runtime_<시각>.json` | 이번 호출의 준비·전체 학습 호출·평가 시간; 설치·CU 제외 |
| `submission/` | 노트북과 동일한 실행 소스·패키지·문서 |

후보 선택은 stop 기준이므로 나머지 A/B/C의 공식 Validation 점수는 기본적으로 계산하지 않습니다.
완료 결과 재사용·중단 재개 시 notebook 시간은 해당 호출만 측정합니다. 과거 단계 시간을 다시 합산하지 않습니다.
`bundle_base=False`이므로 **기존 입력 ZIP과 ver8 결과 폴더 전체를 함께 보관**합니다.

```bash
python results/mission3/ver8/submission/inference.py --audio_dir "" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver8/mission3_ver8_model.ckpt \
  --base_dir results/mission3/ver4 --output outputs/mission3/ver8/mission3.csv \
  --candidate_output outputs/mission3/ver8/candidate_only.csv --batch_size 8 --evaluate
```

노트북 재생성: `python src/mission3/mission3_ver8/build_notebook.py`.
테스트: `PYTHONPATH=src/mission3 python -m pytest src/mission3/mission3_ver8/tests -q`.
작은 CPU 테스트와 실제 대회 데이터의 GPU 학습·F1 검증은 구분합니다.
