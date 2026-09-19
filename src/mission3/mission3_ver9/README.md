# Mission 3 ver9 — Korean DeBERTa-v3 + NB-log-count LR

`../mission3_ver9.ipynb` 한 개를 Colab에 올리고, ver8 때 쓰던 `mission3_ver5_inputs.zip`을 그대로 사용합니다.
**아직 전체 GPU 학습과 ver9 F1은 측정하지 않았습니다.** 새 실행 결과와 기존 ver4 비교를 저장하도록 구현했습니다.

## 실행

1. 공유 ZIP에 **정리 → 바로가기 추가 → 내 드라이브**를 적용합니다. 기본 위치는 `/content/drive/MyDrive/mission3_ver5_inputs.zip`입니다.
2. `mission3_ver9.ipynb`를 Colab에서 열고 GPU 런타임을 선택합니다. L4를 우선 사용하고 T4에서는 자동으로 작은 microbatch를 사용합니다.
3. **모두 실행**합니다. Drive를 연결한 계정에 ZIP 바로가기가 있어야 합니다.
4. 결과는 `내 드라이브/DCC_AIdeas/results/mission3/ver9`에 저장됩니다. 새 설정으로 실험하려면 `OUTPUT_DIR`을 새 폴더로 바꾸세요.

추가 소스 업로드는 필요 없습니다. 실행 코드와 requirements가 노트북에 포함됩니다. 공개 모델만 자동 다운로드하고,
Training·Validation 자료를 외부 API로 보내지 않습니다. 모든 작업은 노트북의 한 절대 마감을 공유합니다.
설정 셀부터 설치, ZIP 복원, 공개 가중치 준비, 학습, 보정, 최종 추론·평가까지 **최대 10,800초 예산**입니다.
Drive 연결 대기나 수동 셀 간 대기 역시 설정 셀 이후면 예산에서 차감됩니다.

최종 추론을 위해 최소 25분, Training 보정을 위해 추가 10분을 남깁니다. 첫 fold의 실제 소요 시간을 보고
두 번째 fold를 실행할지 결정하고, epoch마다 시간 부족 여부를 확인합니다. 외부 작업은 제한 시간이 지나면 자식 프로세스까지 종료합니다.
다운로드·Drive 장애나 GPU 속도 때문에 **3시간 안에 모든 학습과 평가의 완료를 보장하지는 않습니다.**
후보 실행이 늦어지면 먼저 완성해 둔 기존 모델 묶음을 사용하며, 추론 자체가 끝나지 않으면 완료된 모델·기록을 보존합니다.
자동으로 새 3시간 실행을 이어 붙이지 않습니다. 새 Colab 세션에서 사용자가 다시 실행하면 별도 3시간 예산이며 누적 실행 비용은 추가됩니다.

## 변경 내용

- `team-lucid/deberta-v3-base-korean` 공개 encoder 전체 미세조정. 고정 revision은 `47a3b00828c5dc2b8b080c6738ef7a58ebd419d5`입니다.
- CLS + mean pooling, 증상별 window attention; 모든 대화의 512-token window와 64-token overlap을 유지합니다.
- unweighted BCE, encoder LR `2e-5`, head LR `1e-4`, 최대 3 epoch, patience 2. 유효 batch는 32입니다.
- 최대 2개 Training fold의 학습된 최적 모델 확률을 평균합니다. 두 번째 fold 실행 여부는 시간으로만 정합니다.
- 문자·단어 TF-IDF에 증상별 NB log-count-ratio를 곱한 Logistic Regression을 추가합니다. SVM 구현이 아닙니다.
- 새 후보는 `0.75 × DeBERTa 평균 + 0.25 × NB-LR`로 미리 고정합니다.
- 후보와 기존 ver4 혼합은 Training calibration의 grouped cross-fit에서 `F1 +0.002` 및 최소 3개 fold 개선을 모두 만족할 때 채택합니다.
- 조건을 통과하지 못하면 기존 ver4의 원래 가중치·보정값을 그대로 사용합니다. 후보 단독 결과는 별도로 기록합니다.

자세한 기존 결과 분석, 논문, 해외 원문 자료와 규칙 근거는 [RESEARCH.md](RESEARCH.md)에 있습니다.
공식 Validation은 선택 완료 뒤 평가에만 쓰며 학습 함수는 Validation 경로를 받지 않습니다.
NB 모델도 공통 Training calibration을 제외하고 학습합니다. 같은 전사의 그룹은 fit/stop/cal 사이에 섞이지 않게 검증합니다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `mission3_ver9_model.ckpt` | 최종 조합·임계값·상대 경로·SHA256을 담은 JSON manifest |
| `base/` | 기존 ver4의 실제 5개 `.ckpt`, sparse 모델, tokenizer, manifest |
| `candidate_1.ckpt`, 선택적 `candidate_2.ckpt` | 새 DeBERTa의 실제 최적 가중치 |
| `candidate_*_last.ckpt` | 중간 재개용 optimizer 포함 상태; 완료된 모델은 최종 선택 저장 후 정리 |
| `nb_ratio_lr.joblib`, `tokenizer/` | 새 희소 분류기와 DeBERTa tokenizer |
| `run_identity.json`, `candidate_history.json` | 데이터·코드·모델 신원 및 epoch 기록 |
| `training_metrics.json`, `calibration_comparison.json` | Training 내부 선택과 채택/미채택 이유 |
| `inference.csv`, `validation_metrics.json` | 최종 예측 및 최종 Macro F1 |
| `candidate_inference.csv`, `candidate_validation_metrics.json` | 시간이 남을 때만 수행하는 후보 단독 진단 |
| `runtime.json` | 설치부터 마지막 단계까지 실제 시간과 중단 사유 |
| `submission/` | 독립 실행 코드, `model.py`, `inference.py`, requirements 사본 |

결과 폴더는 원래 ver4 형제 폴더 없이 이동할 수 있도록 실제 base 파일을 복사합니다. 따라서 **최소 약 10~15 GB의 추가 Drive 공간**을 권장합니다.
기존 입력 ZIP 용량은 별도입니다. 정확한 필요 공간은 후보 모델 수와 중간 optimizer 저장 상태에 따라 달라집니다.
작은 `.ckpt` manifest 하나만 내려받으면 추론할 수 없습니다. 결과 폴더 전체가 필요합니다.

아래 명령은 문제의 네 인자를 그대로 받으며 경로를 옮겨도 별도 `--base_dir`가 필요 없습니다.

```bash
pip install -r results/mission3/ver9/submission/requirements.txt
python results/mission3/ver9/submission/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver9/mission3_ver9_model.ckpt \
  --output results/mission3/ver9/inference.csv
```

`--audio_dir`는 규정 인터페이스 호환용이며 음성을 읽지 않습니다. 입력은 양측 전사 text뿐입니다.
CSV 헤더는 `label file name,symptom`이며 증상은 지정 9개 중 0~9개의 문자열 목록입니다.
학습/추론은 준비된 로컬 가중치만 읽습니다. 공개 모델 다운로드는 별도 준비 단계에서만 수행합니다.

## 재개와 검증 범위

학습은 완료된 epoch 단위로 재개합니다. 데이터·설정·코드·공개 revision·GPU/정밀도가 달라지면 무조건 이어서 학습하지 않습니다.
중간 epoch는 최적 가중치로 발표하지 않습니다. 완료된 실험은 최종 선택을 재사용하며 Validation 결과로 재선택하지 않습니다.
새로운 실험 설정에는 새 결과 폴더를 사용하세요.

CPU의 작은 DeBERTa와 합성 데이터로 전체 encoder 업데이트, 긴 대화 꼬리 gradient, 정확한 epoch 재개,
부분 epoch 중단, 순수 추론, 희소 분류기, 그룹 보정, 시간 초과 자식 종료와 제출 CSV 흐름을 검증합니다.
이는 실제 한국어 pretrained 가중치의 Colab GPU 속도나 F1 검증을 대신하지 않습니다.

```bash
PYTHONPATH=src/mission3 python -m pytest src/mission3/mission3_ver9/tests -q
python src/mission3/mission3_ver9/build_notebook.py
```

제출 전에는 Colab 실행 결과를 저장한 학습 `.ipynb`, 실제 가중치·추론 코드, requirements, 결과 PPT를 준비해야 합니다.
현재 저장소의 새 노트북은 실행 전 템플릿이며, 측정 점수·학습 출력·PPT를 임의로 만들어 넣지 않았습니다.
