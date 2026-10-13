# Mission 3 ver24 — 의료 사전학습 encoder 비교

`mission3_ver24.ipynb`를 Colab GPU 런타임에서 **모두 실행**합니다. Drive 연결에는 기존 입력 ZIP이 있는 공용 계정을 선택합니다. 추가 업로드는 기존 `mission3_ver5_inputs.zip`만 필요합니다. 이전 버전의 모델 파일은 필요하지 않습니다. 모델 다운로드에는 인터넷 연결이 필요합니다.

## 이번 실험의 목적

의료 문장으로 사전학습된 `silber2258/bi-KM-BERT`가 동일한 분류 학습을 거친 `klue/roberta-base`보다 전체 증상과 오심을 잘 구분하는지 확인합니다. 두 모델의 공개 revision은 `model_specs.json`에 고정합니다. bi-KM-BERT는 한국어·영어 의료 corpus로 추가 사전학습한 BERT이며, 이 과제의 9개 증상을 이미 학습한 분류기는 아닙니다.

이 실험은 이전 ver19~23 모델의 추가 학습이나 친구의 11개 모델 앙상블 재현이 아닙니다. 의료 사전학습의 효과를 비교하는 새 실험입니다. 기존 최고 모델보다 성능이 높다는 가정은 하지 않습니다.

## 데이터와 비교 조건

- 원본 서울 Training 데이터와 9개 정답을 유지합니다. 의심 정답 자동 삭제, 재라벨링, label masking, 외부 학습 데이터 추가를 하지 않습니다.
- 입력은 `utterances[].text`이며 기존 데이터의 `utterences` 표기도 지원합니다. 다른 메타데이터를 입력으로 사용하지 않습니다.
- 같은 원문 대화가 fit과 holdout 양쪽에 들어가지 않는 Training 내부 분할을 seed 42로 만듭니다. holdout 비율은 약 20%입니다. 동일 대화 그룹 보존 때문에 정확한 행 수는 데이터에 따라 달라질 수 있습니다.
- 두 후보의 학습 데이터, 평가 데이터, 학습률, 손실, batch 설정 및 최대 epoch를 맞춥니다. 모델별로 해당 사전학습 tokenizer를 사용합니다. tokenizer의 차이 때문에 같은 512토큰이 포함하는 원문 길이는 다를 수 있습니다.
- 각 대화의 앞부분을 최대 512토큰으로 사용합니다. 모든 발화를 별도 창으로 보존한 ver19와 입력 처리 방식이 다르므로, 내부 점수를 ver19와 동일 조건 비교로 해석하지 않습니다.
- mean pooling, dropout 0.1, 9개 출력으로 증상을 분류합니다. 원문이나 환자별 진단 정보를 진행 로그에 출력하지 않습니다.

손실은 `BCEWithLogitsLoss(pos_weight=...)`입니다. 각 증상의 양성 가중치는 학습에 쓰는 데이터에서 계산한 `clip(sqrt(negative_count / positive_count), 1, 5)`를 사용합니다. 내부 비교에서는 fit 데이터에서만 계산하며, 최종 재학습에서는 전체 Training에서 다시 계산합니다. 양성·음성 수가 0인 경우는 구현의 수치 안전 처리를 따릅니다. 검증 데이터로 가중치를 맞추지 않습니다.

모든 증상의 최종 예측은 **`logit >= 0`**으로 고정합니다. 이는 sigmoid 확률 0.5와 같습니다. threshold 탐색, 증상별 기준 변경, 사후 calibration, 확률 이동, 앙상블 비율 탐색을 적용하지 않습니다.

## 학습 순서

1. Drive 연결 후 180분 공유 예산을 시작하고 기존 버전과 같은 패키지를 설치합니다. 별도 Python 프로세스에서 실행하므로 이미 로드된 Colab 패키지와 새 설치가 섞이지 않도록 합니다.
2. 입력 ZIP과 공개 모델을 준비하고 두 모델의 짧은 실제 학습 속도를 측정합니다. 속도 측정용 학습은 후보의 최종 시작 가중치로 재사용하지 않습니다.
3. 두 후보의 학습·내부 평가와 선택 모델의 전체 Training 재학습까지 고려해 가능한 **공통 최대 epoch 수**를 최대 3 안에서 정합니다. 느린 장치에서는 더 적은 epoch로 진행하거나 명확히 중단합니다.
4. 두 후보를 사전학습 가중치에서 각각 새로 학습합니다. 각 후보의 내부 평가 Macro F1이 가장 좋은 epoch 수·예측·지표를 저장합니다. Drive 저장량을 줄이기 위해 비교용 fit 가중치는 보관하지 않으며, 최종 전체 Training 재학습 가중치만 보관합니다.
5. **의료 후보의 Macro F1이 대조군보다 높고 오심 F1도 낮아지지 않을 때만 의료 후보를 선택**합니다. 그렇지 않으면 이번 실행의 일반 한국어 대조군을 선택합니다.
6. 선택한 모델 종류와 epoch 수로 전체 Training을 사전학습 가중치에서 다시 학습합니다. fit-only 모델을 최종 제출 모델로 표시하지 않습니다.
7. 저장한 모델을 다시 불러와 추론이 일치하는지 검증한 뒤 공식 Validation을 평가하고 완료 표시를 남깁니다. Validation은 후보·epoch 선택에 쓰지 않습니다.

기본 설정은 batch 8, gradient accumulation 2, 학습률 2e-5, weight decay 0.01, warmup 비율 0.1입니다. 학습 속도 추정에는 안전 계수 1.35를 적용하며 최종 평가·저장용으로 20분을 예약합니다. 180분은 설치·다운로드·준비·학습·저장을 합한 실행 예산이며, 모든 GPU에서 모든 단계의 완료를 보장하는 시간이 아닙니다. 부족한 예산에서 완료 결과를 가장하지 않습니다.

학습률 스케줄의 기준 기간은 내부 fit과 전체 Training 재학습 모두 속도 측정 후 정한 `common_epochs`입니다. 선택된 epoch가 이보다 적더라도 스케줄을 그 짧은 기간으로 다시 압축하지 않습니다. 각 단계의 전체 데이터 수에 맞춰 step 수를 계산하므로, 선택 epoch까지의 상대적 학습률 변화를 유지합니다.

## 결과 파일

기본 결과 위치는 `MyDrive/DCC_AIdeas/results/mission3/ver24/<run_id>/`입니다.

| 파일 | 용도 |
|---|---|
| `candidate_summary.csv` | 같은 holdout에서 후보별 Macro F1, 오심 F1, 선택 epoch 비교 |
| `final_selection.json` | 선택 후보와 이유 |
| `training_metrics.json` | 선택한 fit 후보의 내부 holdout 점수 |
| `paired_error_changes.json` | 일반 한국어 모델 대비 의료 모델의 누락 회복·새 누락·오탐 제거·새 오탐 |
| `training_review_candidates.csv`, `review_note.json` | 원래 오심 정답과 두 모델 판단이 많이 다른 Training holdout 검토 후보 최대 200건 |
| `validation_metrics.json` | 전체 Training 재학습 모델의 공식 Validation 점수 |
| `mission3.csv` | 최종 Validation 예측 |
| `mission3_ver24_model.ckpt`, `final_model/` | 최종 모델 manifest와 가중치·tokenizer 등 |
| `runtime.json`, `run_complete.json` | 실행 상태와 완료 표시 |

`run_complete.json`과 최종 CSV가 없으면 완료된 제출 결과로 사용하지 않습니다. `run_validation=False`는 모델 검증용으로 CSV를 만들지 않으므로 기본값 True를 유지합니다. manifest만 단독 보관하지 말고 **실행 폴더 전체를 보관**합니다.

검토 후보 파일에는 정렬된 Training 데이터의 0부터 시작하는 행 번호, 원래 정답, 두 모델의 오심 확률을 저장합니다. 대화 원문과 환자 파일 이름은 포함하지 않습니다. 모델 둘이 원래 정답과 다르게 판단하더라도 정답 오류가 확정된 것은 아닙니다. 원문을 읽어 근거를 검토하기 위한 목록이며, 이 버전은 목록을 정답 수정이나 학습 제외에 사용하지 않습니다.

실패 시 노트북은 GPU 반환 전에 실패 단계·오류 종류·코드 위치를 표시하고 `ver24/_diagnostics/ver24_error_*.json`에 저장합니다. 오류 보고서에는 원문 대화와 개별 환자 파일 이름을 포함하지 않습니다. 학습 중단 후 자동 재개는 지원하지 않습니다. 다시 실행하면 새 예산과 Colab 리소스를 사용합니다. 자동 GPU 반환이 실패하면 Colab에서 직접 연결 해제 및 런타임 삭제를 실행합니다.

## 검증 범위

로컬에서 코드와 노트북 생성, 실행 준비·오류 처리, 데이터 분리·예산·저장 관련 검증을 수행할 수 있습니다. 실제 F1, GPU 학습 속도, Drive 속도, Colab 자동 런타임 반환은 실제 Colab 실행에서 확인해야 합니다. 원문에 증상 단어가 없다는 이유만으로 정답 오류가 확정되는 것은 아니므로, 의심 라벨은 이 버전에서 학습 대상에서 제외하지 않습니다.

공개 모델 출처: [KU-RIAS bi-medical-bert](https://github.com/KU-RIAS/bi-medical-bert), [bi-KM-BERT](https://huggingface.co/silber2258/bi-KM-BERT), [KLUE-RoBERTa-base](https://huggingface.co/klue/roberta-base). 체크포인트별 상세 출처와 revision은 `model_specs.json`을 확인합니다.

검토한 bi-KM-BERT 모델 카드에는 가중치의 명시적 라이선스가 없었습니다. 공개 다운로드 가능 여부와 재배포 허용 여부는 다르며, 이 정보는 `model_specs.json`에도 기록합니다. 해당 모델의 라이선스 조건이 확인됐다고 주장하지 않습니다.

저장된 결과 폴더의 `submission/inference.py`는 `--checkpoint <run>/mission3_ver24_model.ckpt --label_dir <label_directory> --output <output_csv>` 인자로 추론합니다. 실행 위치가 달라도 manifest와 `final_model/`의 상대 위치를 유지해야 합니다.
