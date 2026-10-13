# Mission 3 ver10–ver13 실행 안내

네 노트북은 기존 `mission3_ver5_inputs.zip`과 호환되는 독립 Colab 파일입니다.
다른 Python 파일을 업로드하지 않아도 됩니다. 기존 ver1–ver9는 수정하지 않습니다.

## 실행

1. ZIP 바로가기가 있는 공용 계정으로 Colab에 로그인합니다.
2. `mission3_ver10.ipynb`부터 열어 `런타임 → 모두 실행`을 선택합니다.
3. ver11–ver13은 GPU 런타임이 필요합니다. 기본 배치는 ver11=4, ver12–13=8입니다.
   메모리가 부족하면 batch_size를 절반으로 줄이고 gradient_accumulation을 두 배로 늘립니다.
4. 결과는 `MyDrive/DCC_AIdeas/results/mission3/ver번호/실행시각_고유번호/`에 저장됩니다.

| 파일 | 모델 | 비교 목적 |
| --- | --- | --- |
| mission3_ver10.ipynb | 문자·단어 TF-IDF + LinearSVC / ComplementNB | 보정 없는 빠른 분류 기준 |
| mission3_ver11.ipynb | KLUE-RoBERTa-base + 9개 이진 categorical head | 표준 교차엔트로피 기준 |
| mission3_ver12.ipynb | 같은 RoBERTa + 증상 attention + 대조학습 | 증상별 문맥과 있음/없음 표현 구별 |
| mission3_ver13.ipynb | KcELECTRA v2022 + 같은 attention·대조학습 | 구어체 사전학습 encoder 비교 |

GPU 버전은 내부 선택 학습 뒤 선택된 epoch 수만큼 공개 encoder에서 새로 전체 Training을
학습하므로 수 시간이 걸릴 수 있습니다. 이전 버전의 3시간 예산 제한은 적용하지 않습니다.
`refit_full_training=False`는 전체 재학습을 생략하지만 Training의 80%로 학습한 모델이 최종 모델이 됩니다.
중단 시 완료된 epoch 가중치·로그는 남으며, 자동 재개 기능은 없습니다. 재실행은 새 결과 폴더를 만듭니다.

## 규칙 적용

- 입력은 `utterances[].text`만 사용합니다. 발화자·시간·성별·주소·symptom을 특징에 넣지 않습니다.
- symptom은 Training 정답과 최종 평가 정답으로만 사용합니다.
- 9개 외 정답만 제거하고 통화 자체는 유지합니다. 0개 또는 여러 증상 예측이 가능합니다.
- 공개 pretrained encoder만 활용하며 추가 학습 corpus나 상용 AI API를 사용하지 않습니다.
- 학습·epoch·후보 선택은 Training 내부 분할에서만 수행합니다. 같은 전사는 한쪽 집합에만 둡니다.
- TF-IDF 어휘는 내부 fit 집합에서만 학습하고, 최종 선택 후 전체 Training에서 다시 학습합니다.
- 분류기의 `predict`, 신경망의 증상별 두 class logits의 `argmax`를 그대로 사용합니다.
  threshold 검색·calibration·사후 score 이동·증상별 앙상블 가중치·기존 보정 모델 fallback은 없습니다.
- argmax도 수학적 결정 경계를 갖습니다. 여기서 제거한 것은 사후 임계값 선택·튜닝입니다.
  멘토님의 지시를 따르며 이를 “수학적으로 경계가 전혀 없는 모델”이라고 해석하지 않습니다.
- 공식 Validation은 최종 예측 CSV 저장 뒤 평가만 합니다. 여러 버전 선택에는 Training 지표를 사용합니다.

출제 조건은 `docs/0820_DCC 대학부_예선_출제문제.pdf` 5·9–12쪽에서 확인했습니다.
해당 PDF에 threshold 금지 문구는 없지만 전달받은 멘토 지침을 추가 조건으로 적용했습니다.

## 오심 오류를 먼저 보는 이유

저장된 ver9 공식 Validation의 오심 F1은 0.4031, 정밀도 0.3205, 재현율 0.5431,
오탐 494개, 미탐 196개입니다. 기존 최종 예측은 새 DeBERTa보다 기존 ver4 fallback의 결과입니다.
Training calibration의 기존 보정 모델은 구토만 정답인 311개 중 143개를 오심으로 예측했습니다.
이는 최적화에 사용된 자료의 기술적 분석이며 독립적인 평가 점수가 아닙니다.

Training 전체 29,200개에서 오심 정답은 3,341개, 구토 정답은 4,463개입니다.
넓은 오심 표현 정규식으로도 오심 정답 2,901개(86.8%)는 직접적인 표현을 찾지 못했습니다.
따라서 키워드만으로 라벨을 바꾸거나 샘플을 제거하지 않습니다. 자동 표현 집계는 원인 확정 자료가 아닙니다.

각 실행은 `training_diagnostics.json`, `training_errors/errors_오심.csv` 등을 만듭니다.
오탐과 미탐에서 질문과 실제 답변, 부정 표현, 구토만 있는 경우, 간접 증상 표현을 사람이 검토합니다.
CSV는 실제 전사를 포함하고 화면에는 집계 지표만 출력합니다.

## 대조학습

ver12–ver13은 동일한 Training 통화를 두 번 dropout forward해 두 표현을 얻습니다.
증상별 있음/없음 정답이 같은 다른 표현을 가까이 하고 다른 class를 구별하는 손실을 교차엔트로피에 추가합니다.
각 증상의 양성·음성 anchor 평균을 균등하게 집계하여 다수 음성이 대조학습 항을 지배하지 않도록 합니다.
대조학습은 실제 batch의 통화끼리만 비교합니다. gradient accumulation은 비교집합을 늘리지 않습니다.
불확실한 정답을 자동 수정하지 않습니다. 효과는 Training holdout의 증상별 P/R/F1로 확인해야 합니다.
ver11→ver12는 attention과 손실을 함께 바꾸므로 각각의 효과를 분리한 인과적 실험은 아닙니다.

## 결과와 제출

`.ckpt`는 manifest입니다. 동일 실행 폴더의 `sparse.joblib` 또는 `.pt`와 `assets`를 함께 보관하세요.
최종 모델은 재로딩해 확인한 뒤 Validation CSV를 생성합니다. 새 환경 추론은 모델 다운로드가 필요 없습니다.

```text
python submission/inference.py --audio_dir /path/audio --label_dir /path/json --ckpt_path mission3_ver11_model.ckpt --output mission3.csv
```

CSV 헤더는 `label file name,symptom`이며 기존 string-list 형식을 유지합니다.
학습 로그가 저장된 노트북, 실행 코드, 가중치, requirements, 실제 결과 PPT를 제출 규격대로 준비합니다.
실제 F1 향상과 Colab GPU 실행 시간은 GPU 학습 전에는 검증되지 않습니다.
기존 약 0.6567은 사후 threshold 보정이 포함되어 새 규칙 기준과 동일한 baseline이 아닙니다.

## “co” 힌트와 자료

모델명은 확정할 수 없습니다. 한국어 “코”였다면 KoBERT도 후보입니다. ComplementNB나
contrastive learning을 멘토님이 말한 모델이라고 단정하지 않습니다.
KoBERT는 공식 전용 tokenizer wrapper가 필요해 단순 AutoTokenizer 비교군에서 제외했습니다.

- [KLUE 공식 모델](https://huggingface.co/klue/roberta-base)
- [KcELECTRA 작성자 모델 카드](https://huggingface.co/beomi/KcELECTRA-base)
- [SKT 공식 KoBERT](https://github.com/SKTBrain/KoBERT)
- [Supervised Contrastive Learning 논문](https://arxiv.org/abs/2004.11362)
- [ComplementNB 공식 설명](https://scikit-learn.org/1.6/modules/naive_bayes.html#complement-naive-bayes)

공식 Hugging Face API로 확인한 공개 버전을 고정했습니다.
KLUE-RoBERTa-base: `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`.
KcELECTRA v2022: `4cf9eb1dfeec0a672afe73059882fc998a299ceb`.

생성 노트북은 `build_notebooks.py`에서 검증된 실행 코드를 모두 포함해 만듭니다.
기존 프로젝트와 같은 dependency 범위를 사용하며 루트 requirements는 변경하지 않습니다.
