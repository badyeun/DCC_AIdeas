# Mission 3 멘토링 준비: ver1~ver9 모델 정리와 질문

작성 기준: 2026-09-19. [실험 로그](../src/mission3/experiment_log.md)와 저장된 실행 결과를 바탕으로 정리했습니다. 아래 질문의 원인 가설은 아직 검증된 결론이 아닙니다.

## 1. 멘토님께 먼저 설명할 내용

> 응급 통화의 양측 전사 텍스트에서 9개 증상을 동시에 예측하는 다중 레이블 분류를 하고 있습니다. 서울 Training 29,200개로 학습하고 Validation 3,640개에서 Macro F1을 평가했습니다.
>
> 문자 TF-IDF와 Logistic Regression으로 약 0.6001을 얻었고, RoBERTa-base로 0.6260, RoBERTa-large 5-fold와 TF-IDF 모델을 결합해 약 0.6566까지 올렸습니다.
>
> 이후 증상별 attention, 손실 함수, 부분·전체 미세조정, KoELECTRA, DeBERTa와 NB-LR까지 비교했습니다. 하지만 최종 성능은 약 0.6567에서 정체돼 있습니다. ver6~ver9는 새 후보가 채택 기준을 통과하지 못해 기존 ver4 구성을 유지했습니다.
>
> 다음 실험에 앞서 모델 구조, 데이터·라벨, 학습 설정, 검증 방식 중 무엇을 우선 점검해야 할지 조언을 받고 싶습니다.

대상 증상은 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란입니다. 한 통화에 여러 증상이 있을 수 있으며, 평가 점수는 9개 증상 각각의 F1을 평균한 값입니다. 모델 입력으로 사용한 것은 전사 text이며 음성 WAV는 사용하지 않았습니다.

## 2. 버전별로 사용한 모델과 결과

모든 점수는 공식 Validation Macro F1입니다. Training 내부 stop·calibration 점수와 구분합니다.

| 버전 | 사용한 모델 / 구조 | 시도한 내용 | 최종 F1 | 해석 |
| --- | --- | --- | ---: | --- |
| ver1 | 문자 TF-IDF + 증상별 Logistic Regression 9개 | 문자 2~5-gram 특징으로 기본 모델 구현 | 별도 실행 기록 없음 | 모듈형 baseline 구현 |
| ver2 | ver1과 같은 TF-IDF + Logistic Regression | Colab 단일 노트북으로 학습·추론·평가 통합 | 0.6001 | 최초 보관 baseline; 결과 폴더 이름은 `ver1` |
| ver3 | `klue/roberta-base` + 첫 토큰 표현 + 선형 9-label head | 공개 언어모델 미세조정; 최대 384토큰 입력 | 0.6260 | 내부 최적 epoch 3을 정한 뒤 전체 Training으로 3 epoch 재학습 |
| ver4 | `klue/roberta-large` 5-fold + 문자·단어 TF-IDF/LR | 긴 대화를 512토큰 window로 처리; 5개 모델 평균과 희소 모델을 증상별로 혼합 | 0.656649 | 후속 버전의 비교 기준이 된 구성 |
| ver5 | ver4 fold 1 + 증상별 token attention 추가 출력 | head 학습 1 epoch, 상위 4층과 head 추가 학습 1 epoch; 확률·logit 혼합 비교 | 0.656718 | 추가 학습 전 epoch 0이 최적; logit 혼합 채택, 학습 효과는 확인되지 않음 |
| ver6 | ver4 RoBERTa-large의 동결 특징 + A~F 분류기/손실 | head 비교 후 D를 선택하고 encoder 상위 2층 추가 미세조정 | 0.656670 | D head는 학습됐지만 추가 encoder 미세조정은 미채택; 최종은 ver4 유지 |
| ver7 | `monologg/koelectra-base-v3-discriminator` + 증상별 token attention | 새 encoder와 head 전체를 3 epoch 학습; epoch 2 선택 | 0.656670 | 새 후보 단독 0.638110; 최종은 ver4 유지 |
| ver8 | `klue/roberta-large` + 증상별 직접 token attention | 공개 모델에서 전체 encoder와 head를 3 epoch 학습; epoch 3 선택 기록 | 0.656670 | 새 후보 단독 0.645292; 최종은 ver4 유지 |
| ver9 | `team-lucid/deberta-v3-base-korean` 2-fold + NB-log-count-ratio Logistic Regression | 전체 encoder·head를 fold당 3 epoch 학습; DeBERTa 평균 75% + NB-LR 25% | 0.656766 | 새 후보 단독 0.603570; 최종은 ver4 유지 |

### 모델 이름을 쉬운 말로 설명하면

- **TF-IDF + Logistic Regression:** 어떤 문자 조각·단어가 등장하는지를 수치화하고, 그 특징으로 증상별 유무를 판단하는 모델입니다.
- **RoBERTa / KoELECTRA / DeBERTa:** 전사 텍스트를 문맥이 반영된 벡터로 바꾸는 사전학습 언어모델입니다. 이번 과제에서는 여기에 증상 분류기를 붙여 학습했습니다.
- **Encoder와 head:** encoder는 텍스트의 특징을 만들고, head는 그 특징으로 9개 증상의 점수를 계산합니다. head만 학습한 실험과 encoder까지 학습한 실험을 구분해야 합니다.
- **증상별 attention:** 두통·복통 등 각 증상의 점수를 만들 때 token이나 window에 서로 다른 가중치를 주는 구조입니다.
- **Fold 앙상블:** Training 내부 분할이 다른 여러 모델의 예측을 합치는 방식입니다. ver4는 5개 모델, ver9의 새 neural 후보는 2개 모델을 사용했습니다.
- **NB-LR:** 증상 양성·음성 문서에서 특징이 나타나는 정도의 로그 비율로 TF-IDF를 다시 가중한 뒤 Logistic Regression을 학습합니다. ver9에서 사용한 분류기는 SVM이 아닙니다.
- **임계값과 보정:** 증상 점수가 얼마 이상일 때 양성으로 정할지, 여러 모델을 어떤 비율로 합칠지 Training 내부의 별도 데이터에서 정하는 단계입니다.

### ver6에서 비교한 A~F

아래는 서로 다른 encoder 6개가 아니라, 같은 동결 encoder 특징 위에서 비교한 분류기·손실 설정입니다.

| 후보 | 실제 비교한 설정 |
| --- | --- |
| A | 기존 head + ASL (Asymmetric Loss) |
| B | 기존 head + DB Loss (Distribution-Balanced Loss) |
| C | 기존 head + ASL + 0.2 × soft Macro F1 보조항 |
| D | 직접 label-token attention (LAAT) + weighted BCE |
| E | D + head 수준 R-Drop |
| F | E + fit 데이터에서 계산한 증상 관계 |

D의 head epoch 2가 선택됐습니다. 이후 encoder 상위 2층 미세조정은 성능이 개선되지 않아 그 단계의 epoch 0을 유지했습니다. 따라서 “ver6은 아무것도 학습하지 않았다”는 설명은 틀립니다. 최종 앙상블에 D를 넣을지는 별도 calibration 비교에서 결정했고, 이 단계에서는 D가 제외됐습니다.

## 3. 멘토링에서 정확히 구분해서 말할 결과

1. **확인된 큰 개선은 ver2 → ver3 → ver4입니다.** TF-IDF 약 0.6001 → RoBERTa-base 약 0.6260 → RoBERTa-large 앙상블 약 0.6566입니다. ver4는 모델 크기·긴 대화 처리·앙상블·보정 방식을 함께 바꿨으므로 각각의 기여는 분리되지 않았습니다.
2. **ver6~ver9의 최종 점수는 새 모델 단독 성능이 아닙니다.** 새 후보가 내부 채택 기준을 통과하지 못해 기존 ver4를 쓴 결과입니다. KoELECTRA 0.638110, ver8 RoBERTa 후보 0.645292, ver9 DeBERTa+NB-LR 후보 0.603570을 따로 설명해야 합니다.
3. **ver9의 0.656766을 DeBERTa 개선이라고 말하면 안 됩니다.** 최종 새 후보 가중치는 0입니다. ver8과 비교해 전신쇠약 오탐 2건이 줄었지만 같은 기존 구성을 재추론한 결과의 미세 차이입니다.
4. **실패 원인은 아직 특정되지 않았습니다.** 라벨 오류, 과적합, 학습 부족, 손실 함수, 모델 크기 중 어느 하나가 원인이라고 확정할 근거는 없습니다.
5. **ver8은 결과와 가중치 보관 문제를 구분합니다.** 2026-09-17 로그에는 후보 CSV 점수가 검증됐지만 로컬 후보 가중치가 완료 기록의 epoch·해시와 다르다고 기록돼 있습니다. 이번 멘토링 정리에서는 해당 가중치를 다시 검증하지 않았습니다.

## 4. 우선 질문: 시간이 짧으면 1~3번부터

### 질문 1. 성능 정체의 원인을 어떤 순서로 확인할까요?

> ver4에서 Macro F1 약 0.6566까지 올린 뒤 head, loss, encoder를 바꿔도 개선되지 않았습니다. 모델을 더 바꾸기 전에 데이터·라벨, 입력 표현, 학습 설정, 검증 절차 중 무엇부터 점검하면 좋을까요? 다음 실험 하나만 한다면 어떤 가설을 확인하는 실험이 좋을까요?

근거: ver5~ver9에서 새 학습으로 인한 뚜렷한 최종 개선이 확인되지 않았습니다. 이 질문으로 다음 실험의 우선순위를 정합니다.

### 질문 2. 낮은 점수의 증상을 어떻게 오류 분석하면 좋을까요?

> ver9 최종 결과에서 열상 F1은 0.8936인데 오심은 0.4031, 두통은 0.5497, 전신쇠약은 0.5933입니다. 특히 오심은 precision 0.3205, recall 0.5431로 오탐이 많습니다. 부정 표현, 질문과 실제 증상 진술의 혼동, 과거·현재 증상, 전사와 라벨의 불일치 등을 어떤 기준으로 나눠 검토하면 좋을까요?

후속 질문: “Training 내부 오류 사례를 증상별로 얼마나, 어떤 방식으로 뽑아 검토하면 충분할까요? 라벨 수정이 필요한 것으로 보이는 사례는 대회 규정 안에서 어떻게 처리해야 할까요?”

주의: 위 오류 유형은 분석할 후보이며 실제 오류 원인으로 확인된 분류가 아닙니다.

### 질문 3. 새 후보의 채택 기준과 검증 방식이 적절한가요?

> Training에서 calibration 2,880개를 따로 떼고, 5-fold 교차 보정에서 기존보다 Macro F1이 0.002 이상 오르면서 최소 3개 fold에서 개선될 때 새 후보를 채택했습니다. ver6은 전체 점수가 0.659671에서 0.662117로 올랐지만 개선 fold가 2/5여서 제외했습니다. 이런 상황에서 개선의 신뢰도를 어떻게 판단하고 채택 기준을 정하는 게 좋을까요?

후속 질문: “여러 버전에서 같은 calibration을 반복 사용하고 공식 Validation 결과를 보며 다음 실험을 설계했습니다. 이 반복 비교에서 생기는 선택 편향을 줄이려면 검증 절차를 어떻게 구성해야 할까요?”

주의: 공식 Validation을 가중치·임계값 학습에 사용했다는 뜻은 아닙니다. 반복적인 연구·설계 선택의 영향을 묻는 질문입니다. ver7~ver9는 최선 혼합 점수 자체도 기존보다 낮아서, 채택 기준만 낮춘다고 개선이 입증되는 상황은 아닙니다.

### 질문 4. 모델 자체의 효과를 비교하려면 무엇을 고정해야 하나요?

> 기존 ver4는 RoBERTa-large 5-fold와 TF-IDF의 혼합인데, ver7·ver8은 새 neural 모델 1개, ver9는 DeBERTa 2개와 NB-LR의 혼합입니다. 모델 계열의 효과와 앙상블 수·손실·head의 효과를 분리하려면 어떤 조건을 고정하고 어떤 순서로 비교하면 좋을까요?

후속 질문: “ver9의 후보 단독 0.603570은 DeBERTa와 NB-LR을 75:25로 섞은 점수입니다. 각각 단독과 혼합을 분리해서 보는 것이 다음 원인 분석에 도움이 될까요?”

근거: 같은 stop에서 ver8 후보 F1 0.637975는 기존 ver4 fold 1의 0.639684보다도 조금 낮았습니다. 앙상블 개수 차이만으로 모든 결과를 설명할 수는 없습니다.

### 질문 5. 손실 함수·epoch·임계값은 어떻게 함께 설계해야 하나요?

> weighted BCE, ASL, DB Loss, soft Macro F1 보조항, unweighted BCE를 서로 다른 실험에서 사용했습니다. ver3은 loss가 계속 내려가는데 holdout F1은 epoch 3 이후 낮아졌고, ver9는 두 fold 모두 마지막 epoch 3까지 stop F1이 올랐습니다. 학습 부족과 과적합을 구분하려면 어떤 기록을 더 봐야 하며, Macro F1 과제에서 loss와 epoch 선택 기준, 증상별 임계값을 어떻게 조합하면 좋을까요?

주의: ver6의 ASL·DB Loss·R-Drop 등은 주로 동결 특징 위 head 단계 비교입니다. 이 결과를 전체 encoder 학습에 그대로 일반화하지 않습니다. ver4는 stop Macro AP, ver6 이후는 주로 stop의 threshold 교차평가 Macro F1을 사용해 선택 지표도 다릅니다.

### 질문 6. 시간 예산 안에서 어떤 실험에 시간을 써야 할까요?

> ver4는 기록된 학습 epoch 합계만 약 10시간 30분이고, ver9는 L4에서 설치부터 두 fold 학습과 평가까지 약 117.82분이 걸렸습니다. 후속 실행에 3시간 정도를 쓸 수 있다면 epoch 확대, 반복 seed, 모델 수 확대, 오류 분석과 통제 비교 중 어느 쪽에 우선 투자하면 좋을까요?

주의: 두 시간은 포함 범위와 환경 기록이 달라 직접적인 속도 비교에 사용할 수 없습니다. 질문의 목적은 남은 실험 예산 배분입니다.

## 5. 가져갈 자료

- 이 문서의 버전별 모델·성능 표와 [전체 실험 로그](../src/mission3/experiment_log.md).
- [ver9 최종 증상별 지표](../results/mission3/ver9/validation_metrics.json)와 [새 후보 단독 지표](../results/mission3/ver9/candidate_validation_metrics.json).
- [ver9 epoch 이력](../results/mission3/ver9/candidate_history.json)과 [보정 비교](../results/mission3/ver9/calibration_comparison.json).
- ver6의 전체 점수는 올랐지만 2/5 fold만 개선된 기록: [실험 로그의 EXP-006 절](../src/mission3/experiment_log.md).
- 가능하면 별도 준비: Training 내부에서 뽑은 오심·두통·전신쇠약의 오탐·미탐 사례와, 정답 및 예측을 나란히 놓은 표. 이 문서를 만들면서 개별 통화 내용을 새로 추출하지는 않았습니다.

멘토링을 마칠 때는 **다음에 검증할 가설 1개, 바꿀 요소 1개, 비교 기준, 채택 조건**을 구체적으로 정리해 오면 후속 실험에 바로 연결할 수 있습니다.
