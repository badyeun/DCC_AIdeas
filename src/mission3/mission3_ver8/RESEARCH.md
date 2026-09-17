# Mission 3 ver8 설계 근거

구현일: 2026-09-17. 실제 대규모 GPU 결과는 아직 없습니다.

## 현재 실험으로 확인된 범위

ver5는 잔차 attention·상위 4층 추가 학습, ver6은 동결 encoder 위 A–F 분류기와 상위 2층 미세조정이었습니다.
ver7의 공개 KoELECTRA 전체 학습은 최적 epoch 2를 얻었으나 혼합 채택 기준을 통과하지 못했습니다.
이것만으로 Mission 3의 성능 상한이 0.65라고 결론 내릴 수 없습니다.

Training calibration에서 확신이 큰 오답을 읽었을 때 전사와 저장된 정답의 대응이 불명확한 사례가 있었습니다.
그 사례들은 극단값을 골랐으므로 라벨 오류율의 추정에 사용할 수 없습니다.
ver8은 무작위 FP/FN 검토 template를 제공하며, 자동 재라벨링·표본 삭제·검토 결과의 학습 반영은 하지 않습니다.

## 통제된 비교

3시간 목표에 맞춘 기본 실행은 B 단독, 최대 3 epoch입니다. 아래 A/B/C 비교는 명시적으로 켠 경우에 수행합니다.
B 자체의 학습 데이터·모델·epoch 수는 유지하며 A/C 후보 탐색을 생략합니다.
따라서 기본 실행만으로 A 대비 구조 변경 효과나 C의 TAPT 효과를 분리해서 확인할 수 없습니다.

A/B/C는 동일한 fit/stop, 입력, 토큰화, 지도학습 조건을 사용합니다.
A는 기존 window 분류 구조, B는 직접 token attention, C는 B에 fit-only TAPT를 추가합니다.
A/B는 같은 공개 encoder, B/C는 같은 분류기 초기화 seed를 사용합니다.
각 후보 전체 encoder를 학습합니다. 초기 공개 RoBERTa-large는 과제 학습이 끝난 ver4 checkpoint와 구분합니다.
동일한 stop에서 여러 epoch와 후보를 고르므로 선택 편향과 단일 seed의 한계는 남습니다.

## 1차 출처와 적용 범위

- [KLUE 공개 모델](https://huggingface.co/klue/roberta-large/tree/28d911204e9022eda172571ca8cc61eaffd942f7):
  공개 encoder·MLM head·tokenizer의 고정 revision. 다운로드 후 safetensors 재로딩과 파일 해시를 검사합니다.
- [PLM-ICD, ClinicalNLP 2022](https://aclanthology.org/2022.clinicalnlp-1.2/):
  분할된 문서의 token 표현과 label attention을 사용하는 설계 근거입니다.
  이 논문의 대규모 ICD 코드·영어 임상 사전학습·보고된 성능을 현재 9-label 한국어 과제에 그대로 옮기지 않습니다.
- [Don't Stop Pretraining, ACL 2020](https://aclanthology.org/2020.acl-main.740/):
  과제의 비지도 텍스트로 추가 사전학습하는 TAPT의 근거입니다.
  이번 적용은 서울 Training 내부 fit 전사만 사용하는 bounded MLM 실험입니다.
  논문의 성능 향상폭 또는 기존 실험의 결과가 Mission 3 향상을 보장하지 않습니다.

## 평가 경계

- TAPT: fit 전사만 사용. stop·calibration·Validation text도 사용하지 않음.
- 지도학습: 원본 fit의 9개 정답. 입력은 모든 발화의 text만 사용.
- epoch·A/B/C 선택: 동일한 stop 그룹 3-fold threshold 교차평가 F1; AP는 동점 보조.
- 보정: 선택된 후보 하나와 기존 ver4의 calibration 예측으로 raw/logit 혼합을 비교.
- 채택: 기존 대비 최소 +0.002 F1 및 최소 3/5 fold 개선. 미달이면 ver4 유지.
- 최종 평가: 모델과 임계값 확정 후 Validation에서 최종 구성과 선택 후보 단독 평가.

오답 검토 template는 원래 calibration 전체에 맞춘 baseline threshold를 사용한 기술 통계입니다.
새 모델 성능의 비편향 추정치 또는 라벨 오류 확정 자료로 쓰지 않습니다.
A/B/C 중 나머지 모델을 Validation 점수로 다시 골라 끼우는 기능은 없습니다.
새 5-fold 앙상블·다중 seed 재검증·외부 자료·상용 LLM API·발화자 메타데이터는 이 실행에 포함하지 않습니다.

## 저장·재현

분류 학습은 완료 epoch 단위로 모델·optimizer·scheduler·RNG를 저장합니다.
TAPT는 optimizer step 단위로 같은 상태와 마스킹 RNG를 저장합니다.
완료 marker는 가중치 저장 뒤에 기록하고, 공개 모델·각 후보·tokenizer 해시로 재사용 여부를 확인합니다.
최적 checkpoint에서 optimizer 상태를 제거하여 가중치가 불필요하게 커지지 않게 합니다.
완료된 단계의 재개 상태만 기본적으로 정리하고 선택용 가중치·학습 이력은 남깁니다.
