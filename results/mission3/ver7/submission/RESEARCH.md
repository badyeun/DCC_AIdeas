# Mission 3 ver7: 설계 근거와 검증 범위

검토일: 2026-09-15. ver7은 새로운 공개 한국어 encoder의 전체 미세조정이 주는 효과를 확인한다.
ver4의 학습된 encoder 위에 작은 분류기를 바꾸던 이전 실험에서 출발점을 바꾼다.
실제 전체 Validation F1·Colab 시간·CU는 아직 측정하지 않았다.

## 확인한 1차 출처

| 출처 | 확인 사항 | ver7 적용 |
|---|---|---|
| [KoELECTRA 저자 저장소](https://github.com/monologg/KoELECTRA) | Base discriminator 12층·hidden 768, v3의 추가 사전학습 자료에 구어·메신저 포함 | 한국어 대화 표현에 대한 실험 가설로 공개 v3 encoder 선택 |
| [공식 모델 카드](https://huggingface.co/monologg/koelectra-base-v3-discriminator) · [고정 커밋](https://huggingface.co/monologg/koelectra-base-v3-discriminator/commit/68b30cd259f34a4b5aa8786392612ba2a2617fcc) | `ElectraModel` encoder 사용법, Apache-2.0 표시, tokenizer | 해당 encoder·tokenizer만 다운로드하고 버전 고정 |
| Huang et al., [PLM-ICD, ClinicalNLP 2022](https://aclanthology.org/2022.clinicalnlp-1.2.pdf), §4.2–4.3 | 긴 문서 segment의 token 표현을 합치고 label별 attention으로 분류 | 긴 전사 window와 9개 증상별 attention, encoder와 분류기 공동 학습 |

KoELECTRA v3에 구어·메신저가 포함되었다는 사실이 Mission 3 향상을 뜻하지는 않는다.
새 encoder의 효과는 같은 원래 fold 1 fit/stop에서 기존 단일 모델과 비교하고,
Training calibration에서 후보 단독·기존 앙상블 혼합을 따로 비교한다.

## 구현 범위

```text
모든 발화 text → 동일한 [SEP]로 순서대로 연결 → overlapping windows
  → 공개 KoELECTRA encoder 전체 미세조정
  → window들의 token 표현 → 9-label attention → 9개 logits → sigmoid
```

512-token window와 64-token overlap을 사용한다. attention은 padding과 special token을 제외한 표현을 선택한다.
겹친 token은 서로 다른 window 문맥에서 계산된 표현으로 포함한다. BCE로 encoder와 분류기를 함께 학습한다.
PLM-ICD의 영어 임상 사전학습·대규모 ICD label 체계·의료 corpus와 원문 전체 실험을 재현하는 구현은 아니다.
증상별 attention 가중치를 원인 설명으로 해석하지 않는다.

학습 데이터는 제공된 서울 Training이다. 모든 전사 text를 동일한 규칙으로 처리하며 speaker 필드와
증상 정답에 따른 입력 변경은 없다. 별도 외부 dataset, 의료 사전, LLM 생성 데이터, 재라벨링을 추가하지 않는다.
공개 사전학습 encoder를 가져오는 것은 새로운 지도학습 데이터 추가와 구분한다.

## 비교와 선택

가중치 학습은 기존 fit 21,032개, epoch 비교는 기존 stop 5,288개,
최종 혼합·임계값 선택은 별도 Training calibration 2,880개를 사용한다.
stop의 threshold는 대화 그룹을 나눈 3-fold에서 평가 fold 바깥으로 맞춘다.
같은 stop의 원래 ver4 fold 1 단독 점수는 기존 앙상블 점수와 구분하여 기록한다.

calibration에서는 5-fold 교차평가로 기존 앙상블과 새 혼합을 비교한다.
최종 개선 폭 0.002 이상과 5개 중 3개 fold의 엄격한 개선을 함께 요구한다.
이 기준은 연구용 보수적 채택 조건이며 통계적 유의성 검정이나 일반화 보장은 아니다.
하나의 고정 fold에서 epoch를 고르므로 완전한 모델 OOF가 아니고 내부 선택 편향이 남는다.

공식 Validation 3,640개는 최종 선택 이후 평가한다. 최종 구성과 후보 단독의 두 CSV를 확정하고 점수를 기록한다.
후보 단독 임계값 역시 Training calibration에서 먼저 고정되며, Validation 진단 결과는 선택에 전달하지 않는다.
선택 로직·모델 코드·입력/실행 조건이 바뀌면 새 출력 폴더에서 별도 실험으로 다룬다.

## 재현성과 비용

공개 encoder는 commit SHA를 고정하고 준비 파일의 SHA256을 기록한다. 다운로드 직후
safetensors·config·tokenizer를 로컬로 다시 열어 본 뒤 원자적으로 완료 폴더를 게시한다.
이후 학습·추론은 네트워크를 차단한 별도 프로세스에서 수행한다.

새 encoder는 최대 3 epoch 전체 미세조정하므로 이전 분류기 중심 실험과 비용이 다르다.
원래 5개 모델은 재학습하지 않지만, 캐시가 없으면 기존 모델들의 추론 비용이 든다.
기존 fold 1 stop 비교와 후보 단독 Validation 진단의 추가 순전파 비용도 포함된다.
노트북은 준비·다운로드·학습·추론의 벽시계 시간을 기록하며 CU는 Colab에서 별도로 측정한다.
작은 CPU 테스트는 파일 무결성·학습 경로·재개·캐시·평가 분리를 확인한다. 대규모 GPU 결과를 대신하지 않는다.
