# Mission 3 ver5: 논문 근거와 설계 판단

ver5는 ver4의 학습 결과를 재사용하고, 증상별 토큰 attention을 추가한 후보 하나를 검증하는 설계다.
성능 향상은 아직 실측하지 않았다. Training 내부 비교에서 개선이 확인되지 않으면 ver4를 유지한다.
판단 기준은 `docs/0820_DCC 대학부_예선_출제문제.pdf`의 Mission 3 조건이다.
공식 Validation은 최종 평가에만 사용하며, 서울 Training의 전사 text만 학습·보정에 사용한다.

## 1. 제공 파일 확인

PDF는 3개지만 서로 다른 논문은 2편이다. 아래 쪽수는 각 PDF의 첫 장을 1쪽으로 센다.

| 로컬 파일 | 논문 / 확인 결과 |
|---|---|
| `1-s2.0-S2405844022009719-main.pdf` | Heliyon 2022 논문, 12쪽 |
| `Automatic-symptoms-identification-from-a-massive-volume-of-unstruct_2022_Hel.pdf` | 위 논문의 중복 사본: 12쪽 본문과 이미지 41개가 같음 |
| `2403.06734v1.pdf` | CognitiveEMS 논문 v1, 13쪽, 2024-03-11 |

Heliyon 사본들의 파일 크기와 SHA-256은 다르므로 파일 자체가 바이트 단위로 같다는 뜻은 아니다.
내용이 같은 두 사본을 서로 독립적인 성능 향상 근거로 계산하지 않는다.
원문: [Heliyon DOI](https://doi.org/10.1016/j.heliyon.2022.e09683), [CognitiveEMS arXiv v1](https://arxiv.org/abs/2403.06734v1).

## 2. Heliyon: 대규모 상담에서 증상 식별

제목은 *Automatic symptoms identification from a massive volume of unstructured medical consultations using deep neural and BERT models*이다.
아랍어 원격의료 상담에서 ICD-10 기반 증상을 추천하는 연구다. 한국어 119 신고 분류 실험이 아니다.

- 4·6쪽: AraBERT를 특징 추출기로 사용하며 BERT의 모든 층을 동결한다.
- 추출한 특징에 BiLSTM과 sigmoid 분류기를 연결한다. 분류 대상은 2,348개 증상이다.
- 임베딩 단계의 상담은 3,310,996개이며, 분류 실험의 최종 자료는 501,004개다.
- 희소 label 제거, label별 최대 5,000개 제한, 중복 제거 등 데이터 구성이 현재 과제와 다르다.
- 6쪽의 전체 token embedding 설명과 7쪽의 CLS만 사용하는 설명 사이에는 모호함이 있다.
- 따라서 논문이 특정 token pooling 구현을 일관되게 입증했다고 해석하지 않는다.
- 7–8쪽: 80/10/10 분할, Adam, batch 128을 사용한다. Table 2는 AraBERT 계열 10 epoch, BiLSTM 비교 모델 50 epoch를 제시한다.
- 9–10쪽: top-5 증상 추천을 평가하며, 전체 hit rate는 AraBERTv2 0.564, v1 0.537, Word2Vec 0.535다.
- 전문가가 검토한 500개에서 recall 0.706, precision 0.233, 최소 한 개 정답을 찾은 비율 0.711을 보고한다.
- 이 수치들은 9개 고정 증상의 Macro F1과 정의·label 수·데이터가 모두 달라 직접 비교할 수 없다.

2쪽은 의사 부착 label의 불완전성을 논의한다. 그러나 자동 정답 수정이나 특정 noise-robust loss를 검증한 논문은 아니다.
이 언급을 근거로 제공 정답을 임의 변경하거나, 모델의 오탐을 모두 annotation 오류로 처리하지 않는다.

**적용:** 이미 학습된 표현을 보존한 채 새 분류 부분을 먼저 적응시키는 동결 학습의 취지를 사용한다.
ver5의 encoder 동결 1 epoch는 새 attention head 준비 단계다. 이후 전체 encoder를 미세조정한다.
이는 논문의 영구 동결 AraBERT+BiLSTM을 재현한 것이 아니라 현재 모델에 맞춘 단계적 학습이다.
BiLSTM, 아랍어 모델, top-5 강제 추천, 희소 증상 삭제는 도입하지 않는다.

## 3. CognitiveEMS: 실시간 응급의료 보조 시스템

제목은 *Real-Time Multimodal Cognitive Assistant for Emergency Medical Services*다.
전체 시스템은 영어 음성 인식, EMS 프로토콜 선택, 영상의 처치 행동 인식으로 구성된다.
현재 과제에 직접 참고할 부분은 6–7쪽 §III.C의 프로토콜 다중 label 분류기다.

- TinyClinicalBERT의 마지막 token hidden state로 전사 표현을 만든다.
- 외부 ODEMSA 지침에서 프로토콜·증상·약물·처치를 추출해 이종 지식 그래프를 구성한다.
- GatorTron으로 초기 node embedding을 만들고, 1층 HGT로 관계를 반영한다.
- HLA는 각 프로토콜이 전사의 각 token에 다른 가중치를 주어 관련 표현을 모은다.
- sigmoid와 BCE를 사용한다. 지식 융합은 기본 15M 모델에 약 0.9M parameter를 추가한다.
- Group-wise 학습은 성인/소아의 유사 프로토콜을 합친 후 추론 시 나이 규칙으로 복원한다.
- 7–9쪽: 영어 ePCR 4,417개, 프로토콜 43개를 사용하며 70:30 분할 및 3-fold 평가 평균을 설명한다.
- 9쪽: HGT hidden 256, dropout 0.3, weight decay 1e-5와 조기 종료를 사용한다.
- 학습 batch 4–32, learning rate 1e-6–1e-3 탐색을 수행했으나 총 학습 시간은 제시하지 않는다.

10쪽 Table III는 각 추가 요소를 제거했을 때의 결과를 비교한다.

| 모델 구성 | Micro F1 | Macro F1 | Acc@3 |
|---|---:|---:|---:|
| TinyClinicalBERT | 0.673 | 0.171 | 0.807 |
| + group-wise 학습 | 0.709 | 0.277 | 0.836 |
| + 지식 융합 | 0.743 | 0.395 | 0.898 |
| 두 요소 모두 | 0.756 | 0.499 | 0.904 |

지식 융합의 이점은 이 연구에서 관찰됐지만, 한국어 증상 분류의 향상폭을 예측하는 값은 아니다.
11쪽의 서버 0.01초·edge 0.14초는 프로토콜 **추론** 지연이다. A100 학습 시간으로 해석하지 않는다.
8쪽은 일부 자체 수집 multimodal 시나리오의 전사 정보 부족으로 프로토콜 예측이 잘되지 않았다고 명시한다.
5–6쪽의 GPT-4·ElevenLabs 합성 자료는 ASR용이며, 전사가 제공되는 본 과제에 추가하지 않는다.
공식 [EMS-Pipeline](https://github.com/UVA-DSA/EMS-Pipeline)의 분류 구현 링크는 후속 [DKEC](https://github.com/UVA-DSA/DKEC)로 연결된다.
현재 공개 저장소 전체를 v1 PDF와 동일한 구현이라고 가정하지 않는다.

**적용:** 한국어 encoder를 유지하면서 각 증상이 token 증거를 선택하게 하는 attention을 추가한다.
외부 지식 그래프, 영어 encoder, 처치/나이 규칙, 음성·영상 처리는 가져오지 않는다.
구토와 오심처럼 관련된 증상을 합치는 것은 논문의 성인/소아 프로토콜 통합과 다르므로 수행하지 않는다.

## 4. ver4에서 확인한 개선 지점

ver4는 token을 CLS+평균으로 압축한 뒤 window 사이에서 증상별 attention을 계산한다.
한 window짜리 통화에서는 attention 가중치가 항상 1이므로, 그 attention이 특정 token을 고르지 못한다.
ver5는 기존 전역·window 표현에 masked token-wise label attention을 잔차 형태로 더한다.
새 잔차 출력은 0으로 초기화하여 평가 모드의 최초 예측이 저장된 ver4 fold와 같게 시작한다.
padding과 special token을 attention 증거에서 제외하고, 긴 통화의 모든 window는 계속 처리한다.
원래 BCE, 9개 증상 정의, 전사 전처리, 빈 증상 결과 허용은 유지한다.

이 설계는 논문에서 영감을 받은 변경이다. 외부 지식 융합까지 포함한 CognitiveEMS 재현은 아니다.
새 head의 attention 연산은 encoder를 하나 더 실행하는 것보다 작지만 실제 시간 증가는 측정해야 한다.
별도 BiLSTM, ASL, label 자동 재작성, 여러 backbone·learning rate의 광범위 탐색은 이번 범위에 넣지 않는다.

## 5. 학습·선택·평가 절차

1. ver4의 실제 모델 5개, tokenizer, TF-IDF, 분할 목록, 원시 calibration 예측 확률을 읽는다.
2. 기존 fold 1의 최적 가중치에서 후보 하나를 시작하며 동일한 fit/stopping 자료를 사용한다.
3. 최대 1 epoch는 encoder를 동결해 새 head를 준비하고, 최대 3 epoch는 전체를 미세조정한다.
4. stopping 평가로 후보 가중치를 고른다. calibration 정답은 encoder 학습과 epoch 선택에 쓰지 않는다.
5. 후보·ver4 평균·TF-IDF의 원시 확률로 Training calibration 내부의 혼합·임계값을 비교한다.
6. 각 바깥 calibration fold의 평가 샘플은 안쪽의 혼합 선택과 임계값 학습에서 제외한다.
7. 교차 평가에서 개선이 없으면 원래 ver4 구성으로 돌아가고, 선택 결과와 비교 점수를 함께 기록한다.
8. 배포용 보정은 Training calibration 전체로 결정한 뒤 공식 Validation을 마지막에 한 번 평가한다.

이미 본 공식 Validation 결과를 개선 방향의 근거로 참고하더라도, 그 정답으로 수치 설정을 최적화하지 않는다.
배포 설정을 정한 자료에 다시 계산한 fitted F1은 교차 평가 F1보다 낙관적일 수 있어 별도로 표기한다.
비교 후 선택 자체에도 불확실성이 있으므로 Training 교차 평가를 숨겨진 Test 성능이라고 설명하지 않는다.
다섯 fold를 새로 학습하지 않는다. 기존 ensemble은 기준 모델이며 후보가 도움이 되는지 확인하는 대상이다.
추론에 필요한 원래 fold·후보·sparse·tokenizer·manifest·실행 코드를 함께 보관해야 한다.

## 6. 시간 예산과 결과 해석

ver4 저장 기록의 22 epoch 합은 630.28분, epoch 평균은 28.65분이었다.
이 합에는 epoch별 학습과 stopping 평가가 포함되고, 저장·다운로드·최종 추론 시간은 빠져 있다.
ver5는 A100에서 최대 동결 1+미세조정 3 epoch와 보정·추론을 포함해 **대략 2–4시간을 목표**로 한다.
이는 아직 실행하지 않은 추정이며 배치 처리, 데이터 위치, GPU 메모리, 저장 속도에 따라 달라진다.
노트북은 실제 처리 속도에 따른 ETA를 출력한다. 추정치를 실측 완료 시간으로 보고하지 않는다.
최종 보고에는 선택된 구성, 후보와 기준의 교차 평가, 공식 Validation 결과, 실제 시간을 구분한다.
후보가 채택되지 않으면 “ver4 유지”라고 기록하며, 버전 이름만 바뀐 것을 성능 개선으로 표현하지 않는다.
