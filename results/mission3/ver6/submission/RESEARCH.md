# Mission 3 ver6: 논문 근거와 적용 범위

검토일: 2026-09-15. 사용자가 제공한 `## 지금 상태에서 어디를 건드려야 하는지부터.txt`를 출발점으로 삼고, 아래 원문 PDF의 방법 절과 저자 코드를 확인했다. 논문의 다른 데이터셋 성능을 Mission 3의 예상 상승폭으로 환산하지 않는다. **ver6의 실제 F1과 총 Colab 컴퓨팅 단위는 실행 결과로 확인해야 한다.**

### 대회 원문과 대조한 조건

[예선 출제문제](../../../docs/0820_DCC%20대학부_예선_출제문제.pdf) 5쪽, 9–11쪽을 직접 확인했다. 서울 Training만 학습에 쓰고, 입력은 전사 text만 사용한다. 정답에 따라 입력 전처리를 달리하거나 공식 Validation을 학습에 사용하지 않는다. 지정된 9개 외의 symptom만 제외하며 **샘플 자체는 유지**하므로 9개 target이 모두 0인 통화도 포함한다. 최종 CSV는 label 파일명과 0–9개 symptom 문자열을 출력한다. 소스의 `dialogue_text`와 추론의 `include_targets=False` 경로가 이 분리를 유지한다.

## 1. 논문 원문

PDF 6개를 로컬 저장소의 `mission3_ver6/papers/`에 보관했다. 페이지는 PDF 뷰어의 1부터 시작하는 페이지 번호다. 논문은 개발 근거 자료이며 노트북 실행 입력이나 학습 데이터가 아니다. Colab에는 기존 입력 ZIP과 새 노트북만 있으면 된다. 내보낸 실행 코드에는 PDF를 포함하지 않으며, 그 환경에서는 표의 원문 URL로 확인할 수 있다.

| 원문 | 실제 확인한 범위 | 보관 파일 |
|---|---|---|
| Huang et al., **PLM-ICD**, ClinicalNLP 2022 | PDF 4–6쪽 §3–5, 8쪽 §6: segment pooling, label attention, 한계·ablation | [원문](https://aclanthology.org/2022.clinicalnlp-1.2.pdf) · [로컬](papers/plm_icd_2022.pdf) |
| Vu et al., **LAAT**, IJCAI 2020 | PDF 2–4쪽 §3–4: attention 식 (4)–(6), label classifier, JointLAAT와 학습 설정 | [원문](https://arxiv.org/pdf/2007.06351) · [로컬](papers/laat_2020.pdf) |
| Ben-Baruch, Ridnik et al., **ASL**, ICCV 2021 / arXiv v4 | PDF 2–4쪽 §2: 식 (2)–(7), focusing·clipping·gradient 분석 | [원문](https://arxiv.org/pdf/2009.14119) · [로컬](papers/asl_2021.pdf) |
| Wu et al., **Distribution-Balanced Loss**, ECCV 2020 | PDF 4–7쪽 §3, 저자 `ResampleLoss`와 COCO 설정: rebalancing, bias, negative scaling | [원문](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf) · [로컬](papers/db_loss_2020.pdf) |
| Bénédict et al., **sigmoidF1**, arXiv v3 (2022; 최초 공개 2021) | PDF 7–9쪽 §4.3–5, 25쪽 hyperparameters: soft confusion counts, batch 단위 Macro-F1 | [원문](https://arxiv.org/pdf/2108.10566) · [로컬](papers/sigmoidf1.pdf) |
| Liang et al., **R-Drop**, NeurIPS 2021 | PDF 2–4쪽 §2: 독립 dropout 출력 두 개, symmetric KL, 학습 비용 | [원문](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf) · [로컬](papers/r_drop_2021.pdf) |

### TXT의 2026 IEEE 논문 확인

*Mitigating Multi-label Imbalance in ICD Coding: A Comprehensive Evaluation of Loss Functions and Threshold Optimization*의 제목·저자·IEEE Access 게재 예정 표기를 [저자 연구실의 공식 목록](https://mix.seoultech.ac.kr/publications)에서 확인했다. 목록은 [IEEE 문서 11589268](https://ieeexplore.ieee.org/document/11589268)을 연결한다. IEEE 원문 본문은 이번 환경에서 가져오지 못했으므로, **해당 논문의 수치나 비교 우열을 ver6 설계 근거로 사용하지 않았다.** 위 여섯 원문으로 구현에 필요한 방법을 확인했다.

## 2. 직접 분류하는 label attention

PLM-ICD는 문서의 각 segment를 encoder에 통과시킨 뒤 **token hidden state를 연결**하고, LAAT의 증상별 attention을 적용한다. 아래 식에서 `H`의 열은 token, `A`의 행은 label이다.

```text
H = concat(encoder(window_1), ..., encoder(window_W))
A = softmax_token(U tanh(V H))
R = H A^T
z_k = w_k^T R_k + b_k
p_k = sigmoid(z_k)
```

ver6의 D/E/F 후보는 이 표현으로 **최종 logit을 직접 계산**한다. ver5의 작은 residual 분기와 구조가 다르다. 각 증상 query가 같은 전사 전체를 보며, 정답 label에 따라 입력 token을 바꾸지 않는다. [PLM-ICD §4.2–4.3](https://aclanthology.org/2022.clinicalnlp-1.2.pdf) · [LAAT §3.1](https://arxiv.org/pdf/2007.06351)

Mission 3 적용 차이는 기존 한국어 encoder와 512-token window/64-token overlap 재사용, padding·special token을 attention 후보에서 제외하는 것이다. overlap에 해당하는 token 표현은 서로 다른 문맥의 표현으로 포함된다. 원문의 영어 임상 사전학습, ICD 계층, 추가 의료 corpus는 도입하지 않는다. Attention 가중치 자체를 원인 설명이나 의학적 근거로 간주하지 않는다.

## 3. 비용을 제한한 여섯 가지 비교

기존 ver4의 fold 1 encoder를 고정하고 입력별 token 표현을 한 번 추출한다. A–F는 **같은 캐시와 같은 fit/stop 분할**을 쓰는 작은 분류기 실험이다. 모델 여섯 개의 encoder를 각각 미세조정하는 실험이 아니다.

| 후보 | 분류기와 손실 | 확인하려는 변화 |
|---|---|---|
| A | 기존 window 분류기 + ASL | 구조를 유지한 손실 변경 |
| B | 기존 window 분류기 + DB-Loss | 빈도·공발생을 반영한 손실 변경 |
| C | 기존 window 분류기 + ASL + `0.2 × soft Macro-F1` | 공식 지표에 가까운 보조 학습 목표 |
| D | 직접 label-token attention + weighted BCE | 증상별 token 표현 자체의 효과 |
| E | D + 분류기 단계의 Bernoulli R-Drop | 작은 추가 비용의 출력 일관성 |
| F | E + Training fit으로 추정한 9-label dependency | 공발생 정보의 추가 효과 |

분류기 비교는 후보당 최대 4 epoch, 실제 batch 32, 개선 없는 2 epoch 후 중단이 기본이다. 선택된 분류기 하나만 encoder 상위 2개 층을 최대 1 epoch 미세조정한다. 이때 기본값은 R-Drop과 soft-F1 항을 끈다. 전자는 encoder를 두 번 계산하는 비용, 후자는 작은 GPU microbatch의 F1 통계 불안정을 피하기 위한 절충이다. 캐시 표현을 만든 encoder와 미세조정된 encoder는 다르므로 **미세조정 모델의 추론에 이전 고정 표현을 사용하면 안 된다.** 실제 추론과 미세조정 전후 비교는 선택된 encoder를 직접 실행한다.

### ASL: 음성·양성의 gradient 기여를 다르게 조절

```text
p = sigmoid(z)
p_m = max(p - margin, 0)
loss = -y (1-p)^gamma_pos log(p)
       -(1-y) p_m^gamma_neg log(1-p_m)
```

기본값은 `gamma_pos=0, gamma_neg=2, margin=0`. ASL 후보에는 기존 BCE의 `pos_weight`를 중복 적용하지 않는다. 원문의 clipping은 매우 쉬운 음성뿐 아니라 일부 매우 어려운 음성의 gradient도 줄인다. 따라서 TXT의 “ASL이면 false positive가 줄어든다”는 설명은 검증할 가설이다. 현재 기본값은 clipping을 끄고, 증상별 precision/recall/F1으로 실제 영향을 확인한다. [ASL §2](https://arxiv.org/pdf/2009.14119)

### DB-Loss: 저자 구현의 부호·정규화 기준

fit에서 얻은 클래스 빈도 `n_k`, fit 크기 `N`을 사용한다. 재가중치는 `r_k = (1/n_k) / sum_j(y_j/n_j)`에 `0.1 + sigmoid(10(r_k-0.2))`를 적용한다. 정답이 모두 0인 행은 분모 0을 나누지 않고, 이 식의 극한인 `1.1`을 사용한다.

저자의 COCO 설정을 출발점으로 `negative_scale=2`, `init_bias=0.05`, focal `gamma=2`, multiplier `2`를 쓴다. bias는 `0.05/2 × log(p_k/(1-p_k))`를 logit에 더한다. 음성 logit에는 2를 곱하고 음성 loss weight에는 1/2를 곱한 뒤 BCE와 focal 항을 계산한다. 빈도 0/1은 수치적으로 보호한다. 부호와 bias의 scale 나눗셈은 [저자 구현](https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/models/losses/resample_loss.py)과 [COCO 설정](https://github.com/wutong16/DistributionBalancedLoss/blob/master/configs/coco/LT_resnet50_pfc_DB.py)을 기준으로 확인했다.

원문의 class-aware 재표집까지 재현하지 않고, 같은 입력 순서를 유지하는 **손실만 적용한 비교**다. label에 따른 loss weight는 정답을 쓰는 지도학습 목적함수이며, label에 따른 입력 전처리와 구분한다. [DB-Loss §3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490154.pdf)

### soft Macro-F1: sigmoidF1에서 단순화한 보조항

실제 minibatch에 대해 `s=sigmoid(z)`, `TP_k=sum(s_k*y_k)`, `FP_k=sum(s_k*(1-y_k))`, `FN_k=sum((1-s_k)*y_k)`를 계산하고 `1-mean_k(2TP_k/(2TP_k+FP_k+FN_k+eps))`를 사용한다. 원문의 sigmoid slope/offset은 `1/0`으로 고정한 형태다. 해당 batch에 양성이 없는 label은 이 보조항의 평균에서 제외하며, ASL은 그 음성도 계속 학습한다.

이는 **ASL과 결합한 soft Macro-F1 변형**이며 논문 전체 재현이 아니다. sigmoidF1 원문도 충분한 batch 통계가 필요하다고 설명한다. Gradient accumulation으로 여러 작은 batch의 loss를 합쳐도, 큰 batch에서 F1의 분자·분모를 한 번 합산한 값과 같지 않다. [sigmoidF1 §4.3–4.4](https://arxiv.org/pdf/2108.10566)

### R-Drop과 label dependency의 적용 한계

R-Drop 원문은 같은 입력에 독립적인 dropout을 적용한 두 모델 출력을 일치시킨다. ver6의 E/F는 **고정 encoder 표현 위의 분류기만 두 번 계산**하는 변형이다. 두 개의 지도학습 loss 평균에 각 label의 Bernoulli 분포 `[p, 1-p]` 사이 symmetric KL을 더한다. 9개 label은 동시에 참일 수 있으므로 label 축 softmax-KL을 쓰지 않는다. encoder dropout까지 정규화하는 원문 방식과 같은 효과라고 주장하지 않는다. [R-Drop §2](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf)

F의 작은 dependency 분류기는 fit label의 공발생을 초기 정보로 사용한다. 정답 label을 추론 입력으로 넣지 않는다. 이는 Mission 3용 실험이며 TXT의 *Orderless Recurrent Models*나 LAAT의 ICD 계층 학습을 재현한 것으로 표기하지 않는다. 별도 Longformer·SMART·외부 지식·LLM 데이터 증강은 기본 실행에 포함하지 않는다.

## 4. 모델 선택과 성능 해석

- 기존 fold 1의 **fit 21,032 / stop 5,288**을 유지한다. 새 분류기의 손실·빈도·공발생 학습은 fit만 사용한다.
- stop 안에서 3분할로 임계값을 교차 추정한 Macro-F1을 사용해 후보와 epoch를 비교한다. 임계값을 학습한 동일 행에서 바로 F1을 채점하는 낙관을 줄인다.
- 별도 Training calibration 2,880개에서 기존 ensemble과 신규 후보 혼합을 평가한다. 기본 채택 조건은 기존 baseline 대비 교차평가 Macro-F1 **0.002 이상** 상승과 5분할 중 **3개 이상** 상승이다. 이는 작은 우연한 변동을 거르는 운영 규칙으로, 통계적 유의성 보장이 아니다.
- 이 과정은 **전체 Training으로 신경망 5개를 새로 학습한 완전한 5-fold OOF가 아니다.** stop은 반복적인 후보 선택에 쓰이고, calibration도 이전 버전부터 재사용되어 선택 편향이 남는다.
- 공식 Validation 3,640개는 최종 선택 후 보고용으로만 사용한다. Validation F1을 보고 자동 재학습·threshold 조정·후보 선택을 하지 않는다.
- 모든 후보에 같은 text normalization/tokenization/window 규칙을 적용한다. 표본별 정답을 보고 문장이나 token을 추가·삭제하지 않는다.

현재 baseline은 강하고 ver5의 개선폭은 작았다. ver6는 구조와 손실을 실제로 비교할 수 있게 만드는 변경이다. 새 후보가 조건을 충족하지 못하면 기존 baseline을 유지하며, 큰 F1 상승을 만들어 낸 것으로 기록하지 않는다.

## 5. 원문 파일 검증

공개 원문 URL에서 받은 PDF 6개를 로컬 PDF parser로 열어 페이지 수와 본문을 확인했다. 아래 SHA-256은 보관 파일을 식별한다.

| 파일 | PDF 페이지 수 | SHA-256 |
|---|---:|---|
| `plm_icd_2022.pdf` | 11 | `6b65a85525dc1695fbe000afac28ef8c01d87353f5f3654a7718cefffa943577` |
| `laat_2020.pdf` | 7 | `efe4c78d61c7a12562ed41bfae0b32b44f873d449babee4b264c84bee87e8b9f` |
| `asl_2021.pdf` | 13 | `730d9af30d43febe7ebd7c5ae5c23ed46d35e8c76f1a6b711249346dbcd80c73` |
| `sigmoidf1.pdf` | 28 | `b241be14ef5712aadcf024d224130383c603c91ce7acf9eb7b81e010e5567446` |
| `db_loss_2020.pdf` | 17 | `36640358c039bc0a118434c6ccc31d2dbfba3d6a2c4e48b64269934a3db36439` |
| `r_drop_2021.pdf` | 16 | `4f3dcee5dcc4951586f1b599076e18abb24dca7ae669e62b2b8bafb560d5a331` |
