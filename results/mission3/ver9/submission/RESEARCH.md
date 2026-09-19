# Mission 3 ver9 설계 근거

작성일: 2026-09-17. **ver9의 전체 데이터 GPU 학습, Validation F1, 실제 Colab 소요 시간은 아직 측정하지 않았습니다.**
이 문서는 기존 산출물에서 확인된 사실과 새 실험의 가설을 구분합니다. 성능 개선을 보장하는 문서가 아닙니다.

## 1. ver1~ver8에서 확인된 결과

근거는 [experiment_log.md](../experiment_log.md)의 버전별 기록과 `results/mission3/`의 저장된 평가입니다.
표의 Validation은 서울 3,640개 전체에 대한 Macro F1입니다.

| 버전 | 실험 | 최종 Validation Macro F1 | 해석 |
| --- | --- | --- | --- |
| ver1 | 모듈형 TF-IDF + Logistic Regression | 별도 실행 기록 없음 | 코드 구현과 실제 측정 구분 |
| ver2 | 단일 노트북 TF-IDF + Logistic Regression | 약 0.6001 | 최초 저장 baseline; 결과 폴더 이름은 `ver1` |
| ver3 | KLUE-RoBERTa-base 미세조정 | 약 0.6260 | 전통적 TF-IDF baseline보다 개선 |
| ver4 | KLUE-RoBERTa-large 5-fold + TF-IDF/LR | 0.656648948 | 현재 주력 앙상블 |
| ver5 | 증상 attention 추가 학습·혼합 보정 | 0.656717758 | ver4 대비 약 +0.0069%p; 최적 후보는 추가 학습 전 epoch 0 |
| ver6 | A–F 분류기/손실 비교·D 미세조정 | 0.656669718 | 새 후보 미채택, 기존 ver4 구성 사용 |
| ver7 | KoELECTRA-Base-v3 전체 미세조정 | 0.656669718 | 새 후보 미채택; 후보 단독은 0.638109662 |
| ver8 | RoBERTa-large + 증상별 token attention 전체 미세조정 | 0.656669718 | 새 후보 미채택; 후보 단독은 0.645292165 |

ver6~ver8의 최종 CSV는 동일한 예측입니다. 따라서 세 버전의 최종 점수를 각 새 encoder의 성능으로 설명하면 안 됩니다.
ver5의 저장된 최종 점수가 수치상 가장 높지만, 이 정도 차이만으로 추가 학습 효과나 일반화 개선을 확정할 수 없습니다.

### 실패 원인으로 단정할 수 있는 것과 없는 것

- **채택 기준만 낮추면 해결된다는 증거는 없습니다.** ver8의 최선 혼합 calibration 교차평가 F1은
  0.657450867로 기존 0.659670866보다 낮았습니다. `+0.002` 기준을 제거해도 이번 후보의 혼합 성능은 기존보다 낮습니다.
- ver6의 최선 logit 혼합은 0.662117로 기존보다 높았지만 개선 fold가 2/5여서 일관성 기준을 통과하지 못했습니다.
  특정 fold에서의 개선을 모든 통화로 일반화하지 않도록 이 기준을 유지합니다.
- 새 모델 1개와 ver4의 5개 모델 앙상블은 학습량·분산 감소 효과가 다릅니다. 다만 동일한 stop에서의 비교도
  ver8 후보 0.637975045가 ver4 fold 1 단독 0.639684092보다 낮았습니다. 단순히 앙상블 개수 차이만으로 결과를 설명할 수도 없습니다.
- ver6은 ASL, R-Drop 등을 head 수준에서 이미 비교했습니다. 같은 손실을 이름만 바꾸어 새 개선책으로 제시하지 않습니다.
  [Asymmetric Loss 논문](https://arxiv.org/abs/2009.14119)과 [R-Drop 논문](https://arxiv.org/abs/2106.14448)의
  다른 과제 성과는 이 데이터에서의 개선 보장이 아닙니다. 전체 encoder에 R-Drop을 적용하면 추가 forward 비용도 듭니다.
- 전사와 symptom의 대응이 불명확해 보이는 사례만으로 라벨 오류율이나 F1의 상한을 추정할 수 없습니다.
  이번 실험은 정답 자동 수정·외부 LLM 재라벨링·특정 샘플 제거를 하지 않습니다.
- ver8의 저장된 CSV 점수는 검증되어 있지만 로컬 `trials/B/best.ckpt`는 완료 기록의 epoch 3과 해시가 다릅니다.
  따라서 ver9는 이 불일치 후보를 초기 가중치로 사용하지 않습니다. 공개 고정 모델과 기존 ver4 입력 묶음에서 시작합니다.

## 2. 새 encoder: 한국어 DeBERTa-v3-base

사용 모델은 [team-lucid/deberta-v3-base-korean](https://huggingface.co/team-lucid/deberta-v3-base-korean)입니다.
공개 encoder와 fast tokenizer를 다음 [고정 revision](https://huggingface.co/team-lucid/deberta-v3-base-korean/tree/47a3b00828c5dc2b8b080c6738ef7a58ebd419d5)에서 준비합니다.

```text
model_id: team-lucid/deberta-v3-base-korean
revision: 47a3b00828c5dc2b8b080c6738ef7a58ebd419d5
```

고정 revision의 실제 config/tokenizer를 Transformers 4.57.1에서 읽어 fast tokenizer 생성,
긴 한국어 전사의 overflow window·마지막 문구 보존, 빈 입력, 저장 후 오프라인 재로딩을 확인했습니다.
이는 전체 pretrained 가중치 다운로드나 GPU 학습 검증을 대신하지 않습니다.
config의 최대 위치는 512이고 tokenizer의 `model_max_length`는 실제 제한을 뜻하지 않는 큰 sentinel 값이므로
코드에서 길이 512를 명시합니다. 토큰화 API는 [Transformers tokenizer 문서](https://huggingface.co/docs/transformers/main_classes/tokenizer)를,
관련 config는 [DeBERTa-v2 문서](https://huggingface.co/docs/transformers/model_doc/deberta-v2)를 참고하며,
최신 온라인 문서와 실제 실행에 고정한 4.57.1 버전은 구분합니다.

[DeBERTaV3 원논문](https://arxiv.org/abs/2111.09543)은 replaced-token detection과
gradient-disentangled embedding sharing을 사용하는 사전학습을 설명합니다. 이 구조의 공개 한국어 모델로
encoder를 교체하여 기존 RoBERTa와 다른 표현·오류 패턴을 얻을 수 있는지 확인합니다.
모델 카드의 한국어 benchmark 결과는 모델 후보를 고르는 근거이며, 한국어 응급 통화 9-label F1에 대한 직접 증거는 아닙니다.

이번 분류 학습은 공개 encoder 전체와 새 분류기를 함께 학습하는 **최대 3 epoch 미세조정**입니다.
ver4의 CLS + mean pooling / window attention 분류 구조를 사용하여 새 token attention까지 동시에 탐색하는 비용을 줄입니다.
입력은 전체 양측 전사이며, 발화 사이에 동일한 `[SEP]`를 넣고 **512-token window / 64-token overlap**으로 처리합니다.
발화자 정답이나 증상에 따라 window·전처리를 바꾸지 않습니다.

손실은 기본 **unweighted BCE**입니다. 기존 양성 가중치가 적용된 BCE와 다른 정밀도/재현율 균형을 탐색하고,
증상별 판정 임계값은 별도 Training calibration에서 정합니다. 양성 가중치가 기존 실패의 원인이라고 확정한 것은 아닙니다.
여러 요소가 동시에 바뀌므로 이번 결과 하나로 encoder, 손실, pooling 각각의 인과적 기여를 분리할 수도 없습니다.

기본적으로 첫 번째 fold를 학습합니다. 첫 fold에서 측정한 시간과 남은 전체 예산이 충분할 때만 두 번째 fold를 시작하고,
완료·검증된 neural 모델들의 확률을 평균합니다. 이는 새 5-fold 앙상블 또는 다중 seed 확증 실험과 다릅니다.
두 번째 fold 실행 여부는 시간으로 판단하며 공식 Validation 점수로 결정하지 않습니다.

다른 후보인 [KcELECTRA](https://huggingface.co/beomi/KcELECTRA-base)는 댓글과 응급 통화의 도메인이 다르고,
이번에는 ver7과 다른 encoder 구조를 우선하기 때문에 기본 후보로 채택하지 않았습니다.
[mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base)는
[원논문](https://arxiv.org/abs/2509.06888)의 다국어·긴 문맥 설계가 있지만, 현재 Colab 제한에서 학습 속도와
가속 호환성을 검증하지 않았고 한국어 응급 통화에서의 우위를 확인한 결과도 없어 기본 실행에서 제외했습니다.

## 3. 독립적인 희소 모델: NB log-count-ratio + Logistic Regression

[Wang & Manning, ACL 2012](https://aclanthology.org/P12-2018/)의 NB log-count-ratio 특징 재가중 아이디어를 사용합니다.
[저자 공개 자료](https://nlp.stanford.edu/~sidaw/home/projects%3Anbsvm)도 참고했습니다.
**이 구현의 분류기는 Logistic Regression이며 SVM이 아닙니다.** TF-IDF 질량을 사용하므로 원문의 NB-SVM을 그대로 재현한 구현이라고 부르지 않습니다.

문자 2~5-gram TF-IDF 최대 80,000개와 단어 1~2-gram TF-IDF 최대 20,000개를 사용합니다.
기본 `min_df=2`, `sublinear_tf=True`이며 각 채널의 TF-IDF를 이어 붙입니다.
9개 증상마다 양성/음성 문서의 특징 질량에 1을 더해 평활화하고 각각 합으로 정규화한 뒤 로그 비율을 계산합니다.
이 비율을 희소 특징에 곱한 결과로 `C=1`, `class_weight=None`, `max_iter=500`, `n_jobs=1`의 이진 Logistic Regression을 학습합니다.

목표는 기존 TF-IDF/LR과 다른 증상별 어휘 가중치를 제공하는 것입니다. NB 비율이 항상 일반화를 개선하는 것은 아닙니다.
희귀 특징의 과적합을 줄이기 위한 기본 빈도 제한·평활화는 있지만, 실제 기여는 Training 내부 교차평가와 최종 Validation으로 확인해야 합니다.

이 희소 모델은 **Training calibration을 제외한 Training 전체**에서 vocabulary, IDF, NB 비율, LR을 학습합니다.
일부 텍스트는 neural 모델의 stop에 해당할 수 있지만, 이 NB 모델로 neural epoch를 고르지 않습니다.
모든 최종 후보 모델에서 공통 calibration 전체를 제외한다는 경계는 유지합니다.
추론에서는 `transform`과 `predict_proba`만 호출하며 unseen 입력으로 vocabulary·IDF·분류기를 재학습하지 않습니다.
한 증상에 양성 또는 음성이 전혀 없거나 추출할 특징이 없으면 해당 fit 분할에서 계산한 constant 확률을 사용합니다.

## 4. 혼합·선택·평가 경계

새 후보 확률은 다음 비율로 **미리 고정**합니다. DeBERTa와 NB 비율 자체를 calibration이나 Validation에서 다시 탐색하지 않습니다.

```text
candidate = 0.75 × completed_DeBERTa_probability_mean + 0.25 × NB_logistic_probability
```

- neural 가중치 학습: 각 fold의 fit 데이터만 사용합니다.
- neural epoch 선택: 해당 fold의 stop에서 그룹 분리 threshold 교차평가 Macro F1을 사용합니다. AP는 동점 보조 지표입니다.
- NB 모델 학습: 서울 Training에서 공통 calibration을 제외한 데이터만 사용합니다.
- 최종 보정: 기존 ver4 neural 확률, 기존 ver4 sparse 확률, 위의 고정된 새 후보를 공통 calibration에서 비교합니다.
  calibration 그룹별 교차평가의 학습 쪽에서만 alpha·beta·threshold를 맞추고 제외된 그룹에서 점수를 계산합니다.
  동일 전사 그룹이 보정 fit/score 양쪽으로 나뉘지 않도록 합니다.
- 비교하는 혼합 계열은 raw 확률 평균과 logit 평균입니다. 증상별 가중치 탐색은 각 fold 안에서 수행하고,
  두 계열 중 하나를 전체 교차평가 Macro F1로 선택합니다. 계열 점수가 같으면 raw를 선택하고 동점 가중치는 후보를 덜 쓰는 쪽을 선호합니다.
- 채택은 기존보다 **Macro F1이 0.002 이상 증가**하고 **최소 3개 calibration fold에서 개선**될 때만 합니다.
  실제 fold 수가 적어도 요구 개선 fold 수를 자동으로 낮추지 않습니다. 조건을 통과하지 못하면 후보 비율 0의 기존 ver4 구성을 유지합니다.
- 후보 단독 진단의 임계값도 Training calibration에서 고정합니다.
- 공식 Validation은 모든 가중치·혼합·임계값을 확정한 뒤 최종 CSV와 점수를 만드는 데만 사용합니다.
  Validation 정답을 학습·epoch 선택·혼합 비율·threshold 탐색·예측 캐시 생성에 사용하지 않습니다.

교차평가 점수로 혼합 계열을 다시 선택하므로 선택 편향은 남습니다. `+0.002 / 3 fold` 조건은 보수적 선택 규칙이며
통계적 유의성 검정이나 독립 Test 성능 보장은 아닙니다. 여러 버전의 결과를 보고 다음 설계를 정한 연구 과정 자체도
완전히 새로운 평가 데이터에 대한 확증 실험과 구분해야 합니다.

## 5. 3시간 실행 예산

ver8은 3시간을 목표로 추정했지만 자동 중단이 없었습니다. ver9는 **한 번의 자동 실행 전체에 공유하는 10,800초 절대 마감**을 사용합니다.
자동 준비가 시작될 때 만든 deadline을 패키지 설치·입력 복원·모델 준비·학습·최종 추론 단계로 전달하고,
단계마다 새 3시간을 부여하지 않습니다. 최종 보정·저장·평가 시간을 먼저 남겨두고 추가 fold 또는 epoch의 실행 여부를 판단합니다.
외부 Python 프로세스는 남은 시간으로 제한하며 마감에 도달하면 프로세스 그룹을 종료합니다.

이 제한은 **3시간 안에 2-fold × 3 epoch와 모든 Validation 진단이 반드시 끝난다는 보장이 아닙니다.**
GPU 종류, 통화별 window 수, 다운로드, Drive I/O에 따라 학습 fold/epoch 수 또는 완료 가능한 평가 범위가 달라집니다.
시간이 부족하면 이미 완성된 가중치·기록을 보존하고 추가 작업을 멈춥니다. Colab 런타임 강제 종료, 사용자 Drive 인증 대기,
시스템 자체의 정지와 같은 외부 상황까지 Python 코드가 통제할 수는 없습니다.

기존 로그에서 ver7의 실제 epoch 학습 시간 합계는 약 37.30분, ver8은 약 63.06분이었습니다.
이 값은 해당 실행의 단계 기록이고 설치·재개 이전 전체 시간을 포함하는 새 실행의 총비용 추정치가 아닙니다.
DeBERTa가 더 작은 encoder라는 이유만으로 실제 wall-clock 시간이나 F1 향상을 단정하지 않습니다.
최종 판단은 ver9에 저장된 실행 시간·완료 상태·CSV·평가 JSON으로 합니다.

## 6. 문제 PDF와의 대응

기준은 [0820_DCC 대학부_예선_출제문제.pdf](../../../docs/0820_DCC%20대학부_예선_출제문제.pdf)입니다.

| PDF 페이지 | 조건 | ver9 적용 |
| --- | --- | --- |
| 5 | AI-hub 제공 데이터만 사용, 외부 데이터 추가 금지 | 서울 Training 전사와 정답만 학습에 사용; 외부 학습 표본·LLM 생성 표본 없음 |
| 5 | 공개 사전학습 모델 허용, 별도의 개발 없는 상용 AI API 금지 | 고정 revision 공개 DeBERTa를 로컬에서 미세조정; API 추론·데이터 전송 없음 |
| 5 | Training은 서울만, Validation 학습 금지 | fit/stop/calibration 모두 서울 Training 내부; 공식 Validation은 최종 평가만 |
| 5 | 라벨별 차등 전처리, 평가 데이터만 유리한 전처리 금지 | 모든 표본에 동일한 text 정규화·발화 구분·window 규칙 적용 |
| 9 | 입력은 양측 대화의 전사 텍스트만 | speaker, startAt/endAt, 성별, 파일명 등을 예측 특징으로 사용하지 않음 |
| 9 | 지정 9개 symptom만 처리, 다른 symptom만 제거하고 샘플은 유지 | 9차원 다중 레이블; 빈 정답과 빈 예측도 허용 |
| 10 | 클래스별 F1을 평균 | 9개 증상의 Macro F1을 선택·보고 지표로 사용 |
| 11 | 표준 inference.py 명령 한 번으로 CSV 생성 | audio_dir/label_dir/ckpt_path/output 인터페이스; CSV는 label file name, symptom |
| 12 | 학습 로그 노트북·가중치·모델 코드·requirements 등 제출 | standalone 노트북과 runtime/submission 소스·완료 가중치·기록 보관 |

공개 pretrained 모델의 원래 사전학습 코퍼스를 대회 표본으로 추가하는 것은 아닙니다. 문제 p5의 공개 사전학습 모델 허용 범위에서
준비한 가중치만 사용하며, 추가적인 원문 코퍼스 수집이나 외부 데이터를 이용한 적응 학습은 수행하지 않습니다.
모델 준비 단계에는 대회 전사·정답 경로를 전달하지 않고 준비 후 학습·추론은 로컬 파일만 사용합니다.

최종 제출에는 결과 PPT와 **실행 로그가 남은** 노트북도 필요합니다. 소스 생성만으로 실제 GPU 학습·측정·제출이 끝난 것은 아닙니다.
