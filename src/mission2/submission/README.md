# Mission 2 제출 — 발화 조각의 화자 분류 (119대원 0 / 신고자 1)

## 모델 개요

```text
발화마다 startAt~endAt 구간 음성 ─┬─ ECAPA 화자 임베딩 (192)      ┐
                                  └─ MFCC 통계 (261)              ├→ 통화 단위 트랜스포머 ×10 → 확률 평균 → 0.5 이상이면 1(신고자)
                    통화 안 다른 발화들의 같은 특징 (순서 정보 없음) ┘
```

| 구성 | 내용 |
| --- | --- |
| 발화 특징 | `speechbrain/spkrec-ecapa-voxceleb` 화자 임베딩 192차원(공개 사전학습 모델, 가중치 고정) + MFCC·Δ·Δ²·RMS·ZCR·spectral 통계와 발화 길이 261차원 |
| 판정 모델 | 통화의 발화 전체를 한 시퀀스로 넣는 트랜스포머 인코더(128차원, 3층, 4헤드). **위치 인코딩이 없어 발화 순서를 알 수 없다** — 발화를 어떤 차례로 넣어도 결과가 같다 |
| 앙상블 | 설정을 조금씩 바꾼 트랜스포머 10개(시드 3개, Train 전체 학습, 큰 모델, 같은 화자 보조 과제 4개)의 확률 단순 평균 |
| 체크포인트 | `mission2_final10.ckpt` 한 파일에 트랜스포머 10개 가중치·입력 표준화 값·ECAPA 인코더 파일이 모두 들어 있다. 추론 중 인터넷이 필요 없다 |

### 규칙 준수
- 학습: 서울 Training 29,200통화만 사용. **Validation 3,640통화는 학습(역전파)에 쓰지 않았다.** 모델 선택(최고 epoch)은 Training 통화 10% holdout으로 했다.
- annotation: 추론 입력에는 **`startAt`, `endAt`만** 사용한다. `text`와 발화 순서는 학습·추론 어디에서도 쓰지 않는다. `speaker`는 학습 정답으로만 썼다.
- 같은 통화의 다른 발화 구간 음성을 함께 비교한다(2026-09-19 멘토링에서 원천데이터 음성 비교는 허용된다는 답변을 받아 적용). 비교 연산은 평균·self-attention처럼 순서와 무관하다.
- 사전학습 모델은 공개 모델(ECAPA)만 사용했다. 상용 AI API는 쓰지 않는다.

## 성능 (서울 Validation, 발화 111,947개)

| 단계 | Accuracy |
| --- | ---: |
| MFCC 통계 + RandomForest (기준선) | 0.8372 |
| ECAPA 임베딩 + LightGBM 2단계 | 0.9009 |
| 같은 화자 판별기 2단계 (이전 최고) | 0.9117 |
| 통화 단위 트랜스포머 단독 (10개 각각) | 0.9219 ~ 0.9252 |
| **최종: 트랜스포머 10개 평균** | **0.9277** |

| 최종 모델 상세 | Precision | Recall | F1 | 발화 수 |
| --- | ---: | ---: | ---: | ---: |
| 119대원(0) | 0.9472 | 0.8999 | 0.9229 | 53,845 |
| 신고자(1) | 0.9113 | 0.9535 | 0.9319 | 58,102 |
| **Macro avg** | | | **0.9274** | 111,947 |

최종 0.9277은 아래 공식 명령으로 Validation WAV 전체를 다시 추론해 얻은 값이다(Colab T4, 2시간 5분). 학습 때 저장된 확률로 계산한 값과 같다.

## 실행

```bash
pip install -r requirements.txt
python inference.py \
  --audio_dir "/path/to/wav_folder" \
  --label_dir "/path/to/json_folder" \
  --ckpt_path "./mission2_final10.ckpt" \
  --output "./outputs/mission2.csv"
```

- 출력: `audio file name,startAt,endAt,speaker` 헤더의 CSV. 입력 JSON의 모든 발화 경계마다 한 행이며 `speaker`는 0 또는 1.
- `--audio_dir` 아래의 `.wav`를 하위 폴더까지 모두 찾고, JSON은 `--label_dir` 아래에서 같은 이름으로 짝을 찾는다.
- GPU가 있으면 자동으로 사용한다. Validation 3,640통화 기준 T4에서 약 2시간(Drive 읽기 포함).
- 체크포인트 안의 ECAPA 인코더 파일은 실행 때 임시 폴더에 풀어 쓴다.
- 너무 짧거나 잘못된 구간처럼 처리할 수 없는 발화는 행을 빠뜨리지 않고 Training 다수 클래스(1)로 채운다.

## 폴더 구성

```text
inference.py              공식 진입점 (체크포인트 종류를 읽어 통화 단위 트랜스포머로 추론)
common.py                 오디오 읽기, 구간 자르기, MFCC 통계 특징
speaker_embed.py          ECAPA 인코더 불러오기·임베딩, 체크포인트 안 파일 풀기
call_model.py             통화 단위 트랜스포머 모델 코드, 입력 만들기, 확률 평균
mission2_final10.ckpt     최종 가중치 (트랜스포머 10개 + ECAPA 인코더, 한 파일)
requirements.txt
Mission2_training.ipynb   학습 기록 (학습 코드 실행 로그, 10개 모델 원래 학습 로그, 앙상블·체크포인트 재현 확인)
training_code/            학습 코드
  train_speaker_embed.py, train_mfcc_rf.py   발화별 ECAPA·MFCC 특징 캐시 생성 (Colab)
  train_call_model.py, call_model.py          통화 단위 트랜스포머 학습
  pack_call_model.py                          10개를 체크포인트 한 파일로 묶기
  TRAINING_COMMANDS.md                        최종 10개의 실제 학습 인자와 결과
  experiment_log.md                           전체 실험 기록 (EXP-001 ~ EXP-011)
  build_submission.py, make_training_notebook.py   이 제출 폴더와 노트북을 만든 코드
```
