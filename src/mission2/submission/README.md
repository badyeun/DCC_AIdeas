# Mission 2 제출 — 신고자(1) / 119대원(0) 발화 분류

## 모델 개요

```text
발화마다 startAt~endAt 구간 음성 ─┬─ ECAPA 화자 임베딩 192차원 ─┐
                                  └─ MFCC 통계 261차원 ─────────┴→ 통화 단위 트랜스포머 ×10 → 확률 평균 → 0.5 이상이면 1(신고자)
                                     (같은 통화의 발화 전체를 한 시퀀스로, 순서 정보 없이)
```

| 구성 | 내용 |
| --- | --- |
| 발화 특징 | 공개 사전학습 화자 인식 모델 `speechbrain/spkrec-ecapa-voxceleb`의 임베딩 192차원(가중치 고정) + MFCC·Δ·Δ²·RMS·ZCR·spectral 프레임 통계와 발화 길이 261차원 |
| 판정 모델 | 통화의 발화 전체를 한 시퀀스로 넣는 트랜스포머 인코더(128차원, 3층, 4헤드). self-attention으로 같은 통화의 다른 목소리와 비교해 각 발화가 어느 화자인지 판단한다. **위치 인코딩이 없어 발화 순서를 알 수 없다**(어떤 차례로 넣어도 결과가 같다) |
| 앙상블 | 설정만 조금씩 바꾼 트랜스포머 10개(시드, Train 전체 학습, 큰 모델, 같은 화자 보조 과제)의 확률 단순 평균 |
| 체크포인트 | `mission2_final10.ckpt` 한 파일에 트랜스포머 10개 가중치, MFCC 표준화 값, ECAPA 인코더 파일이 모두 들어 있다. 추론 중 인터넷이 필요 없다 |

### 데이터·규칙
- 학습: 서울 Training 29,200통화만 사용했다. **Validation 3,640통화는 학습과 epoch 선택에 쓰지 않았다.** epoch 선택은 Training 통화 10% holdout으로 했다.
- annotation: 추론 입력에는 **`startAt`, `endAt`만** 사용한다. `text`와 발화 순서는 학습·추론 어디에서도 쓰지 않는다. `speaker`는 학습 정답으로만 썼다.
- 모든 발화를 라벨과 관계없이 같은 방식으로 전처리한다. 사전학습 모델은 공개 모델(ECAPA)만 썼고 상용 AI API는 쓰지 않는다.

## 성능 (서울 Validation, 발화 111,947개)

| 단계 | Accuracy |
| --- | ---: |
| MFCC 통계 + RandomForest (발화 하나만) | 0.8372 |
| ECAPA 임베딩 + 통화 내 비교 2단계 | 0.9009 |
| 같은 화자 판별기 2단계 (트랜스포머 이전 최고) | 0.9117 |
| 통화 단위 트랜스포머 단독 (10개 각각) | 0.9219 ~ 0.9252 |
| **최종: 트랜스포머 10개 평균** | **0.9277** |

| 최종 모델 | Precision | Recall | F1 | 발화 수 |
| --- | ---: | ---: | ---: | ---: |
| 119대원(0) | 0.9472 | 0.8999 | 0.9229 | 53,845 |
| 신고자(1) | 0.9113 | 0.9535 | 0.9319 | 58,102 |
| **Macro avg** | | | **0.9274** | 111,947 |

최종 0.9277은 이 폴더 그대로 아래 공식 명령을 실행해 Validation WAV 전체를 다시 추론해 얻은 값이다(Colab T4 GPU). 학습 때 저장한 확률로 계산한 값과 같다.

## 실행

```bash
pip install -r requirements.txt
python inference.py \
  --audio_dir "/path/to/wav_folder" \
  --label_dir "/path/to/json_folder" \
  --ckpt_path "./mission2_final10.ckpt" \
  --output "./outputs/mission2.csv"
```

- 출력: `audio file name,startAt,endAt,speaker` 헤더의 CSV. 입력 JSON의 발화마다 한 행이며 `speaker`는 0 또는 1.
- `--audio_dir` 아래의 `.wav`를 하위 폴더까지 찾고, JSON은 `--label_dir` 아래에서 같은 이름으로 짝을 찾는다.
- GPU가 있으면 자동으로 쓴다. Validation 3,640통화 기준 Colab T4에서 약 2시간(Drive에서 WAV 읽기 포함).
- 비어 있거나 거꾸로 된 구간처럼 처리할 수 없는 발화는 행을 빠뜨리지 않고 Training 다수 클래스(1)로 채운다.

## 폴더 구성

```text
inference.py              공식 진입점: JSON의 발화 경계로 음성을 잘라 판정하고 CSV 저장
features.py               annotation 읽기, 음성 자르기, MFCC 통계, ECAPA 임베딩 (학습 캐시와 추론이 같이 씀)
model.py                  모델 코드: 통화 단위 트랜스포머, 입력 만들기, 10개 확률 평균
mission2_final10.ckpt     학습 완료 가중치 (트랜스포머 10개 + ECAPA 인코더 파일, 한 파일)
requirements.txt
Mission2_training.ipynb   학습 기록 (데이터 분석, 학습 코드 재실행, 10개 모델 학습 로그, 앙상블·체크포인트 재현, 오류 분석)
training_code/
  make_feature_cache.py       1단계: 발화별 ECAPA·MFCC 특징 캐시 (Colab GPU)
  train_call_model.py         2단계: 통화 단위 트랜스포머 학습 (CPU)
  pack_model.py               3단계: 10개를 체크포인트 한 파일로 묶기
  make_training_notebook.py   학습 기록 노트북 생성·실행
  build_submission.py         이 제출 폴더 조립 (검증한 파일과 sha256이 같은지 확인)
```

## 학습 재현

```bash
python training_code/make_feature_cache.py --data_root <part 폴더들> --encoder_dir <ECAPA 파일 폴더> \
  --mfcc_cache_dir mission2_cache --embed_cache_dir mission2_embed_cache
python training_code/train_call_model.py --embed_cache_dir mission2_embed_cache --mfcc_cache_dir mission2_cache \
  --work_dir work --out A0_base.pt --epochs 20 --threads 14 [아래 추가 인자]
python training_code/pack_model.py --models A0_base.pt A1_seed7.pt ... F1_full_pair.pt --encoder_dir <ECAPA 파일 폴더> \
  --out mission2_final10.ckpt
```

| 모델 | 추가 인자 | 선택 epoch | Train holdout | Validation | 학습 시간 |
| --- | --- | ---: | ---: | ---: | ---: |
| A0_base | (기본) | 18 | 0.9261 | 0.9219 | 72분 |
| A1_seed7 | `--seed 7` | 20 | 0.9249 | 0.9226 | 58분 |
| A3_seed11 | `--seed 11` | 16 | 0.9282 | 0.9240 | 58분 |
| F0_full | `--holdout 0` | 20 (고정) | — | 0.9233 | 64분 |
| A2_wide | `--d_model 192 --ff 384 --layers 4 --dropout 0.15` | 19 | 0.9256 | 0.9234 | 102분 |
| C0_pair | `--pair_weight 0.5` | 18 | 0.9265 | 0.9241 | 69분 |
| C1_pair_seed7 | `--pair_weight 0.5 --seed 7` | 17 | 0.9254 | 0.9244 | 65분 |
| C2_pair_w1 | `--pair_weight 1.0` | 17 | 0.9273 | 0.9236 | 87분 |
| C3_pair_seed11 | `--pair_weight 0.5 --seed 11` | 15 | 0.9275 | 0.9237 | 98분 |
| F1_full_pair | `--holdout 0 --pair_weight 0.5` | 20 (고정) | — | 0.9252 | 131분 |

- 선택 epoch: Train holdout 정확도가 가장 높은 epoch. `--holdout 0`(Train 전체 학습)인 두 모델은 holdout 실험에서 정한 20 epoch를 그대로 학습했다.
- Validation 열은 학습이 끝난 뒤 한 번 계산한 값이다. 앙상블은 10개를 가중치 없이 평균했다(가중치나 부분 조합을 Validation으로 고르지 않음).
- 학습은 로컬 PC(CPU 16코어)에서 했다. 특징 캐시는 Colab GPU에서 만들었다.
