# Mission 1 제출 — 119 신고자 성별 분류 (M/F)

## 모델 개요

두 종류의 모델이 낸 여성(F) 확률을 가중 평균한다.

```text
P(F) = 0.25 × [0.4·Attention+F0 + 0.4·Stats+F0 + 0.2·Stats(No F0)]   ← WavLM 3모델 앙상블
     + 0.75 × ECAPA 로지스틱 회귀                                       ← 화자 인식 임베딩 모델
P(F) ≥ 0.5 이면 F, 아니면 M
```

| 구성 | 입력 | 사전학습 모델 (가중치 고정) |
| --- | --- | --- |
| WavLM 3모델 | 신고자 발화를 이어 붙인 음성의 앞·중간·끝 최대 8초 3구간 + F0 통계 | `microsoft/wavlm-base-plus` |
| ECAPA 로지스틱 회귀 | 신고자 발화별 ECAPA 임베딩(192)의 발화 길이 가중 평균 + 표준편차 | `speechbrain/spkrec-ecapa-voxceleb` |

- annotation에서는 `startAt`, `endAt`, `speaker`만 읽어 신고자(`speaker=1`) 구간을 고른다. `gender`는 학습 정답으로만 썼다.
- 섞는 비율(0.25:0.75)과 로지스틱 회귀의 C는 Training 내부 holdout(2,920통화)으로 골랐다. Validation은 학습에 쓰지 않았다.
- 사전학습 모델 파일은 폴더 안에 들어 있어 추론 중 인터넷 접속이 필요 없다.

## 성능 (서울 Validation 3,640통화)

| 모델 | Accuracy | 오류 |
| --- | ---: | ---: |
| WavLM 3모델 앙상블 | 0.9788 | 77 |
| ECAPA 로지스틱 회귀 | 0.9802 | 72 |
| **최종 (25:75)** | **0.9827** | **63** |

최종 값은 아래 공식 명령으로 Validation WAV 전체를 다시 추론해 얻은 값이다(Colab T4).

## 실행

```bash
pip install -r requirements.txt
python inference.py \
  --audio_dir "/path/to/wav_folder" \
  --label_dir "/path/to/json_folder" \
  --ckpt_path "./ecapa_gender/ecapa_lr.npz" \
  --output "./outputs/mission1.csv"
```

- 체크포인트는 `inference.py`가 있는 폴더 기준 상대 경로로 모두 읽는다. `--ckpt_path`는 공식 명령 형식을 맞추기 위한 인자라 값과 상관없이 동작하며, **폴더 구조를 그대로 유지**해야 한다.
- 여러 분할 폴더는 `--audio_dir a b --label_dir c d`처럼 공백으로 이어 줄 수 있다.
- GPU가 있으면 자동으로 사용한다.
- 출력: `audio file name,gender` 헤더의 CSV, 통화(WAV 파일) 하나당 한 행.
- 파일 하나가 실패해도 전체가 멈추지 않는다. 한쪽 모델만 실패하면 다른 쪽 확률을 쓰고, 둘 다 실패하면 F(Training 다수 클래스)로 둔다.

## 폴더 구성

```text
inference.py                  공식 진입점 → inference_ecapa_blend.py 실행
inference_ecapa_blend.py      최종 추론 (WavLM 3모델 + ECAPA 섞기)
inference_3model_final.py     WavLM 3모델 모델·특징 코드
common.py, model.py           WavLM 쪽 공통 코드
ecapa_gender.py               ECAPA 쪽 모델 코드 (음성 자르기, 임베딩, 특징, 로지스틱 회귀)
attention_f0/ stats_f0/ stats_no_f0/   WavLM 3모델 가중치
pretrained/wavlm-base-plus/   WavLM 사전학습 모델
ecapa_gender/encoder/         ECAPA 사전학습 모델
ecapa_gender/ecapa_lr.npz     로지스틱 회귀 가중치(표준화 평균·표준편차, 계수, 절편)와 섞는 비율
requirements.txt
Mission1_final.ipynb          WavLM 3모델 학습 기록 (학습 로그 포함)
Mission1_ecapa_training.ipynb ECAPA 모델 학습·비율 선택·재현 검증 기록 (학습 로그 포함)
training_code/                ECAPA 쪽 학습 코드 (.py)와 특징 캐시 생성 코드
```
