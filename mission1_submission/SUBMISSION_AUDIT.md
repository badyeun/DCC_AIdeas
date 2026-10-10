# Mission 1 제출 준비 점검 보고서

점검일: 2026-10-10  
근거: `Mission1 (2).ipynb`의 실제 실행 기록 및 현재 작업 폴더의 정적 파일 목록

## 결론

- 제출 후보는 3개 분류기의 확률 가중 앙상블이다.
  - Attention+F0: 40%
  - Stats+F0: 40%
  - Stats Attention (No F0): 20%
- 저장된 확률에 대한 공식 Validation 평가는 **97.8846% (3,563 / 3,640, 오류 77개)**다.
- 내부 Training holdout의 40:40:20 결과는 **97.9110%**다.
- 원본 노트북 실행 기록에서 공식 명령어를 사용한 WAV 1개 추론 및 CSV 생성이 성공한 것을 확인했다.
- 이 수치는 저장된 Validation 확률을 평가한 결과다. 전체 Validation WAV를 최종 스크립트로 재추론한 성능이라고 해석하면 안 된다.

## 정리한 노트북

- 대상: `Mission1_final.ipynb`
- 원본: `Mission1 (2).ipynb` (224 cells: code 218, Markdown 6)
- 원본은 수정하지 않았다.
- 제출용 노트북에는 다음만 유지·통합했다.
  - 데이터 및 전처리 검증
  - 최종 3개 모델의 cache 기반 학습 준비와 학습 코드
  - 실제 epoch 학습 로그
  - 내부 holdout ensemble 선택 결과
  - 저장된 공식 Validation 확률 평가 결과
  - 제출 명령어와 WAV 1개 실행 검증 결과
- 제거·축약 대상은 중복 탐색, 초기 baseline 실행 코드, runtime recovery, 특정 샘플 디버깅, 구버전 inference 생성·패치, 전체 Validation 재추론 및 반복 평가 코드다.

## 최종 학습 코드 위치

정리 노트북에는 아래 원본 코드 셀을 보존했다.

| 모델 | 원본 셀 | 보존 내용 |
| --- | ---: | --- |
| Attention+F0 | 104, 106 | 3-chunk embedding/F0 준비, 3-seed attention+F0 학습 및 checkpoint 저장 |
| Stats Attention (No F0) | 120, 121 | 1,536차원 stats cache 검증·분할·scaler, attention 학습 및 checkpoint 저장 |
| Stats+F0 | 123 | F0 cache 정렬 검증, stats+F0 학습 및 checkpoint 저장 |
| Ensemble 선택 | 204 | 동일 내부 holdout에서 40:40:20 비교 |
| 공식 성능 평가 | 205 | 저장된 공식 Validation 확률을 40:40:20으로 평가 |

학습 로그는 원본 실행 출력 그대로 코드 셀 output으로 보존했다. 새 로그나 새 평가 출력은 생성하지 않았다.

## 데이터 및 규칙 준수 점검

### 확인된 사항

- Training 29,200개와 공식 Validation 3,640개가 분리되어 있다.
- 최종 학습의 내부 split은 Train 26,280개 / Holdout 2,920개이며 인덱스 중복이 없음을 assertion으로 확인한다.
- Scaler는 각 최종 모델에서 내부 Train subset의 유효 chunk에만 fit한다.
- 음성 입력은 JSON의 발화 시간과 화자 구분을 이용해 신고자 구간을 잘라 생성한다.
- WavLM 입력은 16 kHz mono waveform이며 최대 8초, front/middle/end 최대 3 chunks를 사용한다.
- `gender`는 학습의 target label 및 성능 평가에 사용된다. 추론 feature로 사용하는 코드는 최종 학습·추론 경로에서 확인되지 않았다.

### 확인이 필요한 사항

- 현재 작업 폴더에는 실제 제출 Drive의 최종 파일이 없다. 다음 파일의 현행 내용·해시·상호 호환성은 이 환경에서 직접 확인할 수 없다.
  - `inference_3model_final.py`
  - `attention_f0/attention_f0_seed_42.pt`
  - `stats_f0/stats_f0_seed42.pt`
  - `stats_no_f0/stats_attention_seed42.pt`
  - `pretrained/wavlm-base-plus/`
- 원본 노트북의 최종 실행 기록은 위 파일이 Drive에 존재하고, 문법 검사와 WAV 1개 추론이 성공했다고 보고한다. 제출 직전 해당 Drive 폴더에서 다시 파일 존재 여부와 공식 명령어를 확인해야 한다.
- 공식 Validation은 다수 후보 비교에 사용됐다. 향후 가중치와 threshold 선택은 내부 holdout 또는 OOF 예측에서 수행하고, 공식 Validation은 최종 확인용으로 제한하는 것이 안전하다.

## 추론 코드 및 체크포인트 호환성

원본 노트북의 최종 실행 기록은 다음을 확인했다.

- `inference.py`, `inference_3model_final.py`, `common.py`의 AST 문법 검사 통과
- 3개 최종 checkpoint와 로컬 WavLM 폴더 존재 확인
- 공식 명령어로 WAV 1개를 추론해 CSV header `audio file name,gender` 및 M/F 값을 확인
- Stats+F0 loader는 `stats_scaler_mean`, `stats_scaler_scale`을 사용
- Stats No-F0 loader는 `scaler_mean`, `scaler_scale`을 사용

`--ckpt_path`는 공식 명령 형식을 맞추기 위해 전달되지만, 최종 wrapper와 3모델 구현은 제출 폴더 기준 상대 경로로 세 checkpoint를 함께 로드하는 구조다. 따라서 제출 시 checkpoint 하나만 복사하면 안 되며, **제출 폴더 전체 구조**를 보존해야 한다.

## requirements.txt

원본 노트북의 실제 제출 환경에서 기록된 패키지 목록은 다음과 같다.

```text
numpy==2.1.3
torch==2.11.0+cpu
torchaudio==2.11.0+cpu
transformers==5.18.0
librosa==0.11.0
soundfile==0.14.0
scipy==1.16.3
scikit-learn==1.6.1
safetensors==0.8.0
```

현재 작업 폴더의 `requirements.txt`는 이 기록과 다르며 최종 3모델 추론 코드도 존재하지 않아 수정하지 않았다. 특히 `torch==...+cpu` 표기는 package index와 평가 환경에 따라 설치 방식이 달라질 수 있으므로, 실제 평가 환경에서 설치 가능성을 확인하기 전 버전을 임의로 바꾸면 안 된다.

## 제출 직전 체크리스트

1. Drive의 최종 제출 폴더에서 3개 checkpoint, `pretrained/wavlm-base-plus`, `inference.py`, `inference_3model_final.py`, `common.py`가 모두 있는지 확인한다.
2. 각 checkpoint의 scaler key가 모델 loader와 일치하는지 확인한다.
3. 공식 명령어로 WAV 1개를 재실행하고 CSV 행 수·header·M/F 값을 확인한다.
4. 실행 경로가 아닌 `inference.py` 파일 위치를 기준으로 상대 경로가 해석되는지 확인한다.
5. 제출 ZIP에는 전체 모델 폴더와 local pretrained model을 포함한다.
