# Mission 2 실험 기록

> 모델 체크포인트와 원본 데이터는 Git에 포함하지 않습니다. 아래 기록은 같은 코드와
> 데이터 구성에서 실험을 재현하고, 후속 실험과 비교하기 위한 기준선입니다.

## EXP-001 — MFCC + RandomForest baseline

| 항목 | 결과 |
| --- | --- |
| 실행 코드 | `src/mission2/train_mfcc_rf.py` |
| 실행 환경 | Google Colab, CPU 2코어 + 고용량 RAM (GPU 미사용) |
| 학습 데이터 | 서울 Training 통화 29,200개 → 발화 889,358개 |
| 검증 데이터 | 서울 Validation 통화 3,640개 → 발화 111,947개 |
| 데이터 무결성 | Training/Validation 모두 WAV–JSON 1:1 매칭, 짝 없는 파일 0개 |
| 처리 실패 | Train/Validation 모두 **건너뜀 0개** |
| 입력 특징 | 발화 구간의 MFCC(20)·Δ·Δ² + RMS·ZCR·spectral centroid/bandwidth/rolloff |
| 특징 차원 | 261 (65개 계열 × mean/std/p25/p75 + 발화 길이 log) |
| 학습 설정 | `n_estimators=150`, `max_depth=20`, `class_weight="balanced"`, seed 42 |
| 특징 추출 시간 | Train 5시간 13분 28초 / Validation 39분 03초 |
| RandomForest 학습 | 약 1시간 |
| 다수 클래스 기준선 | 신고자(1), Accuracy **0.5190** |
| 최종 선택 지표 | Validation **Accuracy 0.8372** / Macro F1 0.8367 |
| 체크포인트 | `/content/drive/MyDrive/saved_models/mission2/mission2_mfcc_rf.joblib` |

### Validation 상세

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 119대원(0) | 0.8446 | 0.8108 | 0.8273 | 53,845 |
| 신고자(1) | 0.8309 | 0.8617 | 0.8460 | 58,102 |
| **Macro avg** | **0.8377** | **0.8362** | **0.8367** | **111,947** |
| Accuracy | | | **0.8372** | **111,947** |

### 해석

- 다수 클래스 기준선 대비 Accuracy가 **31.82%p** 높습니다.
- 두 클래스 F1이 0.8273 / 0.8460으로 큰 편차 없이 동작했습니다. 클래스 비율도
  48.1% : 51.9%로 균형에 가까워, Accuracy를 그대로 신뢰할 수 있습니다.
- 119대원(0)의 recall이 0.8108로 가장 낮습니다. 대원 발화를 신고자로 잘못 보는
  경우가 상대적으로 많다는 뜻이며, 개선 여지가 여기에 있습니다.
- 예선 문제(p8)가 허용하는 입력만 사용했습니다. `startAt`/`endAt`으로 구간만 자르고
  text와 발화 순서는 읽지 않으며, `speaker`는 학습 정답으로만 사용합니다.

### 규칙 준수 확인

- 서울 데이터만 사용했습니다. 데이터에 존재하는 폴더는
  `TS/TL/VS/VL_서울_구급` 네 개뿐임을 확인했습니다.
- Validation 폴더 데이터는 평가에만 쓰고 학습에 넣지 않았습니다.
- 통화(파일) 단위로 split된 주최 측 폴더를 그대로 사용하므로, 같은 통화의 발화가
  Train/Validation에 흩어지지 않습니다.

### 재현 명령

```bash
python src/mission2/train_mfcc_rf.py \
  --data_root /content/drive/MyDrive/data\(inho\) \
  --ckpt_path /content/drive/MyDrive/saved_models/mission2/mission2_mfcc_rf.joblib \
  --cache_dir /content/drive/MyDrive/mission2_cache \
  --n_estimators 150 \
  --max_depth 20 \
  --seed 42
```

## 실행 중 확인한 문제와 대응

재현 시 같은 곳에서 막히지 않도록 기록합니다. 아래 네 가지를 적용하기 전에는
특징 추출 예상 시간이 10시간을 넘어 Colab 세션 제한 안에 끝나지 않았습니다.

| 문제 | 증상 | 대응 |
| --- | --- | --- |
| Drive 분할 다운로드 | 같은 통화의 WAV와 JSON이 서로 다른 part 폴더에 들어가 `JSON을 찾지 못했습니다` | `collect_pairs`가 전체 part의 라벨을 한 번에 색인 |
| 원격 파일 조각 읽기 | `librosa.load(path)`가 통화 하나에 4.3초 소요 | 파일을 통째로 읽고 메모리에서 디코딩 (`decode_waveform`) |
| STFT 중복 계산 | MFCC·centroid·bandwidth·rolloff가 각자 STFT를 계산해 발화당 38ms | STFT를 한 번 계산해 넷이 공유 (특징값은 동일함을 확인) |
| 읽기 대기 | 계산 중 I/O가 놀고, I/O 중 CPU가 놀아 통화당 0.83초 | 다음 통화 파일을 미리 읽는 `--prefetch` 추가 → 0.36초 |

`--cache_dir`를 지정하면 통화 2,000개마다 특징을 shard로 저장하므로, 세션이 끊겨도
추출을 처음부터 다시 하지 않습니다. RandomForest 하이퍼파라미터만 바꾸는 재실험은
추출을 건너뛰고 즉시 학습 단계로 들어갑니다.

## 다음 실험 계획

현재 기준선은 **EXP-001 (Validation Accuracy 0.8372)** 입니다. 후속 실험은 동일한
Validation split에서 이 수치를 넘는지로 판단합니다. 특징이 캐시에 있으므로 1번과
2번은 추출 없이 학습만 다시 하면 됩니다.

| 순서 | 실험 | 기대 | 비용 |
| ---: | --- | --- | --- |
| 1 | `n_estimators` 300~500, `max_depth` 해제 | 보통 +0.3~1%p | 학습만, 수 시간 |
| 2 | 특징 중요도 확인 후 저기여 특징 제거 | 학습 시간 단축 | 학습만 |
| 3 | 8kHz 원본을 그대로 사용 (16kHz 업샘플링 제거) | 추출 비용 약 절반 | 재추출 필요 |
| 4 | Log-Mel + CNN | 채널·환경 특성을 더 잘 포착할 여지 | 재추출 + GPU |

- 문제(p8)의 힌트대로 판별 신호는 **녹음 채널·배경 소음·발화 스타일**입니다. Noise
  증강을 세게 걸면 그 신호 자체를 지울 수 있으므로, 증강은 신중히 적용합니다.
- 발화 조각이 매우 짧은 경우가 있어(최소 512 sample 미만) padding이 필요합니다.
  현재 구현은 `waveform_to_feature`에서 처리합니다.
