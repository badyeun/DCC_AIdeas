# Mission 2 — 신고자 / 119대원 분류

각 통화의 JSON `utterances`에서 `startAt`~`endAt` 구간을 잘라, 신고자(`1`)와
119대원(`0`)을 구분합니다. 이 구현은 text와 발화 순서를 사용하지 않습니다.

판정은 통화 하나의 발화 전체를 한 번에 모아서 합니다. 발화 조각 하나만 보면
"이 정도면 조용한가"를 절대 기준으로 판단해야 하는데, 통화마다 회선 상태와
주변 소음이 달라 그 기준이 흔들립니다. 같은 통화 안에서 서로 비교하면 그
편차가 사라집니다. 비교에 쓰는 값(중앙값, 순위, 군집)은 모두 발화를 늘어놓은
차례와 무관하므로, 조각을 아무 순서로 섞어 넣어도 결과가 같습니다.

## 학습

```bash
python src/mission2/train_mfcc_rf.py \
  --data_root /path/to/data_root \
  --ckpt_path saved_models/mission2_mfcc_rf.joblib
```

`data_root` 아래의 각 part에서 `대학부 데이터/Training`과 `Validation`의 서울
데이터를 자동으로 찾습니다. 분할 다운로드로 WAV와 JSON이 서로 다른 part에 나뉘어
들어가도 전체 part의 라벨을 한 번에 색인해 짝을 맞춥니다. 검증 데이터는 평가에만
사용하며 체크포인트 학습에는 넣지 않습니다.

## 중단 후 이어받기

`--cache_dir`를 주면 특징을 통화 `--shard_calls`개(기본 2,000)마다 shard로 저장합니다.
세션이 끊겨도 같은 명령을 다시 실행하면 마지막 shard 다음부터 이어서 추출합니다.

```bash
python src/mission2/train_mfcc_rf.py   --data_root /path/to/data_root   --ckpt_path saved_models/mission2_mfcc_rf.joblib   --cache_dir cache/mission2
```

`n_mfcc`, `timestamp_unit`, 통화 수 중 하나라도 달라지면 기존 shard를 버리고 처음부터
다시 추출하므로, 설정이 섞인 특징이 학습에 들어가지 않습니다. RandomForest 파라미터만
바꿔 재실험할 때는 특징 추출을 건너뛰고 바로 학습으로 넘어갑니다.

## 제출 CSV 생성

```bash
python src/mission2/inference.py \
  --audio_dir {wav_folder} \
  --label_dir {json_folder} \
  --ckpt_path saved_models/mission2_mfcc_rf.joblib \
  --output ./outputs/mission2.csv
```

출력 컬럼은 정확히 `audio file name,startAt,endAt,speaker`입니다. 입력 JSON의 모든
발화 경계마다 한 행을 생성하고 `speaker`는 `0` 또는 `1`입니다.

## 특징

MFCC·Δ·Δ²의 분포 통계에 RMS, zero-crossing rate, spectral centroid/bandwidth/rolloff,
발화 길이를 더해 RandomForest로 분류합니다. 발화 하나당 261차원입니다.

## 통화 단위 비교

`--call_context`로 통화 안에서의 상대값을 특징에 더합니다.

| 값 | 차원 | 내용 |
| --- | ---: | --- |
| `none` | 261 | 발화 조각만 본다 (이전 동작) |
| `center` | 522 | 통화 중앙값을 뺀 값을 추가 (기본값) |
| `full` | 588 | 위에 더해 통화 안에서의 순위(0~1)를 추가 |

`--cluster`는 통화의 발화를 음향 유사도로 두 군집에 묶고, 군집 평균
확률로 화자를 정합니다. 같은 사람의 발화는 녹음 환경이 같아 서로 닮으므로,
개별 발화에서 삐끗한 예측이 같은 군집 다수에 의해 교정됩니다. 발화가
`--cluster_min_utterances`(기본 6)보다 적거나 두 군집의 평균 확률 차이가
`--cluster_margin`(기본 0.10)보다 작으면 묶기를 건너뛰고 발화별 예측을 씁니다.
한 사람만 말한 통화를 억지로 둘로 가르지 않기 위한 안전장치입니다.

| `--cluster` | 동작 |
| --- | --- |
| `auto` (기본) | 군집 없음/있음 중 Validation 정확도가 높은 쪽을 체크포인트에 저장 |
| `on` | 항상 군집 묶기 |
| `off` (= `--no_cluster`) | 군집 묶기 안 함 |

실데이터(EXP-002)에서는 군집 묶기가 정확도를 0.8537 → 0.6676으로 떨어뜨렸습니다.
그래서 기본값을 `auto`로 바꿔, 나빠지는 설정이 체크포인트에 저장되지 않게 했습니다.
이전 동작으로 돌리려면 `--call_context none --no_cluster`를 줍니다.

학습은 두 결과를 모두 출력합니다. 군집 묶기가 얼마나 기여했는지 구분해서
기록하기 위해서입니다. 체크포인트에는 선택된 설정이 저장되고, `inference.py`가
그 설정을 읽어 학습과 같은 방식으로 판정합니다.

## 화자 임베딩 + 통화 내 비교 (`train_speaker_embed.py`)

사전학습 화자 인식 모델(SpeechBrain ECAPA-TDNN, VoxCeleb 학습)로 발화마다 192차원
화자 임베딩을 뽑고, 같은 통화의 발화끼리 그 임베딩을 직접 비교합니다. 예선 규칙
(p5)의 "공개 오픈소스 라이브러리(pre-trained) 사용 허용"에 해당합니다.

| 단계 | 입력 | 모델 |
| --- | --- | --- |
| 1단계 | 임베딩, 통화 평균을 뺀 임베딩, 통화 중심·다른 발화와의 유사도 (+ MFCC 522) | LightGBM (EXP-003/004는 HistGradientBoosting) |
| 2단계 | 1단계 확률을 목소리 유사도로 가중한 "신고자 쪽/대원 쪽과 닮은 정도", 가장 닮은 발화 3개의 평균 확률 | HistGradientBoosting |

2단계는 군집처럼 통화를 두 덩어리로 자르지 않고 유사도로 가중만 합니다. 2단계의
학습 입력은 통화 단위 fold로 만든 out-of-fold 확률입니다. `--stage2 auto`(기본)는
1단계만/2단계까지 중 Validation 정확도가 높은 쪽을 저장합니다.

```bash
python src/mission2/train_speaker_embed.py \
  --data_root /path/to/data_root \
  --ckpt_path saved_models/mission2/mission2_speaker.ckpt \
  --embed_cache_dir cache/mission2_embed \
  --mfcc_cache_dir cache/mission2
```

- 임베딩도 shard로 저장되어 중단 후 이어받습니다. **GPU 권장**입니다(CPU는 수십 시간).
- `--mfcc_cache_dir`는 읽기만 합니다. 설정이나 통화 수가 다르면 캐시를 건드리지
  않고 멈춥니다. 빼면 임베딩만으로 학습합니다.
- 체크포인트 옆 `ecapa_voxceleb/` 폴더에 인코더 파일을 복사합니다. 제출 시
  체크포인트와 이 폴더를 함께 두면 `inference.py`가 인터넷 없이 동작합니다.
- `inference.py`는 체크포인트 종류를 읽어 MFCC-RF와 화자 임베딩을 알아서 고릅니다.
- `--bank_clusters N`: Train의 대원/신고자 임베딩을 각각 N개 중심으로 요약한 목소리
  사전을 만들고, 발화가 사전과 얼마나 닮았는지를 1단계 특징에 더합니다. 119 대원은
  여러 통화에 같은 사람이 반복해 나오기 때문입니다. Train 발화의 사전 특징은 자기
  fold를 뺀 사전으로 재고, 사전은 체크포인트에 저장되어 추론에서 그대로 씁니다.
- `--stage2_version 2`(기본): 2단계 입력에 발화 길이, 확실한 발화(확률 0.8 이상/0.2 이하)
  중 가장 닮은 몇 개와의 유사도, 회선 소리(MFCC 평균 25열, 통화 평균을 뺀 값)로 잰
  비교를 더합니다. `--rounds N`(기본 2)은 2단계를 N회차 반복하고, `--stage2 auto`는
  **Train 통화 단위 out-of-fold 정확도**가 가장 높은 회차까지만 저장합니다(EXP-006까지는
  Validation으로 골랐습니다). EXP-003/004와 같은 설정은 `--stage2_version 1 --rounds 1`입니다.
- `--stage2_version 3` (EXP-007): v2 입력에 **같은 화자 판별기**로 잰 비교 11개를 더합니다.
  판별기는 Train 통화 안의 발화 쌍으로 "두 발화가 같은 화자인가"를 학습한 LightGBM입니다
  (입력: 지문별 코사인·통화 평균을 뺀 코사인·상대 코사인, 통화 안 순위, 두 발화 길이,
  회선 소리 차이 27열 등 38개, 모두 쌍의 앞뒤를 바꿔도 같은 값). VoxCeleb 코사인을 그대로
  믿던 v1/v2와 달리, 짧은 발화에서 코사인을 얼마나 믿을지를 이 데이터로 배웁니다.
  Train 통화의 판별 확률은 그 통화를 보지 않은 fold 판별기로 만듭니다. `--pair_samples`
  (기본 120)는 통화당 학습 쌍 수, `--pair_max_iter`(기본 500)는 반복 상한입니다.
  추론은 통화마다 모든 쌍(n(n-1)/2)을 판별합니다. MFCC 캐시가 필요합니다.
- 실행 로그에 `Train out-of-fold Accuracy (1단계)`, `(2단계 N회차)`가 찍힙니다. 개선 판단은
  Validation과 이 값이 같은 방향으로 오르는지로 합니다.
- `--second_encoder wavlm`: Microsoft WavLM-Base-Plus 화자 검증 모델(`microsoft/wavlm-base-plus-sv`)로
  두 번째 목소리 지문(512차원)을 뽑고, PCA로 `--second_dims`(기본 128)차원까지 줄여 ECAPA와
  이어 붙입니다. 두 지문을 각각 길이 1로 맞춰 붙이므로 코사인 유사도는 두 모델 유사도의
  평균이 되고, 1·2단계와 목소리 사전은 코드 변경 없이 두 모델을 함께 씁니다.
  WavLM-Base는 앞단이 GroupNorm이라 padding이 결과를 바꾸므로 발화를 하나씩 넣습니다.
  캐시는 `--second_cache_dir`에 `wavlm_*` 이름으로 저장하고, 파일은 체크포인트 옆
  `wavlm_base_plus_sv/`에 복사합니다.
- `--extract_only`: 임베딩 캐시만 만들고 끝냅니다. 추출은 GPU 런타임, 학습은 CPU 런타임에서
  나눠 돌릴 때 씁니다.
- `--feature_fraction`, `--bagging_fraction`: 1단계 나무마다 쓰는 특징·행 비율. 1보다 작으면
  반복당 시간이 줄어듭니다.
- `--stage1_cache 파일`: 1단계 모델과 확률을 저장해 두고, 1단계 설정이 같으면 다음
  실행에서 1단계 학습(약 40분)을 건너뜁니다. 2단계만 바꾸는 실험이 수 분으로 줄어듭니다.
- 학습이 끝나면 체크포인트 옆에 `*_val_predictions.npz`(정답, 통화 경계, 1·2단계 확률,
  발화 길이)를 저장합니다. 오류 분석용입니다.

## 예전 캐시 이어쓰기

통화 경계를 저장하기 전에 만든 캐시도 그대로 씁니다. 이 경우 JSON만 다시 읽어
경계를 복원하며, WAV 재추출은 일어나지 않습니다. 복원한 발화 수가 캐시의 행
수와 다르면 학습을 시작하지 않고 멈춥니다. 어긋난 경계로 통화를 묶으면 조용히
틀린 결과가 나오기 때문입니다.

특징 캐시는 `--call_context` 값과 무관합니다. 상대값은 캐시를 불러온 뒤에
붙이므로, 모드를 바꿔 다시 실험할 때 추출을 건너뛰고 바로 학습으로 갑니다.

## 메모리

`center`는 특징이 두 배가 됩니다. 서울 Training 88.9만 발화 기준으로 학습 배열이
약 1.9GB이고, 원본 배열과 합쳐 한때 약 2.8GB를 씁니다. RAM이 빠듯하면
`--call_context none`으로 상대값을 빼면 차원이 늘지 않습니다.

예전 캐시는 처음 한 번 JSON에서 통화 경계를 복원해 `{split}_counts.npy`로 저장합니다.
다음 실행부터는 그 파일을 써서 복원(실데이터 약 20분)을 건너뜁니다.
