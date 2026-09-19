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

`--cluster`(기본 켜짐)는 통화의 발화를 음향 유사도로 두 군집에 묶고, 군집 평균
확률로 화자를 정합니다. 같은 사람의 발화는 녹음 환경이 같아 서로 닮으므로,
개별 발화에서 삐끗한 예측이 같은 군집 다수에 의해 교정됩니다. 발화가
`--cluster_min_utterances`(기본 6)보다 적거나 두 군집의 평균 확률 차이가
`--cluster_margin`(기본 0.10)보다 작으면 묶기를 건너뛰고 발화별 예측을 씁니다.
한 사람만 말한 통화를 억지로 둘로 가르지 않기 위한 안전장치입니다.

끄려면 `--call_context none --no_cluster`를 줍니다. 이전 동작과 같아집니다.

학습은 두 결과를 모두 출력합니다. 군집 묶기가 얼마나 기여했는지 구분해서
기록하기 위해서입니다. 체크포인트에는 `--cluster` 설정이 그대로 저장되고,
`inference.py`가 그 설정을 읽어 학습과 같은 방식으로 판정합니다.

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
`--call_context none --cluster`로 군집 묶기만 켜도 됩니다. 이쪽은 차원이 늘지
않습니다.
