# Mission 3 ver5

ver4의 학습된 다섯 모델을 재사용하고, **증상별 token attention을 추가한 모델 한 개**를 학습합니다.
논문 3개 PDF는 서로 다른 연구 2편이며 전체 검토 결과와 적용 근거는 [RESEARCH.md](RESEARCH.md)에 있습니다.
새 구조의 성능 향상은 전체 GPU 실행 전에는 확인되지 않았습니다.

## 실행

### 새 계정에서 파일 하나로 준비

1. `outputs/upload/mission3_ver5_inputs.zip`을 새 계정 Google Drive의 **내 드라이브 바로 아래**에 업로드합니다.
2. 최신 `src/mission3/mission3_ver5.ipynb`를 Colab에서 열고 GPU 런타임을 선택합니다.
3. 모두 실행하고 ZIP을 올린 계정의 Drive를 연결합니다. 설정 셀은 그대로 두어도 됩니다.

ZIP에는 ver4 필수 자산 13개(5개 fold 가중치, manifest, sparse 모델, splits, 보정 예측,
tokenizer 파일 4개)와 **Training JSON 29,200개 + Validation JSON 3,640개**가 들어 있습니다.
노트북은 ZIP의 파일 목록·크기·SHA-256을 확인하며 고정된 `/content/mission3_ver5_inputs`에 풉니다.
직접 압축을 풀거나 음성 WAV를 업로드할 필요는 없습니다. 실행 코드도 노트북에 포함되어 있습니다.
ZIP이 다른 위치에 있다면 첫 셀의 `INPUT_ZIP`에 전체 경로를 지정하세요.

새 결과는 Drive에 저장하고 기존 5개 모델을 다시 복사하지 않습니다(`bundle_base=False`).
**원본 ZIP을 Drive에 계속 보관하세요.** Colab 세션이 새로 시작하면 로컬 입력은 ZIP에서 다시 복원합니다.
결과 폴더만 옮기면 기존 모델이 빠지므로 다른 환경에서도 원본 ZIP이 필요합니다.

### 기존 폴더로 실행

ZIP 대신 `BASE_DIR`, `TRAIN_LABEL_DIR`, `VALID_LABEL_DIR`를 직접 지정해도 됩니다.
음성 WAV는 읽지 않습니다. 기본 설정은 **고정 학습 1 epoch + 상위 4개 층 미세조정 1 epoch**입니다.
기존 폴더 자동 탐색과 Colab 출력 기본 경로:

- 모델: `MyDrive/DCC_AIdeas/results/mission3/ver4`
- 데이터: `MyDrive/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급`
  또는 `MyDrive/DCC_AIdeas/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급`
- 새 결과: `MyDrive/DCC_AIdeas/results/mission3/ver5_efficient`
- 공유 예측 캐시: `MyDrive/DCC_AIdeas/results/mission3/prediction_cache`

ver4 폴더에는 가중치 외에도 `splits.json`, `calibration_predictions.npz`, tokenizer, sparse 모델,
`mission3_ver4_model.ckpt`가 필요합니다. 원래 Training과 source fingerprint가 일치하는지 확인한 뒤 재사용합니다.

## 모델과 비교

1. 저장된 fold 1의 **원래 fit/stop**을 복원하고 fold 1 최적 가중치에서 시작합니다.
2. 기존 CLS/mean/window head를 유지하고, padding·특수 토큰을 제외한 증상별 token attention의 결과를 더합니다.
   추가 출력의 초기값은 0이므로 학습 전에는 기존 fold 1과 같은 예측을 냅니다.
3. encoder 고정 상태로 head를 1 epoch 학습하고, 24개 층 중 마지막 4개 층과 분류기만 1 epoch 미세조정합니다.
   embedding과 하위 20개 층은 가중치와 dropout을 고정하고, 상위 층에만 역전파합니다.
   후보 단독의 stop Macro AP로 최적 epoch를 선택합니다. 초기 모델(epoch 0)도 보관 후보입니다.
4. ver4의 **원시** 5-model 평균 확률·TF-IDF 확률과 새 후보 확률을 Training 보정 집합에서 비교합니다.
   확률 평균과 logit 평균 두 후보를 비교하며, 각 교차평가 fold마다 혼합 비율과 임계값을 해당 fit 부분에서만 선택합니다.
   logit 평균은 확률을 log-odds로 바꿔 평균한 뒤 확률로 되돌립니다. 입력 텍스트 전처리는 모든 샘플에 동일합니다.
5. 후보 절차의 교차평가 Macro F1이 더 높을 때만 채택하며, 동률·하락이면 기존 ver4 보정으로 돌아갑니다.
   이 선택 점수는 비편향 성능 보장이 아닙니다. 공식 Validation은 마지막 CSV 생성 후에만 평가합니다.

다섯 기존 모델을 다시 학습하거나 대규모 학습률 탐색을 하지 않습니다. 혼합 비율 비교는 캐시된 확률에 대한
CPU 계산입니다. 선택된 비율이 0인 성분은 최종 추론에서 생략합니다.

## 반복 실험의 비용과 F1 측정

- 학습 대상 21,032개, stop 5,288개, Training calibration 2,880개를 유지합니다.
- 본체 고정 1회 후 상위 4개 층만 1회 미세조정합니다. 현재 모델에서 약 15.6%의 매개변수를 갱신합니다.
  optimizer 상태·역전파 비용은 줄지만 전체 층의 순전파는 필요합니다. 시간·CU가 84.4% 줄어든다는 뜻은 아닙니다.
  하위 층 고정으로 과적응을 줄이려는 설계이며, 실제 F1 향상은 Colab 실행으로 확인해야 합니다.
  `finetune_top_layers=-1`은 전체 미세조정, `0`은 분류기만 학습하는 설정입니다.
- **서울 Validation 3,640개 전체**로 Macro F1과 증상별 F1·양성 개수를 보고합니다.
  평가 표본을 줄이지 않아 드문 증상의 평가 사례도 유지합니다. 정답은 모델·보정값 선택에 쓰지 않습니다.
- 노트북은 모델별 원시 예측을 공통 캐시에 저장합니다. 기존 5개 모델의 추론은 최초 계산 후 재사용할 수 있고,
  새 후보의 가중치가 바뀌면 후보만 다시 계산합니다. 같은 모델의 재평가도 캐시를 재사용합니다.
- 캐시는 순서가 있는 파일명·전처리된 텍스트, 실제 가중치·tokenizer 내용, 추론 코드, 패키지 버전,
  GPU 종류·정밀도·배치 설정이 같을 때만 사용합니다. 입력 정답과 혼합 비율·임계값은 키에 포함하지 않으며,
  현재 실험에서 고정된 보정값을 원시 예측에 적용합니다. 손상된 캐시는 다시 계산합니다.
- 캐시는 평가/추론 경로에서만 사용하며 학습과 calibration에는 전달하지 않습니다.
  제출에 필요하지 않습니다. CLI에서는 선택 인자 `--cache_dir`로 사용합니다.

다른 설정으로 새 실험을 시작할 때 `OUTPUT_DIR`는 새 폴더로 지정하고 `INFERENCE_CACHE_DIR`는 유지하세요.
기본 출력은 `ver5_efficient`이므로 이전 `ver5`·`ver5_budget` 실행 결과와 충돌하지 않습니다.
Validation 경로가 없으면 학습 시작 전에 알려주며, 평가하려면 전체 JSON 폴더를 준비해야 합니다.

## 시간과 재개

이전 A100 기록은 22 epoch, 학습+stop 평가 합 630.28분입니다. 현재 ver5는 frozen 1 + 상위 4개 층 fine-tune 1 epoch입니다.
현재 설정의 총 시간·CU는 아직 실측하지 않았습니다. GPU·Drive I/O·저장/추론 시간에 따라 달라지며,
노트북에 학습·보정·저장 구간과 최종 추론·평가 구간의 실제 시간을 각각 출력합니다.
학습 횟수를 줄여도 압축 해제, 데이터 읽기, 모델 무결성 확인, 최초 평가 비용은 남습니다.
무결성 확인에는 기존 약 6.3GiB 가중치를 읽는 시간이 필요합니다.
노트북은 약 6.3GiB의 기존 모델 사본을 결과에 중복 저장하지 않습니다. 입력 ZIP 외에 후보 가중치와
재개 상태를 저장할 Drive 여유 공간도 필요합니다. 업로드는 GPU 런타임을 시작하기 전에 완료하세요.

Colab에서는 GPU 메모리에 따라 batch/window를 8/8, 4/4, 2/2로 정하고 accumulation으로 유효 batch 32를 유지합니다.
메모리 부족이 발생하면 새 출력 폴더에서 batch/window를 줄여 실행하세요. 자동 설정이 모든 GPU에서의 성공을 보장하지는 않습니다.

- `candidate_best.ckpt`: stop AP가 가장 좋았던 후보 모델.
- `candidate_best_last.ckpt`: 마지막 **완료 epoch**의 optimizer/scheduler/scaler/RNG를 포함한 재개 상태.
- 중간에 중단하면 완료된 epoch까지 복원하고, 미완료 epoch는 다시 시작합니다.
- 같은 데이터·설정·코드로 재실행하면 완료된 후보 학습과 보정 캐시를 재사용합니다.
- 기존 ver4에서 시작하는 첫 학습은 새 optimizer로 수행하는 추가 학습이며, ver4 학습의 정확한 재개가 아닙니다.

## 제출

노트북 기본 설정의 결과는 기존 모델을 외부 경로로 참조합니다(`portable=False`).
`results/mission3/ver5_efficient` 결과 폴더와 입력 ZIP을 함께 보관하고, 다른 환경에서는 ZIP을 푼 뒤
아래 `--base_dir`에 복원된 ver4 폴더를 지정합니다. 모델 내용이 원래 ver4와 같은지 자동 확인합니다.
`mission3_ver5_model.ckpt` 하나만 복사하지 마세요. 재개용 `*_last.ckpt`는 추론에 필요하지 않습니다.

```bash
python results/mission3/ver5_efficient/submission/inference.py \
  --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \
  --ckpt_path results/mission3/ver5_efficient/mission3_ver5_model.ckpt \
  --base_dir "입력 ZIP을 푼 위치/DCC_AIdeas/results/mission3/ver4" \
  --output mission3.csv
```

학습 CLI의 기본값은 `bundle_base=True`입니다. 이 경우 결과의 `base/`에 기존 가중치와 tokenizer를
복사하므로 결과 폴더만 옮겨 추론할 수 있고, 추론 시 `--base_dir`는 생략합니다.
CLI에서도 중복 복사를 생략하려면 `--config_json` 파일에 `"bundle_base": false`를 지정하세요.
설정이 다른 기존 결과 폴더를 덮어쓰지 말고 새 출력 폴더를 사용합니다.

학습 CLI 예:

```bash
python src/mission3/mission3_ver5/train.py \
  --train_label_dir "Training/2.라벨링데이터/TL_서울_구급" \
  --base_dir results/mission3/ver4 --output_dir results/mission3/ver5_efficient --auto_batch
```

노트북 재생성: `python src/mission3/mission3_ver5/build_notebook.py`.
최종 CSV는 `label file name,symptom`, 증상은 지정된 9개 문자열 중 0~9개입니다.
학습 로그가 포함된 노트북을 저장하고, 대회가 요구하는 결과 정리 PPT는 별도로 준비하세요.
