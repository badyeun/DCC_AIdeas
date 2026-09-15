# Mission 3 ver5

ver4의 학습된 다섯 모델을 재사용하고, **증상별 token attention을 추가한 모델 한 개**를 학습합니다.
논문 3개 PDF는 서로 다른 연구 2편이며 전체 검토 결과와 적용 근거는 [RESEARCH.md](RESEARCH.md)에 있습니다.
새 구조의 성능 향상은 전체 GPU 실행 전에는 확인되지 않았습니다.

## 실행

`src/mission3/mission3_ver5.ipynb`를 Colab에 올리고 GPU 런타임에서 모두 실행하세요.
노트북에는 실행 코드가 포함되어 있습니다. 기존 ver4 결과 폴더 전체와 서울 Training JSON이 필요합니다.
기본 경로:

- 모델: `MyDrive/DCC_AIdeas/results/mission3/ver4`
- 데이터: `MyDrive/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급`
  또는 `MyDrive/DCC_AIdeas/DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급`
- 새 결과: `MyDrive/DCC_AIdeas/results/mission3/ver5`

ver4 폴더에는 가중치 외에도 `splits.json`, `calibration_predictions.npz`, tokenizer, sparse 모델,
`mission3_ver4_model.ckpt`가 필요합니다. 원래 Training과 source fingerprint가 일치하는지 확인한 뒤 재사용합니다.

## 모델과 비교

1. 저장된 fold 1의 **원래 fit/stop**을 복원하고 fold 1 최적 가중치에서 시작합니다.
2. 기존 CLS/mean/window head를 유지하고, padding·특수 토큰을 제외한 증상별 token attention의 결과를 더합니다.
   추가 출력의 초기값은 0이므로 학습 전에는 기존 fold 1과 같은 예측을 냅니다.
3. encoder 고정 상태로 head를 1 epoch 학습하고, 전체 encoder를 최대 3 epoch 미세조정합니다.
   후보 단독의 stop Macro AP로 최적 epoch를 선택합니다. 초기 모델(epoch 0)도 보관 후보입니다.
4. ver4의 **원시** 5-model 평균 확률·TF-IDF 확률과 새 후보 확률을 Training 보정 집합에서 비교합니다.
   각 교차평가 fold마다 혼합 비율과 임계값을 해당 fit 부분에서만 선택합니다.
5. 후보 절차의 교차평가 Macro F1이 더 높을 때만 채택하며, 동률·하락이면 기존 ver4 보정으로 돌아갑니다.
   이 선택 점수는 비편향 성능 보장이 아닙니다. 공식 Validation은 마지막 CSV 생성 후에만 평가합니다.

다섯 기존 모델을 다시 학습하거나 대규모 학습률 탐색을 하지 않습니다. 혼합 비율 비교는 캐시된 확률에 대한
CPU 계산입니다. 선택된 비율이 0인 성분은 최종 추론에서 생략합니다.

## 시간과 재개

이전 A100 기록은 22 epoch, 학습+stop 평가 합 630.28분입니다. ver5는 frozen 1 + fine-tune 최대 3 epoch로,
**전체 2~4시간을 계획용으로 예상**합니다. 아직 실측이 아니며 GPU·Drive I/O·저장/추론 시간에 따라 달라집니다.
첫 실행의 무결성 검사와 독립 제출 묶음 복사는 기존 6.4GiB 가중치를 읽으므로 시간이 필요합니다.
최종 가중치·재개 상태·기본 모델 사본을 위해 기존 ver4 외에 약 12GiB 이상의 추가 저장 공간을 준비하세요.

Colab에서는 GPU 메모리에 따라 batch/window를 8/8, 4/4, 2/2로 정하고 accumulation으로 유효 batch 32를 유지합니다.
메모리 부족이 발생하면 새 출력 폴더에서 batch/window를 줄여 실행하세요. 자동 설정이 모든 GPU에서의 성공을 보장하지는 않습니다.

- `candidate_best.ckpt`: stop AP가 가장 좋았던 후보 모델.
- `candidate_best_last.ckpt`: 마지막 **완료 epoch**의 optimizer/scheduler/scaler/RNG를 포함한 재개 상태.
- 중간에 중단하면 완료된 epoch까지 복원하고, 미완료 epoch는 다시 시작합니다.
- 같은 데이터·설정·코드로 재실행하면 완료된 후보 학습과 보정 캐시를 재사용합니다.
- 기존 ver4에서 시작하는 첫 학습은 새 optimizer로 수행하는 추가 학습이며, ver4 학습의 정확한 재개가 아닙니다.

## 제출

완료된 `results/mission3/ver5` 폴더는 `base/`에 기존 가중치와 tokenizer를 포함하므로 별도로 옮겨도 추론할 수 있습니다.
`mission3_ver5_model.ckpt` 하나만 복사하지 마세요. 재개용 `*_last.ckpt`는 추론에 필요하지 않습니다.

```bash
python results/mission3/ver5/submission/inference.py \
  --audio_dir "WAV 폴더" --label_dir "JSON 폴더" \
  --ckpt_path results/mission3/ver5/mission3_ver5_model.ckpt \
  --output mission3.csv
```

학습 CLI:

```bash
python src/mission3/mission3_ver5/train.py \
  --train_label_dir "Training/2.라벨링데이터/TL_서울_구급" \
  --base_dir results/mission3/ver4 --output_dir results/mission3/ver5 --auto_batch
```

노트북 재생성: `python src/mission3/mission3_ver5/build_notebook.py`.
최종 CSV는 `label file name,symptom`, 증상은 지정된 9개 문자열 중 0~9개입니다.
학습 로그가 포함된 노트북을 저장하고, 대회가 요구하는 결과 정리 PPT는 별도로 준비하세요.
