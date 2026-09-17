# Mission 3 ver4 실행 안내

실행 파일은 [src/mission3_ver4.ipynb](../mission3_ver4.ipynb)입니다. 이 노트북 하나만 Colab에 올리면 필요한 모델·학습·추론 코드를 생성합니다. 이 폴더의 `.py` 파일은 같은 기능을 로컬이나 제출 환경에서 실행하기 위한 코드입니다.

## 기존 결과와 개선 근거

| 실행 버전 | 실제 Validation Macro F1 | 근거 |
|---|---:|---|
| `mission3_all_in_one_ver2` → results/mission3/ver1 | 0.600123 | `results/mission3/ver1/mission3_metrics.json` |
| `mission3_ver3` → results/mission3/ver3 | 0.626015 | `results/mission3/ver3/mission3_ver3_metrics.json` |
| ver4 | 아직 미측정 | 전체 GPU 학습 후 평가 필요 |

ver3의 Training 내부 F1은 3 epoch에서 0.644361로 최고였고, 15 epoch에는 0.601536으로 하락했습니다. 학습 손실은 계속 감소해 과적합 징후가 있습니다. ver4는 별도 stopping 집합의 Macro AP로 최적 epoch를 선택하고 조기 종료합니다. AP는 임계값 없이 예측 순위를 평가하며, 최종 대회 지표는 그대로 **9개 증상의 Macro F1**입니다.

Training 29,200개를 저장된 ver3 tokenizer로 직접 측정했을 때 384토큰을 넘는 대화는 3,292개(11.27%), 512토큰을 넘는 대화는 788개(2.70%)였습니다. 특수 토큰을 포함한 최대 길이는 873토큰입니다. ver4는 긴 대화의 뒷부분도 처리하도록 설계했습니다.

**현재 작성 환경에는 GPU가 없어 ver4 전체 학습과 Validation 성능은 확인하지 못했습니다.** 작은 CPU 모델을 이용한 실행 검증은 코드 경로 확인용입니다. 기존 버전보다 성능이 높다는 판단은 전체 학습·추론이 끝난 뒤 실제 점수로 해야 합니다.

## 모델과 데이터 분리

- `klue/roberta-large`를 5개 fold에서 각각 학습하고 예측 확률을 평균냅니다.
- 한 window는 최대 512토큰이며 인접 window는 64토큰이 겹칩니다. 마지막 window까지 유지하고, 증상별 attention으로 대화 전체를 모은 뒤 **통화당 한 번** BCE 손실을 계산합니다.
- 문자 2–5그램과 단어 1–2그램 TF-IDF + 증상별 Logistic Regression을 함께 학습합니다.
- 증상마다 신경망/TF-IDF의 혼합 비율과 판정 임계값을 Training 내부에서 결정합니다. 어떤 통화도 최소 1개 증상을 출력하도록 강제하지 않습니다.

서울 Training을 약 90% development와 10% calibration으로 먼저 나눕니다. Development 내부 5-fold는 원래 Training 기준으로 각 fold마다 약 72% fit, 18% stopping입니다. 동일한 정규화 대화는 같은 그룹에 두어 분할을 넘지 않게 합니다.

Calibration 10%는 신경망 학습, TF-IDF 학습, epoch 선택에 전혀 사용하지 않습니다. TF-IDF는 development 90%에만 fit합니다. 완성된 5개 신경망의 평균 확률과 TF-IDF 확률을 calibration에 적용한 뒤, 그 안에서 다시 5-fold 교차 보정하여 혼합 비율·임계값 선택의 일반화 점수를 계산합니다. 배포용 보정값은 calibration 전체로 결정합니다. **보정 후 기본 모델을 다시 학습하지 않고 같은 모델들을 그대로 추론에 사용합니다.** 공식 Validation은 이 과정에 들어가지 않습니다.

주요 기본값은 `pipeline.py`의 `DEFAULT_CONFIG`에 있습니다.

| 설정 | 기본값 |
|---|---|
| 모델 / fold | `klue/roberta-large` / 5 |
| 최대 epoch / 조기 종료 patience | 6 / 2 |
| 통화 batch / gradient accumulation | 2 / 16 |
| encoder / head learning rate | `1.5e-5` / `1e-4` |
| 문자 / 단어 최대 특징 수 | 120,000 / 40,000 |

## Colab에서 실행

1. [mission3_ver4.ipynb](../mission3_ver4.ipynb)를 Colab에 업로드합니다.
2. GPU 런타임을 선택합니다. 기본 large 모델의 5-fold 학습에는 **A100 또는 L4를 권장**합니다. 사용 가능한 장치에 따라 시간과 메모리 사용량은 달라집니다.
3. 제공 데이터의 압축을 풀고 `TL_서울_구급`, `VL_서울_구급` 폴더 이름을 유지합니다. Mission 3은 JSON만 읽습니다.
4. 위에서부터 모든 셀을 실행하고 Google Drive를 연결합니다. 노트북은 `/content`를 먼저 탐색하고 Drive의 알려진 데이터 경로를 확인합니다. 자동 탐색에 실패하면 노트북의 데이터 경로 설정에 실제 폴더를 지정합니다.
5. 학습·보정이 끝난 뒤 Validation 추론과 평가 셀을 실행합니다. 실행 로그가 남은 노트북도 저장합니다.

기본 결과 폴더는 Colab에서 `/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver4`, 로컬에서 프로젝트의 `results/mission3/ver4`입니다. 신경망 가중치 5개를 저장하므로 Drive 여유 공간도 확인합니다. 패키지와 최초 공개 pretrained 모델 다운로드에는 인터넷 연결이 필요하며, 학습 완료 후에는 저장된 구성으로 오프라인 추론할 수 있습니다.

## 로컬 또는 제출 환경에서 실행

다음 명령은 `DCC_AIdeas` 프로젝트 루트 기준입니다. 전체 학습 명령은 CUDA GPU를 요구합니다.

```bash
python -m pip install -r src/mission3_ver4/requirements.txt

python src/mission3_ver4/train.py \
  --label_dir "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급" \
  --output_dir results/mission3/ver4

python src/mission3_ver4/inference.py \
  --audio_dir "DATA/대학부 데이터/Validation/1.원천데이터/VS_서울_구급" \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --ckpt_path results/mission3/ver4/mission3_ver4_model.ckpt \
  --output results/mission3/ver4/mission3.csv

python src/mission3_ver4/evaluate.py \
  --label_dir "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급" \
  --prediction results/mission3/ver4/mission3.csv \
  --output results/mission3/ver4/validation_metrics.json
```

`--audio_dir`는 주최 측 인터페이스를 맞추기 위해 받으며 음성 파일을 읽지 않습니다. 추론은 `--device cpu`도 지원하지만 large 모델 5개를 처리하므로 GPU보다 느릴 수 있습니다. 학습 설정은 `train.py --config 설정.json`으로 변경할 수 있습니다. 설정을 바꿀 때는 새 `--output_dir`를 사용합니다.

## 결과 읽기와 파일 보관

| 파일 | 의미 |
|---|---|
| `mission3_ver4_metrics.json` | Training 내부 보정 점수, 증상별 교차 보정 F1, 최종 보정값 |
| `mission3_ver4_history.json` | 각 fold의 epoch별 학습·stopping 기록 |
| `validation_metrics.json` | 위 평가 명령으로 만든 공식 Validation Macro F1와 증상별 점수 |
| `mission3.csv` | `label file name,symptom` 형식의 예측 결과 |
| `run.json`, `splits.json` | 실행 설정·데이터/코드 식별값과 분할 파일 목록 |
| `submission/` | 실행 당시 `.py` 파일, requirements와 실행 안내 사본 |

`calibration_crossfit_macro_f1`는 각 보정 평가 샘플의 정답을 보정값 선택에서 제외하여 계산한 **Training 내부 점수**입니다. `calibration_fit_macro_f1`는 같은 calibration 데이터에서 보정값을 정하고 다시 평가한 점수라 낙관적일 수 있습니다. 둘을 Validation 점수로 보고하면 안 됩니다. 공식 Validation 평가를 모델 선택이나 임계값 재조정에 반복 사용하지 않습니다.

**`mission3_ver4_model.ckpt`는 구성 목록과 보정값을 담은 manifest입니다. 이 파일 하나만 옮기면 추론할 수 없습니다.** 결과 디렉터리 전체를 함께 보관하거나 최소한 다음 구성을 유지합니다.

```text
ver4/
├─ mission3_ver4_model.ckpt
├─ fold_1.ckpt ... fold_5.ckpt
├─ tokenizer/
├─ sparse.joblib
└─ submission/
```

CSV의 symptom 값은 기존 결과와 같은 문자열 목록 표현입니다. 예를 들어 `[]`, `['구토', '복통']`처럼 저장하며 CSV 인용 처리는 코드가 담당합니다. 주최 측 sample submission에서 더 구체적인 형식을 제공하면 해당 규격도 확인합니다.

## 중단 후 다시 실행

같은 데이터·설정·코드와 같은 결과 폴더로 다시 실행하면 `fold_N.ckpt`와 `fold_N_done.json`이 모두 존재하는 **완료된 fold만 재사용**합니다. 실행 도중 끊긴 fold는 처음부터 다시 학습합니다. 마지막 미니배치나 optimizer 상태에서 이어가는 기능은 아닙니다.

데이터·설정·코드의 식별값이 기존 실행과 다르면 혼합된 결과를 만들지 않도록 중단합니다. 이 경우 새 결과 폴더를 지정합니다. 메모리가 부족하면 통화 batch와 window batch를 줄이는 설정을 검토하고 새 폴더에서 시작합니다.

## 대회 조건과 참고 자료

판단 기준은 [docs의 출제 PDF](../../docs/0820_DCC%20대학부_예선_출제문제.pdf) 5쪽, 9–12쪽입니다. 서울 Training만 학습하고 양측의 전사 `text`만 입력으로 사용합니다. `speaker`, 시간, 주소, 성별, 정답 라벨은 입력 특징에서 제외합니다. 대상은 `고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란`이며, 나머지 라벨만 제거하고 통화는 유지합니다. 외부 추가 데이터와 상용 AI API를 사용하지 않습니다.

최종 제출에는 추론 코드, 학습 로그 포함 노트북, 학습 완료 가중치, 모델 코드, requirements 및 결과 정리 PPT가 필요합니다. 이 구현의 학습을 완료하고 로그·가중치를 보관해야 하며 PPT는 별도로 준비합니다.

공식 구현 참고: [KLUE RoBERTa large](https://huggingface.co/klue/roberta-large), [Transformers tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer), [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). 모델 로딩·토큰 처리·혼합 정밀도 API 참고 자료이며 Mission 3의 성능 향상을 보장하는 근거는 아닙니다.
