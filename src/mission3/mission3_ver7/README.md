# Mission 3 ver7

공개 사전학습 **KoELECTRA-Base-v3에서 시작해 전체 encoder와 새 9-label attention 분류기**를 함께 학습한다.
기존 ver4의 5개 모델은 비교와 혼합에 재사용한다. 실행 파일은 `src/mission3/mission3_ver7.ipynb` 하나이고,
기존 **`mission3_ver5_inputs.zip`을 그대로 사용**한다. 새 입력 ZIP이나 ver5·ver6 결과는 필요 없다.

## Colab에서 모두 실행

1. Drive 공유 문서함의 `mission3_ver5_inputs.zip`에서 **정리 → 바로가기 추가 → 내 드라이브**를 선택한다.
2. 내 드라이브 바로 아래에 같은 이름의 바로가기를 둔다.
3. `mission3_ver7.ipynb`를 Colab에서 열고 GPU 런타임을 선택한 뒤 **모두 실행**한다.
4. 바로가기를 만든 계정으로 Drive를 연결한다.

바로가기를 하위 폴더에 두었다면 첫 셀의 `INPUT_ZIP`만 바꾼다. 별도 DATA 폴더 연결이나 WAV는 필요 없다.
Drive 연결 후 바로가기를 추가하여 파일이 보이지 않으면 경로 셀의 `force_remount=True`로 한 번 재연결한다.
필요한 소스는 노트북에 모두 포함되어 있다.

| 용도 | 기본 경로 |
|---|---|
| 입력 ZIP 바로가기 | `/content/drive/MyDrive/mission3_ver5_inputs.zip` |
| 기존 모델 복원 | `/content/mission3_ver5_inputs/DCC_AIdeas/results/mission3/ver4` |
| Training·Validation JSON | `/content/mission3_ver5_inputs/DCC_AIdeas/DATA/대학부 데이터/` |
| 새 공개 encoder·tokenizer | `/content/mission3_ver7_pretrained/koelectra_base_v3` |
| **새 결과** | **`/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver7`** |
| 공유 원시 예측 캐시 | `/content/drive/MyDrive/DCC_AIdeas/results/mission3/prediction_cache` |

`ver4` 입력 폴더의 이름을 바꾸지 않는다. 원래 ZIP은 읽기 전용으로 사용하며 manifest·CRC·SHA256을 검사한다.
같은 런타임에서 검증된 압축 해제 파일은 재사용한다. 새 Colab 세션이면 로컬 복원을 다시 한다.
ver7은 전체 encoder를 학습하므로 frozen feature cache를 만들지 않는다.

## 공개 사전학습 모델 준비

[`monologg/koelectra-base-v3-discriminator`](https://huggingface.co/monologg/koelectra-base-v3-discriminator)의
공식 [커밋 `68b30cd259f34a4b5aa8786392612ba2a2617fcc`](https://huggingface.co/monologg/koelectra-base-v3-discriminator/commit/68b30cd259f34a4b5aa8786392612ba2a2617fcc)을 고정한다.
처음에는 encoder 가중치 약 452 MB와 tokenizer를 자동 다운로드한다. 공개 모델 준비 단계에만 Hugging Face
네트워크를 허용하고, 대회 JSON이나 전사 text는 전달하지 않는다. 이후 학습·추론은 오프라인이다.

`AutoModel`로 encoder를 읽어 `safetensors`로 저장하고, 로컬 재로딩과 파일별 SHA256 검사를 통과한 뒤 완료 manifest를 만든다.
완료 폴더의 모든 파일을 검증하고 재사용하므로 같은 런타임에서 불필요하게 다시 다운로드하지 않는다.
모델·tokenizer는 새 후보용이고, 기존 5개 모델은 ZIP에 들어 있는 원래 tokenizer를 계속 사용한다.

## 실험과 데이터 분할

모든 발화 text를 원래 순서대로 동일한 `[SEP]`로 구분한다. 정답과 speaker에 따라 입력을 바꾸지 않는다.
긴 대화를 512-token window와 64-token overlap으로 처리하고 증상별 token attention에서 9개 확률을 직접 계산한다.
전체 encoder와 분류기를 BCE로 학습한다. 출처와 적용 범위는 [RESEARCH.md](RESEARCH.md)에 기록한다.

| 구간 | 역할 |
|---|---|
| 원래 fold 1 fit 21,032개 | 가중치 학습 |
| 원래 stop 5,288개 | 그룹을 분리한 3-fold threshold 교차평가로 epoch 선택 |
| 같은 stop의 원래 ver4 fold 1 | 기존 단일 모델과의 비교 진단 |
| 별도 Training calibration 2,880개 | 5-fold 교차평가로 기존 앙상블·새 후보 혼합과 임계값 선택 |
| 공식 Validation 3,640개 전체 | 선택 확정 후 최종 구성·후보 단독 평가 |

최종 혼합 후보는 기존 calibration 교차평가 Macro F1보다 **0.002 이상** 좋고,
**5개 fold 중 최소 3개**에서 엄격히 개선되어야 채택한다. 미달·동점이면 기존 ver4 보정을 유지한다.
후보 단독 calibration 결과도 기록한다. Validation은 학습·보정·선택에 사용하지 않는다.

이 실험은 새 encoder 하나의 고정 fit/stop 실험이다. 5개 새 encoder를 각각 학습한 모델 OOF가 아니다.
같은 stop에서 여러 epoch를 비교한 선택 편향과 calibration 반복 연구의 선택 편향이 남는다.
최소 개선 기준은 Validation·Test 성능의 상승을 보장하지 않는다.

## 기본 설정·재개

| 설정 | 기본값 |
|---|---|
| 최대 epoch / patience | 3 / 2 |
| encoder / classifier learning rate | `2e-5` / `1e-4` |
| 유효 batch | 32 |
| GPU 자동 microbatch | T4 4 / L4 8 / A100 16 |
| Validation 최종 추론 batch | 8 |
| `RUN_VALIDATION`, `RUN_CANDIDATE_DIAGNOSTIC` | `True`, `True` |
| `bundle_base` | `False` |

`--auto_batch`가 GPU 메모리에 맞춰 microbatch와 window batch를 정하고 gradient accumulation으로 유효 batch를 유지한다.
**미완료 학습 재개는 같은 GPU 종류·AMP 정밀도·관련 패키지 버전과 같은 코드·데이터·설정이 필요하다.**
완료된 최종 결과는 다른 GPU에서도 재사용할 수 있으며, 조건이 다른 원시 예측 캐시는 다시 계산한다.
새 설정 실험은 다른 `OUTPUT_DIR`을 사용한다. 실제 GPU 학습 시간과 CU는 아직 실측하지 않았다.

## 결과 읽기

- `mission3_ver7_model.ckpt`: 확정한 모델 구성·혼합·임계값 manifest.
- `candidate_best.ckpt`: 선택한 epoch의 새 encoder와 attention 분류기 가중치.
- `mission3_ver7_metrics.json`: 후보 학습 이력, 원래 fold 1 단독 비교, 내부 교차평가, 채택/fallback 이유.
- `predictions/<실행시각>/mission3.csv`, `validation_metrics.json`: 최종 출력과 전체·증상별 precision/recall/F1/support.
- 같은 폴더의 `candidate_only.csv`, `candidate_validation_metrics.json`: **새 후보 단독 진단**.
- `notebook_runtime_<실행시각>.json`: 입력 준비, 공개 모델 준비, 학습·보정, 추론·평가 시간. 패키지 설치와 CU는 제외.
- `submission/`: 실행 가능한 추론 소스·패키지 목록·문서.

후보 단독 진단의 임계값은 Training calibration에서 고정한다. 진단 점수로 모델·혼합·임계값을 다시 고르지 않는다.
최종 구성에 후보가 제외되어도 진단을 위해 후보를 추가 추론하며, 관련 원시 예측 캐시가 있으면 재사용한다.
이전 전체 Validation Macro F1은 ver4 **0.6566489480388843**, ver5 **0.6567177583831921**,
ver6 **0.6566697183939796**이다. 노트북 마지막에 각각과 차이를 표시한다. 현재 GPU·batch 차이로 작은 수치 차이가 생길 수 있다.

기본 `bundle_base=False`는 기존 5개 모델을 Drive에 중복 복사하지 않는다. **원래 입력 ZIP과 ver7 결과를 함께 보관한다.**
새 후보는 결과 폴더에 포함되므로 최종 추론에는 공개 encoder 다운로드가 필요하지 않다.

```bash
python submission/inference.py --audio_dir "" --label_dir "JSON 폴더" \
  --ckpt_path mission3_ver7_model.ckpt --base_dir "ZIP에서 복원한 ver4 폴더" --output mission3.csv
```

`--audio_dir`는 대회 인터페이스 호환용이다. `--evaluate`는 CSV 확정 후 평가만 수행하고,
`--candidate_output candidate_only.csv`는 후보 단독 진단을 추가한다. `--cache_dir`로 공용 원시 예측 캐시를 지정할 수 있다.
캐시에는 정답·보정값을 저장하지 않으며 text·가중치·tokenizer·추론 코드·관련 실행 조건이 같은 경우에만 재사용한다.

노트북 재생성: `python src/mission3/mission3_ver7/build_notebook.py`.
작은 CPU 테스트는 구현·재개·저장·캐시 동작 검증용이며 실제 전체 데이터 F1 검증을 대신하지 않는다.
