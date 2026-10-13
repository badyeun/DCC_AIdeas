# Mission 3 ver25

기본 실행은 **Training 중 약 4,000개로 Qwen과 새 KLUE 대조군을 각각 1 epoch 학습하고, development 결과 저장 후 종료**합니다. 전체 학습은 결과를 검토한 뒤 별도 `full` 실행으로 시작합니다. Qwen3-4B-Instruct-2507의 QLoRA adapter와 9출력 분류 head를 원본 정답으로 학습하며 Qwen 단독, KLUE 단독, 고정 1:1 확률 평균을 비교합니다. 변경된 자동 배치 선택과 예비 학습의 실제 GPU 실행 시간·성능 개선은 아직 확인되지 않았습니다.

## 첫 실행: 4,000개 예비 학습

1. `../mission3_ver25.ipynb`를 Colab에 업로드하고 GPU 런타임을 선택합니다.
2. 기존 `mission3_ver5_inputs.zip`이 있는 계정으로 Drive를 연결합니다. 다른 위치라면 설정의 `input_zip`만 수정합니다.
3. 기본 설정 `run_mode='pilot'`, `pilot_samples=4000`, `max_epochs=1`을 유지하고 **런타임 → 모두 실행**을 누릅니다. 코드가 노트북에 포함되어 있어 저장소 전체 업로드는 필요 없습니다.
4. 결과는 `MyDrive/DCC_AIdeas/results/mission3/ver25/실행시각_고유번호/`에 저장합니다.

Drive 연결 직후, 패키지 설치 전에 작업 예산을 시작합니다. 기본값은 180분이며 `time_budget_minutes`를 실행 전에 유한한 양수로 직접 지정해 늘릴 수 있습니다. 자동 연장하지 않습니다. 기본 25분은 저장·재로딩·결과 검증용으로 예약합니다. 설치·다운로드·준비·속도 측정·학습·평가도 예산에 포함됩니다. 학습은 설치 후 새 Python 프로세스에서 시작합니다. 마지막 셀만 실행하거나 실패한 준비 셀을 건너뛸 수 없습니다. 재실행은 새 실행이며 추가 GPU 리소스를 소비합니다.

**시간 제한 없이 실행하려면 기존 설정 셀 마지막에 `SETTINGS["time_budget_minutes"] = None`을 추가합니다.** 문자열 `"None"`이나 `0`은 사용할 수 없습니다. 이 설정은 예상 시간에 따른 학습 시작 보류, 실행 중 deadline 검사, 패키지 설치와 학습 프로세스의 시간 제한, 시간 만료 GPU 반환 타이머를 끕니다. 배치 속도 측정과 예상 시간 기록은 계속 수행합니다. `AUTO_RELEASE_GPU=True`는 유지하면 됩니다. 완료·오류 시 결과 기록 후 GPU를 반환하며, Colab 자체의 세션 종료는 막을 수 없습니다. 학습 epoch 수와 pilot/full 구분은 그대로 적용됩니다. 결과 JSON의 `remaining_seconds: null`은 시간 제한이 없다는 뜻입니다.

고정된 fit 20,440개 중 중복 그룹과 증상 비율을 고려하여 약 4,000개를 뽑고 두 모델에 같은 표본을 사용합니다. 중복 그룹을 나누지 않으므로 표본 수가 목표에서 조금 벗어날 수 있으며 실제 수와 선택 ID를 저장합니다. development 4,380개는 그대로 유지합니다. **pilot에서는 confirmation을 평가하지 않고, 공식 Validation을 모델 평가에 사용하지 않으며 전체 Training refit을 시작하지 않습니다.** 입력 ZIP 준비 시 Validation 파일도 추출·무결성 검사에 포함될 수 있습니다. `run_validation=True`도 full 단계에만 적용됩니다.

## 배치 자동 선택

Qwen batch 1/2/4/8, KLUE batch 4/8/16을 같은 대표 표본으로 측정합니다. 기본 `benchmark_warmup_examples=16` 이후 `benchmark_examples=64`개에서 실제 forward/backward·optimizer 처리량과 GPU 메모리를 기록합니다. 최대 입력 길이의 메모리 검사도 수행하고 GPU 메모리 여유가 10% 미만이거나 메모리가 부족한 후보를 제외합니다. 성공한 후보 중 가장 빠른 배치를 선택하며 기울기 누적 횟수를 조정하여 `effective_batch_size=16`을 유지합니다. Qwen 1×16, KLUE 8×2 설정값은 측정 전 초기값이며 학습에는 자동 선택 결과를 적용합니다. 배치를 4배로 늘려도 속도가 4배가 된다는 보장은 없습니다.

측정용 모델 업데이트를 본 학습의 시작 상태로 사용하지 않습니다. bf16 지원 시 bf16, 미지원 시 fp16 경로를 측정하며 준비·학습·평가·저장 예상 시간에는 기본 안전계수 1.35를 적용합니다. 시간 제한을 켠 경우 예산이 부족하면 시작을 보류하거나 미완료 결과로 저장하고, 실행 중에도 deadline을 검사합니다. 현재 GPU에서의 실제 처리량으로 시간 추정이 갱신되며 모든 GPU에서 기본 180분 내 완료를 보장하지 않습니다.

## 결과를 보고 전체 학습을 따로 실행하기

1. 완료된 예비 실행의 `candidate_summary.csv`, `final_selection.json`, development 지표·예측을 확인합니다. `pilot_complete.json`이 있어야 full 실행의 출처로 사용할 수 있습니다.
2. 같은 폴더의 `full_run_estimate.json`에서 전체 실행 예상 시간을 확인합니다. GPU·네트워크·저장 속도가 바뀌면 실제 시간도 달라집니다.
3. 노트북의 **기존 실행 설정 셀 하나에서** `run_mode`를 `'full'`로 바꿉니다.
4. `source_run`을 완료된 예비 실행 폴더의 절대 경로로 지정합니다. 예: `/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver25/20261010T000000Z_example` (실제 생성된 폴더로 교체).
5. 시간 제한 없이 끝까지 실행하려면 `time_budget_minutes=None`으로 설정합니다. 시간 제한을 원하면 예상 시간과 준비·저장 여유를 반영해 분 단위 정수로 지정합니다. GPU 런타임에서 처음부터 **모두 실행**합니다. 원래 예비 결과를 덮어쓰지 않고 새 실행 폴더를 만듭니다.

시간 제한 없는 전체 실행은 **수정된 최신 노트북**을 업로드하고 기존 설정 셀의 마지막에 아래를 추가합니다. `source_run`은 `budget_not_started` 실행 폴더가 아니라 `pilot_complete.json`이 있는 완료된 예비 실행 폴더입니다.

```python
SETTINGS["run_mode"] = "full"
SETTINGS["source_run"] = "/content/drive/MyDrive/DCC_AIdeas/results/mission3/ver25/20261010T145455Z_3c1d8643"
SETTINGS["time_budget_minutes"] = None
```

full은 출처 실행의 완료 상태, 데이터·분할·고정 학습 조건과 저장 파일의 무결성을 검증합니다. 예비 실행의 고정 모델 revision, 동결한 후보와 epoch를 그대로 사용하고 현재 GPU에 맞춰 배치를 다시 측정합니다. 동결한 선택을 confirmation에서 한 번 확인한 뒤, 선택 recipe만 공개 초기 가중치에서 전체 Training 29,200개로 새로 학습합니다. 작은 subset의 가중치에서 이어서 학습하거나 더 큰 fit 비교를 다시 실행하지 않습니다. 공식 Validation은 전체 refit 후 `run_validation=True`일 때만 평가합니다.

## 고정 조건

- 입력은 배열 순서의 `utterances[].text`만 사용합니다. `utterences`는 기존 데이터 호환 파서만 지원합니다. 발화 경계는 줄바꿈으로 유지합니다. 메타데이터·파일명·음성은 모델 특징에 넣지 않습니다.
- 라벨 순서는 고열, 구토, 두통, 복통, 어지러움, 열상, 오심, 전신쇠약, 호흡곤란입니다. 다른 라벨만 제외하며 해당 통화는 유지합니다. 샘플 삭제·재라벨링·정답 마스킹·외부 데이터·생성 정답을 사용하지 않습니다.
- 텍스트 중복 그룹을 분리하지 않는 새 70/15/15 fit/development/confirmation 분할을 저장합니다. 그룹 제약에 따른 실제 개수와 digest를 기록합니다. 과거 all-Training 가중치는 비교에 사용하지 않습니다.
- Qwen: NF4 4-bit + double quantization, LoRA r16/alpha32/dropout0.05, 모든 block의 attention/MLP projection과 분류 head 학습, gradient checkpointing, `use_cache=False`, 동적 padding입니다.
- Qwen 입력은 최대 1,024 tokens이며 앞·뒤 전사와 모든 표본에 같은 판정 접미문을 보존합니다. 마지막 non-padding 판정 위치를 pooling합니다. 길이 분포·초과율을 저장합니다. KLUE는 특수 토큰을 포함한 앞부분 최대 512 tokens를 유지하고 초과한 뒷부분을 자릅니다.
- 실제 배치는 위의 자동 측정 결과를 사용하며 유효 batch16을 유지합니다. LoRA/head/KLUE learning rate는 각각 1e-4/3e-4/2e-5, warmup0.05, AdamW weight decay0.01, gradient clip1.0, 기본1 epoch입니다.
- 9개 float32 logits에서 명시적 weighted BCE를 계산합니다. `pos_weight = clip(sqrt(N_negative / max(N_positive, 1)), 1, 5)`는 실제 예비 학습 subset에서만 계산하고 final refit 때 전체 Training으로 다시 계산합니다.
- 판정은 모든 라벨에 `logits >= 0`입니다. threshold·offset·calibration·혼합비 탐색을 하지 않습니다.

## 선택과 해석

각 모델의 epoch와 recipe는 development Macro F1로 선택합니다. 기본값은 1 epoch이며 소규모 1 epoch의 낮은 점수만으로 수렴 실패를 단정하거나 전체 학습 성능을 보장하지 않습니다. 같은 4,000개로 학습한 새 KLUE 대조군과 비교합니다. 오심 소폭 하락만으로 Macro 개선 후보를 탈락시키지 않습니다. Qwen 또는 고정1:1 평균의 대조군 대비 +0.01은 고비용 확장을 위한 추진 기준이며 통계적 유의성 기준이 아닙니다. 선택을 예비 실행에서 동결한 뒤 별도 full 실행의 confirmation에서 한 번만 비교하고 재선택하지 않습니다.

9개 F1/AP, 오심 P/R/TP/FP/FN, 공통 FP/FN 및 새 오답, 통화 단위 paired bootstrap을 저장합니다. confirmation을 후속 실험 설계에 반복 사용하면 개발셋이 된다는 한계가 있습니다. 이번 분할은 완전히 새로운 외부 검증이 아닙니다.

예산이 남아도 pilot에서 전체 Training 학습을 자동으로 시작하지 않습니다. full의 공식 Validation은 이미 반복 관찰된 셋이라는 한계를 남깁니다. weighted BCE만 구현합니다. ASL과3-fold OOF, 친구11개 앙상블의 새 split 재학습은 별도 예산을 정할 후속 실험입니다.

## 완료 여부와 제출

예비 성공은 `runtime.json`의 `status='pilot_complete'`와 `pilot_complete.json`으로 확인합니다. 학습 체크포인트·development 지표/예측·선택 결과·배치 속도 기록·전체 실행 시간 추정을 저장하며 `training_scope='training_subset'`, `submission_ready=false`입니다. **예비 실행은 정상 완료해도 제출 모델이 아닙니다.** `run_complete.json`과 최종 manifest를 만들지 않습니다.

full 최종 완료는 `runtime.json`과 `run_complete.json`으로 확인하며 전체 Training refit, 로컬 재로딩, 결과·파일 검증이 모두 성공했을 때만 기록합니다. `complete_not_improved`도 새 표현 후보가 대조군보다 개선되지 않았다는 표시이며 기존 최고 baseline을 이겼다는 뜻이 아닙니다. 작은 양의 개선도 실제 development 최고 후보를 선택하며, +0.01 미만이면 고비용 확장 권고만 끕니다. `fit_only_budget` 등 진단 완료 상태와 `diagnostic_complete.json`은 예비 성공이나 최종 제출 완료가 아닙니다.

full의 `mission3_ver25_model.ckpt`는 모델 경로와 조합을 담은 JSON manifest입니다. 해당 파일만 옮기지 말고 base·adapter·head·tokenizer·revision·라이선스·실행 코드를 포함한 실행 폴더 전체를 보관합니다. 예비 실행도 출처 검증과 confirmation에 필요한 체크포인트가 있으므로 폴더 전체를 보관해야 합니다. 최종 분류 recipe에 따라 `final_control/`, `final_qwen/`, `shared_base/`가 필요합니다.

제출 폴더의 `inference.py`는 다음 인터페이스를 수용합니다.

```bash
python inference.py --audio_dir /path/to/wav --label_dir /path/to/json --ckpt_path ../mission3_ver25_model.ckpt --output ./outputs/mission3.csv
```

`--checkpoint`는 `--ckpt_path`의 별칭입니다. `audio_dir`는 인터페이스 호환용입니다. 추론은 패키지에 포함된 로컬 가중치만 사용하며 다운로드하지 않습니다. **NF4 Qwen 또는 Qwen을 포함한 앙상블은 오프라인 추론에도 CUDA GPU가 필요합니다.** KLUE 단독 모델은 CPU에서도 추론할 수 있습니다. 출력은 `label file name`과 string symptom list입니다.

실패 시 `_diagnostics/ver25_error_*.json`에 실패 단계·오류 종류·코드 위치를 저장하며 예외 메시지에 포함될 수 있는 원문·개인 경로는 출력하지 않습니다. GPU 반환은 완료·오류·시간 제한 시 요청하며 자동 반환 실패 시 Colab에서 직접 연결을 해제하고 런타임을 삭제합니다. 중단된 학습을 자동 재개하지 않습니다.

## 의존성과 라이선스

ver25 전용 requirements의 기존 고정 버전을 유지합니다 (`transformers==4.57.1`, `peft==0.17.1`, `bitsandbytes==0.48.1`, `accelerate==1.10.1` 등). 이번 변경으로 의존성을 추가·업그레이드하지 않습니다. 노트북 의존성 설치는 Colab에서 실행하며 로컬 전역 환경은 변경하지 않습니다. 실제 Colab GPU 조합에서 변경된 속도 측정·학습의 검증이 필요합니다.

- [Qwen 공식 모델](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Apache-2.0 라이선스](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE): 실행 시작 시 공개 revision을 한 번 immutable SHA로 고정하여 저장·재사용합니다.
- [KLUE 공식 모델](https://huggingface.co/klue/roberta-base): 고정 revision `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`, CC-BY-SA-4.0. 재배포 시 출처·동일조건변경허락 조건을 보존합니다.
- [PEFT quantization](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization), [bitsandbytes 0.48.1](https://pypi.org/project/bitsandbytes/0.48.1/), [accelerate 1.10.1](https://pypi.org/project/accelerate/1.10.1/).

## 로컬 검증

다음 명령은 `DCC_AIdeas` 프로젝트 루트에서 실행합니다. 학습 의존성이 준비된 검증 환경이 필요합니다.

```bash
python src/mission3/mission3_ver25/build_notebook.py
python -m unittest discover -s src/mission3/mission3_ver25 -t . -p "test_*.py" -v
```

실제 로컬 Training 29,200개를 읽어 새 분할 구현을 확인했으며 fit 20,440개 / development 4,380개 / confirmation 4,380개였습니다. 현재 데이터에서는 예비 subset을 정확히 4,000개로 선택했고 fit 대비 증상별 양성 비율의 최대 차이는 약 0.01101%p였습니다. 이는 데이터 파싱과 분할 검증 결과이며 모델 학습·성능 점수가 아닙니다. 정규화한 텍스트의 동일 중복 그룹을 검사하며 가까운 중복의 유사도 분석은 미확인입니다.

작은 fixture 검증은 원본 데이터·대형 공개 모델 다운로드·GPU 학습을 실행하지 않습니다. 이 테스트 통과만으로 실제 GPU 학습 속도·메모리·정확도·Colab 자동 반환이 확인되지는 않습니다.
