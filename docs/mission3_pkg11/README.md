# Mission 3 제출물 — 환자 증상 인식

119 신고 전사 텍스트만 읽고 9개 증상을 예측합니다. 판정 임계값은 **전 증상 0.5**.

공식 Validation 3,640통화 실측 **Macro F1 0.65658**.

---

## 실행

```bash
pip install -r requirements.txt

python inference.py \
    --audio_dir {wav 폴더} \
    --label_dir {주석 JSON 폴더} \
    --ckpt_path ./weights \
    --output ./outputs/mission3.csv
```

Mission 3은 음성을 읽지 않지만 `--audio_dir`를 받습니다(채점 인터페이스 호환).

`--device cpu`로 CPU 실행도 됩니다. GPU에서 3,640통화에 5분 내외입니다.

**네트워크 불필요** — 체크포인트가 인코더 설정을 직접 들고 있어서 사전학습
가중치를 받지 않습니다. `HF_HUB_OFFLINE=1`에서도 동일한 결과가 나옵니다.

## 구성

| 파일 | 역할 |
|---|---|
| `inference.py` | 한 줄 실행 진입점 |
| `normalize.py` | 동의어 통합 전처리 (일부 멤버가 사용) |
| `tfidf_model.py` | TF-IDF 모델 로드·예측 |
| `common_local.py` | 공통 상수 (증상 목록) |
| `weights/*.ckpt` | 신경망 11개 |
| `weights/tfidf.pkl` | TF-IDF + 로지스틱 회귀 |
| `weights/tokenizer/` | KLUE 토크나이저 |

## 앙상블 구성 — 일곱 계열 11개 멤버

| 계열 | 멤버 | 무엇이 다른가 |
|---|---:|---|
| base mean lr2e-5 | 2 | 기준 레시피 |
| base attn lr2e-5 | 2 | 증상별 attention 헤드 |
| base mean lr3e-5 | 1 | 학습률 |
| base attn lr3e-5 | 1 | 학습률 + 헤드 |
| 정규화 텍스트 | 2 | 동의어 통합 입력 |
| 도메인 적응 | 2 | 전사로 MLM 추가 학습 후 fine-tune |
| CLS 풀링 | 1 | `[CLS]` 토큰 사용 |

공통: `klue/roberta-base`, 3 epoch, lr 표기값, `pos_weight = clip(sqrt(음성/양성), 1, 5)`,
batch 16 × accum 2, AdamW, bf16, **Training 29,200통화 전부**.

최종 확률은 11개의 **단순 평균에 TF-IDF를 10% 혼합**하고, 전 증상 **0.5**에서 판정합니다.

체크포인트마다 자기 전처리(`normalize`)와 인코더 설정을 기록하고 있어서,
추론 코드가 멤버별로 맞는 입력을 만들어 넣습니다.

## 규칙 준수

- 입력은 `utterances[].text`만. speaker·startAt·endAt·gender·address·triage 등 미사용
- 출력은 지정된 9개 증상. 대상 외 증상은 라벨 목록에서 제거(샘플 유지)
- Training은 서울 데이터만. 외부 데이터 없음
- **공식 Validation은 학습·보정에 사용하지 않음**
- 판정 임계값 전 증상 0.5. 증상별 임계값 없음
- 상용 AI API 미사용. 오픈소스 사전학습 체크포인트만
- 도메인 적응 MLM도 Training 전사만 사용(라벨 미사용)

## 코드 전체

학습 코드, 분석 스크립트, 실험 기록은 아래에 있습니다.

https://github.com/badyeun/DCC_AIdeas/tree/inho
