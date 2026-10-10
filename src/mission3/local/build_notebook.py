"""Build the submission notebook, with the real training logs embedded.

제출 요건: "학습 코드 파일 (학습 로그가 포함된 .ipynb)".

ver1~ver8 노트북은 전부 셀 출력이 비어 있어 이 요건을 못 채웁니다. 여기서는
실제 학습이 남긴 `final_train_log.md`를 읽어 출력 셀로 심으므로, 노트북을 다시
실행하지 않아도 로그가 보입니다.

Run:  python src/mission3/local/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "src" / "mission3" / "local"
RESULTS = ROOT / "results" / "mission3" / "local"
OUT = LOCAL / "mission3_final.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(source, output_text=""):
    outputs = []
    if output_text:
        outputs = [{"output_type": "stream", "name": "stdout",
                    "text": output_text.splitlines(True)}]
    return {"cell_type": "code", "execution_count": 1, "metadata": {},
            "outputs": outputs, "source": source.strip().splitlines(True)}


def read(path, limit=None):
    if not path.exists():
        return f"(파일 없음: {path.name})"
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit and len(text) > limit:
        head = text[: limit // 2]
        tail = text[-limit // 2:]
        return f"{head}\n\n... (중략) ...\n\n{tail}"
    return text


cells = [
    md("""
# Mission 3 — 환자 증상 인식 (최종 제출)

119 신고 대화의 전사 텍스트만 읽고 9개 증상을 예측합니다. 평가 지표는 증상별
F1의 평균(Macro F1)이고, 판정 임계값은 **전 증상 0.5**입니다.

이 노트북은 최종 모델의 **실제 학습 로그**를 담고 있습니다. 아래 출력은
`train_local.py`가 GPU에서 남긴 기록을 그대로 옮긴 것입니다.
"""),
    md("""
## 1. 어떻게 이 구성에 도달했는가

약 45회의 짝비교를 Training 내부 holdout(5,840통화)에서 수행했습니다.
공식 Validation은 선택 과정에서 한 번도 사용하지 않았습니다.

### 효과가 있었던 것

| 방법 | Δ Macro F1 | 검증 방식 |
|---|---:|---|
| **앙상블 (확률 단순 평균)** | **+0.0089** | 크기별 부분집합 150개의 평균 |
| **전체 데이터 학습** | **+0.0010** | 데이터 크기 곡선, 최상단 구간 기울기 |
| **이질 구성 (계열 섞기)** | **+0.0016** | 무작위 조합 200개 중 200개 승 |
| **TF-IDF 10% 혼합** | **+0.0012** | 반분 검증 80회, 91% 승 |

앙상블은 **10개에서 포화**합니다 (10개 0.6462 / 16개 0.6463 / 23개 0.6462).
그래서 멤버를 늘리는 대신 **계열을 섞었습니다**. 같은 6개라도 한 레시피로
채우면 0.6448, 계열마다 하나씩 뽑으면 0.6464입니다. 멤버 단독 품질은
같고(0.6374 vs 0.6381) 상호 불일치만 높습니다(0.0295 → 0.0358).

### 효과가 없었던 것

| 축 | 시도 | 결과 |
|---|---|---|
| 전처리 | 텍스트 정규화 | −0.0003 |
| 모델 | 백본 5종 | 전부 klue/roberta-base 이하 |
| 모델 | 구어체 도메인 적응 (MLM) | −0.0008 (6회) |
| 라벨 | 자기 증류 · 라벨 스무딩 · 증상별 손실 가중치 | 전부 음수 |
| 구조 | 로짓 혼합층 · 조합/개수 보조과제 | −0.0034 ~ −0.0073 |
| 분포 | 희귀 증상 오버샘플링 | −0.0109 |
| 손실 | ASL · Focal · 소프트 macro-F1 | 전부 음수 |
| 결합 | 스태킹 · 증상별 혼합비율 | 전부 음수 |
| 후처리 | 최소 1개 증상 강제 | −0.0008 |

### 측정의 한계를 함께 적습니다

| 각 설정의 실행 수 | 탐지 가능 최소 효과 |
|---:|---:|
| 3 (주로 사용) | 0.0044 |
| 40 | 0.0012 |

잡음은 seed(표준편차 0.0023)와 평가(0.0015)에서 옵니다. **"효과 없음" 중
상당수는 "이 설계로 못 쟀다"입니다.** TF-IDF와 이질 구성만 별도 검증을
거쳤기에 확정이라 적었습니다.
"""),
    md("""
## 2. 데이터 분석이 알려준 것

### 모델은 전사에서 읽을 수 있는 것을 거의 다 읽습니다

놓친 정답(FN)과 잘못 짚은 것(FP)에 전사 근거가 있는지 셌습니다.

| 증상 | FN 수 | FN에 표현 있음 | FP 수 | FP에 표현 있음 |
|---|---:|---:|---:|---:|
| 구토 | 404 | **1.5%** | 280 | 50.4% |
| 두통 | 309 | 10.4% | 240 | 56.2% |
| 호흡곤란 | 210 | 33.3% | 313 | **86.9%** |

**양쪽 다 어노테이션 행동입니다.** 구토는 놓친 404건 중 6건에만 구토 표현이
있습니다 — 읽고도 못 맞힌 게 아니라 읽을 게 없습니다. 반대로 호흡곤란 FP의
87%에 "숨"이 나옵니다.

### 오심만 질적으로 다릅니다

표현이 있는 통화 안에서 라벨 유무를 구분할 수 있는가(조건부 AUC):

| 열상 | 고열 | 복통 | 호흡곤란 | 전신쇠약 | 두통 | 어지러움 | 구토 | **오심** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.980 | 0.940 | 0.897 | 0.871 | 0.842 | 0.777 | 0.774 | 0.726 | **0.567** |

오심 전체 AUC 0.777은 거의 전부 "이게 구토·어지러움 통화인가"를 맞히는 데서
나옵니다. 그 안에서 오심이 추가로 붙었는지는 **동전던지기**입니다.
"메스꺼워요"가 전사에 또박또박 있어도 오심 라벨은 **63%만** 붙습니다.

오심이 붙은 통화와 안 붙은 통화를 가르는 단어를 가정 없이 채굴해도,
올라오는 것은 `어지럽고`(공동출현 재확인)와 `노원구`·`위치` 같은 주소·상투어
입니다 — 신호가 없을 때 나오는 모습입니다.

### 임계값 0.5 고정의 비용은 +0.0062

증상별 임계값을 자유롭게 골랐다면 0.6441 → 0.6503입니다. 단조 보정은 순위를
못 바꾸므로 **확률 보정으로 얻을 수 있는 것도 이게 전부**입니다.
"""),
    md("## 3. 모델 코드"),
    code(read(LOCAL / "train_local.py", 14000)),
    md("""
## 4. 최종 학습 설정 — 한 레시피가 아니라 일곱 계열

앙상블은 10개에서 포화하므로(10개 0.6462 / 16개 0.6463 / 23개 0.6462) 멤버를
늘리는 길은 닫혀 있습니다. 남은 지렛대는 **계열 수**였습니다.

| 계열 | 멤버 | 무엇이 다른가 |
|---|---:|---|
| base mean lr2e-5 | 2 | 기준 레시피 |
| base attn lr2e-5 | 2 | 증상별 attention 헤드 |
| base mean lr3e-5 | 1 | 학습률 |
| base attn lr3e-5 | 1 | 학습률 + 헤드 |
| 정규화 텍스트 | 2 | 동의어 통합 + 필러 제거 |
| 도메인 적응 | 2 | 전사 29,200개로 MLM 3 epoch 후 fine-tune |
| CLS 풀링 | 1 | `[CLS]` 토큰 사용 |

공통: 3 epoch · `pos_weight = clip(sqrt(음성/양성), 1, 5)` · batch 16 × accum 2 ·
AdamW · bf16 · Training 29,200개 전부 · **확률 단순 평균 + TF-IDF 10%** ·
**판정 임계값 전 증상 0.5** · EMA 없음 · 후처리 없음.

**전체 데이터로 학습하므로 로컬 검증 세트가 없습니다.** 아래 로그의 Macro F1은
학습에 포함된 통화로 잰 값이라 일반화 성능이 아니며, 수렴 확인용입니다.
구성은 모두 holdout 실험에서 먼저 확정했습니다 — 같은 계열 구성을 holdout
학습 모델로 조립한 대리값이 **0.64666**이고, 한 레시피로만 채운 구성은
0.64484입니다.
"""),
    md("## 5. 실제 학습 로그"),
    code("# 아래는 학습 중 GPU가 남긴 실제 출력입니다. 제출 모델은 두 번의\n"
         "# 학습에서 나옵니다 — 기존 레시피 4개와 계열을 넓힌 7개.\n"
         "for f in ('final_train_log.md', 'final_diverse_log.md'):\n"
         "    print(open(f'results/mission3/local/{f}', encoding='utf-8').read())",
         read(RESULTS / "final_train_log.md", 9000)
         + "\n\n========== 계열 확장 학습 ==========\n\n"
         + read(RESULTS / "final_diverse_log.md", 11000)),
    md("""
## 6. 추론

제출 규격(출제문제 11쪽)에 맞춘 한 줄 실행입니다.

```bash
python inference.py --audio_dir {wav folder} --label_dir {json folder} \\
    --ckpt_path {checkpoint folder} --output ./outputs/mission3.csv
```

Mission 3은 음성을 읽지 않지만 `--audio_dir`를 받습니다. `--ckpt_path`에
폴더를 주면 안의 모든 체크포인트를 앙상블합니다.
"""),
    code(read(LOCAL / "inference.py", 9000)),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"작성 완료: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes, 셀 {len(cells)}개)")
print(f"  학습 로그 포함: {(RESULTS / 'final_train_log.md').exists()}")
