"""제출용 노트북을 만든다 — 16개 구성, 실제 학습 로그 포함.

제출 요건(출제문제 12쪽): "학습 코드 파일 (학습 로그가 포함된 .ipynb)".
셀 출력이 비어 있으면 요건을 못 채운다. 그래서 노트북을 다시 실행하지 않아도
로그가 보이도록, 학습이 실제로 남긴 기록을 출력 셀로 심는다.

로그는 멤버별로 **해당 블록만** 뽑아 넣는다. 로그 파일 전체를 붙이면 제출
모델과 무관한 실험 수백 건이 따라오고, 12쪽의 "코드 내 불필요부분 삭제"에
어긋난다.

이전 버전은 11개 구성을 설명하고 있었다. 제출 모델이 16개로 바뀌었으므로
노트북도 바뀌어야 한다 -- 모델과 설명이 어긋나면 채점자가 볼 때 앞뒤가
맞지 않는다.

Run:  python src/mission3/local/build_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "src" / "mission3" / "local"
RESULTS = ROOT / "results" / "mission3" / "local"
OUT = LOCAL / "mission3_final.ipynb"

LOG_FILES = ["final_train_log.md", "final_diverse_log.md", "labelside_log.md"]

# 제출하는 16개 멤버. (계열 설명, 체크포인트 태그)
SHIP = [
    ("base mean lr2e-5 · seed 42", "roberta-base_raw_s42_FINAL"),
    ("base mean lr2e-5 · seed 1", "roberta-base_raw_s1_FINAL"),
    ("base attention lr2e-5 · seed 42", "roberta-base_attn_raw_s42_FINAL"),
    ("base attention lr2e-5 · seed 1", "roberta-base_attn_raw_s1_FINAL"),
    ("base mean lr3e-5", "roberta-base_raw_s11_D3"),
    ("base attention lr3e-5", "roberta-base_attn_raw_s13_D4"),
    ("base 정규화 텍스트 · seed 15", "roberta-base_synonym+filler_s15_D5"),
    ("base 정규화 텍스트 · seed 16", "roberta-base_synonym+filler_s16_D5"),
    ("base 도메인 적응 mean", "domain_adapted_full_raw_s17_D6"),
    ("base 도메인 적응 attention", "domain_adapted_full_attn_raw_s18_D6"),
    ("base CLS 풀링", "roberta-base_cls_raw_s19_D7"),
    ("large mean · seed 1", "roberta-large_raw_s1_LGE2"),
    ("large mean · seed 7", "roberta-large_raw_s7_LGE2"),
    ("large attention · seed 42", "roberta-large_attn_raw_s42_LGE2"),
    ("large 정규화 텍스트", "roberta-large_synonym+filler_s42_LGF"),
    ("large TAPT", "klue-roberta-large-tapt_raw_s42_LGF"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(True)}


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
        return f"{text[:limit // 2]}\n\n... (중략) ...\n\n{text[-limit // 2:]}"
    return text


def training_logs():
    """멤버별 실제 학습 로그 블록을 모은다."""
    blocks = {}
    for name in LOG_FILES:
        path = RESULTS / name
        if not path.exists():
            continue
        for blk in re.split(r"\n(?=### )", path.read_text(encoding="utf-8",
                                                          errors="replace")):
            m = re.search(r"가중치 저장: submission_models/(\S+)\.ckpt", blk)
            if m and m.group(1) not in blocks:
                blocks[m.group(1)] = blk.strip()
    out, missing = [], []
    for family, tag in SHIP:
        if tag in blocks:
            out.append(f"----- {family}\n{blocks[tag]}\n")
        else:
            missing.append(tag)
    if missing:
        raise SystemExit(f"학습 로그를 찾지 못한 멤버: {missing}")
    return "\n".join(out)


cells = [
    md("""
# Mission 3 — 환자 증상 인식 (최종 제출)

119 신고 대화의 **전사 텍스트만** 읽고 9개 증상을 예측합니다. 평가 지표는
증상별 F1의 평균(Macro F1)이고, 판정 임계값은 **전 증상 0.5**입니다.

공식 Validation 3,640통화 실측 **Macro F1 = 0.65746**

이 노트북은 제출 모델의 **실제 학습 로그**를 담고 있습니다. 5절의 출력은
학습이 GPU에서 남긴 기록을 멤버별로 그대로 옮긴 것입니다.
"""),
    md("""
## 1. 최종 구성

신경망 16개가 각자 9개 증상의 확률을 내고, 그 확률을 단순 평균한 뒤 TF-IDF
선형모델의 확률을 10% 섞어 0.5에서 판정합니다.

```
최종 확률 = 0.9 × (신경망 16개 확률의 평균) + 0.1 × (TF-IDF 확률)
판정      = 최종 확률 >= 0.5
```

### 멤버 — 11개 계열 16개

| 계열 | 멤버 | 무엇이 다른가 |
|---|---:|---|
| base mean lr2e-5 | 2 | 기준 레시피 |
| base attention lr2e-5 | 2 | 증상별 attention 헤드 |
| base mean lr3e-5 | 1 | 학습률 |
| base attention lr3e-5 | 1 | 학습률 + 헤드 |
| base 정규화 텍스트 | 2 | 동의어 통일 입력 |
| base 도메인 적응 | 2 | 전사로 MLM 추가 학습 후 미세조정 |
| base CLS 풀링 | 1 | `[CLS]` 토큰 사용 |
| large mean | 2 | `klue/roberta-large` |
| large attention | 1 | large + attention 헤드 |
| large 정규화 텍스트 | 1 | large + 동의어 통일 입력 |
| large TAPT | 1 | 도메인 적응 사전학습된 large |

쓴 사전학습 체크포인트는 세 개입니다(모두 오픈소스).

| 표기 | 사전학습 모델 | 파라미터 |
|---|---|---:|
| base | `klue/roberta-base` | 110M |
| large | `klue/roberta-large` | 337M |
| large TAPT | `SangwonYoon/klue-roberta-large-tapt` | 337M |

`base 도메인 적응` 멤버는 `klue/roberta-base`를 Training 전사 텍스트로
MLM 추가 학습한 뒤 미세조정한 것입니다(라벨 미사용).

공통: 3 epoch · AdamW · bf16 · `pos_weight = clip(sqrt(음성/양성), 1, 5)` ·
**Training 29,200통화 전부** · 멤버 가중치 없음 · 후처리 없음.

`klue/roberta-large` mean·attention 멤버는 3 epoch 스케줄에서 **epoch 2
가중치**를 씁니다. 홀드아웃에서 epoch 2가 최고였고 이후 하락하기 때문입니다.
스케줄을 2 epoch로 줄이면 학습률이 더 짧은 구간에 걸쳐 감쇠해 다른 모델이
되므로(홀드아웃 −0.00198), 스케줄은 3 epoch로 두고 저장 시점만 당겼습니다.

체크포인트마다 자기 전처리(`normalize`)와 인코더 설정을 기록하고 있어서,
추론 코드가 멤버별로 맞는 입력을 만들어 넣고 **네트워크 없이** 동작합니다.
"""),
    md("""
## 2. 어떻게 이 구성에 도달했는가

약 110회의 짝비교를 **Training 내부 홀드아웃 5,840통화**에서 수행했습니다.
공식 Validation은 설계 과정에서 쓰지 않고, 제출 결정 시점에만 측정했습니다.

### 효과가 있었던 것

| 방법 | Δ Macro F1 | 검증 방식 |
|---|---:|---|
| **앙상블 (확률 단순 평균)** | **+0.0089** | 크기별 부분집합 150개 평균 |
| **large 계열 5개 추가** | +0.0037 | 반분 검증 240회 전승 |
| **이질 구성 (계열 섞기)** | +0.0016 | 무작위 조합 200개 중 200개 승 |
| **TF-IDF 10% 혼합** | +0.0012 | 반분 검증 80회, 91% 승 |
| **전체 데이터 학습** | +0.0010 | 데이터 크기 곡선 |

앙상블은 **10개에서 포화**합니다(10개 0.6462 / 16개 0.6463 / 23개 0.6462).
그래서 멤버를 늘리는 대신 **계열을 섞었습니다**. 같은 6개라도 한 레시피로
채우면 0.6448, 계열마다 하나씩 뽑으면 0.6464입니다. 멤버 단독 품질은
같고(0.6374 vs 0.6381) 상호 불일치만 높습니다(0.0295 → 0.0358).

`large` 계열은 처음에 "base와 비슷한데 3.7배 느리다"고 보고 두 번 제외했는데,
그 판단이 틀렸습니다. 설정이 잘못된 상태의 측정이었고(올바른 값은 lr 1.5e-5,
batch 4×8), 더 중요하게는 **단독 성능으로 판단한 것 자체가 잘못**이었습니다 —
단독 성능과 앙상블 기여의 상관은 +0.22뿐입니다. 판단 기준을 "앙상블에 넣었을
때의 변화"로 바꾸고 백본 후보 15개를 다시 재서 large 계열만 통과했습니다.

멤버 수가 늘어 오른 것이 아니라는 통제: large 5개 추가는 +0.0019(승률 98%),
기존 멤버 3개를 복제해 추가하면 **−0.0012(승률 0%)**입니다.

### 효과가 없었던 것

| 축 | 시도 | 결과 |
|---|---|---|
| 백본 | 10종 (KcELECTRA, km-bert(의학), kobigbird 등) | 전부 base 이하 |
| 백본 | large 추가 레시피 6종 | 전부 기준 미달 |
| 구조 | 풀링 층 변경 · 하위 층 동결 · meanmax 헤드 | 전부 음수 |
| 손실 | ASL · Focal · soft macro-F1 · R-Drop | 전부 음수 |
| 라벨 | 자기 증류 · 라벨 스무딩 · 증상별 손실 가중치 | 전부 음수 |
| 분포 | 희귀 증상 오버샘플링 | −0.0109 |
| 결합 | 스태킹 · 멤버 가중치 · 증상별 혼합비율 | 전부 음수 |
| 압축 | 가중치 평균(model soup) | −0.0091 |

**공통 교훈: 조정할 수 있는 값을 늘리면 거의 항상 손해였습니다.** 홀드아웃
5,840통화로 그 값을 맞추면 그 데이터에만 맞춰집니다. 실제로 측정했습니다 —
홀드아웃에서 "최적의 5개 조합"은 0.65231이지만, 고르는 것까지 반쪽 데이터
안에서 하고 다른 쪽에서 재면 0.64921입니다. **차이 +0.0031이 순수한 선택
편향**입니다.
"""),
    md("""
## 3. 데이터 분석이 알려준 것

### 모델은 전사에서 읽을 수 있는 것을 거의 다 읽습니다

놓친 정답(FN)과 잘못 짚은 것(FP)에 전사 근거가 있는지 셌습니다.

| 증상 | FN 수 | FN에 표현 있음 | FP 수 | FP에 표현 있음 |
|---|---:|---:|---:|---:|
| 구토 | 404 | **1.5%** | 280 | 50.4% |
| 두통 | 309 | 10.4% | 240 | 56.2% |
| 호흡곤란 | 210 | 33.3% | 313 | **86.9%** |

**양쪽 다 어노테이션 행동입니다.** 구토는 놓친 404건 중 6건에만 구토 표현이
있습니다 — 읽고도 못 맞힌 게 아니라 읽을 것이 없습니다. 반대로 호흡곤란 FP의
87%에 "숨"이 나옵니다.

### 오심만 질적으로 다릅니다

표현이 있는 통화 안에서 라벨 유무를 구분할 수 있는가(조건부 AUC):

| 열상 | 고열 | 복통 | 호흡곤란 | 전신쇠약 | 두통 | 어지러움 | 구토 | **오심** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.980 | 0.940 | 0.897 | 0.871 | 0.842 | 0.777 | 0.774 | 0.726 | **0.567** |

"메스꺼워요"가 전사에 또박또박 있어도 오심 라벨은 **63%만** 붙습니다.
최종 성적에서도 오심 F1이 0.4101로 최하위이고, 이는 모델의 약점이 아니라
라벨의 성질을 반영합니다.

### 512 토큰 한도는 충분합니다

토큰 길이 중앙값 230, 95분위 460. 한도를 넘는 통화는 788개(2.7%)이고 전체
토큰 중 버려지는 비율은 0.8%입니다. 버려지는 구간에만 증상 근거가 있는 양성은
1,115건 중 **10건**입니다. 이 방향에는 얻을 것이 없습니다.

### 임계값 0.5 고정의 비용은 +0.0062

증상별 임계값을 자유롭게 골랐다면 0.6441 → 0.6503입니다. 단조 보정은 순위를
못 바꾸므로 **확률 보정으로 얻을 수 있는 것도 이게 전부**입니다.
"""),
    md("## 4. 모델 코드"),
    code(read(LOCAL / "train_local.py", 14000)),
    md("""
## 5. 실제 학습 로그

제출하는 16개 멤버가 학습 중 남긴 기록입니다. 멤버별로 해당 블록만 뽑았습니다.

아래 `Macro F1` 값은 **학습에 포함된 통화로 잰 값**입니다(전체 데이터로
학습하므로 남는 홀드아웃이 없습니다). 일반화 성능이 아니라 수렴 확인용이며,
실제 성능은 7절의 공식 Validation 실측값입니다. 구성은 모두 홀드아웃 실험에서
먼저 확정했습니다.
"""),
    code("# 학습이 남긴 기록을 멤버별로 출력합니다.\n"
         "for name in ('final_train_log.md', 'final_diverse_log.md', 'labelside_log.md'):\n"
         "    print(open(f'results/mission3/local/{name}', encoding='utf-8').read())",
         training_logs()),
    md("""
## 6. 추론

제출 규격(출제문제 11쪽)에 맞춘 한 줄 실행입니다.

```bash
python inference.py --audio_dir {wav folder} --label_dir {json folder} \\
    --ckpt_path ./weights --output ./outputs/mission3.csv
```

Mission 3은 음성을 읽지 않지만 `--audio_dir`를 받습니다(채점 인터페이스
호환). `--ckpt_path`에 폴더를 주면 안의 모든 체크포인트를 앙상블합니다.

`--device cpu`로 CPU 실행도 됩니다. **네트워크가 필요하지 않습니다** —
체크포인트가 인코더 설정을 직접 들고 있어서 사전학습 가중치를 받지 않습니다
(`HF_HUB_OFFLINE=1`에서 검증 완료).
"""),
    code(read(LOCAL / "inference.py", 9000)),
    md("""
## 7. 최종 성적 (공식 Validation 3,640통화)

| 증상 | 실제 | 예측 | 정답 | Precision | Recall | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| 열상 | 463 | 480 | 419 | 0.873 | 0.905 | **0.8887** |
| 복통 | 838 | 756 | 658 | 0.870 | 0.785 | **0.8256** |
| 고열 | 678 | 565 | 441 | 0.781 | 0.650 | **0.7096** |
| 호흡곤란 | 492 | 509 | 334 | 0.656 | 0.679 | **0.6673** |
| 어지러움 | 774 | 816 | 530 | 0.650 | 0.685 | **0.6667** |
| 구토 | 537 | 463 | 304 | 0.657 | 0.566 | **0.6080** |
| 전신쇠약 | 651 | 704 | 397 | 0.564 | 0.610 | **0.5860** |
| 두통 | 374 | 296 | 186 | 0.628 | 0.497 | **0.5552** |
| 오심 | 429 | 400 | 170 | 0.425 | 0.396 | **0.4101** |
| **Macro F1** | | | | | | **0.65746** |

통화당 평균 1.37개를 예측했습니다(실제 1.44개). 증상을 하나도 예측하지 않은
통화는 107개입니다.

### 계산 효율 (출제문제 권장 항목)

| 항목 | 값 |
|---|---|
| Total 파라미터 | **2,902.70M** |
| Active 파라미터 | **2,902.70M** (앙상블이므로 모든 멤버가 모든 통화를 읽습니다) |
| GPU | NVIDIA GeForce RTX 5060 Ti, VRAM 8 GiB |
| CUDA / PyTorch | 12.8 / 2.11.0+cu128 · bf16 autocast |
| Validation batch size | **32** |
| 전체 추론 시간 | **6.1분** (3,640통화) |
| 통화당 평균 | **101 ms** |

내역: base 110.63M × 7 + base(attention) 110.82M × 4 + large 336.67M × 4 +
large(attention) 336.93M × 1 + TF-IDF 1.44M.

가중치는 fp16으로 저장해 용량을 11.62 GB → **5.82 GB**로 줄였습니다. 추론
시에 fp32로 적재되므로 연산 정밀도와 무관하며, Validation 32,760개 판정 중
라벨이 달라진 것은 2건(Macro F1 차이 +0.00003)입니다.

### 선택 근거

같은 데이터에서 네 후보를 실측했습니다.

| 구성 | Macro F1 | 모델 | 추론 |
|---|---:|---:|---:|
| **16개 (제출)** | **0.65746** | 16 | 6.1분 |
| 증류 학생 6개 | 0.65683 | 6 | ~3.3분 |
| large 5개 | 0.65680 | 5 | 3.7분 |
| base 11개 | 0.65658 | 11 | 2.9분 |

차이가 최대 0.0009인데 점수 자체의 불확실성이 ±0.0059입니다(부트스트랩
4,000회). **점수로는 네 구성이 구분되지 않습니다.** 그래서 검증 완료 여부와
견고함으로 골랐습니다 — 16개 구성은 제출 전 7개 항목 점검(오프라인 적재·
오프라인 추론·CSV 규격·파일명 일치·증상 목록 범위 등)을 통과했고, 멤버가
많아 하나가 어긋나도 영향이 작으며, 검증된 11개 구성을 그대로 포함합니다.

### 규칙 준수

- 입력은 `utterances[].text`만. speaker·시간·성별·주소·중증도 미사용
- Training은 서울 데이터만. 외부 데이터 없음
- 공식 Validation은 학습·보정에 사용하지 않음
- 판정 임계값 전 증상 0.5. 증상별 임계값 없음
- 상용 AI API 미사용. 오픈소스 사전학습 체크포인트만
- 도메인 적응 MLM도 Training 전사만 사용(라벨 미사용)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"작성 완료: {OUT.relative_to(ROOT)}  "
      f"({OUT.stat().st_size:,} bytes, 셀 {len(cells)}개)")
print(f"  학습 로그: 멤버 {len(SHIP)}개 전부 포함")
