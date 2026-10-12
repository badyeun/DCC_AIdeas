"""Mission 2 학습 노트북(학습 로그 포함)을 만들고 실행해 출력까지 저장한다.

제출 규칙상 학습 코드는 실행 로그가 남은 .ipynb여야 한다. 최종 10개 트랜스포머를 노트북에서 전부 다시
학습하면 로컬 CPU로 약 12시간이 걸려, 이 노트북은 (1) 학습 코드를 그대로 불러 1 epoch를 실제로 다시 돌려
코드가 동작함을 보이고, (2) 원래 학습 실행의 로그 파일을 그대로 읽어 보여 주며, (3) 저장된 예측과
제출 체크포인트로 Validation 정확도를 다시 계산해 학습 결과와 제출 파일이 같음을 확인한다.

사용: python make_training_notebook.py --out Mission2_training.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

CELLS = [
    ("md", """# Mission 2 — 통화 단위 트랜스포머 학습 기록 (EXP-011)

**최종 제출 모델**: 통화 단위 트랜스포머 10개의 확률 평균 (`mission2_final10.ckpt`, 서울 Validation Accuracy **0.9277**).

**구성**
1. 발화마다 `startAt`~`endAt` 구간을 잘라 **ECAPA 화자 임베딩**(192, `speechbrain/spkrec-ecapa-voxceleb`, 가중치 고정)과
   **MFCC 통계**(261: MFCC·Δ·Δ² + RMS·ZCR·spectral 통계 + 발화 길이)를 만든다.
2. 통화 하나의 발화 전체를 한 시퀀스로 넣는 트랜스포머(128차원 3층, **위치 인코딩 없음**)가 발화마다 신고자(1) 확률을 낸다.
   위치 인코딩이 없어 발화 순서를 알 수 없다(순서를 섞어도 결과가 같음을 시험으로 확인).
3. 설정을 조금씩 바꾼 10개(시드, Train 전체 학습, 큰 모델, 같은 화자 보조 과제)의 확률을 단순 평균한다.

**규칙 준수**
- 학습: 서울 Training 29,200통화만. Validation 3,640통화는 학습(역전파)에 쓰지 않았다. 모델 선택은 Training 내부 holdout(통화 10%).
- annotation: 입력에는 `startAt`, `endAt`만 사용. `text`, 발화 순서는 쓰지 않는다. `speaker`는 학습 정답으로만 사용.
- 사전학습 모델은 공개 모델(ECAPA)만, 상용 API 없음.

**이 노트북에 대해**: 10개를 전부 다시 학습하면 로컬 CPU로 약 12시간이 걸려 다시 돌리지 않았다. 대신
① 같은 학습 코드로 1 epoch를 실제로 다시 실행하고, ② 원래 학습 실행의 로그 파일을 그대로 읽어 보여 주며,
③ 저장된 예측과 제출 체크포인트로 Validation 정확도를 다시 계산해 학습 결과와 제출 파일이 같은지 확인한다."""),
    ("code", """import sys, json, time, subprocess, argparse
from pathlib import Path
import numpy as np

SRC = Path(r"C:/유니짜장/DCC_Aideas/DCC_AIdeas/src/mission2")
M2 = Path(r"C:/유니짜장/DCC_Aideas/mission2_local")
RUNS = M2 / "runs"                                   # 원래 학습 실행 결과 (로그, 모델, Validation 확률)
CKPT = M2 / "final" / "mission2_final10.ckpt"        # Colab에서 inference.py로 검증한 제출 체크포인트
WORK = M2 / "work_notebook"                          # 이 노트북 전용 작업 폴더
sys.path.insert(0, str(SRC))
from train_call_model import load_split, make_call
FINAL = ["A0_base", "A1_seed7", "A3_seed11", "F0_full", "A2_wide",
         "C0_pair", "C1_pair_seed7", "C2_pair_w1", "C3_pair_seed11", "F1_full_pair"]"""),
    ("md", """## 1. 데이터 정렬 검증
ECAPA 캐시와 MFCC 캐시(둘 다 Colab에서 `train_speaker_embed.py`, `train_mfcc_rf.py`로 만든 발화별 캐시)를 읽어
같은 발화 순서인지 확인한다. 정답(speaker)과 통화 경계가 다르면 `load_split`이 멈춘다."""),
    ("code", """paths = argparse.Namespace(embed_cache_dir=M2 / "mission2_embed_cache", mfcc_cache_dir=M2 / "mission2_cache",
                           work_dir=WORK, second_cache_dir=None)
train = load_split(paths, "train")
val = load_split(paths, "val")
for name, d in (("Train", train), ("Validation", val)):
    print(f"{name}: 통화 {len(d['counts']):,}, 발화 {len(d['labels']):,}, 신고자 비율 {d['labels'].mean():.4f}")"""),
    ("md", """## 2. 학습 코드 실행 확인 (1 epoch 재실행)
최종 모델과 같은 학습 코드(`train_call_model.py`)를 기본 설정으로 1 epoch만 실제로 실행한다."""),
    ("code", """start = time.time()
result = subprocess.run([sys.executable, str(SRC / "train_call_model.py"),
                         "--embed_cache_dir", str(M2 / "mission2_embed_cache"), "--mfcc_cache_dir", str(M2 / "mission2_cache"),
                         "--work_dir", str(WORK), "--out", str(WORK / "demo_1epoch.pt"), "--epochs", "1", "--threads", "8"],
                        capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
print("\\n".join(line for line in result.stdout.splitlines() if "Warning" not in line) or result.stderr[-3000:])
print(f"소요 {time.time() - start:.0f}초")"""),
    ("md", """## 3. 최종 10개 학습 로그 (원래 실행 로그 파일)
각 모델은 로컬 PC(CPU 16코어)에서 `train_call_model.py`로 20 epoch 학습했다(모델당 50분~2시간).
아래는 실행 때 저장된 로그 파일을 그대로 읽은 것이다."""),
    ("code", """rows = {json.loads(l)["name"]: json.loads(l) for l in (RUNS / "sweep.jsonl").read_text(encoding="utf-8").splitlines()}
for name in FINAL:
    print("=" * 30, name, "| 인자:", rows[name]["args"])
    for line in (RUNS / f"{name}.log").read_text(encoding="utf-8").splitlines():
        if any(key in line for key in ("모델:", "[epoch", "최고 epoch", "Validation Accuracy (트랜스포머)", "저장:")):
            print(line)"""),
    ("md", """## 4. 앙상블 정확도 (저장된 Validation 확률로 계산)
섞는 규칙은 결과를 보기 전에 정했다: 트랜스포머끼리 단순 평균."""),
    ("code", """labels = val["labels"]
probs = {name: np.load(RUNS / f"{name}.val_probability.npy") for name in FINAL}
acc = lambda p: float(((p >= .5) == labels).mean())
for name in FINAL:
    print(f"{name:<16} 단독 Validation {acc(probs[name]):.4f}")
three = np.mean([probs[n] for n in FINAL[:3]], axis=0)
final = np.mean([probs[n] for n in FINAL], axis=0)
print(f"\\n3개 평균 (A0·A1·A3): {acc(three):.4f}")
print(f"최종 10개 평균:      {acc(final):.4f}")
for label, value in (("119대원(0)", 0), ("신고자(1)", 1)):
    print(f"  {label} recall {((final >= .5)[labels == value] == value).mean():.4f}")"""),
    ("md", """## 5. 제출 체크포인트 재현 확인
`pack_call_model.py`로 묶은 제출 체크포인트 안의 10개 트랜스포머로 Validation을 다시 계산해, 위의 학습 결과(4번)와
같은지 확인한다. 추론 코드와 같은 함수(`call_model.load_transformers`, `call_inputs`)를 쓴다."""),
    ("code", """import joblib
from call_model import load_transformers, predict_calls
checkpoint = joblib.load(CKPT)
print("체크포인트:", checkpoint["model_type"], "| 트랜스포머", len(checkpoint["transformers"]), "개 | blend", checkpoint["blend"],
      "| 기존 파이프라인 포함:", checkpoint["base"] is not None)
loaded = load_transformers(checkpoint["transformers"])
outputs = []
for model, entry in loaded:
    inputs = [make_call(val, c, entry["scale"], None, None, bool(entry.get("context", True))) for c in range(len(val["counts"]))]
    outputs.append(np.concatenate(predict_calls(model, inputs)))
reproduced = np.mean(outputs, axis=0)
print(f"체크포인트로 다시 계산한 Validation 정확도: {acc(reproduced):.4f}")
print(f"학습 때 확률과 최대 차이: {np.abs(reproduced - final).max():.2e}, 판정이 다른 발화: {int(((reproduced >= .5) != (final >= .5)).sum())}")"""),
    ("md", """## 6. 실데이터 추론 검증 (Colab T4, 공식 명령)

`python inference.py --audio_dir <Validation WAV> --label_dir <Validation JSON> --ckpt_path mission2_final10.ckpt --output ./outputs/mission2.csv`

| 체크포인트 | 행 수 | Accuracy | 추론 시간 |
| --- | ---: | ---: | ---: |
| 3개 평균 (A0·A1·A3) | 111,947 | 0.9263 | — |
| **최종 10개 평균** | **111,947** | **0.9277** | **2시간 5분** |

두 값 모두 위 4번에서 학습 때 확률로 계산한 값과 같다.

## 결과 요약 (서울 Validation 발화 111,947개)

| 단계 | Accuracy |
| --- | ---: |
| EXP-001 MFCC + RandomForest (발화 하나만) | 0.8372 |
| EXP-003 ECAPA 임베딩 + 2단계 LightGBM | 0.9009 |
| EXP-009 같은 화자 판별기 2단계 (이전 최고) | 0.9117 |
| EXP-011 통화 단위 트랜스포머 단독 | 0.9219~0.9252 |
| **최종: 트랜스포머 10개 평균** | **0.9277** |"""),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
                      for kind, text in CELLS]
    ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(args.out.parent)}})
    nbformat.write(notebook, args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
