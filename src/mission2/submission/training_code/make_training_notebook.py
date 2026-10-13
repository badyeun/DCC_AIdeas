"""제출용 학습 노트북(Mission2_training.ipynb)을 만들고 실행해 출력(학습 로그)까지 저장한다.

python make_training_notebook.py --out <제출 폴더>/Mission2_training.ipynb --verified_accuracy 0.9277 --verified_time "2시간 5분"

최종 10개 모델을 노트북 안에서 다시 학습하면 CPU로 약 13시간이 걸려, 노트북은
(1) 학습 데이터 분석, (2) 학습 코드로 1 epoch를 실제로 다시 돌려 원래 학습 로그와 같은지 확인,
(3) 원래 학습 실행 로그(10개), (4) 저장된 Validation 확률로 앙상블 정확도, (5) 제출 체크포인트로 재현,
(6) 오류 분석을 보여 준다. 노트북은 제출 폴더 안(training_code/ 옆)에서 실행된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

INTRO = """# Mission 2 — 신고자(1) / 119대원(0) 발화 분류: 학습 기록

**최종 모델**: 통화 단위 트랜스포머 10개의 확률 평균 (`mission2_final10.ckpt`). 서울 Validation 발화 111,947개 Accuracy **{accuracy}**.

| 단계 | 코드 | 내용 |
| --- | --- | --- |
| 1. 특징 캐시 | `training_code/make_feature_cache.py` (Colab GPU) | 발화마다 `startAt`~`endAt`을 잘라 ECAPA 화자 임베딩 192차원(`speechbrain/spkrec-ecapa-voxceleb`, 가중치 고정)과 MFCC 통계 261차원을 저장 |
| 2. 학습 | `training_code/train_call_model.py` (CPU) | 통화의 발화 전체를 한 시퀀스로 넣는 트랜스포머(위치 인코딩 없음) 10개를 설정만 바꿔 학습 |
| 3. 묶기 | `training_code/pack_model.py` | 10개 가중치 + ECAPA 인코더 파일을 체크포인트 한 파일로 |
| 추론 | `inference.py` (+ `features.py`, `model.py`) | 공식 명령 한 번으로 CSV 생성 |

**데이터·annotation 사용**
- 학습: 서울 Training 29,200통화만 사용. Validation 3,640통화는 학습(역전파)과 epoch 선택에 쓰지 않았다. epoch 선택은 Training 통화 10% holdout으로 했다.
- 입력: annotation에서는 `startAt`, `endAt`(발화 자르기)만 쓴다. `text`와 발화 순서는 쓰지 않는다. `speaker`는 학습 정답으로만 쓴다.
- 모델에 위치 인코딩이 없어 발화를 어떤 차례로 넣어도 결과가 같다(발화 순서를 쓸 수 없는 구조).
- 사전학습 모델은 공개 모델(ECAPA)만 사용했고 상용 AI API는 쓰지 않는다.

**이 노트북에 대해**: 10개를 노트북 안에서 다시 학습하면 CPU로 약 13시간이 걸려 다시 돌리지 않았다. 대신 학습 코드로
1 epoch를 실제로 다시 실행해 원래 로그와 같은 값이 나오는지 확인하고(2절), 원래 학습 실행의 로그 파일을 그대로 보여 주며(3절),
저장된 결과와 제출 체크포인트로 정확도를 다시 계산한다(4~5절)."""

CELLS = [
    ("md", None),  # INTRO
    ("code", """import sys, json, time, subprocess, argparse, os
from pathlib import Path
import numpy as np

SUBMISSION = Path.cwd()                              # 이 노트북이 있는 제출 폴더
TRAINING = SUBMISSION / "training_code"
DATA = Path(r"C:/유니짜장/DCC_Aideas/mission2_local") # 특징 캐시와 원래 학습 실행 결과가 있는 폴더
RUNS = DATA / "runs"                                 # 원래 학습 실행: 로그(.log), 결과(.json), Validation 확률(.npy)
WORK = DATA / "work_notebook"                        # 이 노트북 전용 작업 폴더
sys.path[:0] = [str(SUBMISSION), str(TRAINING)]
from train_call_model import load_split, make_call
from model import CallEnsemble, predict_calls
FINAL = ["A0_base", "A1_seed7", "A3_seed11", "F0_full", "A2_wide",
         "C0_pair", "C1_pair_seed7", "C2_pair_w1", "C3_pair_seed11", "F1_full_pair"]
caches = argparse.Namespace(embed_cache_dir=DATA / "mission2_embed_cache", mfcc_cache_dir=DATA / "mission2_cache", work_dir=WORK)
train = load_split(caches, "train")
val = load_split(caches, "val")
print("캐시 정렬 확인 완료 (ECAPA와 MFCC의 정답·통화 경계가 같음)")"""),
    ("md", """## 1. 데이터 분석
발화 길이는 MFCC 특징의 마지막 값(log(1 + 길이 초))에서 되돌린다. 정답(speaker)은 분석과 학습 정답에만 쓴다."""),
    ("code", """def describe(name, data):
    labels, counts = data["labels"], data["counts"]
    seconds = np.expm1(np.asarray(data["mfcc"][:, -1], dtype=np.float64))
    call_id = np.repeat(np.arange(len(counts)), counts)
    caller_share = (np.bincount(call_id, weights=labels, minlength=len(counts)) / np.maximum(counts, 1))[counts > 0]
    print(f"[{name}] 통화 {len(counts):,} | 발화 {len(labels):,} | 신고자(1) {labels.mean():.1%} / 119대원(0) {1 - labels.mean():.1%}")
    print(f"  통화당 발화 수: 중앙값 {np.median(counts):.0f}, 평균 {counts.mean():.1f}, 90% {np.percentile(counts, 90):.0f}, 최대 {counts.max()}")
    print(f"  통화 안 신고자 발화 비율: 중앙값 {np.median(caller_share):.2f} (한 화자만 있는 통화 {np.mean((caller_share == 0) | (caller_share == 1)):.1%})")
    for value, who in ((0, "119대원"), (1, "신고자")):
        s = seconds[labels == value]
        print(f"  {who} 발화 길이(초): 중앙값 {np.median(s):.2f}, 평균 {s.mean():.2f}, 1초 미만 {np.mean(s < 1):.1%}")
describe("Training", train)
describe("Validation", val)"""),
    ("md", """## 2. 학습 코드 실행 확인 (A0_base 설정으로 1 epoch 재실행)
최종 모델 A0_base와 같은 설정(seed 42, 20 epoch 스케줄, CPU 14스레드)으로 첫 epoch만 실제로 다시 돌린다.
같은 코드라면 원래 학습 로그의 첫 epoch 줄과 loss·holdout 정확도가 같아야 한다."""),
    ("code", """start = time.time()
result = subprocess.run([sys.executable, str(TRAINING / "train_call_model.py"),
                         "--embed_cache_dir", str(caches.embed_cache_dir), "--mfcc_cache_dir", str(caches.mfcc_cache_dir),
                         "--work_dir", str(WORK / "rerun"), "--out", str(WORK / "rerun" / "A0_epoch1.pt"),
                         "--epochs", "20", "--run_epochs", "1", "--threads", "14"],
                        capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
rerun = [line for line in result.stdout.splitlines() if "Warning" not in line]
print("\\n".join(rerun) or result.stderr[-3000:])
print(f"소요 {time.time() - start:.0f}초\\n")
original = next(line for line in (RUNS / "A0_base.log").read_text(encoding="utf-8").splitlines() if line.startswith("[epoch  1]"))
again = next(line for line in rerun if line.startswith("[epoch  1]"))
print("원래 로그:", original)
print("재실행  :", again)
print("loss·holdout 정확도 일치:", original.split(" (")[0] == again.split(" (")[0])"""),
    ("md", """## 3. 최종 10개 학습 로그 (원래 실행 로그 파일)
로컬 PC(CPU 16코어)에서 `train_call_model.py`로 학습했다(모델당 1~2시간, 합계 약 13시간). 아래는 실행 때 저장된 로그 파일을 그대로 읽은 것이다.
Train 전체 학습 2개(F0, F1)는 holdout 실험에서 정한 20 epoch를 그대로 학습하고 마지막 epoch를 썼다."""),
    ("code", """sweep = {json.loads(l)["name"]: json.loads(l) for l in (RUNS / "sweep.jsonl").read_text(encoding="utf-8").splitlines()}
for name in FINAL:
    print("=" * 30, name, "| 추가 인자:", sweep[name]["args"].replace("--epochs 20 --threads 14", "").strip() or "(기본)")
    for line in (RUNS / f"{name}.log").read_text(encoding="utf-8").splitlines():
        if line.startswith(("모델:", "[epoch", "holdout이", "최고 epoch", "Validation Accuracy (트랜스포머)", "  119대원", "  신고자")):
            print(line)"""),
    ("md", """## 4. 앙상블 정확도 (학습 때 저장한 Validation 확률)
섞는 규칙은 미리 정했다: 트랜스포머끼리 가중치 없이 단순 평균, 0.5 이상이면 신고자(1)."""),
    ("code", """from sklearn.metrics import classification_report
labels = val["labels"]
probs = {name: np.load(RUNS / f"{name}.val_probability.npy") for name in FINAL}
acc = lambda p: float(((p >= .5) == labels).mean())
for name in FINAL:
    print(f"{name:<16} 단독 Validation {acc(probs[name]):.4f}")
final = np.mean([probs[n] for n in FINAL], axis=0)
print(f"\\n3개 평균 (A0·A1·A3): {acc(np.mean([probs[n] for n in FINAL[:3]], axis=0)):.4f}")
print(f"최종 10개 평균:      {acc(final):.4f}\\n")
print(classification_report(labels, (final >= .5).astype(int), target_names=["119대원(0)", "신고자(1)"], digits=4))"""),
    ("md", """## 5. 제출 체크포인트 재현 확인
`pack_model.py`로 묶은 제출 체크포인트를 추론 코드와 같은 모델 코드(`model.py`)로 불러 Validation 캐시에서 다시 계산한다.
4절의 학습 결과와 같아야 한다."""),
    ("code", """import joblib
checkpoint = joblib.load(SUBMISSION / "mission2_final10.ckpt")
print("체크포인트:", checkpoint["model_type"], "| 트랜스포머", len(checkpoint["transformers"]), "개:",
      ", ".join(entry["source"] for entry in checkpoint["transformers"]))
ensemble = CallEnsemble(checkpoint["transformers"])
outputs = []
for model, scale in ensemble.models:
    inputs = [make_call(val, call, scale) for call in range(len(val["counts"]))]
    outputs.append(np.concatenate(predict_calls(model, inputs)))
reproduced = np.mean(outputs, axis=0)
print(f"체크포인트로 다시 계산한 Validation 정확도: {acc(reproduced):.4f}")
print(f"학습 때 확률과 최대 차이: {np.abs(reproduced - final).max():.2e}, 판정이 다른 발화: {int(((reproduced >= .5) != (final >= .5)).sum())}")"""),
    ("md", None),  # 실데이터 검증
    ("md", """## 7. 오류 분석 (최종 10개 평균, Validation)"""),
    ("code", """seconds = np.expm1(np.asarray(val["mfcc"][:, -1], dtype=np.float64))
correct = (final >= .5) == labels
print("발화 길이별 정확도")
for low, high in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)):
    mask = (seconds >= low) & (seconds < high)
    print(f"  {low:>4}~{'' if high > 1e8 else high:<4}초: 발화 {mask.sum():>6,} ({mask.mean():5.1%}), 정확도 {correct[mask].mean():.4f}")
size = np.repeat(val["counts"], val["counts"])
print("통화 발화 수별 정확도")
for low, high in ((1, 6), (6, 16), (16, 31), (31, 61), (61, 10_000)):
    mask = (size >= low) & (size < high)
    print(f"  {low:>3}~{high - 1:<4}개: 발화 {mask.sum():>6,} ({mask.mean():5.1%}), 정확도 {correct[mask].mean():.4f}")
call_id = np.repeat(np.arange(len(val["counts"])), val["counts"])
call_accuracy = (np.bincount(call_id, weights=correct, minlength=len(val["counts"])) / np.maximum(val["counts"], 1))[val["counts"] > 0]
print(f"통화별 정확도: 전부 맞힌 통화 {np.mean(call_accuracy == 1):.1%}, 90% 이상 {np.mean(call_accuracy >= .9):.1%}, 절반 이하 {np.mean(call_accuracy <= .5):.1%}")
confident = np.abs(final - .5) >= .4
print(f"확신 높은 발화(확률 0.1 이하·0.9 이상) {confident.mean():.1%}: 정확도 {correct[confident].mean():.4f} / 나머지 정확도 {correct[~confident].mean():.4f}")"""),
    ("md", """## 결과 요약 (서울 Validation 발화 111,947개)

| 단계 | Accuracy |
| --- | ---: |
| MFCC 통계 + RandomForest (발화 하나만) | 0.8372 |
| ECAPA 임베딩 + 통화 내 비교 2단계 | 0.9009 |
| 같은 화자 판별기 2단계 (트랜스포머 이전 최고) | 0.9117 |
| 통화 단위 트랜스포머 단독 | 0.9219 ~ 0.9252 |
| **최종: 트랜스포머 10개 평균** | **{accuracy}** |"""),
]

VERIFIED = """## 6. 실데이터 추론 검증 (Colab T4 GPU, 공식 명령)

```
python inference.py --audio_dir <Validation WAV> --label_dir <Validation JSON> --ckpt_path mission2_final10.ckpt --output ./outputs/mission2.csv
```

| 항목 | 결과 |
| --- | --- |
| 입력 | Validation 통화 3,640개 WAV·JSON (WAV를 새로 읽어 특징을 처음부터 계산) |
| 출력 | `audio file name,startAt,endAt,speaker` 헤더, 111,947행 |
| **Accuracy** | **{accuracy}** (4절 학습 결과와 같음) |
| 추론 시간 | {time} (Drive에서 WAV 읽기 포함) |
| 환경 | Python 3.13, torch 2.11.0+cu130, speechbrain 1.1.1, librosa 0.11.0, numpy 2.1.3 (`requirements.txt`로 설치) |"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="제출 폴더 안의 노트북 경로")
    parser.add_argument("--verified_accuracy", required=True, help="Colab 실데이터 추론 정확도")
    parser.add_argument("--verified_time", required=True, help="Colab 실데이터 추론 시간")
    args = parser.parse_args()
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    texts = [INTRO.format(accuracy=args.verified_accuracy) if index == 0
             else VERIFIED.format(accuracy=args.verified_accuracy, time=args.verified_time) if text is None
             else text.replace("{accuracy}", args.verified_accuracy)
             for index, (_, text) in enumerate(CELLS)]
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
                      for (kind, _), text in zip(CELLS, texts)]
    ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(args.out.parent)}})
    nbformat.write(notebook, args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
