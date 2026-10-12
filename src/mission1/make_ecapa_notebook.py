"""Mission 1 ECAPA 학습 노트북(학습 로그 포함)을 만들고 실행해 출력까지 저장한다.

제출 규칙상 학습 코드는 실행 로그가 남은 .ipynb여야 한다. 노트북 셀은 src/mission1의 학습 코드를
그대로 불러 실행하므로, 노트북과 .py가 서로 다른 계산을 할 일이 없다.

사용: python make_ecapa_notebook.py --out Mission1_ecapa_training.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

CELLS = [
    ("md", """# Mission 1 — ECAPA 화자 임베딩 성별 모델 학습 기록 (EXP-012)

최종 제출 모델 = **예은 WavLM 3모델 앙상블 25% + ECAPA 로지스틱 회귀 75%** (확률 가중 평균).
WavLM 3모델의 학습 기록은 `Mission1_final.ipynb`에 있고, 이 노트북은 **ECAPA 쪽 학습과 섞는 비율 선택**을 기록한다.

**규칙 준수**
- 학습 데이터: 서울 Training 29,200통화만 사용. Validation 3,640통화는 학습(역전파·적합)에 쓰지 않고 마지막 확인에만 사용.
- annotation: 입력에는 `startAt`, `endAt`, `speaker`만 사용(신고자 구간 선택). `gender`는 학습 정답과 채점에만 사용.
- 사전학습 모델: `speechbrain/spkrec-ecapa-voxceleb`(공개 모델, 가중치 고정, 미세조정 없음). 상용 API 없음.
- 모델 선택(C, 섞는 비율)은 Training 내부 holdout 2,920통화로만 했다(예은 앙상블을 고른 것과 같은 holdout).

**특징 캐시**: 발화별 ECAPA 192차원 임베딩과 MFCC 통계는 Mission 2 때 Colab GPU에서
`training_code/mission2_ecapa_cache/train_speaker_embed.py`, `train_mfcc_rf.py`로 만든 캐시를 그대로 쓴다
(speaker 0/1 발화를 JSON 차례대로 잘라 32개씩 ECAPA, float16 저장). 통화별 `gender`와 파일 이름은
`export_labels_for_local.py`(JSON만 읽음)로 같은 통화 순서에 맞춰 붙였다."""),
    ("code", """import sys, time
from pathlib import Path
import numpy as np

SRC = Path(r"C:/유니짜장/DCC_Aideas/DCC_AIdeas/src/mission1")
M2_LOCAL = Path(r"C:/유니짜장/DCC_Aideas/mission2_local")          # Mission 2 발화별 캐시 (memmap)
EXPORT = Path(r"C:/유니짜장/DCC_Aideas/mission1_local/export_raw/mission1_export")  # 통화별 gender·stem, 예은 확률
WORK = Path(r"C:/유니짜장/DCC_Aideas/mission1_local/notebook_run")
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
from gender_data import load_split, yeeun_holdout_mask, pooled_features"""),
    ("md", """## 1. 데이터 정렬 검증
JSON에서 센 통화별 발화 수가 캐시의 통화 경계와 **모든 통화에서** 같아야 한다(다르면 `load_split`이 멈춘다).
예은 holdout은 stem 오름차순 + `train_test_split(test_size=0.1, random_state=42, stratify=gender)`로 재현하고,
예은이 저장한 split 파일과 같은지 확인한다."""),
    ("code", """train = load_split(M2_LOCAL, EXPORT, "train")
val = load_split(M2_LOCAL, EXPORT, "val")
for name, d in (("Train", train), ("Validation", val)):
    print(f"{name}: 통화 {len(d['counts']):,}, 발화 {int(d['counts'].sum()):,}, 신고자 발화 비율 {d['speaker'].mean():.4f}, F 비율 {d['gender'].mean():.4f}")
held = yeeun_holdout_mask(train["stems"], train["gender"], EXPORT)
print(f"내부 학습 {int((~held).sum()):,}통화 / 내부 holdout {int(held.sum()):,}통화 (F 비율 {train['gender'][held].mean():.4f})")"""),
    ("md", """## 2. 통화 요약 특징
신고자(`speaker=1`) 발화들의 ECAPA 임베딩(길이 1로 정규화)을 발화 길이로 가중 평균한 192차원 + 표준편차 192차원 = 384차원.
발화 순서는 쓰지 않는다(평균·표준편차는 순서와 무관)."""),
    ("code", """start = time.time()
pooled_train = pooled_features(train)
pooled_val = pooled_features(val)
np.save(WORK / "pooled_train.npy", pooled_train)
np.save(WORK / "pooled_val.npy", pooled_val)
print("특징 계산", pooled_train.shape, pooled_val.shape, f"{time.time() - start:.0f}초")"""),
    ("md", """## 3. 로지스틱 회귀 학습
표준화(내부 학습 통화로만 fit) → 로지스틱 회귀. C는 내부 holdout 정확도(동률이면 log loss)로 선택.
Validation은 선택이 끝난 모델로 한 번만 판정한다. 비교용으로 MFCC·대원 특징을 넣은 변형도 같은 절차로 돌린다."""),
    ("code", """import subprocess
RUNS = WORK / "runs"
for blocks in (["caller_ecapa"], ["caller_mfcc"], ["caller_ecapa", "caller_mfcc"], ["caller_ecapa", "caller_mfcc", "agent", "counts"]):
    print("=" * 20, "+".join(blocks))
    result = subprocess.run([sys.executable, str(SRC / "train_gender_pooled.py"), "--m2_local", str(M2_LOCAL),
                             "--export_dir", str(EXPORT), "--pooled_dir", str(WORK), "--blocks", *blocks,
                             "--out", str(RUNS / ("pooled_" + "+".join(blocks)))], capture_output=True, text=True, encoding="utf-8")
    print(result.stdout.strip() or result.stderr[-2000:])"""),
    ("md", """## 4. 예은 3모델 앙상블과 섞는 비율 선택
예은 3모델(40:40:20)의 내부 holdout·공식 Validation 확률(예은이 저장한 파일)과 같은 통화끼리 맞춰
`P(F) = (1-w)·예은 + w·ECAPA`의 오류를 계산한다. **w는 holdout 오류가 최소인 구간의 가운데로 고른다.**
Validation 열은 참고로만 출력하며 선택에 쓰지 않았다."""),
    ("code", """result = subprocess.run([sys.executable, str(SRC / "blend_with_yeeun.py"), "--export_dir", str(EXPORT),
                         "--runs", str(RUNS / "pooled_caller_ecapa.npz")], capture_output=True, text=True, encoding="utf-8")
print(result.stdout.strip() or result.stderr[-2000:])"""),
    ("md", """## 5. 제출용 가중치와 재현 확인
제출 폴더의 `ecapa_gender/ecapa_lr.npz`에는 표준화 평균·표준편차, 로지스틱 회귀 계수·절편과 섞는 비율(0.75)만
numpy 배열로 들어 있다(sklearn 버전과 무관). 제출 코드(`ecapa_gender.py`)의 특징·확률 계산이 학습 때와 같은지 확인한다."""),
    ("code", """SUB = Path(r"C:/유니짜장/DCC_Aideas/mission1_local/mission1_submission_v2")
sys.path.insert(0, str(SUB))
from ecapa_gender import EcapaGender, caller_feature
model = EcapaGender.__new__(EcapaGender)   # 인코더 없이 숫자만 불러 계산 확인
with np.load(SUB / "ecapa_gender" / "ecapa_lr.npz") as z:
    model.mean, model.scale = z["scaler_mean"].astype(float), z["scaler_scale"].astype(float)
    model.coef, model.intercept, model.blend = z["coef"].astype(float), float(z["intercept"][0]), float(z["blend"])
    print("C =", float(z["C"]), "/ ECAPA 비율 =", model.blend)
trained = np.load(RUNS / "pooled_caller_ecapa.npz")
probs = []
for c in range(len(val["counts"])):
    span = slice(val["offsets"][c], val["offsets"][c + 1]); caller = val["speaker"][span] == 1
    feature = caller_feature(np.asarray(val["ecapa"][span], np.float32)[caller], np.asarray(val["mfcc"][span][:, -1], np.float32)[caller])
    probs.append(model.probability_from_feature(feature))
probs = np.array(probs)
print("제출 코드 확률 vs 학습 확률 최대 차이:", np.abs(probs - trained["val_prob"]).max(),
      "/ 판정이 다른 통화:", int(((probs >= .5) != (trained["val_prob"] >= .5)).sum()))"""),
    ("md", """## 6. 실데이터 추론 검증 (Colab T4)
제출 폴더 v2를 공식 명령 그대로 Validation WAV 3,640개에 실행한 결과(`outputs` CSV와 파일별 확률)를 채점한다.
예은 3모델 확률이 예은 저장값과, ECAPA 확률이 위 학습 값과 같은지도 비교한다."""),
    ("code", """import csv
RESULT = Path(r"C:/유니짜장/DCC_Aideas/mission1_local/colab_full_result")
with open(RESULT / "mission1.csv", encoding="utf-8") as f:
    rows = list(csv.reader(f))
truth = dict(zip(val["stems"], val["gender"]))
pred = {Path(n).stem: int(g == "F") for n, g in rows[1:]}
wrong = sum(pred[s] != truth[s] for s in truth)
print("헤더", rows[0], "/ 행", len(rows) - 1)
print(f"공식 명령 재추론 Validation 정확도 {1 - wrong / len(truth):.4f} (오류 {wrong}/{len(truth)})")
ref = np.load(Path(r"C:/유니짜장/DCC_Aideas/mission1_local/colab_v2/verify/v2_reference_val.npz"))
got = np.load(RESULT / "probs.npz")
index = np.array([{str(s): i for i, s in enumerate(ref["stems"])}[Path(str(n)).stem] for n in got["names"]])
w = np.array([.4, .4, .2])
print("예은 3모델 확률 최대 차이:", np.abs(got["yeeun"] - ref["yeeun"][index]).max(axis=0),
      "/ 예은 앙상블 단독 오류:", int(((got["yeeun"] @ w >= .5) != ref["y"][index]).sum()))
print("ECAPA 확률 최대 차이:", float(np.abs(got["ecapa"] - ref["ecapa"][index]).max()),
      "/ ECAPA 단독 오류:", int(((got["ecapa"] >= .5) != ref["y"][index]).sum()))
print("최종 판정이 로컬 계산과 다른 통화:", int(((got["final"] >= .5) != (ref["final"][index] >= .5)).sum()))"""),
    ("md", """## 결과 요약

| 모델 | 내부 holdout 오류 (2,920) | Validation 오류 (3,640) | Validation Accuracy |
| --- | ---: | ---: | ---: |
| 예은 WavLM 3모델 40:40:20 | 61 | 77 | 0.9788 |
| ECAPA + 로지스틱 회귀 | 54 | 72 | 0.9802 |
| **최종: 예은 25% + ECAPA 75%** | **49** | **63** | **0.9827** |

두 모델은 서로 다른 통화에서 틀린다(Validation: 둘 다 오류 37, 예은만 40, ECAPA만 35). 그래서 섞으면 오류가 줄어든다.
통화 단위 트랜스포머(신고자·대원 발화 전체 입력)도 시험했지만 holdout 기준 이득이 없어 쓰지 않았다
(`experiment_log.md` EXP-012)."""),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
                      for kind, text in CELLS]
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(args.out.parent)}})
    nbformat.write(notebook, args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
