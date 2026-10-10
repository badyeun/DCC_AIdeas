"""오심 라벨이 전사 밖 정보로 설명되는가.

JSON에는 triage, urgencyLevel, disasterMedium, sentiment, gender, 발화자 번호,
발화 타이밍이 들어 있습니다. 규정상 Mission 3의 입력은 utterances[].text
뿐이라 **모델은 이것들을 절대 쓸 수 없습니다.** 여기서는 오직 진단 목적으로만
읽습니다 — 라벨이 이것들로 설명된다면 "허용된 입력 밖에 답이 있다"는 뜻이고,
설명되지 않는다면 순수 잡음이라는 쪽이 굳어집니다.
"""
import sys, json, re, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load

names, texts, Y = load(TRAIN_DIR, "train")
by_name = {p.name: p for p in TRAIN_DIR.rglob("*.json")}
FIELDS = ("disasterMedium", "urgencyLevel", "sentiment", "triage", "gender", "mediaType")
meta, spk_nausea, n_turns, dur = [], [], [], []
NAU = re.compile(r"울렁|메스|미식|느글|토할|구역")
for n in names:
    a = json.loads(by_name[n].read_text(encoding="utf-8"))
    meta.append([str(a.get(f, "")) for f in FIELDS])
    us = a.get("utterances") or a.get("utterences") or []
    n_turns.append(len(us))
    dur.append((a.get("endAt") or 0) - (a.get("startAt") or 0))
    who = -1
    for u in us:
        if isinstance(u, dict) and isinstance(u.get("text"), str) and NAU.search(u["text"]):
            who = u.get("speaker", -1); break
    spk_nausea.append(who)
meta = np.array(meta, dtype=object)
O = TARGETS.index("오심")
y = Y[:, O]

print("=== 1. 메타데이터만으로 오심을 맞힐 수 있나 (모델은 못 쓰는 정보) ===")
X = OneHotEncoder(handle_unknown="ignore").fit_transform(meta)
p = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                      X, y, cv=5, method="predict_proba")[:, 1]
print(f"  범주형 6개(triage 등)만으로 오심 AUC = {roc_auc_score(y, p):.3f}  (기저율 {y.mean():.1%})")
num = np.column_stack([n_turns, dur, [len(t) for t in texts]]).astype(float)
p2 = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                       num, y, cv=5, method="predict_proba")[:, 1]
print(f"  길이·발화수·통화시간만으로      오심 AUC = {roc_auc_score(y, p2):.3f}")
print("  참고: 전사 전체를 쓰는 신경망 앙상블의 오심 AUC는 0.777입니다.\n")

print("=== 2. 필드별 오심 양성률 ===")
for j, f in enumerate(FIELDS):
    vals = Counter(meta[:, j]).most_common(5)
    print(f"  {f}:")
    for v, c in vals:
        m = meta[:, j] == v
        print(f"    {v[:28]:<30}{c:>6}건 → 오심 {y[m].mean():>6.1%}")

print("\n=== 3. 오심성 표현을 '누가' 말했는가 ===")
spk = np.array(spk_nausea)
has = spk >= 0
print(f"  오심성 표현이 있는 통화 {int(has.sum()):,}건")
for s in sorted(set(spk[has].tolist())):
    m = spk == s
    role = "접수자(119)" if s == 0 else f"신고자 {s}"
    print(f"    {role:<14}{int(m.sum()):>5}건 → 오심 {y[m].mean():>6.1%}")
print(f"  표현 없음                {int((~has).sum()):>5}건 → 오심 {y[~has].mean():>6.1%}")
