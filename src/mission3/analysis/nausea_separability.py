"""구토가 분명한 통화 안에서, 오심이 붙는 쪽과 안 붙는 쪽을 가르는 신호가 있는가.

있으면 모델이 아직 못 배운 것이고, 없으면 어노테이션 동전던지기라 끝입니다.
구토 라벨이 있는 통화만 남기고 오심=1 대 오심=0을 분류해 봅니다. 이 안에서
AUC가 0.5 근처면 "구분 불가"가 증명됩니다.
"""
import sys, re, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load

names, texts, Y = load(TRAIN_DIR, "train")
O, G = TARGETS.index("오심"), TARGETS.index("구토")
texts = np.array(texts, dtype=object)

def probe(mask, label):
    sub, y = texts[mask], Y[mask, O]
    if y.sum() < 40 or (1 - y).sum() < 40:
        print(f"{label}: 표본 부족"); return
    X = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                        max_features=80000, sublinear_tf=True).fit_transform(sub)
    p = cross_val_predict(LogisticRegression(C=4, max_iter=2000, class_weight="balanced"),
                          X, y, cv=5, method="predict_proba")[:, 1]
    print(f"{label}")
    print(f"  통화 {len(sub):,}건 · 오심 양성 {int(y.sum()):,} ({y.mean():.1%})"
          f" · AUC {roc_auc_score(y, p):.3f} · AP {average_precision_score(y, p):.3f}"
          f" (기저 {y.mean():.3f})")
    return y, p

print("=== 오심 분리 가능성 ===\n")
probe(np.ones(len(texts), bool), "전체 통화에서 오심 분류")
print()
probe(Y[:, G] == 1, "구토 라벨이 있는 통화 안에서만 오심 분류")
print()
VOM = re.compile(r"토했|토하|구토|게워|욱질")
probe(np.array([bool(VOM.search(t)) for t in texts]), "구토성 표현이 있는 통화 안에서만")
print()
DIZ = TARGETS.index("어지러움")
probe(Y[:, DIZ] == 1, "어지러움 라벨이 있는 통화 안에서만 오심 분류")

print("\n=== 구토 통화 안에서 오심을 가르는 단어 (있다면) ===")
m = Y[:, G] == 1
pos = [set(re.findall(r"[가-힣]{2,}", t)) for t, y in zip(texts[m], Y[m, O]) if y == 1]
neg = [set(re.findall(r"[가-힣]{2,}", t)) for t, y in zip(texts[m], Y[m, O]) if y == 0]
cp, cn = Counter(), Counter()
for s in pos: cp.update(s)
for s in neg: cn.update(s)
rows = [(np.log((cp[w]/len(pos) + 1e-3) / (cn.get(w, 0)/len(neg) + 1e-3)), w, cp[w])
        for w in cp if cp[w] >= 25]
rows.sort(reverse=True)
print(f"오심=1 {len(pos)}건 / 오심=0 {len(neg)}건")
print(f"{'단어':<12}{'오심1':>9}{'오심0':>9}{'로그비':>8}")
for lr, w, c in rows[:12]:
    print(f"{w:<12}{c/len(pos):>9.1%}{cn.get(w,0)/len(neg):>9.1%}{lr:>8.2f}")
