"""Two questions the model experiments cannot answer.

(1) How noisy is the annotation? No two transcripts are identical, so we use
    near-duplicates instead: pairs of calls whose wording is almost the same.
    If their labels still disagree, no model reading the text could get both
    right, and that disagreement rate bounds the achievable F1.
(2) Do the other eight symptoms predict 오심? If they do, the independent
    nine-sigmoid head is leaving information on the table and a correlation
    term should pay. If they do not, that experiment is dead on arrival.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load

names, texts, Y = load(TRAIN_DIR, "train")
print(f"Training {len(texts):,}건\n")

print("=== 1. 거의 같은 통화끼리 라벨이 갈리는가 ===")
X = TfidfVectorizer(min_df=3, max_features=60000, sublinear_tf=True).fit_transform(texts)
import sklearn.preprocessing as pp
X = pp.normalize(X)
best_sim = np.zeros(len(texts), np.float32); best_j = np.zeros(len(texts), np.int64)
for s in range(0, X.shape[0], 1000):
    S = (X[s:s+1000] @ X.T).toarray()
    for r in range(S.shape[0]):
        S[r, s + r] = -1
    best_j[s:s+1000] = S.argmax(1)
    best_sim[s:s+1000] = S.max(1)
print(f"최근접 유사도 중앙값 {np.median(best_sim):.3f}")
for lo in (0.95, 0.90, 0.85, 0.80):
    m = best_sim >= lo
    if m.sum() < 50: continue
    agree = (Y[m] == Y[best_j[m]]).mean(0)
    print(f"\n  유사도 ≥ {lo}: {m.sum():,}쌍 · 9개 라벨 전부 일치 "
          f"{(Y[m] == Y[best_j[m]]).all(1).mean():.1%}")
    print("   " + "  ".join(f"{t} {a:.0%}" for t, a in zip(TARGETS, agree)))

print("\n  → 라벨 불일치율 d인 증상의 F1 상한 ≈ 1-d. 아래가 그 추정치:")
m = best_sim >= 0.85
for i, t in enumerate(TARGETS):
    a, b = Y[m][:, i], Y[best_j[m]][:, i]
    both = (a == 1) & (b == 1); either = (a == 1) | (b == 1)
    print(f"    {t:<8} 양성끼리 일치(Jaccard) {both.sum()/max(either.sum(),1):.3f}")

print("\n=== 2. 다른 8개 증상만으로 오심을 맞힐 수 있나 ===")
def f1(t, p):
    tp = int((t & p).sum()); d = int(t.sum()) + int(p.sum())
    return 2 * tp / d if d else 0.0
for name in ("오심", "구토", "두통", "전신쇠약"):
    i = TARGETS.index(name)
    others = np.delete(Y, i, axis=1)
    pred = cross_val_predict(LogisticRegression(max_iter=1000), others, Y[:, i],
                             cv=5, method="predict_proba")[:, 1]
    print(f"  {name:<8} 다른 라벨만으로 F1@0.5 = {f1(Y[:, i] == 1, pred >= .5):.3f}"
          f"  (기저율 {Y[:, i].mean():.1%})")
    co = [(np.corrcoef(Y[:, i], Y[:, j])[0, 1], TARGETS[j])
          for j in range(len(TARGETS)) if j != i]
    co.sort(reverse=True)
    print("           상관 상위: " + ", ".join(f"{t} {c:+.2f}" for c, t in co[:3]))
