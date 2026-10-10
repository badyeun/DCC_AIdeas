"""How much of the label signal is learnable at all?

Two independent estimates of the annotation noise floor:
  (1) Exact-duplicate transcripts that carry different labels. Same input,
      different answer -- no model can get both right.
  (2) Near-duplicate pairs (high TF-IDF cosine), same idea, softer.
Then: which words actually separate 오심=1 from 오심=0, mined rather than
assumed, so we can tell whether the keyword list simply missed the wording.
"""
import sys, re, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, load, group_ids, TRAIN_DIR

names, texts, Y = load(TRAIN_DIR, "train")
print(f"Training {len(texts):,}건\n")

print("=== 1. 같은 전사, 다른 라벨 (순수 어노테이션 잡음) ===")
g = group_ids(texts)
buckets = defaultdict(list)
for i, k in enumerate(g):
    buckets[k].append(i)
dups = {k: v for k, v in buckets.items() if len(v) > 1}
print(f"중복 전사 그룹 {len(dups):,}개 / 포함 통화 {sum(len(v) for v in dups.values()):,}건")
disagree = np.zeros(len(TARGETS), int); total = np.zeros(len(TARGETS), int)
for v in dups.values():
    sub = Y[v]
    for i in range(len(TARGETS)):
        total[i] += 1
        if sub[:, i].min() != sub[:, i].max():
            disagree[i] += 1
if len(dups):
    print(f"{'증상':<8}{'불일치 그룹':>12}{'비율':>8}{'달성가능 F1 상한':>18}")
    for i, s in enumerate(TARGETS):
        r = disagree[i] / max(total[i], 1)
        print(f"{s:<8}{disagree[i]:>12}{r:>8.1%}")
    print(f"\n전체 그룹 중 라벨이 완전히 같은 비율: "
          f"{sum(1 for v in dups.values() if (Y[v]==Y[v][0]).all())/len(dups):.1%}")
print()

print("=== 2. 오심을 가르는 단어 (가정 없이 채굴) ===")
def toks(t):
    return set(re.findall(r"[가-힣]{2,}", t))
oi, gi = TARGETS.index("오심"), TARGETS.index("구토")
pos = [toks(t) for t, y in zip(texts, Y[:, oi]) if y == 1]
neg = [toks(t) for t, y in zip(texts, Y[:, oi]) if y == 0]
cp, cn = Counter(), Counter()
for s in pos: cp.update(s)
for s in neg: cn.update(s)
np_, nn = len(pos), len(neg)
rows = []
for w, c in cp.items():
    if c < 60: continue
    a, b = c / np_, cn.get(w, 0) / nn
    rows.append((np.log((a + 1e-4) / (b + 1e-4)), w, c, a, b))
rows.sort(reverse=True)
print(f"오심=1 {np_:,}건 / 오심=0 {nn:,}건")
print(f"{'단어':<12}{'오심1 출현율':>12}{'오심0 출현율':>12}{'로그비':>9}")
for lr, w, c, a, b in rows[:20]:
    print(f"{w:<12}{a:>12.1%}{b:>12.1%}{lr:>9.2f}")
print()

print("=== 3. '토' 관련 표현이 있는 통화에서 오심/구토가 갈리는가 ===")
VOM = re.compile(r"토했|토하|구토|게워|욱질|울렁|메스|미식|느글|토할")
has = np.array([bool(VOM.search(t)) for t in texts])
print(f"구토성 표현 포함 {has.sum():,}건 ({has.mean():.1%})")
for a in (1, 0):
    for b in (1, 0):
        n = ((Y[:, oi] == a) & (Y[:, gi] == b) & has).sum()
        print(f"  오심={a} 구토={b}: {n:>6}건 ({n/has.sum():>5.1%})")
print("\n개별 표현별 오심 양성률:")
for pat in ("울렁", "메스", "미식", "느글", "토할 것", "토할거", "속이 안", "토했", "구토", "게워"):
    m = np.array([pat in t for t in texts])
    if m.sum() < 30: continue
    print(f"  {pat:<10} {m.sum():>6}건 → 오심 {Y[m, oi].mean():>6.1%} · 구토 {Y[m, gi].mean():>6.1%}")
