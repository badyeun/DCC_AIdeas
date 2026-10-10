"""놓친 정답(FN)에 전사 근거가 있는가 — 증상별로.

조건부 AUC는 "표현이 있는 통화 안에서 구분 가능한가"를 재지만, 모델이
실제로 놓친 건들이 어느 쪽인지는 말해주지 않습니다. FN에 표현이 있으면
모델이 읽고도 못 맞힌 것이라 고칠 여지가 있고, 없으면 애초에 못 맞힙니다.
"""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load, split, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
P = np.load(R / "_baseline_ens_probs.npy")
names, texts, Yall = load(TRAIN_DIR, "train")
tr, te = split(texts, Yall, seed=2026)
ht = np.array([texts[i] for i in te], dtype=object)
pred = (P >= .5).astype(np.int8)

W = {  # 넓은 쪽으로 (부록 C에서 좁은 정규식이 과소평가를 낳은 것을 반영)
    "고열": r"열이|고열|열나|미열|체온|발열|뜨거|도까지|도예요|도에요",
    "구토": r"토했|토하|구토|게워|욱질|토를",
    "두통": r"머리|두통|골치|관자|뒷골",
    "복통": r"배|복통|속.{0,3}아프|명치|윗배|아랫배",
    "어지러움": r"어지러|어지럼|현기증|핑 도|빙빙",
    "열상": r"찢어|베였|베어|상처|피가|출혈|열상|칼에|긁",
    "오심": r"울렁|메스|미식|느글|토할|구역|속.{0,3}안 좋",
    "전신쇠약": r"기운|힘이.{0,3}없|쇠약|축.{0,2}처|기력|탈진|못 일어|거동|식사|밥을|무기력",
    "호흡곤란": r"숨|호흡|헐떡|가쁘|산소",
}
_, per = macro_f1(Y, pred, per_symptom=True)

print("=== 놓친 정답(FN)과 잘못 짚은 것(FP)에 표현이 있는가 ===")
print(f"{'증상':<8}{'F1':>7}{'FN수':>6}{'FN에 표현':>10}{'FP수':>6}{'FP에 표현':>10}   진단")
for i, t in enumerate(TARGETS):
    pat = re.compile(W[t])
    has = np.array([bool(pat.search(x)) for x in ht])
    fn = (pred[:, i] == 0) & (Y[:, i] == 1)
    fp = (pred[:, i] == 1) & (Y[:, i] == 0)
    fnw, fpw = has[fn].mean(), has[fp].mean()
    if fnw > .6:
        d = "읽고도 놓침 — 모델 여지"
    elif fnw < .35:
        d = "근거 없는 정답 — 불가"
    else:
        d = "절반씩"
    print(f"{t:<8}{per[i]:>7.3f}{int(fn.sum()):>6}{fnw:>10.1%}{int(fp.sum()):>6}{fpw:>10.1%}   {d}")

print("\n=== 표현이 있는데 놓친 건의 확률 분포 ===")
print("모델이 '거의 맞힐 뻔'했는지, 아예 안 보였는지")
print(f"{'증상':<8}{'표현O FN':>9}{'확률 중앙값':>12}{'0.4 이상':>9}{'0.2 미만':>9}")
for i, t in enumerate(TARGETS):
    pat = re.compile(W[t])
    has = np.array([bool(pat.search(x)) for x in ht])
    m = (pred[:, i] == 0) & (Y[:, i] == 1) & has
    if m.sum() < 10:
        continue
    q = P[m, i]
    print(f"{t:<8}{int(m.sum()):>9}{np.median(q):>12.3f}{(q >= .4).mean():>9.1%}{(q < .2).mean():>9.1%}")

print("\n=== 두통: 표현이 있는데 놓친 통화에서 '머리'가 어떻게 쓰였나 ===")
i = TARGETS.index("두통")
pat = re.compile(W["두통"])
has = np.array([bool(pat.search(x)) for x in ht])
miss = (pred[:, i] == 0) & (Y[:, i] == 1) & has
hit = (pred[:, i] == 1) & (Y[:, i] == 1) & has
from collections import Counter
def ctx(mask, n=600):
    c = Counter()
    for x in ht[mask]:
        for m in re.finditer(r"머리", x):
            c[x[max(0, m.start()-6):m.start()+10].replace("\n", " ")] += 1
    return c
print("  [놓친 건의 '머리' 주변]")
for s, n in ctx(miss).most_common(10):
    print(f"    {n:>3}회  …{s}…")
print("  [맞힌 건의 '머리' 주변]")
for s, n in ctx(hit).most_common(10):
    print(f"    {n:>3}회  …{s}…")
