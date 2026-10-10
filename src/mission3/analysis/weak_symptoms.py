"""오심 말고 나머지 약한 증상들도 같은 병인가.

오심은 "전사에 말이 있어도 라벨이 63%만 붙는" 어노테이션 문제였습니다.
두통(F1 0.521) · 구토(0.577) · 전신쇠약(0.597)도 같은 병인지, 아니면 다른
이유로 낮은지 가립니다. 다르다면 공략법도 달라집니다.
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
ht = [texts[i] for i in te]

WORDS = {
    "고열":   r"열이|고열|열나|미열|체온|발열|뜨거",
    "구토":   r"토했|토하|구토|게워|욱질",
    "두통":   r"머리.{0,3}아프|두통|머리.{0,3}지끈|골치|머리가 깨",
    "복통":   r"배.{0,3}아프|복통|속.{0,3}아프|명치|윗배|아랫배",
    "어지러움": r"어지러|어지럼|현기증|핑 도",
    "열상":   r"찢어|베였|베어|상처|피가 나|출혈|열상|칼에",
    "오심":   r"울렁|메스|미식|느글|토할|구역",
    "전신쇠약": r"기운.{0,3}없|힘이.{0,3}없|쇠약|축 처|기력|탈진|못 일어",
    "호흡곤란": r"숨.{0,3}차|호흡곤란|숨.{0,3}쉬|헐떡|숨.{0,3}막",
}

print("=== 증상별: 전사에 말이 있을 때 라벨이 붙는 비율 ===")
print(f"{'증상':<8}{'F1':>7}{'기저율':>8}{'표현 있음':>10}{'→라벨 붙음':>11}"
      f"{'표현 없는데 라벨':>14}{'라벨 중 근거있음':>15}")
rows = []
for i, t in enumerate(TARGETS):
    m = np.array([bool(re.search(WORDS[t], x)) for x in texts])
    y = Yall[:, i] == 1
    prec = y[m].mean()                       # 표현 있을 때 라벨 붙는 비율
    recall = m[y].mean()                     # 라벨 중 근거 있는 비율
    _, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)
    rows.append((t, per[i], y.mean(), m.sum(), prec, (y & ~m).sum(), recall))
    print(f"{t:<8}{per[i]:>7.3f}{y.mean():>8.1%}{m.sum():>10,}{prec:>11.1%}"
          f"{(y & ~m).sum():>14,}{recall:>15.1%}")

print("\n=== '말이 있는데 라벨이 안 붙은' 경우가 핵심입니다 ===")
print(f"{'증상':<8}{'표현 O 라벨 X':>14}{'그 비율':>9}   해석")
for t, f1v, base, nm, prec, nolab, rec in rows:
    i = TARGETS.index(t)
    m = np.array([bool(re.search(WORDS[t], x)) for x in texts])
    n = int((m & (Yall[:, i] == 0)).sum())
    verdict = ("어노테이션 흔들림 큼" if prec < .55 else
               "중간" if prec < .75 else "전사와 라벨이 잘 맞음")
    print(f"{t:<8}{n:>14,}{n/max(m.sum(),1):>9.1%}   {verdict}")

print("\n=== 조건부 분리 가능성: 표현이 있는 통화 안에서 라벨 유무를 가를 수 있나 ===")
print(f"{'증상':<8}{'표현 있는 홀드아웃':>16}{'그 안 기저율':>12}{'그 안 AUC':>10}")
from sklearn.metrics import roc_auc_score
for i, t in enumerate(TARGETS):
    m = np.array([bool(re.search(WORDS[t], x)) for x in ht])
    if m.sum() < 60 or Y[m, i].sum() < 15 or (Y[m, i] == 0).sum() < 15:
        print(f"{t:<8}{int(m.sum()):>16}{'표본 부족':>12}")
        continue
    print(f"{t:<8}{int(m.sum()):>16}{Y[m,i].mean():>12.1%}{roc_auc_score(Y[m,i],P[m,i]):>10.3f}")
print("\nAUC가 0.5에 가까울수록 '말이 있어도 라벨 유무는 동전던지기'입니다.")
