"""'전사에 근거가 없다'는 말이 정말인가, 아니면 내 정규식이 좁은 건가.

이 캠페인 내내 "오심 라벨의 84%에 전사 근거가 없다" 같은 숫자를 썼는데,
그 '근거'는 사람이 손으로 적은 정규식입니다. 라벨은 있는데 정규식에 안
걸린 통화들의 실제 어휘를 채굴해서, 빠진 표현이 있는지 확인합니다.
있다면 근거 비율 숫자가 과소평가된 것이고, 없다면 진짜 근거가 없는 것입니다.
"""
import sys, re, numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load

names, texts, Y = load(TRAIN_DIR, "train")
WORDS = {
    "고열": r"열이|고열|열나|미열|체온|발열|뜨거",
    "구토": r"토했|토하|구토|게워|욱질",
    "두통": r"머리.{0,3}아프|두통|머리.{0,3}지끈|골치|머리가 깨",
    "복통": r"배.{0,3}아프|복통|속.{0,3}아프|명치|윗배|아랫배",
    "어지러움": r"어지러|어지럼|현기증|핑 도",
    "열상": r"찢어|베였|베어|상처|피가 나|출혈|열상|칼에",
    "오심": r"울렁|메스|미식|느글|토할|구역",
    "전신쇠약": r"기운.{0,3}없|힘이.{0,3}없|쇠약|축 처|기력|탈진|못 일어",
    "호흡곤란": r"숨.{0,3}차|호흡곤란|숨.{0,3}쉬|헐떡|숨.{0,3}막",
}

def tok(t):
    return set(re.findall(r"[가-힣]{2,}", t))

for name in ("두통", "전신쇠약", "오심"):
    i = TARGETS.index(name)
    pat = re.compile(WORDS[name])
    has = np.array([bool(pat.search(t)) for t in texts])
    missing = (Y[:, i] == 1) & ~has            # 라벨은 있는데 정규식 미적중
    other = (Y[:, i] == 0) & ~has              # 라벨도 없고 정규식도 미적중
    print(f"\n=== {name}: 라벨 있고 정규식 미적중 {int(missing.sum()):,}건 "
          f"(라벨 전체의 {missing.sum()/max((Y[:,i]==1).sum(),1):.0%}) ===")
    cp, cn = Counter(), Counter()
    for t in np.array(texts, dtype=object)[missing]: cp.update(tok(t))
    for t in np.array(texts, dtype=object)[other]: cn.update(tok(t))
    np_, nn = int(missing.sum()), int(other.sum())
    rows = [(np.log((c/np_ + 1e-4)/(cn.get(w,0)/nn + 1e-4)), w, c)
            for w, c in cp.items() if c >= 40]
    rows.sort(reverse=True)
    print(f"  {'단어':<12}{'이 집단':>9}{'비교군':>9}{'로그비':>8}")
    for lr, w, c in rows[:14]:
        print(f"  {w:<12}{c/np_:>9.1%}{cn.get(w,0)/nn:>9.1%}{lr:>8.2f}")

print("\n\n=== 빠진 표현을 정규식에 넣으면 근거 비율이 얼마나 오르나 ===")
EXTRA = {
    "두통": r"머리|두통약|편두통|뒷골|관자|어지럽",
    "전신쇠약": r"처져|축.{0,2}늘어|기운|힘없|못 움직|의식.{0,3}없|쳐져|무기력|몸살",
    "오심": r"속.{0,3}안 좋|속.{0,3}불편|토기|토할것|구역질",
}
print(f"{'증상':<8}{'기존 근거율':>12}{'확장 후':>10}{'변화':>9}")
for name, extra in EXTRA.items():
    i = TARGETS.index(name)
    y = Y[:, i] == 1
    a = np.array([bool(re.search(WORDS[name], t)) for t in texts])
    b = np.array([bool(re.search(WORDS[name] + "|" + extra, t)) for t in texts])
    print(f"{name:<8}{a[y].mean():>12.1%}{b[y].mean():>10.1%}{b[y].mean()-a[y].mean():>+9.1%}")
    # 확장이 정밀도를 얼마나 희생하는지도 함께
    print(f"{'':8}  확장 표현 포함 통화에서 라벨 붙는 비율 "
          f"{a.sum():,}건 {y[a].mean():.1%} → {b.sum():,}건 {y[b].mean():.1%}")
