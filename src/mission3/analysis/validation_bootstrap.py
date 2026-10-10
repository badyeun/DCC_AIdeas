"""두 후보의 Validation 차이가 구분 가능한 크기인가.

0.65658 대 0.65591 = +0.00067. 3,640통화에서 이 정도면 통화 두세 건입니다.
같은 부트스트랩 표본에서 두 예측을 나란히 채점해 차이의 분포를 냅니다.
"""
import ast, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")


def read_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for name, sym in list(csv.reader(f))[1:]:
            out[name] = set(ast.literal_eval(sym))
    return out


truth = {}
for p in sorted(VAL.rglob("*.json")):
    ann = json.loads(p.read_text(encoding="utf-8"))
    truth[p.name] = {str(s).strip() for s in (ann.get("symptom") or [])}

names = sorted(truth)
Y = np.array([[1 if s in truth[n] else 0 for s in TARGETS] for n in names], np.int8)


def to_matrix(pred):
    return np.array([[1 if s in pred[n] else 0 for s in TARGETS] for n in names], np.int8)


def macro(y, p):
    out = []
    for i in range(9):
        tp = int(((y[:, i] == 1) & (p[:, i] == 1)).sum())
        d = int(y[:, i].sum()) + int(p[:, i].sum())
        out.append(2 * tp / d if d else 0.0)
    return float(np.mean(out))


E = ROOT / "results" / "mission3" / "ver4_eval"
A = to_matrix(read_csv(E / "pred_refit05.csv"))     # ver4, 0.5 + 보정 재적합
B = to_matrix(read_csv(E / "pred_mine.csv"))        # 내 11개 앙상블
C = to_matrix(read_csv(E / "pred_orig.csv"))        # ver4, 증상별 임계값

print(f"ver4 재적합 {macro(Y, A):.5f} · 내것 {macro(Y, B):.5f} · ver4 원본 {macro(Y, C):.5f}\n")
rng = np.random.default_rng(2026)
d_mine, d_free = [], []
for _ in range(2000):
    i = rng.integers(0, len(names), len(names))
    y = Y[i]
    d_mine.append(macro(y, B[i]) - macro(y, A[i]))
    d_free.append(macro(y, C[i]) - macro(y, A[i]))
d_mine, d_free = np.array(d_mine), np.array(d_free)

for lab, d in (("내것 − ver4재적합", d_mine), ("ver4원본 − ver4재적합 (임계값 제약 비용)", d_free)):
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{lab}")
    print(f"  평균 {d.mean():+.5f} · 표준편차 {d.std():.5f} · 95% [{lo:+.5f}, {hi:+.5f}]")
    print(f"  0보다 큰 비율 {(d > 0).mean():.0%} · 판정 "
          f"{'구분 가능' if lo > 0 or hi < 0 else '구분 불가'}\n")

print("해석용: ver4 가 임계값 제약으로 잃고 있는 양이 '재학습이 회수할 수 있는 최대치'입니다.")
