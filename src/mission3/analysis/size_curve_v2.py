"""앙상블 크기 곡선을 다시 — 이번엔 신뢰구간과 함께.

1라운드는 "9개에서 포화"로 결론 냈습니다. 그때는 쓸 수 있는 멤버가 적었고
F1@0.5 하나로만 봤습니다. 지금은 홀드아웃 실행이 40개 넘게 쌓였으니,
'비슷하게 좋은' 멤버 풀에서 크기 k의 부분집합을 반복 추출해 기댓값과
퍼짐을 함께 봅니다. 최고 조합을 고르는 게 아니라 **크기 k의 평균**을
보는 것이라 홀드아웃에 맞추는 일이 없습니다.

제출을 12개에서 더 키울 가치가 있는지가 걸린 질문입니다.
"""
import sys, glob, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")


def f1(P):
    return macro_f1(Y, (P >= .5).astype(np.int8))


# 명백히 진 설정은 제외: 다른 백본, 손실 변형, 오버샘플, 보조과제, 혼합층.
BAD = re.compile(r"bert-base|KcELECTRA|koelectra|KR-ELECTRA|asl|focal|ema|llrd|"
                 r"pwnone|pwpow0(65|80)|_f0\.|os(full|sqrt)|aux|corr|lw(invfreq|custom)|"
                 r"soft|constant|e6|lr(1e-5|5e-5|8e-5)|sf1")
pool, names = [], []
for p in sorted(glob.glob(str(R / "probs_*.npy"))):
    n = Path(p).stem
    if BAD.search(n):
        continue
    q = np.load(p)
    if q.shape != Y.shape:
        continue
    pool.append(q); names.append(n[6:])
solo = np.array([f1(q) for q in pool])
keep = solo >= solo.max() - 0.008           # 비슷하게 좋은 것만
pool = [q for q, k in zip(pool, keep) if k]
names = [n for n, k in zip(names, keep) if k]
print(f"멤버 풀 {len(pool)}개 (단독 F1 {min(solo[keep]):.5f} ~ {max(solo[keep]):.5f})")
for n in names:
    print(f"  {n}")

rng = np.random.default_rng(5)
print(f"\n{'크기':>4}{'평균 F1':>11}{'표준편차':>10}{'평균 +TFIDF':>13}{'표준편차':>10}")
prev = None
for k in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, min(18, len(pool)), len(pool)):
    if k > len(pool):
        continue
    a, b = [], []
    reps = 1 if k == len(pool) else 150
    for _ in range(reps):
        idx = rng.choice(len(pool), k, replace=False)
        P = np.mean([pool[i] for i in idx], axis=0)
        a.append(f1(P)); b.append(f1(.9 * P + .1 * T))
    mark = ""
    if prev is not None and np.mean(a) - prev < 0.0003:
        mark = "  <- 포화"
    print(f"{k:>4}{np.mean(a):>11.5f}{np.std(a):>10.5f}{np.mean(b):>13.5f}{np.std(b):>10.5f}{mark}")
    prev = np.mean(a)
print("\n각 행은 그 크기 부분집합 150개의 평균이라 '운 좋은 조합'이 아닌 기댓값입니다.")
