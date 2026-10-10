"""DAPT 모델을 앙상블에 넣는 것이 진짜 이득인가 — 부트스트랩 검정.

겉보기로는 +0.0009였습니다. 그런데 홀드아웃 부트스트랩 표준편차가 0.0044라
그 크기는 그냥 보고 판단할 수 없습니다. 같은 부트스트랩 표본에서 두 구성을
나란히 채점해 차이의 분포를 냅니다 (짝비교라 공통 잡음이 상쇄됩니다).

이 절차를 쓰는 이유: 이번 캠페인에서 "작지만 양수고 이론적으로 말이 되는"
결과를 두 번 믿었다가 두 번 틀렸습니다 (정규화 +0.0067 → −0.0003,
라벨 스무딩 +0.0009 → −0.0016).
"""
import sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
mean3 = [np.load(R / f"probs_roberta-base_raw_s{s}_lr2e-5e3.npy") for s in (42, 1, 7)]
attn3 = [np.load(R / f"probs_roberta-base_attn_raw_s{s}.npy") for s in (42, 1, 7)]
dapt = [np.load(p) for p in sorted(glob.glob(str(R / "probs_domain_adapted_*_DAPT_LS.npy")))]
print(f"DAPT 모델 {len(dapt)}개 사용\n")

def blend(members, w=0.10):
    return (1 - w) * np.mean(members, axis=0) + w * T

CAND = {
    "기준 6개 + TFIDF": blend(mean3 + attn3),
    f"+ DAPT {len(dapt)}개 + TFIDF": blend(mean3 + attn3 + dapt),
    f"기준 6개 + DAPT {len(dapt)}개 (TFIDF 없음)": np.mean(mean3 + attn3 + dapt, axis=0),
}
print(f"{'구성':<34}{'실측 F1':>10}")
for k, p in CAND.items():
    print(f"{k:<34}{macro_f1(Y, (p >= .5).astype(np.int8)):>10.5f}")

rng = np.random.default_rng(23)
A = CAND["기준 6개 + TFIDF"]
B = CAND[f"+ DAPT {len(dapt)}개 + TFIDF"]
d = []
for _ in range(600):
    idx = rng.integers(0, len(Y), len(Y))
    yb = Y[idx]
    d.append(macro_f1(yb, (B[idx] >= .5).astype(np.int8))
             - macro_f1(yb, (A[idx] >= .5).astype(np.int8)))
d = np.array(d)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\n부트스트랩 600회, 짝지은 차이 (DAPT 추가 − 기준)")
print(f"  평균 {d.mean():+.5f} · 표준편차 {d.std():.5f}")
print(f"  95% 구간 [{lo:+.5f}, {hi:+.5f}]")
print(f"  0보다 큰 비율 {(d > 0).mean():.1%}")
print(f"  판정: {'○ 이득이 실재' if lo > 0 else '✗ 손해가 실재' if hi < 0 else '− 구분 불가'}")
print("\n주의: 부트스트랩은 '이 홀드아웃에서의 차이'만 봅니다. seed를 다시")
print("뽑았을 때의 흔들림(표준편차 0.0023)은 여기 들어 있지 않습니다.")
