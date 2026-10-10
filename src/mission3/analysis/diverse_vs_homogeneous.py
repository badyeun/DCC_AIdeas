"""같은 크기라면, 같은 레시피 6개와 서로 다른 계열 6개 중 어느 쪽인가.

크기 곡선이 10개에서 포화하므로 멤버를 늘리는 건 끝났습니다. 남은 질문은
**같은 예산을 어떻게 쓰느냐**입니다. 현 제출은 한 레시피(lr 2e-5)에 헤드만
둘로 나눈 동질적 구성입니다. 풀에는 정규화·DAPT·large·다른 lr 계열이
있는데, 거기서 하나씩 뽑은 이질적 구성과 비교합니다.

구성은 **미리 정합니다** — 홀드아웃에서 최고 조합을 찾는 게 아닙니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")

HOMO = [  # 현 제출과 같은 모양: 한 레시피, 헤드만 둘
    "roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
    "roberta-base_raw_s7_lr2e-5e3",
    "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
    "roberta-base_attn_raw_s7_lr2e5_LS",
]
DIVERSE = [  # 계열마다 하나씩
    "roberta-base_raw_s42_lr2e-5e3",      # mean, lr 2e-5
    "roberta-base_attn_raw_s1",           # attention, lr 3e-5
    "roberta-base_synonym+filler_s7",     # 정규화 텍스트
    "domain_adapted_raw_s42_DAPT_LS",     # 도메인 적응 mean
    "domain_adapted_attn_raw_s42_DAPT_LS",# 도메인 적응 attention
    "roberta-large_raw_s42",              # 더 큰 모델
]
sets = {"동질 6개 (현 제출 모양)": HOMO, "이질 6개 (계열마다 하나)": DIVERSE}
P = {k: 0.9 * np.mean([L(n) for n in v], axis=0) + 0.1 * T for k, v in sets.items()}
raw = {k: np.mean([L(n) for n in v], axis=0) for k, v in sets.items()}

print(f"{'구성':<26}{'F1':>10}{'+TFIDF':>10}{'멤버 단독 평균':>14}{'상호 불일치':>12}")
for k, v in sets.items():
    members = [L(n) for n in v]
    solo = np.mean([macro_f1(Y, (m >= .5).astype(np.int8)) for m in members])
    dis = np.mean([np.mean((members[i] >= .5) != (members[j] >= .5))
                   for i in range(len(members)) for j in range(i + 1, len(members))])
    print(f"{k:<26}{macro_f1(Y, (raw[k] >= .5).astype(np.int8)):>10.5f}"
          f"{macro_f1(Y, (P[k] >= .5).astype(np.int8)):>10.5f}{solo:>14.5f}{dis:>12.4f}")

rng = np.random.default_rng(41)
A, B = P["동질 6개 (현 제출 모양)"], P["이질 6개 (계열마다 하나)"]
d = []
for _ in range(800):
    i = rng.integers(0, len(Y), len(Y))
    yb = Y[i]
    d.append(macro_f1(yb, (B[i] >= .5).astype(np.int8))
             - macro_f1(yb, (A[i] >= .5).astype(np.int8)))
d = np.array(d)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\n부트스트랩 800회 (이질 − 동질): {d.mean():+.5f} · 95% [{lo:+.5f}, {hi:+.5f}]"
      f" · 이긴 비율 {(d>0).mean():.0%}")
print(f"판정: {'○ 이질이 낫다' if lo > 0 else '✗ 동질이 낫다' if hi < 0 else '− 구분 불가'}")
