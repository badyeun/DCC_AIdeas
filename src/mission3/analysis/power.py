"""이 설계로 잴 수 있는 가장 작은 효과는 얼마인가.

37개쯤 실험했는데 앙상블 말고는 전부 "잡음 안"으로 끝났습니다. 그게 효과가
없어서인지, 설계가 둔해서인지 가립니다. 잡음은 두 갈래입니다:

  seed 잡음  — 같은 설정을 다시 돌렸을 때의 흔들림
  평가 잡음  — 홀드아웃에 어떤 통화가 들어갔느냐에 따른 흔들림

짝비교(같은 홀드아웃에서 두 설정을 비교)는 평가 잡음을 상당히 상쇄하지만
완전히는 아닙니다. 둘을 실제로 재서, n회 실행으로 탐지 가능한 최소 효과를
냅니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
G = {"mean lr3e-5": ["raw_s42", "raw_s1", "raw_s7"],
     "attn lr3e-5": ["attn_raw_s42", "attn_raw_s1", "attn_raw_s7"],
     "mean lr2e-5": ["raw_s42_lr2e-5e3", "raw_s1_lr2e-5e3", "raw_s7_lr2e-5e3"],
     "attn lr2e-5": ["attn_raw_s42_lr2e5_LS", "attn_raw_s1_lr2e5_LS", "attn_raw_s7_lr2e5_LS"]}
load = {k: [np.load(R / f"probs_roberta-base_{t}.npy") for t in v] for k, v in G.items()}

print("=== 1. seed 잡음 (같은 설정, seed만 다름) ===")
sds = []
for k, v in load.items():
    s = [macro_f1(Y, (p >= .5).astype(np.int8)) for p in v]
    sds.append(np.std(s, ddof=1))
    print(f"  {k:<14} {' '.join(f'{x:.5f}' for x in s)}   표준편차 {np.std(s, ddof=1):.5f}")
seed_sd = float(np.mean(sds))
print(f"  → 평균 seed 표준편차 {seed_sd:.5f}")

print("\n=== 2. 짝비교에서 남는 잡음 (부트스트랩) ===")
rng = np.random.default_rng(11)
pairs = [("mean lr3e-5", "mean lr2e-5"), ("attn lr3e-5", "attn lr2e-5")]
resid = []
for a, b in pairs:
    Pa, Pb = np.mean(load[a], axis=0), np.mean(load[b], axis=0)
    d = []
    for _ in range(300):
        idx = rng.integers(0, len(Y), len(Y))
        d.append(macro_f1(Y[idx], (Pa[idx] >= .5).astype(np.int8))
                 - macro_f1(Y[idx], (Pb[idx] >= .5).astype(np.int8)))
    resid.append(np.std(d))
    print(f"  {a} vs {b}: 짝지은 차이의 표준편차 {np.std(d):.5f}")
eval_sd = float(np.mean(resid))
print(f"  → 평균 {eval_sd:.5f}  (짝비교라 평가 잡음이 상당히 상쇄된 값)")

total = float(np.hypot(seed_sd, eval_sd))
print(f"\n=== 3. 단일 짝비교 1회의 총 불확실성 ≈ {total:.5f} ===")
print(f"\n{'실행 수(각 팔)':<14}{'탐지 가능 최소 효과 (80% 검정력)':>30}")
for n in (1, 2, 3, 5, 10, 20, 40):
    mde = 2.8 * total / np.sqrt(n)
    print(f"{n:<14}{mde:>30.5f}")

print("\n=== 4. 이번 캠페인에서 실제로 관측된 효과 크기 ===")
for lab, val in (("앙상블 1개 → 9개", 0.0089), ("전체 데이터 학습(외삽)", 0.0040),
                 ("TF-IDF 혼합", 0.0012), ("헤드 혼합", 0.0014),
                 ("학습률 3e-5 → 2e-5", 0.0013), ("도메인 적응(1 seed)", -0.0025)):
    n_need = (2.8 * total / abs(val)) ** 2 if val else float("inf")
    print(f"  {lab:<24}{val:+.4f}   80% 검정력에 필요한 seed 수 {n_need:>7.0f}")
