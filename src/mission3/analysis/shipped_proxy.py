"""제출할 구성을 홀드아웃 모델로 똑같이 조립해 점수를 추정합니다.

제출 모델은 자체검증 통화까지 학습에 넣으므로 로컬에서 직접 잴 수 없습니다.
대신 **같은 계열 구성**을 홀드아웃 학습 모델로 만들어 대리값을 냅니다.
지금까지 보고한 0.64644 는 6멤버 대리값이었고, 실제 제출은 11멤버 7계열이라
그 구성 그대로 다시 잽니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
f1 = lambda P: macro_f1(Y, (P >= .5).astype(np.int8))

SHIPPED = {                                   # 제출 구성과 1:1 대응
    "base mean lr2e-5": ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3"],
    "base attn lr2e-5": ["roberta-base_attn_raw_s42_lr2e5_LS",
                         "roberta-base_attn_raw_s1_lr2e5_LS"],
    "base mean lr3e-5": ["roberta-base_raw_s11"] if (R / "probs_roberta-base_raw_s11.npy").exists()
                        else ["roberta-base_raw_s42"],
    "base attn lr3e-5": ["roberta-base_attn_raw_s7"],
    "정규화 텍스트":      ["roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1"],
    "도메인 적응":        ["domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS"],
    "CLS 풀링":          ["roberta-base_cls_raw_s42_LS"],
}
CURRENT = {                                   # 지금 제출물 (동질 12개)
    "base mean lr2e-5": ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
                         "roberta-base_raw_s7_lr2e-5e3"],
    "base attn lr2e-5": ["roberta-base_attn_raw_s42_lr2e5_LS",
                         "roberta-base_attn_raw_s1_lr2e5_LS",
                         "roberta-base_attn_raw_s7_lr2e5_LS"],
}

def build(spec):
    mem = [L(n) for v in spec.values() for n in v]
    return mem, np.mean(mem, axis=0)

for lab, spec in (("지금 제출물 (동질)", CURRENT), ("새 제출물 (이질 7계열)", SHIPPED)):
    mem, P = build(spec)
    solo = np.mean([f1(m) for m in mem])
    dis = np.mean([np.mean((mem[i] >= .5) != (mem[j] >= .5))
                   for i in range(len(mem)) for j in range(i + 1, len(mem))])
    print(f"{lab}")
    print(f"  멤버 {len(mem)}개 · 계열 {len(spec)}개 · 멤버 단독 평균 {solo:.5f} · 상호 불일치 {dis:.4f}")
    print(f"  F1 {f1(P):.5f}   +TF-IDF {f1(.9 * P + .1 * T):.5f}\n")

_, A = build(CURRENT); _, B = build(SHIPPED)
A, B = .9 * A + .1 * T, .9 * B + .1 * T
rng = np.random.default_rng(97)
d = []
for _ in range(800):
    i = rng.integers(0, len(Y), len(Y)); yb = Y[i]
    d.append(macro_f1(yb, (B[i] >= .5).astype(np.int8))
             - macro_f1(yb, (A[i] >= .5).astype(np.int8)))
d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
print(f"짝지은 부트스트랩 800회 (새 − 지금): {d.mean():+.5f} · 95% [{lo:+.5f}, {hi:+.5f}]"
      f" · 이긴 비율 {(d > 0).mean():.0%}")
print(f"판정: {'○ 개선 실재' if lo > 0 else '✗ 악화 실재' if hi < 0 else '− 구분 불가'}")
print("\n증상별 (새 제출물):")
_, per = macro_f1(Y, (B >= .5).astype(np.int8), per_symptom=True)
for t, v in zip(TARGETS, per):
    print(f"  {t:<8}{v:.4f}")
