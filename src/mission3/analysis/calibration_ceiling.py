"""Has the model already extracted everything the text holds about 오심?

If the predicted probabilities are calibrated -- when the model says 0.3, the
label really is positive 30% of the time -- then the remaining error is noise
in the annotation, not something a better model would fix. The expected F1 of
a perfectly calibrated predictor is computable, so we can compare it against
the F1 actually achieved and see how much of the gap is ours to close.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
mem = ["probs_roberta-base_raw_s42.npy", "probs_roberta-base_raw_s1.npy",
       "probs_roberta-base_raw_s7.npy", "probs_roberta-base_attn_raw_s42.npy",
       "probs_roberta-base_attn_raw_s1.npy", "probs_roberta-base_attn_raw_s7.npy",
       "probs_roberta-base_raw_s42_lr2e-5e3.npy", "probs_roberta-base_raw_s1_lr2e-5e3.npy",
       "probs_roberta-base_raw_s7_lr2e-5e3.npy"]
P = np.mean([np.load(R / m) for m in mem], axis=0)
_, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)

print("=== 보정 상태: 모델이 p라고 말할 때 실제 양성률은? ===")
edges = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])
for i in (TARGETS.index("오심"), TARGETS.index("두통"), TARGETS.index("구토"),
          TARGETS.index("열상")):
    print(f"\n{TARGETS[i]}  (F1 {per[i]:.3f})")
    print(f"  {'구간':<12}{'건수':>7}{'예측확률':>9}{'실제양성률':>11}{'치우침':>9}")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (P[:, i] >= a) & (P[:, i] < b)
        if m.sum() < 15: continue
        q, r = P[m, i].mean(), Y[m, i].mean()
        print(f"  {a:.2f}-{b:<8.2f}{m.sum():>7}{q:>9.3f}{r:>11.3f}{r-q:>+9.3f}")

print("\n\n=== 완벽히 보정된 예측기의 기대 F1 (잡음만 남았을 때의 상한) ===")
print("  확률이 참값이라면 임계 t로 자른 집합의 기대 F1 = 2*sum(p in S) / (sum(p)+|S|)")
print(f"  {'증상':<8}{'실제F1@0.5':>11}{'기대F1@0.5':>11}{'기대F1@최적':>12}{'남은여지':>10}")
room = 0
for i, t in enumerate(TARGETS):
    p = P[:, i]
    exp_at = lambda th: (2 * p[p >= th].sum() / (p.sum() + (p >= th).sum())
                         if (p >= th).sum() else 0.0)
    e05 = exp_at(.5)
    ebest = max(exp_at(th) for th in np.arange(.05, .96, .01))
    room += ebest - per[i]
    print(f"  {t:<8}{per[i]:>11.3f}{e05:>11.3f}{ebest:>12.3f}{ebest-per[i]:>+10.3f}")
print(f"\n  보정된 확률로 얻을 수 있는 Macro F1 상한 ≈ {np.mean(per) + room/9:.4f}"
      f"  (현재 {np.mean(per):.4f})")
print("  → 이 상한은 '지금 모델의 확률이 참값일 때'의 값입니다. 실제 확률이 더")
print("    날카로워지면 상한 자체가 올라가므로, 이건 '현 상태의 잡음 하한'입니다.")
