"""Label-side diagnosis on the shared Training-internal holdout.

Everything here is scored at the mandated 0.5. Where a different threshold is
printed it is a DIAGNOSTIC only -- it says how far the model's 0.5 crossing
sits from the F1-optimal cut, which tells us whether a loss-side fix could
recover that gap without ever moving the threshold.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, load, split, macro_f1, TRAIN_DIR

R = Path(r"C:\DevBox\projects\DCC_AIdeas") / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")

# The 14-model validated ensemble (EMA excluded -- it hurt).
members = ["probs_roberta-base_raw_s42.npy", "probs_roberta-base_raw_s1.npy",
           "probs_roberta-base_raw_s7.npy", "probs_roberta-base_attn_raw_s42.npy",
           "probs_roberta-base_attn_raw_s1.npy", "probs_roberta-base_attn_raw_s7.npy",
           "probs_roberta-base_raw_s42_lr2e-5e3.npy", "probs_roberta-base_raw_s1_lr2e-5e3.npy",
           "probs_roberta-base_raw_s7_lr2e-5e3.npy"]
P = np.mean([np.load(R / m) for m in members if (R / m).exists()], axis=0)
print(f"앙상블 {sum((R/m).exists() for m in members)}개 · holdout {len(Y):,}건\n")

print("=== 1. 0.5에서의 증상별 성적 ===")
print(f"{'증상':<8}{'정답수':>7}{'예측수':>7}{'정밀도':>8}{'재현율':>8}{'F1':>8}{'FP':>6}{'FN':>6}")
base, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)
for i, s in enumerate(TARGETS):
    t, p = Y[:, i] == 1, P[:, i] >= .5
    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    print(f"{s:<8}{t.sum():>7}{p.sum():>7}{pr:>8.3f}{rc:>8.3f}{per[i]:>8.3f}{fp:>6}{fn:>6}")
print(f"{'Macro':<8}{'':>7}{'':>7}{'':>8}{'':>8}{base:>8.5f}\n")

print("=== 2. 0.5가 F1 최적점에서 얼마나 떨어져 있나 (진단) ===")
print(f"{'증상':<8}{'F1@0.5':>9}{'최적임계':>9}{'최적F1':>9}{'손실':>9}  방향")
gap_total = 0
for i, s in enumerate(TARGETS):
    best_t, best_f = .5, per[i]
    for t in np.arange(.05, .96, .01):
        pred = P[:, i] >= t
        tp = (pred & (Y[:, i] == 1)).sum(); d = Y[:, i].sum() + pred.sum()
        f = 2 * tp / d if d else 0.
        if f > best_f: best_t, best_f = t, f
    gap_total += best_f - per[i]
    arrow = "과소예측(임계↓가 유리)" if best_t < .5 else ("과대예측" if best_t > .5 else "-")
    print(f"{s:<8}{per[i]:>9.3f}{best_t:>9.2f}{best_f:>9.3f}{best_f-per[i]:>9.3f}  {arrow}")
print(f"\n자유 임계값이었다면 Macro F1 = {base + gap_total/9:.5f}  (+{gap_total/9:.5f})")
print("→ 이 격차는 '모델이 못 배운 것'이 아니라 '0.5에서 잘리는 위치' 문제입니다.\n")

print("=== 3. 오심 / 구토 결합 구조 (정답 기준) ===")
o, g = Y[:, TARGETS.index("오심")], Y[:, TARGETS.index("구토")]
for a in (0, 1):
    for b in (0, 1):
        n = ((o == a) & (g == b)).sum()
        print(f"  오심={a} 구토={b}: {n:>6}건 ({n/len(Y)*100:>5.1f}%)")
print(f"  P(오심=1 | 구토=1) = {o[g==1].mean():.3f}")
print(f"  P(구토=1 | 오심=1) = {g[o==1].mean():.3f}")
print(f"  피어슨 상관 = {np.corrcoef(o, g)[0,1]:.3f}\n")

print("=== 4. 오심을 완벽히 맞히면 Macro F1은 얼마가 되나 (상한) ===")
for name in ("오심", "두통", "전신쇠약"):
    i = TARGETS.index(name)
    pred = (P >= .5).astype(np.int8); pred[:, i] = Y[:, i]
    print(f"  {name} 완벽 → Macro F1 {macro_f1(Y, pred):.5f}  (+{macro_f1(Y,pred)-base:.5f})")
