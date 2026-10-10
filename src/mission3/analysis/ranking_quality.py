"""Separate 'where the cut sits' from 'how well the model ranks'.

A monotone recalibration cannot reorder anything, so the best F1 reachable by
ANY per-symptom threshold is already the ceiling for a given ranking. That
ceiling was measured at 0.6503 against 0.6441 at the fixed 0.5 -- six
thousandths. Everything else has to come from a better ordering of the calls.

Average precision measures exactly that ordering and ignores where the cut
sits, so it is both the right target and a far quieter yardstick than F1@0.5:
it does not flip on a handful of calls sitting astride the threshold.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")

def summarise(tag, P):
    f1, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)
    ap = [average_precision_score(Y[:, i], P[:, i]) for i in range(9)]
    return f1, np.mean(ap), per, ap

singles = ["probs_roberta-base_raw_s42_lr2e-5e3.npy",
           "probs_roberta-base_raw_s1_lr2e-5e3.npy",
           "probs_roberta-base_raw_s7_lr2e-5e3.npy"]
print("=== 단일 모델 3개: F1@0.5는 흔들리고 AP는 안정적인가 ===")
print(f"{'실행':<34}{'F1@0.5':>9}{'평균 AP':>10}")
rows = [summarise(s, np.load(R / s)) for s in singles]
for s, (f1, ap, _, _) in zip(singles, rows):
    print(f"{s[6:-4]:<34}{f1:>9.5f}{ap:>10.5f}")
f1s = [r[0] for r in rows]; aps = [r[1] for r in rows]
print(f"{'편차 (최대-최소)':<34}{max(f1s)-min(f1s):>9.5f}{max(aps)-min(aps):>10.5f}")
print(f"→ AP의 seed 편차가 F1의 {(max(f1s)-min(f1s))/(max(aps)-min(aps)):.1f}분의 1."
      f" 같은 효과를 그만큼 적은 실행으로 잴 수 있습니다.\n")

mem = ["probs_roberta-base_raw_s42.npy", "probs_roberta-base_raw_s1.npy",
       "probs_roberta-base_raw_s7.npy", "probs_roberta-base_attn_raw_s42.npy",
       "probs_roberta-base_attn_raw_s1.npy", "probs_roberta-base_attn_raw_s7.npy"] + singles
P = np.mean([np.load(R / m) for m in mem], axis=0)
f1, ap, per, aps = summarise("ens", P)
print(f"=== 9모델 앙상블 기준선: F1@0.5 {f1:.5f} · 평균 AP {ap:.5f} ===")
print(f"{'증상':<8}{'F1@0.5':>9}{'AP':>9}{'AUC':>9}{'기저율':>9}   AP가 기저율의 몇 배")
for i, t in enumerate(TARGETS):
    base = Y[:, i].mean()
    print(f"{t:<8}{per[i]:>9.3f}{aps[i]:>9.3f}"
          f"{roc_auc_score(Y[:, i], P[:, i]):>9.3f}{base:>9.3f}{aps[i]/base:>14.1f}")
np.save(R / "_baseline_ens_probs.npy", P)
print(f"\n기준선 확률 저장: _baseline_ens_probs.npy")
