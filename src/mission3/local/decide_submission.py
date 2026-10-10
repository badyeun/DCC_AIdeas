"""제출 구성을 고릅니다 — 미리 정한 후보만 비교합니다.

부분집합을 전수 탐색해 최고를 고르면 그 최고값은 홀드아웃에 맞춰진 값이라
제출 성적이 그만큼 나오지 않습니다. 그래서 후보를 **먼저 적어두고** 그
안에서만 고릅니다. 새 실험이 이긴 경우에만 후보를 추가합니다.

판정은 전 증상 0.5에서의 Macro F1입니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy") if (R / "tfidf_holdout_probs.npy").exists() else None

POOL = {
    "mean3e5": ["raw_s42", "raw_s1", "raw_s7"],
    "attn3e5": ["attn_raw_s42", "attn_raw_s1", "attn_raw_s7"],
    "mean2e5": ["raw_s42_lr2e-5e3", "raw_s1_lr2e-5e3", "raw_s7_lr2e-5e3"],
    "attn2e5": ["attn_raw_s42_lr2e5_LS", "attn_raw_s1_lr2e5_LS", "attn_raw_s7_lr2e5_LS"],
}


def get(group):
    out = []
    for t in POOL[group]:
        p = R / f"probs_roberta-base_{t}.npy"
        if p.exists():
            out.append(np.load(p))
    return out


CANDIDATES = {
    "A. mean2e5 + attn3e5 (현 제출에 가장 가까움)": ["mean2e5", "attn3e5"],
    "B. mean2e5 + attn2e5 (현 제출 구성 그대로)": ["mean2e5", "attn2e5"],
    "C. mean3e5 + attn3e5": ["mean3e5", "attn3e5"],
    "D. 네 그룹 전부": ["mean3e5", "attn3e5", "mean2e5", "attn2e5"],
    "E. attn3e5 단독": ["attn3e5"],
    "F. mean2e5 단독": ["mean2e5"],
}

print(f"{'후보':<44}{'모델수':>5}{'TF-IDF 없음':>13}{'TF-IDF 0.10':>13}")
rows = []
for name, groups in CANDIDATES.items():
    mem = [m for g in groups for m in get(g)]
    if not mem:
        print(f"{name:<44}{'(아직 없음)':>31}")
        continue
    P = np.mean(mem, axis=0)
    a = macro_f1(Y, (P >= .5).astype(np.int8))
    b = macro_f1(Y, ((.9 * P + .1 * T) >= .5).astype(np.int8)) if T is not None else float("nan")
    rows.append((b, name, len(mem), a))
    print(f"{name:<44}{len(mem):>5}{a:>13.5f}{b:>13.5f}")

if rows:
    best = max(rows)
    print(f"\n권장: {best[1]}  (모델 {best[2]}개 + TF-IDF 0.10 → {best[0]:.5f})")
    print("\n증상별:")
    groups = CANDIDATES[best[1]]
    P = np.mean([m for g in groups for m in get(g)], axis=0)
    if T is not None:
        P = .9 * P + .1 * T
    _, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)
    for t, v in zip(TARGETS, per):
        print(f"  {t:<8}{v:.4f}")
    print("\n이 값은 23,360개로 학습한 모델의 홀드아웃 점수입니다. 제출 모델은")
    print("29,200개 전부로 학습하므로 데이터 크기 곡선상 +0.004 정도가 더 붙습니다.")
    print("그 보정분은 측정값이 아니라 외삽입니다.")
