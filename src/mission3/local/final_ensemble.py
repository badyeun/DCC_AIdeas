"""Build the submission ensemble from validated configurations only.

Throwing every trained model into the average is wrong: the pool includes runs
that were deliberately bad (weak pos_weight, ASL, 8e-5, small data fractions,
weak backbones). Those were experiments, not candidates. Only configurations
that survived their own A/B belong here.

Which members to include is itself a choice, so it is made from the rules the
experiments established -- not by scanning the holdout for the best subset.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1                                   # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"

# Every entry is a configuration that either won or tied its paired A/B.
# Excluded on evidence: weak/strong pos_weight, ASL, focal, lr 1e-5/5e-5/8e-5,
# 6 epochs, bagging, other backbones, reduced data.
CANDIDATES = {
    "mean 3e-5 s1":   "probs_roberta-base_raw_s1.npy",
    "mean 3e-5 s7":   "probs_roberta-base_raw_s7.npy",
    "mean 3e-5 s42":  "probs_roberta-base_raw_s42.npy",
    "mean 2e-5 s1":   "probs_roberta-base_raw_s1_lr2e-5e3.npy",
    "mean 2e-5 s7":   "probs_roberta-base_raw_s7_lr2e-5e3.npy",
    "mean 2e-5 s42":  "probs_roberta-base_raw_s42_lr2e-5e3.npy",
    "attn 3e-5 s1":   "probs_roberta-base_attn_raw_s1.npy",
    "attn 3e-5 s7":   "probs_roberta-base_attn_raw_s7.npy",
    "attn 3e-5 s42":  "probs_roberta-base_attn_raw_s42.npy",
    "norm 3e-5 s1":   "probs_roberta-base_synonym+filler_s1.npy",
    "norm 3e-5 s7":   "probs_roberta-base_synonym+filler_s7.npy",
    "norm 3e-5 s42":  "probs_roberta-base_synonym+filler_s42.npy",
    "large 3e-5 s42": "probs_roberta-large_raw_s42.npy",
    "large norm s42": "probs_roberta-large_synonym+filler_s42.npy",
    "EMA 2e-5 s1":    "probs_roberta-base_ema0.999_raw_s1.npy",
    "EMA 2e-5 s7":    "probs_roberta-base_ema0.999_raw_s7.npy",
    "EMA 2e-5 s42":   "probs_roberta-base_ema0.999_raw_s42.npy",
}


def score(P, Y):
    return macro_f1(Y, (P >= 0.5).astype(np.int8))


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    runs, missing = {}, []
    for name, fn in CANDIDATES.items():
        path = OUT / fn
        (runs.__setitem__(name, np.load(path)) if path.exists() else missing.append(name))
    if missing:
        print(f"없는 파일: {', '.join(missing)}\n")

    print(f"후보 {len(runs)}개 · 자체검증 {len(Y):,}통화 · 임계값 0.5\n")
    singles = {n: score(P, Y) for n, P in runs.items()}
    print("=" * 56)
    print("개별 (검증된 설정만)")
    print("=" * 56)
    for n in sorted(singles, key=lambda k: -singles[k]):
        print(f"  {n:<18} {singles[n]:.5f}")
    print(f"\n  평균 {np.mean(list(singles.values())):.5f} · 최고 {max(singles.values()):.5f}")

    def ens(names):
        return score(np.mean([runs[n] for n in names], axis=0), Y)

    groups = {
        "mean 헤드만 (6)":      [n for n in runs if n.startswith("mean")],
        "attn 헤드만 (3)":      [n for n in runs if n.startswith("attn")],
        "EMA만 (3)":            [n for n in runs if n.startswith("EMA")],
        "mean + attn (9)":      [n for n in runs if n.startswith(("mean", "attn"))],
        "EMA 제외 전부":        [n for n in runs if not n.startswith("EMA")],
        "검증 후보 전부":       list(runs),
    }
    print("\n" + "=" * 56)
    print("미리 정한 구성")
    print("=" * 56)
    for label, names in groups.items():
        if len(names) >= 2:
            print(f"  {label:<22} {len(names):>2}개  {ens(names):.5f}")

    print("\n" + "=" * 56)
    print("크기별 기댓값 (무작위 조합 60개 평균)")
    print("=" * 56)
    keys, rng = sorted(runs), np.random.default_rng(0)
    for size in range(2, len(keys) + 1):
        total = 1
        for i in range(size):
            total = total * (len(keys) - i) // (i + 1)
        picks = (list(combinations(keys, size)) if total <= 60 else
                 [tuple(keys[i] for i in rng.choice(len(keys), size, replace=False))
                  for _ in range(60)])
        values = [ens(p) for p in picks]
        print(f"  {size:>2}개  평균 {np.mean(values):.5f}  (최저 {min(values):.5f} / 최고 {max(values):.5f})")


if __name__ == "__main__":
    main()
