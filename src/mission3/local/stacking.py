"""Does a learned combination beat a plain average of the ensemble?

The holdout is split in half: one half fits the combiner, the other scores it.
Both halves are Training-internal, so official Validation stays untouched, and
the reported number is not the one the weights were fitted on.

Threshold stays 0.5 everywhere -- the combiner changes how the probabilities
are pooled, never where the cut sits.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1                                   # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"


def load_runs():
    runs = {}
    for path in sorted(OUT.glob("probs_*.npy")):
        m = re.fullmatch(
            r"probs_(.+?)_(raw|synonym\+filler|synonym|privacy[^_]*)_s(\d+)(_\w+)?", path.stem)
        if m:
            runs[path.stem[6:]] = np.load(path)
    return runs


def score(P, Y):
    return macro_f1(Y, (P >= 0.5).astype(np.int8))


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    runs = load_runs()
    names = sorted(runs)
    stack = np.stack([runs[n] for n in names])        # (models, calls, symptoms)
    print(f"모델 {len(names)}개 · 자체검증 {len(Y):,}통화\n")

    rng = np.random.default_rng(7)
    order = rng.permutation(len(Y))
    fit, test = order[: len(Y) // 2], order[len(Y) // 2:]
    print(f"결합 학습 {len(fit):,} / 채점 {len(test):,}  (둘 다 Training 내부)\n")

    print("=" * 58)
    print("채점 절반 기준 비교")
    print("=" * 58)
    plain = score(stack.mean(0)[test], Y[test])
    print(f"  단순 평균                  {plain:.5f}")

    # (a) one logistic regression per symptom over the models' probabilities
    learned = np.zeros((len(test), len(TARGETS)))
    for k in range(len(TARGETS)):
        Xf, Xt = stack[:, fit, k].T, stack[:, test, k].T
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xf, Y[fit][:, k])
        learned[:, k] = clf.predict_proba(Xt)[:, 1]
    print(f"  증상별 로지스틱 결합        {score(learned, Y[test]):.5f}")

    # (b) same, but every symptom shares one weight vector over models
    flatX_f = stack[:, fit, :].transpose(1, 2, 0).reshape(-1, len(names))
    flatX_t = stack[:, test, :].transpose(1, 2, 0).reshape(-1, len(names))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(flatX_f, Y[fit].reshape(-1))
    shared = clf.predict_proba(flatX_t)[:, 1].reshape(len(test), len(TARGETS))
    print(f"  전 증상 공통 가중치 결합    {score(shared, Y[test]):.5f}")

    # (c) drop the members that hurt, keep the plain average
    single = {n: score(runs[n][fit], Y[fit]) for n in names}
    keep = [n for n in names if single[n] >= np.median(list(single.values()))]
    trimmed = np.stack([runs[n] for n in keep]).mean(0)
    print(f"  상위 절반만 단순 평균 ({len(keep)}개) {score(trimmed[test], Y[test]):.5f}")
    print("    (어느 모델을 남길지는 결합 학습 절반에서만 고름)")

    print()
    best = max(plain, score(learned, Y[test]), score(shared, Y[test]),
               score(trimmed[test], Y[test]))
    print(f"  최고 {best:.5f} · 단순 평균 대비 {best - plain:+.5f}")


if __name__ == "__main__":
    main()
