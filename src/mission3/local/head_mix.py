"""Are mean-pooling and label-attention models good ensemble partners?

Experiment 6 taught that an ensemble member has to be BOTH similarly good AND
different. Weak backbones failed the first test. Attention heads pass it, so
the question is whether they are different enough to be worth mixing.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import macro_f1                                            # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
SEEDS = (1, 7, 42)


def score(P, Y):
    return macro_f1(Y, (P >= 0.5).astype(np.int8))


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    mean = {s: np.load(OUT / f"probs_roberta-base_raw_s{s}.npy") for s in SEEDS}
    attn = {s: np.load(OUT / f"probs_roberta-base_attn_raw_s{s}.npy") for s in SEEDS}

    print(f"자체검증 {len(Y):,}통화 · 임계값 0.5\n")
    print("=" * 62)
    print("1. 개별")
    print("=" * 62)
    for s in SEEDS:
        a, b = score(mean[s], Y), score(attn[s], Y)
        print(f"  seed {s:<3} mean {a:.5f}   attention {b:.5f}   {b - a:+.5f}")
    print(f"  {'평균':<8} mean {np.mean([score(mean[s], Y) for s in SEEDS]):.5f}   "
          f"attention {np.mean([score(attn[s], Y) for s in SEEDS]):.5f}")

    print("\n" + "=" * 62)
    print("2. 예측이 얼마나 다른가 (판정 불일치율)")
    print("=" * 62)
    hard_m = {s: mean[s] >= 0.5 for s in SEEDS}
    hard_a = {s: attn[s] >= 0.5 for s in SEEDS}
    same_head = [float(np.mean(hard_m[x] != hard_m[y])) for x, y in combinations(SEEDS, 2)]
    same_head += [float(np.mean(hard_a[x] != hard_a[y])) for x, y in combinations(SEEDS, 2)]
    cross = [float(np.mean(hard_m[x] != hard_a[y])) for x in SEEDS for y in SEEDS]
    print(f"  같은 헤드끼리 (seed만 다름) : {np.mean(same_head):.3%}")
    print(f"  헤드가 다른 것끼리          : {np.mean(cross):.3%}")
    print(f"  차이                        : {np.mean(cross) - np.mean(same_head):+.3%}")
    print("  (실험 6의 백본 다양화는 +0.8%p였는데 개별 성능이 나빠 실패했습니다)")

    print("\n" + "=" * 62)
    print("3. 같은 크기 앙상블")
    print("=" * 62)

    def ens(members):
        return score(np.mean(members, axis=0), Y)

    print(f"  mean 3개        : {ens([mean[s] for s in SEEDS]):.5f}")
    print(f"  attention 3개   : {ens([attn[s] for s in SEEDS]):.5f}")
    mixed3 = [ens([mean[a], attn[b], attn[c]]) for a in SEEDS
              for b, c in combinations(SEEDS, 2)]
    print(f"  혼합 3개 (평균) : {np.mean(mixed3):.5f}   "
          f"(최저 {min(mixed3):.5f} / 최고 {max(mixed3):.5f})")

    print("\n  6개 전부:")
    print(f"  mean 3 + attention 3 : {ens(list(mean.values()) + list(attn.values())):.5f}")
    print(f"  (참고) mean 3개만    : {ens([mean[s] for s in SEEDS]):.5f}")


if __name__ == "__main__":
    main()
