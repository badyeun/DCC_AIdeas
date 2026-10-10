"""Does bagging ensemble better than seed variation at the same size?

Both arms use the same recipe and the same holdout. Bagging members each see a
different 80% of the train split; seed members all see the same 100% of it.
Experiment 6 showed diversity only pays when member quality survives, so the
individual scores are reported alongside.
"""
from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import macro_f1                                           # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"


def score(P, Y):
    return macro_f1(Y, (P >= 0.5).astype(np.int8))


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    bags = {int(re.search(r"b(\d+)", p.stem).group(1)): np.load(p)
            for p in sorted(OUT.glob("probs_roberta-base_f0.8b*_raw_s42.npy"))}
    seeds = {s: np.load(OUT / f"probs_roberta-base_raw_s{s}_lr2e-5e3.npy")
             for s in (1, 7, 42)}
    if len(bags) < 3:
        print(f"배깅 멤버가 {len(bags)}개뿐입니다.")
        return

    print(f"자체검증 {len(Y):,}통화 · 임계값 0.5\n")
    print("=" * 60)
    print("1. 개별 성능")
    print("=" * 60)
    for b, P in sorted(bags.items()):
        print(f"  bag{b} (데이터 80%)   {score(P, Y):.5f}")
    bag_mean = np.mean([score(P, Y) for P in bags.values()])
    print(f"  {'배깅 평균':<18} {bag_mean:.5f}")
    for s, P in sorted(seeds.items()):
        print(f"  seed{s} (데이터 100%)  {score(P, Y):.5f}")
    seed_mean = np.mean([score(P, Y) for P in seeds.values()])
    print(f"  {'seed 평균':<18} {seed_mean:.5f}")
    print(f"\n  개별 품질 차이: {bag_mean - seed_mean:+.5f}  "
          f"(데이터 20%를 덜 본 대가)")

    print("\n" + "=" * 60)
    print("2. 예측 불일치율 (다양성)")
    print("=" * 60)
    hb = {b: P >= 0.5 for b, P in bags.items()}
    hs = {s: P >= 0.5 for s, P in seeds.items()}
    d_bag = [float(np.mean(hb[x] != hb[y])) for x, y in combinations(sorted(bags), 2)]
    d_seed = [float(np.mean(hs[x] != hs[y])) for x, y in combinations(sorted(seeds), 2)]
    print(f"  배깅끼리   {np.mean(d_bag):.3%}")
    print(f"  seed끼리   {np.mean(d_seed):.3%}")
    print(f"  차이       {np.mean(d_bag) - np.mean(d_seed):+.3%}")

    print("\n" + "=" * 60)
    print("3. 같은 크기 앙상블")
    print("=" * 60)
    size = min(len(bags), len(seeds))
    bag_keys = sorted(bags)[:size]
    bag_ens = score(np.mean([bags[b] for b in bag_keys], axis=0), Y)
    seed_ens = score(np.mean(list(seeds.values()), axis=0), Y)
    print(f"  배깅 {size}개   {bag_ens:.5f}")
    print(f"  seed {size}개   {seed_ens:.5f}")
    print(f"  차이         {bag_ens - seed_ens:+.5f}")
    print(f"  → {'배깅이 낫습니다' if bag_ens > seed_ens else 'seed 다양성이 낫거나 같습니다'}")

    if len(bags) > size:
        full = score(np.mean(list(bags.values()), axis=0), Y)
        print(f"\n  배깅 전체 {len(bags)}개  {full:.5f}")
    mixed = score(np.mean(list(bags.values()) + list(seeds.values()), axis=0), Y)
    print(f"  배깅+seed 전부 {len(bags) + len(seeds)}개  {mixed:.5f}")


if __name__ == "__main__":
    main()
