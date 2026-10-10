"""Does backbone diversity ensemble better than seed diversity?

Same-sized ensembles, so the only difference is where the diversity comes from.
Also reports how much the members actually disagree: a weak model only helps an
ensemble if it is wrong in different places.

Run:  PYTHONPATH=src/mission3/local python src/mission3/local/diversity.py
"""
from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1                                   # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"

SEED_FAMILY = [("roberta-base", "raw", s) for s in (1, 7, 42)]
BACKBONES = ["KcELECTRA-base-v2022", "koelectra-base-v3-discriminator",
             "bert-base", "kcbert-base"]


def load_runs():
    runs = {}
    for path in sorted(OUT.glob("probs_*.npy")):
        match = re.fullmatch(
            r"probs_(.+?)_(raw|synonym\+filler|synonym|privacy[^_]*)_s(\d+)(_\w+)?", path.stem)
        if match:
            model, norm, seed, suffix = match.groups()
            runs[(model, norm + (suffix or ""), int(seed))] = np.load(path)
    return runs


def score(P, Y):
    return macro_f1(Y, (P >= 0.5).astype(np.int8))


def ensemble(runs, keys, Y):
    return score(np.mean([runs[k] for k in keys], axis=0), Y)


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    runs = load_runs()

    print(f"자체검증 {len(Y):,}통화 · 임계값 0.5 · 공식 Validation 미사용\n")

    print("=" * 66)
    print("1. 개별 성능")
    print("=" * 66)
    singles = {k: score(P, Y) for k, P in runs.items()}
    for key in sorted(singles, key=lambda k: -singles[k]):
        print(f"  {key[0]:<34} {key[1]:<16} s{key[2]:<5} {singles[key]:.5f}")

    new = [k for k in runs if k[0] in BACKBONES]
    seeds = [k for k in SEED_FAMILY if k in runs]
    if len(new) < 2 or len(seeds) < 2:
        print("\n비교에 필요한 실행이 아직 모자랍니다.")
        return

    print("\n" + "=" * 66)
    print("2. 같은 크기 앙상블: 백본 다양성 vs seed 다양성")
    print("=" * 66)
    size = min(len(new), len(seeds))
    a = ensemble(runs, new[:size], Y)
    b = ensemble(runs, seeds[:size], Y)
    print(f"  백본 {size}종  : {a:.5f}   ({', '.join(k[0] for k in new[:size])})")
    print(f"  seed {size}개  : {b:.5f}   (klue/roberta-base)")
    print(f"  차이         : {a - b:+.5f}")
    print(f"  → {'백본 다양성이 낫습니다' if a > b else 'seed 다양성이 낫거나 같습니다'}")

    print("\n" + "=" * 66)
    print("3. 예측이 실제로 얼마나 다른가 (겹치지 않는 판정 비율)")
    print("=" * 66)
    print("  낮은 점수라도 서로 다르게 틀리면 앙상블에 기여합니다.\n")
    hard = {k: (P >= 0.5) for k, P in runs.items()}

    def disagreement(k1, k2):
        return float(np.mean(hard[k1] != hard[k2]))

    print("  같은 모델 seed끼리:")
    for k1, k2 in combinations(seeds, 2):
        print(f"    s{k1[2]} vs s{k2[2]:<4}  {disagreement(k1, k2):.3%}")
    base = seeds[0]
    print(f"\n  klue/roberta-base(s{base[2]}) 대비 다른 백본:")
    for k in new:
        print(f"    {k[0]:<34} {disagreement(base, k):.3%}")

    print("\n" + "=" * 66)
    print("4. 전체 앙상블 (미리 정한 구성만)")
    print("=" * 66)
    everything = sorted(runs)
    roberta_only = [k for k in runs if k[0] in ("roberta-base", "roberta-large")]
    for label, keys in (("klue 계열만", roberta_only),
                        ("새 백본만", new),
                        ("전부", everything)):
        if len(keys) >= 2:
            print(f"  {label:<14} {len(keys):>2}개  {ensemble(runs, keys, Y):.5f}")
    print(f"\n  단일 최고: {max(singles.values()):.5f}")


if __name__ == "__main__":
    main()
