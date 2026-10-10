"""Do the raw and normalised arms end up with the same holdout calls?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TRAIN_DIR, load, split
from normalize import VARIANTS, normalize

names, texts, Y = load(TRAIN_DIR, "train")
tr_raw, te_raw = split(texts, Y, seed=2026)

norm = [normalize(t, **VARIANTS["synonym+filler"]) for t in texts]
tr_n, te_n = split(norm, Y, seed=2026)

print(f"raw  holdout: {len(te_raw):,}")
print(f"norm holdout: {len(te_n):,}")
overlap = len(set(te_raw) & set(te_n))
print(f"겹치는 통화 : {overlap:,}  ({overlap / len(te_raw):.1%})")
print(f"동일한가    : {np.array_equal(te_raw, te_n)}")
print()
print("각 holdout의 양성 개수 (증상별):")
print("  raw :", Y[te_raw].sum(axis=0).tolist())
print("  norm:", Y[te_n].sum(axis=0).tolist())
