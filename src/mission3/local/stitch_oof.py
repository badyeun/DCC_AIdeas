"""Stitch the fold predictions into one soft-target array.

Each training call gets a probability from the one fold model that did NOT
train on it, so the soft targets carry no memory of that call's own label.
Holdout rows keep their hard labels; the student never trains on them.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, TRAIN_DIR, load, macro_f1, split

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
names, texts, Y = load(TRAIN_DIR, "train")
tr, te = split(texts, Y, seed=2026)

soft = Y.astype(np.float32).copy()
covered = np.zeros(len(Y), bool)
folds = sorted(R.glob("oof_*_k4f*.npy"))
for p in folds:
    idx = np.load(R / p.name.replace("oof_", "oofidx_"))
    soft[idx] = np.load(p)
    covered[idx] = True
print(f"폴드 {len(folds)}개 · 커버 {covered.sum():,} / 학습분할 {len(tr):,}")
if covered.sum() != len(tr):
    raise SystemExit("OOF가 학습 분할을 전부 덮지 못했습니다.")

s, per = macro_f1(Y[tr], (soft[tr] >= .5).astype(np.int8), per_symptom=True)
print(f"교사 OOF Macro F1 = {s:.5f}   " + " ".join(f"{t} {v:.3f}" for t, v in zip(TARGETS, per)))
print("증상별 평균 소프트 확률 (양성/음성):")
for i, t in enumerate(TARGETS):
    a = soft[tr][Y[tr][:, i] == 1, i].mean(); b = soft[tr][Y[tr][:, i] == 0, i].mean()
    print(f"  {t:<8} 양성 {a:.3f} · 음성 {b:.3f} · 분리도 {a-b:.3f}")
out = R / "soft_targets_oof.npy"
np.save(out, soft)
print(f"\n저장: {out.name}  {soft.shape}")
