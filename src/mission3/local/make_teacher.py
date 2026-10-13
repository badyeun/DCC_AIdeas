"""앙상블 증류용 교사 확률을 만든다 — 전체 29,200통화, load() 순서 정렬.

학습기의 --soft_targets 는 (29200, 9) 를 요구하고 train split 로 색인한다.
교사는 홀드아웃 분할에 학습된 세 모델의 확률 평균이므로, train 구간에서는
표본 내(일반적인 증류 상황), 홀드아웃 구간에서는 표본 외다. 학생은 train
구간만 보고 학습하고 홀드아웃에서 정직하게 채점된다.

Run:  python src/mission3/local/make_teacher.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_local import TRAIN_DIR, load                                   # noqa: E402

R = HERE.parents[2] / "results" / "mission3" / "local"
SRC = R / "submission_models"
# 기본: 홀드아웃 분할에 학습한 3개 (메커니즘 확인용).
PARTS = ["roberta-base_raw_s42_SOUP.ckpt", "roberta-base_raw_s1_SOUP.ckpt",
         "roberta-base_raw_s7_SOUP.ckpt"]
OUT = R / "soft_targets_ens3.npy"
# --dir 를 주면 그 제출 폴더의 모든 체크포인트를 교사로 쓴다.
#
# 16개 구성을 교사로 쓸 때는 평가를 **공식 Validation 으로 해야 한다**:
# 교사가 Training 전체(홀드아웃 포함)를 봤으므로, 학생을 홀드아웃에서 재면
# 교사를 통해 홀드아웃 정보가 새어 들어간다. 교사가 본 적 없는 Validation
# 에서만 정직한 숫자가 나온다.


def main():
    import argparse
    import inference
    global PARTS, OUT, SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="교사로 쓸 제출 폴더")
    ap.add_argument("--out", default="", help="저장할 .npy 이름")
    ap.add_argument("--per_model_dir", default="",
                    help="모델별 Training 확률을 이 폴더에 따로 저장한다. "
                         "교사를 쪼개 여러 학생에게 다른 교사를 주려면 "
                         "평균만으로는 안 되고 멤버별 확률이 필요하다.")
    a = ap.parse_args()
    if a.dir:
        SRC = Path(a.dir)
        PARTS = [f.name for f in sorted(SRC.glob("*.ckpt"))]
    if a.out:
        OUT = R / a.out
    _, texts, Y = load(TRAIN_DIR, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"통화 {len(texts):,}개 · 교사 {len(PARTS)}개 · {device}", flush=True)
    per_dir = Path(a.per_model_dir) if a.per_model_dir else None
    if per_dir:
        per_dir.mkdir(parents=True, exist_ok=True)
    total = None
    for i, p in enumerate(PARTS, 1):
        t0 = time.time()
        cache = per_dir / f"{Path(p).stem}.npy" if per_dir else None
        if cache is not None and cache.exists():
            q = np.load(cache).astype(np.float64)
            print(f"  [{i}/{len(PARTS)}] {p}  (캐시)", flush=True)
        else:
            q = inference.probabilities(SRC / p, texts, device, 32)
            if cache is not None:
                np.save(cache, q.astype(np.float32))
            print(f"  [{i}/{len(PARTS)}] {p}  {time.time()-t0:.0f}초", flush=True)
        total = q if total is None else total + q
    soft = (total / len(PARTS)).astype(np.float32)
    assert soft.shape == Y.shape, (soft.shape, Y.shape)
    np.save(OUT, soft)
    print(f"\n저장: {OUT.name}  {soft.shape}")
    print(f"  확률 평균 {soft.mean():.4f} · 0.5 이상 비율 {(soft>=.5).mean():.4f}")
    print(f"  하드 라벨 양성 비율 {Y.mean():.4f}")


if __name__ == "__main__":
    main()
