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
PARTS = ["roberta-base_raw_s42_SOUP.ckpt", "roberta-base_raw_s1_SOUP.ckpt",
         "roberta-base_raw_s7_SOUP.ckpt"]
OUT = R / "soft_targets_ens3.npy"


def main():
    import inference
    _, texts, Y = load(TRAIN_DIR, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"통화 {len(texts):,}개 · 교사 {len(PARTS)}개 · {device}", flush=True)
    total = None
    for i, p in enumerate(PARTS, 1):
        t0 = time.time()
        q = inference.probabilities(SRC / p, texts, device, 32)
        total = q if total is None else total + q
        print(f"  [{i}/{len(PARTS)}] {p}  {time.time()-t0:.0f}초", flush=True)
    soft = (total / len(PARTS)).astype(np.float32)
    assert soft.shape == Y.shape, (soft.shape, Y.shape)
    np.save(OUT, soft)
    print(f"\n저장: {OUT.name}  {soft.shape}")
    print(f"  확률 평균 {soft.mean():.4f} · 0.5 이상 비율 {(soft>=.5).mean():.4f}")
    print(f"  하드 라벨 양성 비율 {Y.mean():.4f}")


if __name__ == "__main__":
    main()
