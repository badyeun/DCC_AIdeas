"""전체 데이터 체크포인트별 Validation 확률을 한 번만 계산해 캐시한다.

조합을 바꿔가며 재려면 모델마다 추론을 다시 돌릴 수 없다. 모델별 확률을
저장해 두면 어떤 부분집합이든 즉시 평가된다.

주의: 이 파일들로 **조합을 고르면** Validation 에 맞추는 것이 된다. 선택은
Training 내부 홀드아웃에서 하고, 여기 저장된 값은 미리 정한 후보를 확인하는
데만 쓴다. (얼마나 차이나는지 보이기 위해 'Validation 최적'도 함께 보고하되,
그것을 제출 근거로 쓰지는 않는다.)

Run:  python src/mission3/local/val_probs.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
VAL = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
OUT = ROOT / "results" / "mission3" / "local" / "val_probs"
SRC = ROOT / "results" / "mission3" / "local" / "submission_models"
# LGE2 가 대체한 2-epoch 구버전은 제외한다 (앙상블 이득의 절반을 잃음).
SKIP = {"roberta-large_raw_s1_LGF.ckpt", "roberta-large_raw_s7_LGF.ckpt",
        "roberta-large_attn_raw_s42_LGF.ckpt"}


def main():
    import inference
    OUT.mkdir(parents=True, exist_ok=True)
    names, texts = inference.read_inputs(VAL)
    np.save(OUT / "_names.npy", np.array(names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [p for p in sorted(SRC.glob("*.ckpt")) if p.name not in SKIP]
    print(f"통화 {len(names):,}개 · 체크포인트 {len(todo)}개 · {device}", flush=True)
    for i, p in enumerate(todo, 1):
        dst = OUT / f"{p.stem}.npy"
        if dst.exists():
            print(f"  [{i}/{len(todo)}] {p.stem}  (캐시 있음)", flush=True)
            continue
        t0 = time.time()
        prob = inference.probabilities(p, texts, device, 32)
        np.save(dst, prob.astype(np.float32))
        print(f"  [{i}/{len(todo)}] {p.stem}  {time.time()-t0:.0f}초", flush=True)
    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
