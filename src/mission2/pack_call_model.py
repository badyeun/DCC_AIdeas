"""통화 트랜스포머를 제출용 체크포인트 한 파일로 묶는다.

- `--models`: train_call_model.py가 저장한 .pt (여러 개면 시드 앙상블로 확률을 평균)
- `--base_ckpt`: 기존 화자 임베딩 체크포인트(예: EXP-009). ECAPA 인코더 파일을 여기서 가져온다.
- `--blend`: 최종 확률 = blend × 기존 파이프라인 + (1 − blend) × 트랜스포머 평균.
  0이면 트랜스포머만 쓰고, 기존 파이프라인(목소리 사전 포함)은 체크포인트에 넣지 않는다.
  단, 1단계 확률을 쌓은 트랜스포머가 있으면 그 1단계가 필요하므로 기존 파이프라인을 넣는다.

결과 체크포인트는 inference.py가 `mission2_call_transformer`로 알아보고, 인터넷 없이 한 파일로 추론한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="통화 트랜스포머 제출용 체크포인트 만들기")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--base_ckpt", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import torch

    from speaker_embed import pack_folder

    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend는 0~1이어야 합니다.")
    entries = []
    for path in args.models:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("model_type") != "mission2_call_transformer":
            raise ValueError(f"통화 트랜스포머가 아닙니다: {path}")
        entries.append({
            "config": saved["config"], "scale": saved["scale"], "use_stage1": bool(saved["use_stage1"]),
            "state_dict": {key: value.cpu().numpy() for key, value in saved["state_dict"].items()},
            "source": path.name, "holdout_accuracy": saved.get("holdout_accuracy"),
            "validation_accuracy": saved.get("validation_accuracy"),
        })
    base = joblib.load(args.base_ckpt)
    if base.get("model_type") != "mission2_speaker_embedding":
        raise ValueError("--base_ckpt는 화자 임베딩 체크포인트여야 합니다.")
    if base.get("second_encoder"):
        raise ValueError("트랜스포머는 ECAPA만으로 학습했습니다. WavLM을 쓰는 체크포인트는 base로 쓸 수 없습니다.")
    encoder_files = base.get("encoder_files") or pack_folder(args.base_ckpt.parent / str(base["encoder_dir"]))
    needs_base = args.blend > 0 or any(entry["use_stage1"] for entry in entries)
    checkpoint = {
        "model_type": "mission2_call_transformer",
        "transformers": entries,
        "blend": float(args.blend),
        "base": base if needs_base else None,
        "encoder_files": encoder_files,
        "encoder_dir": str(base.get("encoder_dir", "ecapa_voxceleb")),
        "n_mfcc": int(base.get("n_mfcc", 20)),
        "timestamp_unit": str(base.get("timestamp_unit", "auto")),
        "fallback_speaker": int(base.get("fallback_speaker", 1)),
        "batch_size": int(base.get("batch_size", 32)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, args.out)
    size = args.out.stat().st_size / 2 ** 20
    print(f"저장: {args.out} ({size:.1f}MB) — 트랜스포머 {len(entries)}개, "
          f"기존 파이프라인 {'포함' if needs_base else '제외'}, blend {args.blend}")


if __name__ == "__main__":
    main()
