"""학습한 트랜스포머 10개와 ECAPA 인코더 파일을 제출용 체크포인트 한 파일로 묶는다.

python pack_model.py --models A0_base.pt ... F1_full_pair.pt --encoder_dir <ECAPA 파일 폴더> --out mission2_final10.ckpt

- `--models`: train_call_model.py가 저장한 .pt. inference.py가 이들의 확률을 단순 평균한다.
- `--encoder_dir`: `speechbrain/spkrec-ecapa-voxceleb` 파일 폴더(hyperparams.yaml, embedding_model.ckpt 등).
  체크포인트 안에 바이트로 넣어, 추론 때 인터넷 없이 임시 폴더에 풀어 쓴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

N_MFCC = 20              # features.mfcc_features의 MFCC 개수 (캐시와 같은 값)
ECAPA_BATCH = 32         # 통화 안 발화를 ECAPA에 묶어 넣는 크기 (캐시와 같은 값)
FALLBACK_SPEAKER = 1     # 처리할 수 없는 발화의 답: Training 다수 클래스(신고자 비율 0.520)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Mission 2 제출용 체크포인트 만들기")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--encoder_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.encoder_dir / "hyperparams.yaml").is_file():
        raise FileNotFoundError(f"ECAPA 인코더 파일이 없습니다: {args.encoder_dir}")
    entries = []
    for path in args.models:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        # model.py는 ECAPA + MFCC 906차원 입력만 받는다. 다른 입력으로 학습한 모델은 묶지 않는다.
        if saved["config"]["in_dim"] != 906 or saved.get("use_stage1") or saved.get("use_second") or saved.get("context") is False:
            raise ValueError(f"제출 모델과 입력 구성이 다릅니다: {path}")
        entries.append({
            "config": saved["config"], "scale": saved["scale"],
            # 가중치는 numpy 배열로 넣는다(체크포인트를 읽을 때 학습 코드가 필요 없게).
            "state_dict": {key: value.cpu().numpy() for key, value in saved["state_dict"].items()},
            "source": path.name, "holdout_accuracy": saved.get("holdout_accuracy"),
            "validation_accuracy": saved.get("validation_accuracy"),
        })
    checkpoint = {
        "model_type": "mission2_call_transformer",
        "transformers": entries,
        "encoder_files": {path.name: path.resolve().read_bytes() for path in sorted(args.encoder_dir.iterdir())
                          if path.is_file() or path.is_symlink()},
        "encoder_dir": "ecapa_voxceleb",
        "n_mfcc": N_MFCC,
        "timestamp_unit": "auto",
        "fallback_speaker": FALLBACK_SPEAKER,
        "batch_size": ECAPA_BATCH,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, args.out)
    print(f"저장: {args.out} ({args.out.stat().st_size / 2 ** 20:.1f}MB) — 트랜스포머 {len(entries)}개")


if __name__ == "__main__":
    main()
