"""한 번의 명령으로 Mission 1 제출 CSV를 생성합니다."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import torch

from common import (
    INDEX_TO_GENDER,
    MAX_FRAMES,
    SimpleCNN,
    annotation_path_for,
    load_annotation,
    load_caller_audio,
    waveform_to_mfcc_feature,
    waveform_to_mel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 1 gender inference.")
    # 주최 측이 지정한 실행 명령의 인자명과 일치시킵니다.
    parser.add_argument("--audio_dir", type=Path, required=True, help="WAV input folder")
    parser.add_argument("--label_dir", type=Path, required=True, help="JSON annotation folder")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Trained .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[SimpleCNN, int, str]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # 구버전 PyTorch 호환을 위한 처리입니다.
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint must be created by train.py and contain model_state_dict."
        )
    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        int(checkpoint.get("max_frames", MAX_FRAMES)),
        str(checkpoint.get("timestamp_unit", "auto")),
    )


def load_mfcc_rf_checkpoint(path: Path) -> tuple[object, int, str]:
    """MFCC-랜덤 포레스트 체크포인트와 전처리 설정을 불러옵니다."""
    checkpoint = joblib.load(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "mfcc_random_forest":
        raise ValueError("MFCC-랜덤 포레스트 체크포인트 형식이 올바르지 않습니다.")
    return (
        checkpoint["classifier"],
        int(checkpoint.get("n_mfcc", 20)),
        str(checkpoint.get("timestamp_unit", "auto")),
    )


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir() or not args.label_dir.is_dir():
        raise NotADirectoryError("audio_dir and label_dir must both be existing directories")
    if not args.ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    is_mfcc_rf = args.ckpt_path.suffix.lower() == ".joblib"
    if is_mfcc_rf:
        model, n_mfcc, timestamp_unit = load_mfcc_rf_checkpoint(args.ckpt_path)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, max_frames, timestamp_unit = load_checkpoint(args.ckpt_path, device)
    audio_files = sorted(args.audio_dir.rglob("*.wav"))
    if not audio_files:
        raise ValueError(f"No .wav files found in {args.audio_dir}")

    rows: list[tuple[str, str]] = []
    for audio_path in audio_files:
        json_path = annotation_path_for(audio_path, args.audio_dir, args.label_dir)
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation for {audio_path}: {json_path}")
        waveform = load_caller_audio(
            audio_path, load_annotation(json_path), timestamp_unit=timestamp_unit
        )
        if is_mfcc_rf:
            feature = waveform_to_mfcc_feature(waveform, n_mfcc).reshape(1, -1)
            gender = INDEX_TO_GENDER[int(model.predict(feature)[0])]
        else:
            feature = waveform_to_mel(waveform, max_frames).unsqueeze(0).to(device)
            with torch.inference_mode():
                gender = INDEX_TO_GENDER[model(feature).argmax(dim=1).item()]
        rows.append((audio_path.name, gender))

    if args.output.parent != Path("."):
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["audio file name", "gender"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
