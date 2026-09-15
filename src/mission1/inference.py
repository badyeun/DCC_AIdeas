"""한 번의 명령으로 Mission 1 제출 CSV를 생성합니다.

평가 환경에서는 어떤 파일이 들어올지 알 수 없습니다. 파일 하나가 잘못되어도
전체가 중단되면 CSV가 만들어지지 않아 0점이 되므로, 이 스크립트는 개별 파일의
실패를 흡수하고 **입력 WAV 개수와 항상 같은 수의 행**을 기록합니다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import torch

from common import (
    INDEX_TO_GENDER,
    MAX_FRAMES,
    SimpleCNN,
    build_annotation_index,
    load_annotation,
    load_caller_audio,
    resolve_annotation_path,
    waveform_to_mel_windows,
    waveform_to_mfcc_feature,
)


DEFAULT_FALLBACK_GENDER = "F"
WINDOW_BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 1 gender inference.")
    # 주최 측이 지정한 실행 명령의 인자명과 일치시킵니다.
    parser.add_argument("--audio_dir", type=Path, required=True, help="WAV input folder")
    parser.add_argument("--label_dir", type=Path, required=True, help="JSON annotation folder")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Trained checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[SimpleCNN, int, int, str, str]:
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
        int(checkpoint.get("window_hop_frames", MAX_FRAMES // 2)),
        str(checkpoint.get("timestamp_unit", "auto")),
        str(checkpoint.get("fallback_gender", DEFAULT_FALLBACK_GENDER)),
    )


def load_mfcc_rf_checkpoint(path: Path) -> tuple[object, int, bool, str, str]:
    """MFCC-랜덤 포레스트 체크포인트와 전처리 설정을 불러옵니다."""
    checkpoint = joblib.load(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "mfcc_random_forest":
        raise ValueError("MFCC-랜덤 포레스트 체크포인트 형식이 올바르지 않습니다.")
    return (
        checkpoint["classifier"],
        int(checkpoint.get("n_mfcc", 20)),
        # F0 도입 이전 체크포인트에는 이 키가 없으므로 기존 동작(False)을 따릅니다.
        bool(checkpoint.get("use_f0", False)),
        str(checkpoint.get("timestamp_unit", "auto")),
        str(checkpoint.get("fallback_gender", DEFAULT_FALLBACK_GENDER)),
    )


def find_audio_files(audio_dir: Path) -> list[Path]:
    """대소문자를 가리지 않고 WAV 파일을 모읍니다."""
    files = {
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    }
    return sorted(files)


def predict_with_cnn(
    model: SimpleCNN,
    waveform,
    max_frames: int,
    window_hop_frames: int,
    device: torch.device,
) -> str:
    """모든 sliding window의 확률을 합산해 통화 하나의 성별을 정합니다."""
    windows = waveform_to_mel_windows(waveform, max_frames, window_hop_frames)
    probability_sum = torch.zeros(len(INDEX_TO_GENDER), device=device)
    with torch.inference_mode():
        for window_batch in windows.split(WINDOW_BATCH_SIZE):
            probabilities = torch.softmax(model(window_batch.to(device)), dim=1)
            probability_sum += probabilities.sum(dim=0)
    return INDEX_TO_GENDER[int(probability_sum.argmax().item())]


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir():
        raise NotADirectoryError(f"audio_dir is not a directory: {args.audio_dir}")
    if not args.label_dir.is_dir():
        raise NotADirectoryError(f"label_dir is not a directory: {args.label_dir}")
    if not args.ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    is_mfcc_rf = args.ckpt_path.suffix.lower() == ".joblib"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_mfcc_rf:
        model, n_mfcc, use_f0, timestamp_unit, fallback_gender = load_mfcc_rf_checkpoint(
            args.ckpt_path
        )
        max_frames, window_hop_frames = MAX_FRAMES, MAX_FRAMES // 2
    else:
        (
            model,
            max_frames,
            window_hop_frames,
            timestamp_unit,
            fallback_gender,
        ) = load_checkpoint(args.ckpt_path, device)
        n_mfcc, use_f0 = 20, False

    audio_files = find_audio_files(args.audio_dir)
    if not audio_files:
        warn(f"No .wav files found in {args.audio_dir}. Writing an empty CSV.")
    annotation_index = build_annotation_index(args.label_dir)

    rows: list[tuple[str, str]] = []
    failures = 0
    for audio_path in audio_files:
        gender = fallback_gender
        try:
            json_path = resolve_annotation_path(
                audio_path, args.audio_dir, args.label_dir, annotation_index
            )
            if json_path is None:
                raise FileNotFoundError(f"annotation not found for {audio_path.name}")

            waveform = load_caller_audio(
                audio_path, load_annotation(json_path), timestamp_unit=timestamp_unit
            )
            if is_mfcc_rf:
                feature = waveform_to_mfcc_feature(
                    waveform, n_mfcc, use_f0=use_f0
                ).reshape(1, -1)
                gender = INDEX_TO_GENDER[int(model.predict(feature)[0])]
            else:
                gender = predict_with_cnn(
                    model, waveform, max_frames, window_hop_frames, device
                )
        except Exception as error:  # 개별 파일 실패가 전체를 중단시키지 않게 합니다.
            failures += 1
            gender = fallback_gender
            warn(f"{audio_path.name}: {type(error).__name__}: {error} -> '{gender}' 사용")

        rows.append((audio_path.name, gender))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["audio file name", "gender"])
        writer.writerows(rows)

    assert len(rows) == len(audio_files), "CSV 행 수가 입력 WAV 개수와 다릅니다."
    print(f"Saved {len(rows)} predictions to {args.output}")
    if failures:
        warn(f"{failures}/{len(rows)} files fell back to '{fallback_gender}'.")


if __name__ == "__main__":
    main()
