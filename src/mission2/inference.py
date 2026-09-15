"""Mission 2 제출 CSV를 한 번의 명령으로 생성한다."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib

from common import (
    build_annotation_index, detect_timestamp_unit, find_audio_files, get_utterances,
    load_annotation, load_waveform, resolve_annotation_path, slice_utterance,
    waveform_to_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 2 speaker inference.")
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir() or not args.label_dir.is_dir() or not args.ckpt_path.is_file():
        raise FileNotFoundError("audio_dir, label_dir, ckpt_path를 확인하세요.")
    checkpoint = joblib.load(args.ckpt_path)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "mission2_mfcc_random_forest":
        raise ValueError("Mission 2 MFCC-RandomForest 체크포인트가 아닙니다.")
    classifier = checkpoint["classifier"]
    n_mfcc = int(checkpoint.get("n_mfcc", 20))
    configured_unit = str(checkpoint.get("timestamp_unit", "auto"))
    fallback = int(checkpoint.get("fallback_speaker", 0))
    index = build_annotation_index(args.label_dir)
    rows: list[tuple[str, object, object, int]] = []
    failures = 0
    for audio_path in find_audio_files(args.audio_dir):
        label_path = resolve_annotation_path(audio_path, args.audio_dir, args.label_dir, index)
        if label_path is None:
            print(f"[warn] {audio_path.name}: annotation 없음", file=sys.stderr)
            continue
        try:
            annotation = load_annotation(label_path)
            # 추론에서 사용하는 annotation 정보는 발화 경계뿐이다.
            utterances = get_utterances(annotation)
            waveform = load_waveform(audio_path)
            unit = detect_timestamp_unit(annotation, len(waveform) / 16_000) if configured_unit == "auto" else configured_unit
        except (OSError, ValueError) as error:
            print(f"[warn] {audio_path.name}: {error}", file=sys.stderr)
            continue
        for utterance in utterances:
            start, end = utterance.get("startAt"), utterance.get("endAt")
            try:
                feature = waveform_to_feature(slice_utterance(waveform, utterance, unit), n_mfcc).reshape(1, -1)
                speaker = int(classifier.predict(feature)[0])
            except (ValueError, IndexError) as error:
                failures += 1
                speaker = fallback
                print(f"[warn] {audio_path.name} {start}~{end}: {error} -> {fallback}", file=sys.stderr)
            rows.append((audio_path.name, start, end, speaker))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["audio file name", "startAt", "endAt", "speaker"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} utterance predictions to {args.output}")
    if failures:
        print(f"[warn] fallback applied to {failures} utterances", file=sys.stderr)


if __name__ == "__main__":
    main()
