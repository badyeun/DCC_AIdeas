"""Mission 2 제출 CSV를 한 번의 명령으로 생성한다."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

from common import (
    build_annotation_index, detect_timestamp_unit, find_audio_files, get_utterances,
    load_annotation, load_waveform, predict_call, resolve_annotation_path, slice_utterance,
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
    # 통화 맥락 설정이 없는 예전 체크포인트는 발화 하나만 보던 동작을 그대로 따른다.
    call_context = str(checkpoint.get("call_context", "none"))
    cluster = bool(checkpoint.get("cluster", False))
    min_utterances = int(checkpoint.get("cluster_min_utterances", 6))
    margin = float(checkpoint.get("cluster_margin", 0.10))
    seed = int(checkpoint.get("seed", 42))
    print(f"통화 맥락: {call_context}, 군집 묶기: {cluster}")
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
        # 같은 통화의 발화를 한 번에 모아 서로 비교한다. 발화 순서는 쓰지 않는다.
        entries: list[tuple[object, object, np.ndarray | None]] = []
        for utterance in utterances:
            start, end = utterance.get("startAt"), utterance.get("endAt")
            try:
                feature = waveform_to_feature(slice_utterance(waveform, utterance, unit), n_mfcc)
            except (ValueError, IndexError) as error:
                failures += 1
                feature = None
                print(f"[warn] {audio_path.name} {start}~{end}: {error} -> {fallback}", file=sys.stderr)
            entries.append((start, end, feature))
        speakers = [fallback] * len(entries)
        usable = [position for position, entry in enumerate(entries) if entry[2] is not None]
        if usable:
            base = np.vstack([entries[position][2] for position in usable])
            decided = predict_call(classifier, base, call_context, cluster, min_utterances, margin, seed)
            for order, position in enumerate(usable):
                speakers[position] = int(decided[order])
        for (start, end, _), speaker in zip(entries, speakers):
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
