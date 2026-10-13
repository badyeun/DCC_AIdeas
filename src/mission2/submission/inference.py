"""Mission 2 추론: 발화 조각마다 신고자(1) / 119대원(0)을 판정해 CSV로 저장한다.

python inference.py --audio_dir {wav folder} --label_dir {json folder} --ckpt_path mission2_final10.ckpt \
    --output ./outputs/mission2.csv

처리 순서 (통화 = WAV 파일 하나):
1. JSON에서 발화마다 startAt, endAt만 읽어 음성을 자른다. text, speaker 등 다른 annotation은 읽지 않는다.
2. 발화마다 ECAPA 화자 임베딩(192)과 MFCC 통계(261)를 만든다.
3. 통화의 발화 전체를 통화 단위 트랜스포머 10개(model.py)에 넣어 확률을 평균하고, 0.5 이상이면 1(신고자).
출력 CSV: audio file name, startAt, endAt, speaker (입력 JSON의 발화마다 한 행)
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

from features import (
    EcapaEncoder, build_annotation_index, detect_timestamp_unit, find_audio_files, fit_length, get_utterances,
    load_annotation, load_waveform, mfcc_features, resolve_annotation_path, slice_utterance, unpack_folder,
)
from model import CallEnsemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2: 신고자(1) / 119대원(0) 발화 분류")
    parser.add_argument("--audio_dir", type=Path, required=True, help="WAV 폴더 (하위 폴더까지 찾음)")
    parser.add_argument("--label_dir", type=Path, required=True, help="JSON 폴더 (WAV와 같은 이름으로 짝을 찾음)")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="mission2_final10.ckpt")
    parser.add_argument("--output", type=Path, required=True, help="결과 CSV 경로")
    return parser.parse_args()


class Mission2Predictor:
    """체크포인트 한 파일(트랜스포머 10개 + ECAPA 인코더 파일 + MFCC 표준화 값)로 통화 단위 판정을 한다."""

    def __init__(self, checkpoint: dict) -> None:
        import torch

        if checkpoint.get("model_type") != "mission2_call_transformer":
            raise ValueError("Mission 2 통화 트랜스포머 체크포인트가 아닙니다.")
        self.n_mfcc = int(checkpoint["n_mfcc"])
        self.batch_size = int(checkpoint["batch_size"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 체크포인트 안의 ECAPA 인코더 파일을 임시 폴더에 풀어 쓴다(인터넷 접속 없음).
        folder = unpack_folder(checkpoint["encoder_files"], Path(tempfile.mkdtemp(prefix="mission2_")) / checkpoint["encoder_dir"])
        self.encoder = EcapaEncoder(folder, self.device)
        self.ensemble = CallEnsemble(checkpoint["transformers"], self.device)
        print(f"통화 트랜스포머 {len(self.ensemble.models)}개 평균, 장치: {self.device}")

    def prepare(self, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """발화 하나의 (음성 조각, MFCC 통계). 쓸 수 없는 조각은 여기서 ValueError로 걸러 fallback으로 보낸다."""
        fit_length(segment)
        return segment, mfcc_features(segment, self.n_mfcc)

    def decide(self, prepared: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """통화 하나의 발화들 → 발화별 0/1."""
        segments = [segment for segment, _ in prepared]
        embeddings = self.encoder.embed(segments, self.batch_size)
        mfcc = np.vstack([feature for _, feature in prepared])
        probability = self.ensemble.probability(embeddings, mfcc)
        return (probability >= 0.5).astype(np.int64)


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir() or not args.label_dir.is_dir() or not args.ckpt_path.is_file():
        raise FileNotFoundError("audio_dir, label_dir, ckpt_path를 확인하세요.")
    checkpoint = joblib.load(args.ckpt_path)
    predictor = Mission2Predictor(checkpoint)
    configured_unit = str(checkpoint.get("timestamp_unit", "auto"))
    # 처리할 수 없는 발화(빈 구간 등)는 행을 빠뜨리지 않고 Training 다수 클래스(1)로 채운다.
    fallback = int(checkpoint["fallback_speaker"])
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
            utterances = get_utterances(annotation)
            waveform = load_waveform(audio_path)
            unit = detect_timestamp_unit(annotation, len(waveform) / 16_000) if configured_unit == "auto" else configured_unit
        except (OSError, ValueError) as error:
            print(f"[warn] {audio_path.name}: {error}", file=sys.stderr)
            continue
        # 같은 통화의 발화를 한 번에 모아 판정한다. 발화 순서는 쓰지 않는다.
        entries: list[tuple[object, object, tuple | None]] = []
        for utterance in utterances:
            start, end = utterance.get("startAt"), utterance.get("endAt")
            try:
                prepared = predictor.prepare(slice_utterance(waveform, utterance, unit))
            except (ValueError, IndexError) as error:
                failures += 1
                prepared = None
                print(f"[warn] {audio_path.name} {start}~{end}: {error} -> {fallback}", file=sys.stderr)
            entries.append((start, end, prepared))
        speakers = [fallback] * len(entries)
        usable = [position for position, entry in enumerate(entries) if entry[2] is not None]
        if usable:
            decided = predictor.decide([entries[position][2] for position in usable])
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
