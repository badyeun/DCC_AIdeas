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


class RandomForestPredictor:
    """EXP-001/002 MFCC + RandomForest 체크포인트."""

    def __init__(self, checkpoint: dict) -> None:
        self.classifier = checkpoint["classifier"]
        self.n_mfcc = int(checkpoint.get("n_mfcc", 20))
        # 통화 맥락 설정이 없는 예전 체크포인트는 발화 하나만 보던 동작을 그대로 따른다.
        self.call_context = str(checkpoint.get("call_context", "none"))
        self.cluster = bool(checkpoint.get("cluster", False))
        self.min_utterances = int(checkpoint.get("cluster_min_utterances", 6))
        self.margin = float(checkpoint.get("cluster_margin", 0.10))
        self.seed = int(checkpoint.get("seed", 42))
        print(f"MFCC-RandomForest, 통화 맥락: {self.call_context}, 군집 묶기: {self.cluster}")

    def prepare(self, segment: np.ndarray) -> np.ndarray:
        return waveform_to_feature(segment, self.n_mfcc)

    def decide(self, prepared: list[np.ndarray]) -> np.ndarray:
        return predict_call(self.classifier, np.vstack(prepared), self.call_context, self.cluster,
                            self.min_utterances, self.margin, self.seed)


class SpeakerEmbeddingPredictor:
    """화자 임베딩 + 통화 내 비교 체크포인트 (train_speaker_embed.py)."""

    def __init__(self, checkpoint: dict, ckpt_dir: Path) -> None:
        import torch

        from speaker_embed import load_encoder

        self.checkpoint = checkpoint
        self.n_mfcc = int(checkpoint.get("n_mfcc", 20))
        self.use_mfcc = bool(checkpoint.get("use_mfcc", False))
        self.batch_size = int(checkpoint.get("batch_size", 32))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 학습 때 체크포인트 옆에 복사해 둔 인코더를 쓰므로 인터넷이 필요 없다.
        self.encoder = load_encoder(ckpt_dir / str(checkpoint["encoder_dir"]), device)
        print(f"화자 임베딩, 장치: {device}, MFCC: {self.use_mfcc}, 2단계: {bool(checkpoint.get('use_stage2'))}")

    def prepare(self, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        from speaker_embed import fit_length

        fit_length(segment)  # 쓸 수 없는 조각은 여기서 걸러 fallback으로 보낸다.
        return segment, waveform_to_feature(segment, self.n_mfcc) if self.use_mfcc else None

    def decide(self, prepared: list[tuple[np.ndarray, np.ndarray | None]]) -> np.ndarray:
        from speaker_embed import embed_segments, predict_call as predict_embedding_call

        embeddings = embed_segments(self.encoder, [segment for segment, _ in prepared], self.batch_size)
        # 학습은 float16으로 캐시한 임베딩을 썼다. 같은 값으로 맞춰야 Validation 점수가 재현된다.
        embeddings = embeddings.astype(np.float16).astype(np.float32)
        mfcc = np.vstack([feature for _, feature in prepared]) if self.use_mfcc else None
        return predict_embedding_call(self.checkpoint, embeddings, mfcc)


def main() -> None:
    args = parse_args()
    if not args.audio_dir.is_dir() or not args.label_dir.is_dir() or not args.ckpt_path.is_file():
        raise FileNotFoundError("audio_dir, label_dir, ckpt_path를 확인하세요.")
    checkpoint = joblib.load(args.ckpt_path)
    model_type = checkpoint.get("model_type") if isinstance(checkpoint, dict) else None
    if model_type == "mission2_mfcc_random_forest":
        predictor = RandomForestPredictor(checkpoint)
    elif model_type == "mission2_speaker_embedding":
        predictor = SpeakerEmbeddingPredictor(checkpoint, args.ckpt_path.parent)
    else:
        raise ValueError("Mission 2 체크포인트가 아닙니다.")
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
        # 같은 통화의 발화를 한 번에 모아 서로 비교한다. 발화 순서는 쓰지 않는다.
        entries: list[tuple[object, object, object]] = []
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
