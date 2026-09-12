"""Mission 2의 발화 단위 오디오 처리 공통 코드.

추론에서는 annotation의 ``startAt``/``endAt``만 읽어 발화 위치를 얻습니다.
``speaker``는 Training/Validation에서 정답 레이블로만 사용합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np


SAMPLE_RATE = 16_000
SPEAKER_TO_INDEX = {"0": 0, "1": 1}
UTTERANCE_KEYS = ("utterances", "utterences")


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation must be an object: {path}")
    return annotation


def get_utterances(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """발화 목록을 반환한다. 허용되지 않은 필드는 여기서 읽지 않는다."""
    for key in UTTERANCE_KEYS:
        items = annotation.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def find_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def build_annotation_index(label_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(label_dir.rglob("*.json")):
        index.setdefault(path.stem, path)
    return index


def resolve_annotation_path(
    audio_path: Path, audio_dir: Path, label_dir: Path, index: dict[str, Path]
) -> Path | None:
    try:
        same_path = label_dir / audio_path.relative_to(audio_dir).with_suffix(".json")
    except ValueError:
        same_path = None
    if same_path is not None and same_path.is_file():
        return same_path
    return index.get(audio_path.stem)


def detect_timestamp_unit(annotation: dict[str, Any], duration_seconds: float) -> str:
    ends: list[float] = []
    for utterance in get_utterances(annotation):
        try:
            ends.append(float(utterance.get("endAt")))
        except (TypeError, ValueError):
            pass
    if not ends:
        return "milliseconds"
    largest_end = max(ends)
    return (
        "milliseconds"
        if abs(largest_end / 1_000 - duration_seconds) < abs(largest_end - duration_seconds)
        else "seconds"
    )


def timestamp_to_sample(value: Any, timestamp_unit: str) -> int:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp: {value!r}") from error
    if timestamp_unit == "milliseconds":
        seconds /= 1_000
    return max(0, int(round(seconds * SAMPLE_RATE)))


def slice_utterance(
    waveform: np.ndarray, utterance: dict[str, Any], timestamp_unit: str
) -> np.ndarray:
    start = min(len(waveform), timestamp_to_sample(utterance.get("startAt"), timestamp_unit))
    end = min(len(waveform), timestamp_to_sample(utterance.get("endAt"), timestamp_unit))
    if end <= start:
        raise ValueError(f"Empty or reversed segment: {utterance.get('startAt')}~{utterance.get('endAt')}")
    return waveform[start:end]


def load_waveform(path: Path) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("Empty audio")
    return waveform


def waveform_to_feature(waveform: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """MFCC 및 통화 환경 단서(RMS/ZCR/spectral) 통계 특징을 만든다."""
    if waveform.size < 512:
        waveform = np.pad(waveform, (0, 512 - waveform.size))
    mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    frame_count = mfcc.shape[1]
    if frame_count < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        width = min(9, frame_count if frame_count % 2 else frame_count - 1)
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, width=width, order=2)
    spectral = np.vstack((
        librosa.feature.rms(y=waveform),
        librosa.feature.zero_crossing_rate(waveform),
        librosa.feature.spectral_centroid(y=waveform, sr=SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(y=waveform, sr=SAMPLE_RATE),
        librosa.feature.spectral_rolloff(y=waveform, sr=SAMPLE_RATE),
    ))
    values = np.concatenate((mfcc, delta, delta2, spectral), axis=0)
    return np.concatenate((
        values.mean(axis=1), values.std(axis=1),
        np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1),
        np.array([np.log1p(len(waveform) / SAMPLE_RATE)], dtype=np.float32),
    )).astype(np.float32)
