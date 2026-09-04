"""Mission 1의 공통 데이터 처리 및 모델 코드.

추론 시 annotation 파일에서는 신고자 구간을 위한 ``startAt``, ``endAt``,
``speaker`` 필드만 읽습니다. ``gender``는 학습 정답으로만 읽습니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn


SAMPLE_RATE = 16_000
N_MELS = 64
MAX_FRAMES = 300
GENDER_TO_INDEX = {"M": 0, "F": 1}
INDEX_TO_GENDER = {value: key for key, value in GENDER_TO_INDEX.items()}
UTTERANCE_KEYS = ("utterances", "utterences")


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Annotation must be a JSON object: {path}")
    return data


def annotation_path_for(audio_path: Path, audio_dir: Path, label_dir: Path) -> Path:
    """하위 디렉터리 구조를 유지한 대응 JSON 경로를 반환합니다."""
    relative = audio_path.relative_to(audio_dir).with_suffix(".json")
    return label_dir / relative


def get_utterances(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in UTTERANCE_KEYS:
        utterances = annotation.get(key)
        if isinstance(utterances, list):
            return [item for item in utterances if isinstance(item, dict)]
    return []


def is_caller(speaker: Any) -> bool:
    return str(speaker).strip() in {"1", "caller", "신고자"}


def timestamp_to_samples(value: Any, *, timestamp_unit: str) -> int:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp: {value!r}") from error

    if timestamp_unit == "milliseconds":
        timestamp /= 1_000
    return max(0, int(round(timestamp * SAMPLE_RATE)))


def load_caller_audio(
    audio_path: Path,
    annotation: dict[str, Any],
    *,
    timestamp_unit: str = "auto",
) -> np.ndarray:
    """통화에서 신고자(``speaker=1``) 발화 구간만 이어 붙여 불러옵니다."""
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_seconds = len(waveform) / SAMPLE_RATE
    all_utterances = get_utterances(annotation)
    utterances = [
        utterance
        for utterance in all_utterances
        if is_caller(utterance.get("speaker", utterance.get("Speaker")))
    ]
    if timestamp_unit == "auto":
        end_times: list[float] = []
        for utterance in all_utterances:
            try:
                end_times.append(float(utterance.get("endAt")))
            except (TypeError, ValueError):
                continue
        largest_end = max(end_times, default=0.0)
        # 전체 통화 길이와 가장 가까운 단위를 선택해, 발화가 이른 시점에만
        # 있는 신고자도 밀리초 단위를 초로 잘못 해석하지 않도록 합니다.
        seconds_error = abs(largest_end - duration_seconds)
        milliseconds_error = abs(largest_end / 1_000 - duration_seconds)
        timestamp_unit = "milliseconds" if milliseconds_error < seconds_error else "seconds"
    segments: list[np.ndarray] = []

    for utterance in utterances:
        start = timestamp_to_samples(
            utterance.get("startAt"),
            timestamp_unit=timestamp_unit,
        )
        end = timestamp_to_samples(
            utterance.get("endAt"),
            timestamp_unit=timestamp_unit,
        )
        start, end = min(start, len(waveform)), min(end, len(waveform))
        if end > start:
            segments.append(waveform[start:end])

    if not segments:
        raise ValueError(f"No valid caller (speaker=1) segments: {audio_path}")
    return np.concatenate(segments)


def waveform_to_mel_windows(
    waveform: np.ndarray,
    max_frames: int = MAX_FRAMES,
    hop_frames: int | None = None,
) -> torch.Tensor:
    """Mel-spectrogram을 padding 또는 sliding window로 고정 길이 창으로 나눕니다."""
    if max_frames < 1:
        raise ValueError("max_frames는 양수여야 합니다.")
    if hop_frames is None:
        hop_frames = max_frames // 2
    if hop_frames < 1:
        raise ValueError("hop_frames는 양수여야 합니다.")

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    frame_count = mel_db.shape[1]
    if frame_count <= max_frames:
        padding = np.full((N_MELS, max_frames - mel_db.shape[1]), mel_db.min())
        mel_db = np.concatenate((mel_db, padding), axis=1)
        return torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    starts = list(range(0, frame_count - max_frames + 1, hop_frames))
    final_start = frame_count - max_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = np.stack(
        [mel_db[:, start : start + max_frames] for start in starts], axis=0
    )
    return torch.from_numpy(windows.astype(np.float32)).unsqueeze(1)


def waveform_to_mfcc_feature(waveform: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """MFCC와 1·2차 변화량의 시간 통계치를 RandomForest 입력으로 만듭니다."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    frame_count = mfcc.shape[1]
    if frame_count < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta_width = min(9, frame_count if frame_count % 2 else frame_count - 1)
        delta = librosa.feature.delta(mfcc, width=delta_width)
        delta2 = librosa.feature.delta(mfcc, width=delta_width, order=2)
    stacked = np.concatenate((mfcc, delta, delta2), axis=0)
    return np.concatenate(
        (
            stacked.mean(axis=1),
            stacked.std(axis=1),
            np.percentile(stacked, 25, axis=1),
            np.percentile(stacked, 75, axis=1),
        )
    ).astype(np.float32)


class SimpleCNN(nn.Module):
    """입력 길이에 독립적인 풀링을 사용하는 경량 CNN 베이스라인입니다."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 16)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, len(GENDER_TO_INDEX)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
