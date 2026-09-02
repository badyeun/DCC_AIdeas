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

# F0(기본주파수) 추출 범위입니다. 성인 남성 약 85~180Hz, 여성 약 165~255Hz를
# 모두 포함하도록 여유를 두었습니다.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
F0_FEATURE_SIZE = 6


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


def build_annotation_index(label_dir: Path) -> dict[str, Path]:
    """``label_dir``의 JSON을 한 번만 훑어 파일명(stem) -> 경로 사전을 만듭니다.

    같은 stem이 여러 번 나오면 경로가 짧은(상위에 있는) 파일을 우선합니다.
    """
    index: dict[str, Path] = {}
    for json_path in sorted(label_dir.rglob("*.json")):
        index.setdefault(json_path.stem, json_path)
    return index


def resolve_annotation_path(
    audio_path: Path,
    audio_dir: Path,
    label_dir: Path,
    annotation_index: dict[str, Path] | None = None,
) -> Path | None:
    """WAV에 대응하는 JSON을 찾습니다. 못 찾으면 ``None``을 반환합니다.

    주최 측 평가 폴더의 하위 구조를 알 수 없으므로 다음 순서로 시도합니다.

    1. 동일한 하위 경로 (``audio_dir/a/b/x.wav`` -> ``label_dir/a/b/x.json``)
    2. 파일명(stem) 일치 (하위 구조가 달라도 찾을 수 있음)
    """
    try:
        same_structure = annotation_path_for(audio_path, audio_dir, label_dir)
    except ValueError:  # audio_path가 audio_dir 하위가 아닌 경우입니다.
        same_structure = None
    if same_structure is not None and same_structure.is_file():
        return same_structure

    if annotation_index is None:
        annotation_index = build_annotation_index(label_dir)
    return annotation_index.get(audio_path.stem)


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


def detect_timestamp_unit(annotation: dict[str, Any], duration_seconds: float) -> str:
    """``startAt``/``endAt``의 단위를 판별합니다.

    신고자 발화만 보면 신고자가 통화 앞부분에서만 말한 경우 통화 길이의 1% 안에
    끝나 ``seconds``로 오판할 수 있습니다. 그래서 화자를 가리지 않고 **모든 발화**와
    최상위 ``endAt``까지 함께 봅니다. 이 값들의 최댓값은 통화 길이에 가까우므로
    밀리초라면 초 단위 길이의 약 1000배가 되어 안전하게 구분됩니다.
    """
    candidates = [annotation.get("endAt")]
    candidates.extend(utterance.get("endAt") for utterance in get_utterances(annotation))

    largest_end = 0.0
    for value in candidates:
        try:
            largest_end = max(largest_end, float(value))
        except (TypeError, ValueError):
            continue

    return "milliseconds" if largest_end > duration_seconds * 10 else "seconds"


def load_caller_audio(
    audio_path: Path,
    annotation: dict[str, Any],
    *,
    timestamp_unit: str = "auto",
) -> np.ndarray:
    """통화에서 신고자(``speaker=1``) 발화 구간만 이어 붙여 불러옵니다."""
    waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration_seconds = len(waveform) / SAMPLE_RATE
    utterances = [utterance for utterance in get_utterances(annotation) if is_caller(utterance.get("speaker"))]
    if timestamp_unit == "auto":
        timestamp_unit = detect_timestamp_unit(annotation, duration_seconds)
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


def waveform_to_mel(waveform: np.ndarray, max_frames: int = MAX_FRAMES) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] >= max_frames:
        mel_db = mel_db[:, :max_frames]
    else:
        padding = np.full((N_MELS, max_frames - mel_db.shape[1]), mel_db.min())
        mel_db = np.concatenate((mel_db, padding), axis=1)
    return torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0)


def waveform_to_f0_feature(waveform: np.ndarray) -> np.ndarray:
    """기본주파수(F0) 통계치를 만듭니다.

    MFCC는 DCT 과정에서 스펙트럼 포락선만 남기고 F0를 상당 부분 버립니다.
    그런데 성별을 가르는 가장 강한 단서가 F0이므로(남성 약 85~180Hz,
    여성 약 165~255Hz) 별도 특징으로 더해 줍니다.

    반환값은 log2 스케일의 평균·중앙값·표준편차·25%·75% 분위수와 유성음 비율
    총 ``F0_FEATURE_SIZE``개입니다. 옥타브가 선형이 되도록 log2를 사용합니다.
    """
    empty = np.zeros(F0_FEATURE_SIZE, dtype=np.float32)
    if waveform.size < 2048:
        return empty

    try:
        f0 = librosa.yin(
            waveform, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=SAMPLE_RATE, frame_length=2048
        )
    except Exception:  # 무음 등으로 추정에 실패하면 0 벡터를 사용합니다.
        return empty

    f0 = np.asarray(f0, dtype=np.float64)
    # yin은 유성음 판정을 하지 않아 무성 구간에서 경계값을 반환합니다. 범위 밖과
    # 비정상 값을 걸러 실제 목소리 구간만 남깁니다.
    voiced = f0[np.isfinite(f0) & (f0 > F0_MIN_HZ * 1.01) & (f0 < F0_MAX_HZ * 0.99)]
    voiced_ratio = float(voiced.size) / float(f0.size) if f0.size else 0.0
    if voiced.size == 0:
        return np.array([0, 0, 0, 0, 0, voiced_ratio], dtype=np.float32)

    log_f0 = np.log2(voiced)
    return np.array(
        [
            log_f0.mean(),
            np.median(log_f0),
            log_f0.std(),
            np.percentile(log_f0, 25),
            np.percentile(log_f0, 75),
            voiced_ratio,
        ],
        dtype=np.float32,
    )


def waveform_to_mfcc_feature(
    waveform: np.ndarray, n_mfcc: int = 20, *, use_f0: bool = True
) -> np.ndarray:
    """MFCC와 1·2차 변화량의 시간 통계치를 RandomForest 입력으로 만듭니다.

    ``use_f0``가 True면 F0 통계치를 뒤에 이어 붙입니다. 체크포인트에 이 설정을
    저장해 두므로 학습과 추론에서 항상 같은 차원이 만들어집니다.
    """
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
    feature = np.concatenate(
        (
            stacked.mean(axis=1),
            stacked.std(axis=1),
            np.percentile(stacked, 25, axis=1),
            np.percentile(stacked, 75, axis=1),
        )
    ).astype(np.float32)

    if not use_f0:
        return feature
    return np.concatenate((feature, waveform_to_f0_feature(waveform))).astype(np.float32)


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
