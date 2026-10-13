"""WavLM 3모델 쪽 공통 데이터 처리 (신고자 음성 추출, F0 특징).

추론 시 annotation 파일에서는 신고자 구간을 위한 ``startAt``, ``endAt``,
``speaker`` 필드만 읽습니다. ``gender``는 학습 정답으로만 읽습니다.
(학습 노트북에서 쓰던 공통 코드 중 최종 추론에 쓰는 부분만 남겼습니다.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np


SAMPLE_RATE = 16_000
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

    신고자 발화만 보면 신고자가 통화 앞부분에서만 말한 경우 마지막 발화가 통화
    길이에 한참 못 미쳐 단위를 오판할 수 있습니다. 그래서 화자를 가리지 않고
    **모든 발화**와 최상위 ``endAt``까지 후보로 삼습니다.

    그 최댓값을 초로 읽었을 때와 밀리초로 읽었을 때 중 실제 통화 길이에 더
    가까운 쪽을 고릅니다.
    """
    candidates = [annotation.get("endAt")]
    candidates.extend(utterance.get("endAt") for utterance in get_utterances(annotation))

    largest_end = 0.0
    for value in candidates:
        try:
            largest_end = max(largest_end, float(value))
        except (TypeError, ValueError):
            continue

    if largest_end <= 0:
        # 판단 근거가 없으면 이 데이터셋의 실제 단위인 밀리초로 둡니다.
        return "milliseconds"

    seconds_error = abs(largest_end - duration_seconds)
    milliseconds_error = abs(largest_end / 1_000 - duration_seconds)
    return "milliseconds" if milliseconds_error < seconds_error else "seconds"


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
