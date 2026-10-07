"""Mission 2의 발화 단위 오디오 처리 공통 코드.

추론에서는 annotation의 ``startAt``/``endAt``만 읽어 발화 위치를 얻습니다.
``speaker``는 Training/Validation에서 정답 레이블로만 사용합니다.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile
from sklearn.cluster import KMeans


SAMPLE_RATE = 16_000
SPEAKER_TO_INDEX = {"0": 0, "1": 1}
UTTERANCE_KEYS = ("utterances", "utterences")


def decode_annotation(data: bytes) -> dict[str, Any]:
    """이미 읽어둔 바이트에서 annotation을 만든다."""
    annotation = json.loads(data.decode("utf-8"))
    if not isinstance(annotation, dict):
        raise ValueError("Annotation must be an object")
    return annotation


def load_annotation(path: Path) -> dict[str, Any]:
    try:
        return decode_annotation(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{error}: {path}") from error


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


def decode_waveform(data: bytes) -> np.ndarray:
    """이미 읽어둔 바이트를 16kHz mono 파형으로 디코딩한다.

    Google Drive 같은 원격 파일시스템은 조각 단위 읽기마다 왕복이 생기므로,
    호출하는 쪽에서 파일을 통째로 읽어 두고 이 함수에 넘긴다.
    """
    with io.BytesIO(data) as buffer:
        waveform, sample_rate = soundfile.read(buffer, dtype="float32", always_2d=False)
    if waveform.ndim > 1:  # soundfile은 (frame, channel) 순서로 돌려준다.
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if waveform.size == 0:
        raise ValueError("Empty audio")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_waveform(path: Path) -> np.ndarray:
    try:
        return decode_waveform(path.read_bytes())
    except RuntimeError:  # soundfile이 못 읽는 포맷은 librosa에 맡긴다.
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if waveform.size == 0:
            raise ValueError("Empty audio")
        return np.ascontiguousarray(waveform, dtype=np.float32)


def waveform_to_feature(waveform: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """MFCC 및 통화 환경 단서(RMS/ZCR/spectral) 통계 특징을 만든다.

    MFCC와 spectral 특징이 같은 STFT를 공유한다. 각자 `y=`로 호출하면
    동일한 STFT를 네 번 반복 계산하게 되어 발화당 비용이 몇 배로 늘어난다.
    """
    if waveform.size < 512:
        waveform = np.pad(waveform, (0, 512 - waveform.size))
    magnitude = np.abs(librosa.stft(waveform))
    mel = librosa.feature.melspectrogram(S=magnitude ** 2, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
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
        librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE),
        librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE),
    ))
    values = np.concatenate((mfcc, delta, delta2, spectral), axis=0)
    return np.concatenate((
        values.mean(axis=1), values.std(axis=1),
        np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1),
        np.array([np.log1p(len(waveform) / SAMPLE_RATE)], dtype=np.float32),
    )).astype(np.float32)


PROFILE_BANDS = 40
PROFILE_SCALARS = (
    "energy_p5", "energy_p50", "energy_p95", "energy_range", "quiet_fraction", "zero_crossing",
    "peak_db", "clip_fraction", "dc_offset", "noise_flatness", "speech_to_noise", "speech_high_ratio",
)
PROFILE_DIM = PROFILE_BANDS * 3 + len(PROFILE_SCALARS)
_PROFILE_MEL: np.ndarray | None = None


def channel_profile(waveform: np.ndarray) -> np.ndarray:
    """발화 하나의 녹음 환경(채널·배경 소음) 프로파일 (EXP-010).

    문제 p8 힌트대로 대원/신고자를 가르는 단서는 녹음 채널과 배경 소음이다. 그런데 화자
    임베딩은 채널 차이를 지우도록 학습되었고, MFCC 통계는 말소리와 쉬는 구간을 섞어
    평균한다. 그래서 프레임을 에너지로 나눠, 가장 조용한 20%의 평균 스펙트럼(배경 소음)과
    가장 큰 30%의 평균 스펙트럼(말소리)을 따로 잰다. 프레임은 에너지 값으로만 고르므로
    발화 안의 시간 순서와도 무관하다.

    반환: 소음 log-mel 40 + 말소리 log-mel 40 + 전체 log-mel 표준편차 40 + 지표 12.
    """
    global _PROFILE_MEL
    if _PROFILE_MEL is None:
        # 원본은 8kHz라 4kHz 위에는 정보가 없다.
        _PROFILE_MEL = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=512, n_mels=PROFILE_BANDS, fmax=4000).astype(np.float64)
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.size < 512:
        signal = np.pad(signal, (0, 512 - signal.size))
    frames = librosa.util.frame(signal, frame_length=512, hop_length=160).T
    power = np.abs(np.fft.rfft(frames * np.hanning(512), axis=1)) ** 2
    band = 10.0 * np.log10(power @ _PROFILE_MEL.T + 1e-10)
    energy = 10.0 * np.log10(power.sum(axis=1) + 1e-10)
    ordered = np.sort(energy)
    count = len(energy)
    quiet = energy <= ordered[max(1, int(round(0.2 * count))) - 1]
    loud = energy >= ordered[count - max(1, int(round(0.3 * count)))]
    noise = band[quiet].mean(axis=0)
    speech = band[loud].mean(axis=0)
    p5, p50, p95 = np.percentile(energy, (5, 50, 95))
    quiet_power = power[quiet][:, :129] + 1e-10  # 0~4kHz 구간
    flatness = float(np.mean(np.exp(np.log(quiet_power).mean(axis=1)) / quiet_power.mean(axis=1)))
    loud_power = power[loud]
    high = float(loud_power[:, 96:129].sum() / (loud_power[:, :129].sum() + 1e-10))  # 3~4kHz 비율
    peak = float(np.abs(signal).max())
    scalars = np.array([
        p5, p50, p95, p95 - p5, float(np.mean(energy < p95 - 30.0)),
        float(np.mean(np.abs(np.diff(np.sign(signal))) > 0)),
        20.0 * np.log10(peak + 1e-10), float(np.mean(np.abs(signal) >= 0.999)),
        abs(float(signal.mean())), flatness, float(speech.mean() - noise.mean()), high,
    ])
    return np.concatenate((noise, speech, band.std(axis=0), scalars)).astype(np.float32)


# 발화 특징의 앞 65개 열은 MFCC(20)·Δ(20)·Δ²(20)·spectral(5)의 프레임 평균이다.
# 통화 안에서 화자를 갈라내는 데 가장 직접적인 단서라 상대 비교의 기준으로 쓴다.
MEAN_BLOCK = 65
CALL_CONTEXT_CHOICES = ("none", "center", "full")


def call_context_width(width: int, mode: str) -> int:
    """통화 맥락을 붙였을 때의 특징 차원."""
    if mode == "none":
        return width
    if mode == "center":
        return width * 2
    if mode == "full":
        return width * 2 + MEAN_BLOCK + 1
    raise ValueError(f"Unknown call context mode: {mode}")


def rank_within_call(values: np.ndarray) -> np.ndarray:
    """열마다 통화 안에서의 상대 위치를 0~1로 돌려준다.

    같은 값은 같은 순위를 받는다. 동점을 앞뒤 자리로 가르면 발화를 늘어놓은
    차례가 결과에 스며들어, 순서를 쓰지 않는다는 규칙이 깨진다.
    """
    count = values.shape[0]
    if count < 2:
        return np.full(values.shape, 0.5, dtype=np.float32)
    ranks = np.empty(values.shape, dtype=np.float32)
    for column in range(values.shape[1]):
        item = values[:, column]
        ordered = np.sort(item)
        left = np.searchsorted(ordered, item, side="left")
        right = np.searchsorted(ordered, item, side="right")
        ranks[:, column] = (left + right - 1) / 2.0 / (count - 1)
    return ranks


def add_call_context(features: np.ndarray, mode: str) -> np.ndarray:
    """한 통화의 발화 특징에 통화 안에서의 상대값을 덧붙인다.

    조각 하나만 보면 "이 정도면 조용한가"를 절대 기준으로 판단해야 하는데,
    통화마다 회선 상태와 주변 소음이 달라 그 기준이 흔들린다. 같은 통화의
    중앙값을 빼면 그 편차가 사라지고 "이 통화 안에서 어느 쪽인가"만 남는다.

    발화 순서는 사용하지 않는다. 중앙값과 순위는 모두 발화를 늘어놓은 차례와
    무관한 값이라, 조각을 아무 순서로 섞어 넣어도 결과가 같다.
    """
    base = np.asarray(features, dtype=np.float32)
    if mode == "none":
        return base
    if base.ndim != 2:
        raise ValueError("Call features must be 2-dimensional")
    centered = base - np.median(base, axis=0, keepdims=True)
    if mode == "center":
        return np.hstack((base, centered)).astype(np.float32)
    if mode != "full":
        raise ValueError(f"Unknown call context mode: {mode}")
    columns = np.concatenate((np.arange(MEAN_BLOCK), np.array([base.shape[1] - 1])))
    ranks = rank_within_call(base[:, columns])
    return np.hstack((base, centered, ranks)).astype(np.float32)


def speaker_probability(classifier, features: np.ndarray) -> np.ndarray:
    """신고자(1)일 확률을 돌려준다. 열 순서를 classes_로 확인한다."""
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("Classifier was not trained on speaker label 1")
    return classifier.predict_proba(features)[:, classes.index(1)].astype(np.float64)


def cluster_speakers(
    base_features: np.ndarray,
    probabilities: np.ndarray,
    min_utterances: int = 6,
    margin: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """통화 안의 발화를 두 군집으로 묶고 군집 단위로 화자를 정한다.

    같은 사람의 발화는 녹음 환경이 같아 음향이 서로 닮는다. 군집 평균 확률로
    라벨을 정하면 개별 발화에서 삐끗한 예측이 같은 군집 다수에 의해 교정된다.

    발화가 너무 적거나 두 군집의 평균 확률이 비슷해 어느 쪽이 신고자인지
    분명하지 않으면 묶기를 포기하고 발화별 예측을 그대로 둔다. 한 사람만
    말한 통화를 억지로 둘로 가르지 않기 위한 안전장치다.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    single = (probabilities >= 0.5).astype(np.int64)
    if len(probabilities) < min_utterances:
        return single
    base = np.asarray(base_features, dtype=np.float32)[:, :MEAN_BLOCK]
    spread = base.std(axis=0)
    spread[spread < 1e-6] = 1.0
    scaled = (base - base.mean(axis=0)) / spread
    # KMeans의 초기값은 행 순서에 따라 달라진다. 값으로만 정해지는 차례로 한 번
    # 정렬한 뒤 군집을 나누고 되돌리면, 발화를 어떤 순서로 넣어도 결과가 같다.
    order = np.lexsort(scaled.T[::-1])
    inverse = np.argsort(order)
    try:
        assignment = KMeans(n_clusters=2, n_init=4, random_state=seed).fit_predict(scaled[order])[inverse]
    except ValueError:
        return single
    first = probabilities[assignment == 0]
    second = probabilities[assignment == 1]
    if first.size == 0 or second.size == 0:
        return single
    if abs(float(first.mean()) - float(second.mean())) < margin:
        return single
    caller = 0 if first.mean() > second.mean() else 1
    return (assignment == caller).astype(np.int64)


def predict_call(
    classifier,
    base_features: np.ndarray,
    call_context: str,
    cluster: bool,
    min_utterances: int = 6,
    margin: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """통화 하나의 발화 전체를 한 번에 판정한다."""
    context = add_call_context(base_features, call_context)
    probabilities = speaker_probability(classifier, context)
    if not cluster:
        # 학습의 Validation 점수와 같은 규칙(확률 0.5 이상이면 신고자)으로 판정한다.
        # classifier.predict는 0.5 동점을 대원(0)으로 보내 점수가 어긋난다.
        return (probabilities >= 0.5).astype(np.int64)
    return cluster_speakers(base_features, probabilities, min_utterances, margin, seed)
