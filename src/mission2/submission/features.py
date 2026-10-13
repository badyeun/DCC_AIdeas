"""Mission 2 발화 특징 추출: annotation 읽기, 발화 구간 자르기, MFCC 통계, ECAPA 화자 임베딩.

annotation에서는 발화 경계(`startAt`, `endAt`)만 읽는다. `text`, `speaker` 등 다른 필드는 읽지 않는다.
학습 캐시(training_code/)와 추론(inference.py)이 이 파일의 같은 함수로 특징을 만든다.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile

SAMPLE_RATE = 16_000
# 데이터에 따라 발화 목록 키가 오타(utterences)로 들어 있는 파일이 있다.
UTTERANCE_KEYS = ("utterances", "utterences")
ECAPA_DIM = 192
# ECAPA는 0.5초보다 짧으면 임베딩이 불안정하다. 짧은 조각은 반복해서 늘린다.
MIN_SAMPLES = SAMPLE_RATE // 2
# 긴 조각은 가운데 20초만 쓴다. 화자 정보는 충분하고 GPU 메모리가 일정해진다.
MAX_SAMPLES = SAMPLE_RATE * 20


# ---------------------------------------------------------------- 파일 찾기, annotation

def find_audio_files(audio_dir: Path) -> list[Path]:
    """audio_dir 아래(하위 폴더 포함)의 .wav 파일을 이름 순으로 찾는다."""
    return sorted(path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")


def build_annotation_index(label_dir: Path) -> dict[str, Path]:
    """파일 이름(확장자 제외) → JSON 경로. WAV와 JSON의 폴더 구조가 달라도 짝을 찾기 위해 쓴다."""
    index: dict[str, Path] = {}
    for path in sorted(label_dir.rglob("*.json")):
        index.setdefault(path.stem, path)
    return index


def resolve_annotation_path(audio_path: Path, audio_dir: Path, label_dir: Path, index: dict[str, Path]) -> Path | None:
    """같은 상대 경로의 JSON을 먼저 찾고, 없으면 같은 이름의 JSON을 쓴다."""
    try:
        same_path = label_dir / audio_path.relative_to(audio_dir).with_suffix(".json")
    except ValueError:
        same_path = None
    if same_path is not None and same_path.is_file():
        return same_path
    return index.get(audio_path.stem)


def load_annotation(path: Path) -> dict[str, Any]:
    annotation = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation must be an object: {path}")
    return annotation


def get_utterances(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """발화 목록. 이후에는 각 발화의 startAt, endAt만 읽는다."""
    for key in UTTERANCE_KEYS:
        items = annotation.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def detect_timestamp_unit(annotation: dict[str, Any], duration_seconds: float) -> str:
    """startAt/endAt이 밀리초인지 초인지, 가장 늦은 endAt을 음성 길이와 비교해 정한다."""
    ends: list[float] = []
    for utterance in get_utterances(annotation):
        try:
            ends.append(float(utterance.get("endAt")))
        except (TypeError, ValueError):
            pass
    if not ends:
        return "milliseconds"
    largest_end = max(ends)
    return "milliseconds" if abs(largest_end / 1_000 - duration_seconds) < abs(largest_end - duration_seconds) else "seconds"


# ---------------------------------------------------------------- 음성, 발화 구간

def timestamp_to_sample(value: Any, timestamp_unit: str) -> int:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp: {value!r}") from error
    if timestamp_unit == "milliseconds":
        seconds /= 1_000
    return max(0, int(round(seconds * SAMPLE_RATE)))


def slice_utterance(waveform: np.ndarray, utterance: dict[str, Any], timestamp_unit: str) -> np.ndarray:
    """startAt~endAt 구간을 잘라 낸다. 비어 있거나 거꾸로 된 구간은 ValueError."""
    start = min(len(waveform), timestamp_to_sample(utterance.get("startAt"), timestamp_unit))
    end = min(len(waveform), timestamp_to_sample(utterance.get("endAt"), timestamp_unit))
    if end <= start:
        raise ValueError(f"Empty or reversed segment: {utterance.get('startAt')}~{utterance.get('endAt')}")
    return waveform[start:end]


def load_waveform(path: Path) -> np.ndarray:
    """16kHz mono float32 파형. 파일을 통째로 읽은 뒤 디코딩한다(Google Drive에서 조각 읽기 왕복을 줄인다)."""
    try:
        with io.BytesIO(path.read_bytes()) as buffer:
            waveform, sample_rate = soundfile.read(buffer, dtype="float32", always_2d=False)
        if waveform.ndim > 1:  # soundfile은 (frame, channel) 순서로 돌려준다.
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    except RuntimeError:  # soundfile이 못 읽는 포맷은 librosa에 맡긴다.
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("Empty audio")
    return np.ascontiguousarray(waveform, dtype=np.float32)


# ---------------------------------------------------------------- MFCC 통계 (261차원)

def mfcc_features(waveform: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """MFCC·Δ·Δ² 60행 + RMS·ZCR·spectral centroid/bandwidth/rolloff 5행의 프레임 통계와 발화 길이.

    65행 × (평균, 표준편차, 25%, 75%) = 260 + log(1 + 길이 초) 1 = 261차원.
    MFCC와 spectral 특징은 같은 STFT를 공유해 한 번만 계산한다.
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


# ---------------------------------------------------------------- ECAPA 화자 임베딩 (192차원)

def fit_length(segment: np.ndarray) -> np.ndarray:
    """짧은 조각은 반복해 늘리고 긴 조각은 가운데를 잘라 길이를 맞춘다."""
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        raise ValueError("Empty segment")
    if segment.size < MIN_SAMPLES:
        segment = np.tile(segment, int(np.ceil(MIN_SAMPLES / segment.size)))[:MIN_SAMPLES]
    if segment.size > MAX_SAMPLES:
        start = (segment.size - MAX_SAMPLES) // 2
        segment = segment[start:start + MAX_SAMPLES]
    return segment


def unpack_folder(files: dict[str, bytes], target: Path) -> Path:
    """체크포인트에 바이트로 넣어 둔 인코더 파일들을 폴더에 풀어 쓴다."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    return target


class EcapaEncoder:
    """공개 사전학습 화자 인식 모델 SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 가중치 고정).

    folder에 들어 있는 파일(hyperparams.yaml, embedding_model.ckpt 등)만 읽어 인터넷 없이 동작한다.
    """

    def __init__(self, folder: Path, device: str) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain 0.5 계열
            from speechbrain.pretrained import EncoderClassifier
        folder = Path(folder)
        if not (folder / "hyperparams.yaml").is_file():
            raise FileNotFoundError(f"ECAPA 인코더 파일이 없습니다: {folder}")
        self.model = EncoderClassifier.from_hparams(source=str(folder), savedir=str(folder), run_opts={"device": device})

    def embed(self, segments: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """발화 조각들의 임베딩 (발화 수, 192). 길이가 비슷한 것끼리 묶어 padding 낭비를 줄인다."""
        import torch

        prepared = [fit_length(segment) for segment in segments]
        output = np.zeros((len(prepared), ECAPA_DIM), dtype=np.float32)
        if not prepared:
            return output
        # 길이로 정렬하는 것은 값에 따른 차례라 발화 순서와 무관하다. 결과는 제자리로 되돌린다.
        order = np.argsort([segment.size for segment in prepared], kind="stable")
        for begin in range(0, len(order), batch_size):
            chosen = order[begin:begin + batch_size]
            longest = max(prepared[index].size for index in chosen)
            batch = np.zeros((len(chosen), longest), dtype=np.float32)
            lengths = np.zeros(len(chosen), dtype=np.float32)
            for row, index in enumerate(chosen):
                batch[row, :prepared[index].size] = prepared[index]
                lengths[row] = prepared[index].size / longest  # 상대 길이: padding 부분은 계산에서 빠진다
            with torch.no_grad():
                vectors = self.model.encode_batch(torch.from_numpy(batch), torch.from_numpy(lengths))
            output[chosen] = vectors.reshape(len(chosen), -1).float().cpu().numpy()
        return output
