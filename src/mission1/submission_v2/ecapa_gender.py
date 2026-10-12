"""Mission 1 ECAPA 성별 모델 (제출 폴더용, 단독 모듈).

학습은 Mission 2 때 만든 발화별 캐시로 했다(EXP-012). 여기서는 그 캐시를 만든 과정을
**그대로** 다시 밟아 같은 숫자를 만든다. 하나라도 다르면 Validation 점수가 재현되지 않는다.

1. 파형: soundfile로 읽어 mono 평균, 16kHz가 아니면 librosa로 resample (Mission 2 `decode_waveform`).
2. 시간 단위: 가장 늦은 endAt이 ms/초 중 어느 쪽으로 해석할 때 통화 길이에 가까운지 (`detect_timestamp_unit`).
3. 발화: speaker가 "0"/"1"인 발화를 JSON에 적힌 차례대로 자른다. 대원 발화도 함께 임베딩하는 이유는
   캐시가 그렇게 만들어져서다. ECAPA는 32개씩 묶어 padding해 돌리므로, 묶음 구성이 같아야 값이 같다.
   (차례는 묶음 구성을 맞추는 데만 쓰이고 판정에는 쓰이지 않는다. 판정은 신고자 발화의 평균·표준편차뿐이다.)
4. ECAPA 192 → float16 반올림(캐시 형식) → 신고자 발화만 길이 1로 맞춤.
5. 발화 길이: Mission 2 MFCC 특징의 마지막 열 log1p(초)를 float16으로 반올림한 값(512 sample 미만은 512로 계산).
6. 특징 384 = 신고자 ECAPA의 발화 길이 가중 평균 192 + 표준편차 192 → 표준화 → 로지스틱 회귀.

추론에서 annotation은 `startAt`, `endAt`, `speaker`만 읽는다(Mission 1 규칙상 허용).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
SPEAKERS = ("0", "1")
UTTERANCE_KEYS = ("utterances", "utterences")
MIN_SAMPLES = SAMPLE_RATE // 2   # ECAPA 입력 최소 0.5초 (짧으면 반복해서 늘림)
MAX_SAMPLES = SAMPLE_RATE * 20   # 긴 발화는 가운데 20초
MFCC_MIN_SAMPLES = 512           # Mission 2 MFCC 특징이 짧은 발화를 늘리던 길이
EMBED_DIM = 192
BATCH_SIZE = 32


# ---------------------------------------------------------------- 파형·발화 (Mission 2 common.py와 같은 동작)

def decode_waveform(path: Path) -> np.ndarray:
    import librosa
    import soundfile

    data = Path(path).read_bytes()
    try:
        with io.BytesIO(data) as buffer:
            waveform, sample_rate = soundfile.read(buffer, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    except RuntimeError:  # soundfile이 못 읽는 포맷
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("Empty audio")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def get_utterances(annotation: dict) -> list[dict]:
    for key in UTTERANCE_KEYS:
        items = annotation.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def detect_timestamp_unit(annotation: dict, duration_seconds: float) -> str:
    ends = []
    for utterance in get_utterances(annotation):
        try:
            ends.append(float(utterance.get("endAt")))
        except (TypeError, ValueError):
            pass
    if not ends:
        return "milliseconds"
    largest = max(ends)
    return "milliseconds" if abs(largest / 1_000 - duration_seconds) < abs(largest - duration_seconds) else "seconds"


def timestamp_to_sample(value, unit: str) -> int:
    seconds = float(value)
    if unit == "milliseconds":
        seconds /= 1_000
    return max(0, int(round(seconds * SAMPLE_RATE)))


def speaker_of(utterance: dict) -> str:
    return str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()


def labeled_segments(waveform: np.ndarray, annotation: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """speaker 0/1 발화 조각과 역할(1=신고자). 캐시처럼 잘못된 구간을 만나면 거기서 멈춘다."""
    unit = detect_timestamp_unit(annotation, len(waveform) / SAMPLE_RATE)
    segments, roles = [], []
    for utterance in get_utterances(annotation):
        speaker = speaker_of(utterance)
        if speaker not in SPEAKERS:
            continue
        try:
            start = min(len(waveform), timestamp_to_sample(utterance.get("startAt"), unit))
            end = min(len(waveform), timestamp_to_sample(utterance.get("endAt"), unit))
        except (TypeError, ValueError):
            break
        if end <= start:
            break
        segments.append(waveform[start:end])
        roles.append(1 if speaker == "1" else 0)
    return segments, np.asarray(roles, dtype=np.int64)


# ---------------------------------------------------------------- ECAPA (Mission 2 speaker_embed.py와 같은 동작)

def load_encoder(folder: Path, device: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # speechbrain 0.5 계열
        from speechbrain.pretrained import EncoderClassifier
    folder = Path(folder)
    if not (folder / "hyperparams.yaml").is_file():
        raise FileNotFoundError(f"ECAPA 인코더 파일이 없습니다: {folder}")
    return EncoderClassifier.from_hparams(source=str(folder), savedir=str(folder), run_opts={"device": device})


def fit_length(segment: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        raise ValueError("Empty segment")
    if segment.size < MIN_SAMPLES:
        segment = np.tile(segment, int(np.ceil(MIN_SAMPLES / segment.size)))[:MIN_SAMPLES]
    if segment.size > MAX_SAMPLES:
        start = (segment.size - MAX_SAMPLES) // 2
        segment = segment[start:start + MAX_SAMPLES]
    return segment


def embed_segments(encoder, segments: list[np.ndarray], batch_size: int = BATCH_SIZE) -> np.ndarray:
    import torch

    prepared = [fit_length(segment) for segment in segments]
    output = np.zeros((len(prepared), EMBED_DIM), dtype=np.float32)
    order = np.argsort([segment.size for segment in prepared], kind="stable")
    for begin in range(0, len(order), batch_size):
        chosen = order[begin:begin + batch_size]
        longest = max(prepared[index].size for index in chosen)
        batch = np.zeros((len(chosen), longest), dtype=np.float32)
        lengths = np.zeros(len(chosen), dtype=np.float32)
        for row, index in enumerate(chosen):
            batch[row, :prepared[index].size] = prepared[index]
            lengths[row] = prepared[index].size / longest
        with torch.no_grad():
            vectors = encoder.encode_batch(torch.from_numpy(batch), torch.from_numpy(lengths))
        output[chosen] = vectors.reshape(len(chosen), -1).float().cpu().numpy()
    return output


# ---------------------------------------------------------------- 특징·판정

def cached_log_seconds(sizes: np.ndarray) -> np.ndarray:
    """캐시의 발화 길이 값: float32 log1p(초)를 float16으로 저장한 것."""
    seconds = np.maximum(np.asarray(sizes), MFCC_MIN_SAMPLES) / SAMPLE_RATE
    return np.log1p(seconds).astype(np.float32).astype(np.float16).astype(np.float32)


def caller_feature(caller_ecapa: np.ndarray, caller_log_seconds: np.ndarray) -> np.ndarray:
    """학습 `gender_data.pooled_features`의 앞 384열과 같은 계산 (float32)."""
    ecapa = np.asarray(caller_ecapa, dtype=np.float32).copy()
    ecapa /= np.maximum(np.linalg.norm(ecapa, axis=1, keepdims=True), 1e-8)
    seconds = np.expm1(np.asarray(caller_log_seconds, dtype=np.float32))
    weight = seconds / seconds.sum()
    return np.concatenate([(ecapa * weight[:, None]).sum(0), ecapa.std(0)]).astype(np.float32)


class EcapaGender:
    """`folder/encoder/`(ECAPA 파일)와 `folder/ecapa_lr.npz`(로지스틱 회귀 숫자)를 읽는다."""

    def __init__(self, folder: Path, device: str) -> None:
        folder = Path(folder)
        with np.load(folder / "ecapa_lr.npz") as data:
            self.mean = data["scaler_mean"].astype(np.float64)
            self.scale = data["scaler_scale"].astype(np.float64)
            self.coef = data["coef"].astype(np.float64)
            self.intercept = float(data["intercept"][0])
            self.blend = float(data["blend"])
        self.encoder = load_encoder(folder / "encoder", device)

    def probability_from_feature(self, feature: np.ndarray) -> float:
        z = ((np.asarray(feature, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))

    def probability(self, audio_path: Path, annotation: dict) -> float:
        """통화 하나의 여성(F) 확률."""
        waveform = decode_waveform(audio_path)
        segments, roles = labeled_segments(waveform, annotation)
        if not (roles == 1).any():
            raise ValueError("신고자(speaker=1) 발화가 없습니다")
        embeddings = embed_segments(self.encoder, segments).astype(np.float16).astype(np.float32)
        sizes = np.asarray([len(segment) for segment in segments])
        caller = roles == 1
        return self.probability_from_feature(caller_feature(embeddings[caller], cached_log_seconds(sizes[caller])))
