"""Mission 2 화자 임베딩 모델 코드.

사전학습 화자 인식 모델(SpeechBrain ECAPA-TDNN, VoxCeleb)로 발화마다 192차원
"목소리 지문"을 뽑고, 같은 통화 안의 발화끼리 그 지문을 직접 비교해 화자를 정한다.

판정은 두 단계다.

1. 1단계: 발화 임베딩과 통화 안에서의 상대값(통화 평균을 뺀 값, 통화 중심과의
   유사도)으로 신고자 확률을 낸다.
2. 2단계: 같은 통화의 다른 발화들이 1단계에서 받은 확률을, 나와 목소리가 얼마나
   닮았는지로 가중 평균해 다시 판정한다. "나와 닮은 발화들이 신고자 쪽이면 나도
   신고자"라는 뜻이다.

EXP-002의 군집 묶기는 통화를 두 덩어리로 딱 잘라 덩어리째 뒤집었고, 덩어리가
화자가 아닌 기준으로 갈리면 크게 틀렸다. 2단계는 자르지 않고 유사도로 가중만
하므로, 비슷한 발화가 애매하면 영향도 작다.

발화 순서는 쓰지 않는다. 모든 비교는 유사도 행렬과 합·평균으로만 이루어져,
발화를 어떤 차례로 넣어도 결과가 같다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from common import SAMPLE_RATE, add_call_context


ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBED_DIM = 192
# ECAPA는 0.5초보다 짧으면 임베딩이 불안정하다. 짧은 조각은 반복해서 늘린다.
MIN_SAMPLES = SAMPLE_RATE // 2
# 긴 조각은 가운데 20초만 쓴다. 화자 정보는 충분하고 GPU 메모리가 일정해진다.
MAX_SAMPLES = SAMPLE_RATE * 20
# 2단계에서 "나와 가장 닮은 발화"로 볼 개수.
NEIGHBORS = 3
STAGE2_COLUMNS = ("p", "similar_to_caller", "similar_to_operator", "difference", "neighbor_vote", "others_mean")


def load_encoder(savedir: Path, device: str):
    """ECAPA 인코더를 불러온다. savedir에 받아 둔 파일이 있으면 인터넷 없이 쓴다."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # speechbrain 0.5 계열
        from speechbrain.pretrained import EncoderClassifier
    savedir = Path(savedir)
    local = (savedir / "hyperparams.yaml").is_file()
    return EncoderClassifier.from_hparams(
        source=str(savedir) if local else ECAPA_SOURCE,
        savedir=str(savedir),
        run_opts={"device": device},
    )


def bundle_encoder(savedir: Path, target: Path) -> None:
    """추론이 인터넷 없이 돌도록 인코더 파일을 체크포인트 옆에 복사한다."""
    target.mkdir(parents=True, exist_ok=True)
    for path in Path(savedir).iterdir():
        if path.is_file() or path.is_symlink():
            # 허깅페이스 캐시는 심볼릭 링크로 받아지므로 실제 파일 내용을 복사한다.
            shutil.copyfile(path.resolve(), target / path.name)


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


def embed_segments(encoder, segments: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """발화 조각들의 임베딩을 돌려준다. 길이가 비슷한 것끼리 묶어 padding 낭비를 줄인다."""
    import torch

    prepared = [fit_length(segment) for segment in segments]
    output = np.zeros((len(prepared), EMBED_DIM), dtype=np.float32)
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
            lengths[row] = prepared[index].size / longest
        with torch.no_grad():
            vectors = encoder.encode_batch(torch.from_numpy(batch), torch.from_numpy(lengths))
        output[chosen] = vectors.reshape(len(chosen), -1).float().cpu().numpy()
    return output


def normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norm, 1e-8)).astype(np.float32)


def stage1_width(use_mfcc: bool, mfcc_width: int = 261) -> int:
    return EMBED_DIM * 2 + 2 + (mfcc_width * 2 if use_mfcc else 0)


def stage1_features(embeddings: np.ndarray, mfcc: np.ndarray | None = None) -> np.ndarray:
    """통화 하나의 1단계 특징. 임베딩 + 통화 평균 기준 상대값 (+ MFCC와 그 상대값)."""
    unit = normalize(np.asarray(embeddings, dtype=np.float32))
    count = len(unit)
    mean = unit.mean(axis=0, keepdims=True)
    centroid = normalize(mean)
    similarity = unit @ unit.T
    to_centroid = unit @ centroid.T
    if count > 1:
        to_others = (similarity.sum(axis=1) - np.diag(similarity)) / (count - 1)
    else:
        to_others = np.ones(count, dtype=np.float32)
    parts = [unit, unit - mean, to_centroid.reshape(count, 1), to_others.reshape(count, 1)]
    if mfcc is not None:
        parts.append(add_call_context(mfcc, "center"))
    return np.hstack(parts).astype(np.float32)


# 2단계 v2에서 더한 입력. EXP-004 오류 분석에서 오류의 76%가 0.5~2초 발화에 몰려
# 있었다. 짧은 발화는 임베딩이 흔들리므로, 길이를 알려 주고(얼마나 남을 믿을지),
# 확실한 발화(앵커)와 직접 비교하고, 목소리 대신 회선 소리(MFCC 평균)로도 비교한다.
STAGE2_EXTRA_COLUMNS = (
    "log_duration", "anchor_caller", "anchor_operator", "anchor_difference",
    "channel_to_caller", "channel_to_operator", "channel_difference",
)
# 회선 비교에 쓰는 MFCC 열: MFCC 평균 20개 + spectral 평균 5개. Δ·Δ² 평균은 거의 0이라
# 표준화하면 잡음만 키운다(EXP-002 군집 실패의 원인 후보).
CHANNEL_COLUMNS = tuple(range(20)) + tuple(range(60, 65))
ANCHOR = 0.8
ANCHOR_TOP = 3


def stage2_aux(mfcc: np.ndarray | None) -> np.ndarray | None:
    """2단계 v2가 쓰는 MFCC 열만 남긴다. 회선 열 25개 + 마지막 열(log 발화 길이)."""
    if mfcc is None:
        return None
    return np.ascontiguousarray(mfcc[:, list(CHANNEL_COLUMNS) + [mfcc.shape[1] - 1]], dtype=np.float32)


def fit_channel_scale(aux: np.ndarray) -> dict[str, np.ndarray]:
    channel = aux[:, :-1]
    return {"mean": channel.mean(axis=0), "std": np.maximum(channel.std(axis=0), 1e-6)}


def weighted_to_sides(similarity: np.ndarray, probabilities: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """신고자 같은 발화들 / 대원 같은 발화들과의 확률 가중 평균 유사도."""
    caller_weight = probabilities[None, :] * others
    operator_weight = (1.0 - probabilities)[None, :] * others
    to_caller = (similarity * caller_weight).sum(axis=1) / np.maximum(caller_weight.sum(axis=1), 1e-8)
    to_operator = (similarity * operator_weight).sum(axis=1) / np.maximum(operator_weight.sum(axis=1), 1e-8)
    return to_caller, to_operator


def top_mean(similarity: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """mask 안에서 가장 닮은 몇 개의 평균 유사도. 값만 정렬하므로 발화 순서와 무관하다."""
    masked = np.where(mask, similarity, -np.inf)
    ordered = -np.sort(-masked, axis=1)[:, :ANCHOR_TOP]
    finite = np.isfinite(ordered)
    sums = np.where(finite, ordered, 0.0).sum(axis=1)
    sizes = finite.sum(axis=1)
    return np.where(sizes > 0, sums / np.maximum(sizes, 1), fallback)


def stage2_features(
    embeddings: np.ndarray, probabilities: np.ndarray, aux: np.ndarray | None = None,
    channel_scale: dict[str, np.ndarray] | None = None, version: int = 1,
) -> np.ndarray:
    """같은 통화의 다른 발화가 받은 확률을 목소리 유사도로 가중해 모은다.

    - similar_to_caller: 신고자 같은 발화들과 평균적으로 얼마나 닮았는가
    - similar_to_operator: 대원 같은 발화들과 평균적으로 얼마나 닮았는가
    - neighbor_vote: 나와 가장 닮은 발화 몇 개의 평균 확률
    - (v2) anchor_*: 확률이 0.8 이상/0.2 이하인 확실한 발화 중 가장 닮은 몇 개와의 유사도
    - (v2) channel_*: 목소리 대신 회선 소리(MFCC 평균)로 잰 같은 비교
    """
    base = stage2_base(embeddings, probabilities)
    if version == 1:
        return base
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(probabilities)
    extra = np.zeros((count, len(STAGE2_EXTRA_COLUMNS)), dtype=np.float32)
    if aux is not None:
        extra[:, 0] = aux[:, -1]
    if count < 2:
        return np.hstack((base, extra))
    unit = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float64)
    similarity = unit @ unit.T
    others = ~np.eye(count, dtype=bool)
    neutral = (similarity * others).sum(axis=1) / (count - 1)
    extra[:, 1] = top_mean(similarity, others & (probabilities >= ANCHOR)[None, :], neutral)
    extra[:, 2] = top_mean(similarity, others & (probabilities <= 1.0 - ANCHOR)[None, :], neutral)
    extra[:, 3] = extra[:, 1] - extra[:, 2]
    if aux is not None and channel_scale is not None:
        channel = (aux[:, :-1] - channel_scale["mean"]) / channel_scale["std"]
        # 통화 평균을 빼면 두 화자가 함께 겪는 회선 상태는 지워지고 둘의 차이만 남는다.
        channel = normalize(channel - channel.mean(axis=0, keepdims=True)).astype(np.float64)
        to_caller, to_operator = weighted_to_sides(channel @ channel.T, probabilities, others)
        extra[:, 4] = to_caller
        extra[:, 5] = to_operator
        extra[:, 6] = to_caller - to_operator
    return np.hstack((base, extra))


def stage2_base(embeddings: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """EXP-003/004의 2단계 입력 6개."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(probabilities)
    output = np.zeros((count, len(STAGE2_COLUMNS)), dtype=np.float32)
    output[:, 0] = probabilities
    if count < 2:
        output[:, 4] = probabilities
        output[:, 5] = probabilities
        return output
    unit = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float64)
    similarity = unit @ unit.T
    others = ~np.eye(count, dtype=bool)
    to_caller, to_operator = weighted_to_sides(similarity, probabilities, others)
    # 가장 닮은 k개는 k번째 유사도 이상인 발화 전부로 정한다. 동점을 차례로 가르지
    # 않으므로 발화 순서가 결과에 스며들지 않는다.
    k = min(NEIGHBORS, count - 1)
    vote = np.empty(count, dtype=np.float64)
    for row in range(count):
        values = similarity[row, others[row]]
        threshold = np.partition(values, count - 1 - k)[count - 1 - k]
        vote[row] = probabilities[others[row]][values >= threshold].mean()
    output[:, 1] = to_caller
    output[:, 2] = to_operator
    output[:, 3] = to_caller - to_operator
    output[:, 4] = vote
    output[:, 5] = (probabilities.sum() - probabilities) / (count - 1)
    return output


BANK_COLUMNS = ("operator_max", "operator_top", "caller_max", "caller_top", "operator_minus_caller")
BANK_TOP = 5


def fit_bank(embeddings: np.ndarray, labels: np.ndarray, clusters: int, seed: int) -> dict[str, np.ndarray]:
    """Train 발화 임베딩을 화자 종류별로 요약한 목소리 사전을 만든다.

    119 대원은 서울 상황실의 한정된 인원이라 여러 통화에 같은 사람이 반복해 나온다.
    대원 임베딩을 군집 중심 몇백 개로 요약해 두면, 새 발화가 "이미 들어 본 대원
    목소리"와 얼마나 닮았는지 잴 수 있다. 신고자 쪽 사전은 비교 기준으로 함께 둔다.
    """
    from sklearn.cluster import MiniBatchKMeans

    unit = normalize(embeddings)
    bank: dict[str, np.ndarray] = {}
    for name, value in (("operator", 0), ("caller", 1)):
        rows = unit[labels == value]
        count = min(clusters, len(rows))
        model = MiniBatchKMeans(n_clusters=count, batch_size=4096, n_init=3, random_state=seed).fit(rows)
        bank[name] = normalize(model.cluster_centers_.astype(np.float32))
    return bank


def bank_features(embeddings: np.ndarray, bank: dict[str, np.ndarray], chunk: int = 50_000) -> np.ndarray:
    """목소리 사전과의 유사도. 가장 닮은 중심, 상위 몇 개 평균, 대원 쪽과 신고자 쪽의 차이."""
    unit = normalize(np.asarray(embeddings, dtype=np.float32))
    output = np.zeros((len(unit), len(BANK_COLUMNS)), dtype=np.float32)
    for begin in range(0, len(unit), chunk):
        block = unit[begin:begin + chunk]
        for column, name in ((0, "operator"), (2, "caller")):
            similarity = block @ bank[name].T
            top = min(BANK_TOP, similarity.shape[1])
            best = np.partition(similarity, similarity.shape[1] - top, axis=1)[:, -top:]
            output[begin:begin + chunk, column] = best.max(axis=1)
            output[begin:begin + chunk, column + 1] = best.mean(axis=1)
    output[:, 4] = output[:, 0] - output[:, 2]
    return output


def iter_calls(counts: np.ndarray):
    """통화 경계로 행 구간을 돌려준다."""
    start = 0
    for value in counts:
        size = int(value)
        yield start, start + size
        start += size


def build_stage1(embeddings: np.ndarray, counts: np.ndarray, mfcc: np.ndarray | None) -> np.ndarray:
    width = stage1_width(mfcc is not None, 0 if mfcc is None else mfcc.shape[1])
    output = np.empty((len(embeddings), width), dtype=np.float32)
    for start, end in iter_calls(counts):
        if end > start:
            output[start:end] = stage1_features(embeddings[start:end], None if mfcc is None else mfcc[start:end])
    return output


def stage2_width(version: int) -> int:
    return len(STAGE2_COLUMNS) + (len(STAGE2_EXTRA_COLUMNS) if version >= 2 else 0)


def build_stage2(
    embeddings: np.ndarray, probabilities: np.ndarray, counts: np.ndarray,
    aux: np.ndarray | None = None, channel_scale: dict[str, np.ndarray] | None = None, version: int = 1,
) -> np.ndarray:
    output = np.empty((len(embeddings), stage2_width(version)), dtype=np.float32)
    for start, end in iter_calls(counts):
        if end > start:
            output[start:end] = stage2_features(
                embeddings[start:end], probabilities[start:end],
                None if aux is None else aux[start:end], channel_scale, version,
            )
    return output


def positive_probability(classifier, features: np.ndarray) -> np.ndarray:
    """신고자(1)일 확률. 열 순서를 classes_로 확인한다."""
    classes = list(classifier.classes_)
    return classifier.predict_proba(features)[:, classes.index(1)].astype(np.float64)


def predict_call(checkpoint: dict, embeddings: np.ndarray, mfcc: np.ndarray | None) -> np.ndarray:
    """통화 하나의 발화 전체를 한 번에 판정한다."""
    features = stage1_features(embeddings, mfcc)
    if checkpoint.get("bank") is not None:
        features = np.hstack((features, bank_features(embeddings, checkpoint["bank"])))
    probabilities = positive_probability(checkpoint["stage1"], features)
    if checkpoint.get("use_stage2"):
        # 예전 체크포인트는 2단계 모델 하나(v1)만 있다. v2는 여러 회차를 차례로 적용한다.
        version = int(checkpoint.get("stage2_version", 1))
        models = checkpoint.get("stage2_models") or [checkpoint["stage2"]]
        aux = stage2_aux(mfcc) if version >= 2 else None
        for model in models:
            stage2 = stage2_features(embeddings, probabilities, aux, checkpoint.get("channel_scale"), version)
            probabilities = positive_probability(model, stage2)
    return (probabilities >= 0.5).astype(np.int64)
