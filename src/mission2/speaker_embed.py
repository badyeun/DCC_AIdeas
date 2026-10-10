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


class EcapaEncoder:
    """SpeechBrain ECAPA-TDNN (VoxCeleb). EXP-003부터 쓰는 첫 번째 목소리 지문."""

    source = ECAPA_SOURCE
    dim = EMBED_DIM
    cache_prefix = "embed"

    def __init__(self, savedir: Path, device: str) -> None:
        self.savedir = Path(savedir)
        self.model = load_encoder(self.savedir, device)

    def embed(self, segments: list[np.ndarray], batch_size: int) -> np.ndarray:
        return embed_segments(self.model, segments, batch_size)

    @staticmethod
    def bundle_from(savedir: Path, target: Path) -> None:
        """받아 둔 인코더 파일을 체크포인트 옆으로 복사한다. 모델을 메모리에 올릴 필요가 없다."""
        bundle_encoder(Path(savedir), target)


WAVLM_SOURCE = "microsoft/wavlm-base-plus-sv"
WAVLM_DIM = 512


class WavLMEncoder:
    """Microsoft WavLM-Base-Plus 화자 검증 모델(x-vector). EXP-006에서 더한 두 번째 지문.

    ECAPA와 학습 방식이 달라(자기지도 사전학습 9.4만 시간 + VoxCeleb 화자 검증),
    짧은 발화에서 두 모델이 서로 다른 곳에서 틀릴 것을 기대한다.
    """

    source = WAVLM_SOURCE
    dim = WAVLM_DIM
    cache_prefix = "wavlm"

    def __init__(self, savedir: Path, device: str) -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.savedir = Path(savedir)
        local = (self.savedir / "config.json").is_file()
        source = str(self.savedir) if local else WAVLM_SOURCE
        self.extractor = AutoFeatureExtractor.from_pretrained(source)
        self.model = WavLMForXVector.from_pretrained(source).to(device).eval()
        self.device = device
        if not local:
            # 다음 실행과 추론이 인터넷 없이 돌도록 받은 파일을 savedir에 저장한다.
            self.savedir.mkdir(parents=True, exist_ok=True)
            self.extractor.save_pretrained(self.savedir)
            self.model.save_pretrained(self.savedir)

    def embed(self, segments: list[np.ndarray], batch_size: int) -> np.ndarray:
        """발화를 하나씩 넣는다. 묶어서 넣지 않는다.

        WavLM-Base는 맨 앞 합성곱 층이 GroupNorm이라 padding으로 채운 0까지 정규화에
        섞인다. 2초 발화를 같은 길이만큼 0으로 채워 함께 넣으면 혼자 넣었을 때와
        코사인 유사도가 0.93까지 떨어졌다(5%만 채워도 0.987). 통화 안의 발화 길이는
        제각각이라 묶으면 짧은 발화가 가장 크게 망가지므로, 하나씩 넣어 padding을 없앤다.
        """
        import torch

        output = np.zeros((len(segments), WAVLM_DIM), dtype=np.float32)
        for index, segment in enumerate(segments):
            inputs = self.extractor(fit_length(segment), sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                vectors = self.model(input_values=inputs["input_values"].to(self.device)).embeddings
            output[index] = vectors[0].float().cpu().numpy()
        return output


def normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norm, 1e-8)).astype(np.float32)


def fit_projection(raw: np.ndarray, dims: int, seed: int, sample: int = 200_000) -> dict[str, np.ndarray]:
    """두 번째 지문(512차원)을 PCA로 줄인다. 1단계 행렬과 메모리를 감당할 수 있는 크기로."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(raw), size=min(sample, len(raw)), replace=False)
    pca = PCA(n_components=dims, random_state=seed).fit(normalize(raw[np.sort(rows)]))
    return {"mean": pca.mean_.astype(np.float32), "components": pca.components_.astype(np.float32)}


def project(raw: np.ndarray, projection: dict[str, np.ndarray], chunk: int = 100_000) -> np.ndarray:
    output = np.empty((len(raw), len(projection["components"])), dtype=np.float32)
    for begin in range(0, len(raw), chunk):
        block = normalize(raw[begin:begin + chunk]) - projection["mean"]
        output[begin:begin + chunk] = block @ projection["components"].T
    return output


def combine_embeddings(primary: np.ndarray, secondary: np.ndarray | None) -> np.ndarray:
    """두 지문을 각각 길이 1로 맞춰 이어 붙인다. 이어 붙인 벡터의 코사인 유사도는 두
    모델 유사도의 평균이 되므로, 뒤의 1·2단계 코드는 바꾸지 않고 두 모델을 함께 쓴다."""
    if secondary is None:
        return primary
    return (np.hstack((normalize(primary), normalize(secondary))) / np.sqrt(2.0)).astype(np.float32)


def stage1_width(use_mfcc: bool, mfcc_width: int = 261, embed_dim: int = EMBED_DIM) -> int:
    return embed_dim * 2 + 2 + (mfcc_width * 2 if use_mfcc else 0)


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
    channel_scale: dict[str, np.ndarray] | None = None, version: int = 1, same: np.ndarray | None = None,
) -> np.ndarray:
    """같은 통화의 다른 발화가 받은 확률을 목소리 유사도로 가중해 모은다.

    - similar_to_caller: 신고자 같은 발화들과 평균적으로 얼마나 닮았는가
    - similar_to_operator: 대원 같은 발화들과 평균적으로 얼마나 닮았는가
    - neighbor_vote: 나와 가장 닮은 발화 몇 개의 평균 확률
    - (v2) anchor_*: 확률이 0.8 이상/0.2 이하인 확실한 발화 중 가장 닮은 몇 개와의 유사도
    - (v2) channel_*: 목소리 대신 회선 소리(MFCC 평균)로 잰 같은 비교
    - (v3) pair_*: 코사인 대신 학습한 같은 화자 판별기의 확률 same으로 잰 비교
    """
    if version >= 3:
        if same is None:
            raise ValueError("2단계 v3 이상에는 같은 화자 확률 행렬이 필요합니다.")
        v2 = stage2_features(embeddings, probabilities, aux, channel_scale, 2)
        parts = [v2, pair_stage2_features(same, probabilities)]
        if version >= 4:
            parts.append(joint_call_features(same, probabilities))
        return np.hstack(parts)
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


def pack_folder(folder: Path) -> dict[str, bytes]:
    """인코더 폴더의 파일을 바이트로 읽는다. 체크포인트 한 파일에 넣어 폴더 없이 추론하게 한다."""
    return {path.name: path.resolve().read_bytes() for path in sorted(Path(folder).iterdir())
            if path.is_file() or path.is_symlink()}


def unpack_folder(files: dict[str, bytes], target: Path) -> Path:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target / name).write_bytes(data)
    return target


def pack_checkpoint(source: Path, target: Path) -> None:
    """예전 체크포인트(인코더가 옆 폴더에 있는 형식)를 인코더까지 담은 한 파일로 다시 저장한다."""
    import joblib

    source, target = Path(source), Path(target)
    checkpoint = joblib.load(source)
    checkpoint["encoder_files"] = pack_folder(source.parent / str(checkpoint["encoder_dir"]))
    if checkpoint.get("second_encoder"):
        checkpoint["second_encoder_files"] = pack_folder(source.parent / str(checkpoint["second_encoder_dir"]))
    joblib.dump(checkpoint, target)


def iter_calls(counts: np.ndarray):
    """통화 경계로 행 구간을 돌려준다."""
    start = 0
    for value in counts:
        size = int(value)
        yield start, start + size
        start += size


def build_stage1(
    embeddings: np.ndarray, counts: np.ndarray, mfcc: np.ndarray | None, extra_columns: int = 0,
) -> np.ndarray:
    """1단계 행렬. extra_columns만큼 빈 열을 미리 두어, 뒤에 열을 붙일 때 전체를 복사하지 않는다."""
    width = stage1_width(mfcc is not None, 0 if mfcc is None else mfcc.shape[1], embeddings.shape[1])
    output = np.zeros((len(embeddings), width + extra_columns), dtype=np.float32)
    for start, end in iter_calls(counts):
        if end > start:
            output[start:end, :width] = stage1_features(embeddings[start:end], None if mfcc is None else mfcc[start:end])
    return output


class BoosterClassifier:
    """LightGBM Booster를 scikit-learn 분류기처럼 쓰게 감싼다 (`classes_`, `predict_proba`)."""

    classes_ = np.array([0, 1])

    def __init__(self, booster) -> None:
        self.booster = booster
        self.n_iter_ = booster.current_iteration()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = self.booster.predict(np.asarray(features, dtype=np.float32))
        return np.column_stack((1.0 - positive, positive))


def stage2_width(version: int) -> int:
    return (len(STAGE2_COLUMNS) + (len(STAGE2_EXTRA_COLUMNS) if version >= 2 else 0)
            + (len(PAIR_STAGE2_COLUMNS) if version >= 3 else 0)
            + (len(JOINT_COLUMNS) if version >= 4 else 0))


def build_stage2(
    embeddings: np.ndarray, probabilities: np.ndarray, counts: np.ndarray,
    aux: np.ndarray | None = None, channel_scale: dict[str, np.ndarray] | None = None, version: int = 1,
    same: list[np.ndarray] | None = None,
) -> np.ndarray:
    output = np.empty((len(embeddings), stage2_width(version)), dtype=np.float32)
    for index, (start, end) in enumerate(iter_calls(counts)):
        if end > start:
            output[start:end] = stage2_features(
                embeddings[start:end], probabilities[start:end],
                None if aux is None else aux[start:end], channel_scale, version,
                None if same is None else same[index],
            )
    return output


# ---------------------------------------------------------------------------
# 2단계 v3: 통화 안 "같은 화자 판별기" (EXP-007)
#
# 2단계 v1/v2는 VoxCeleb으로 학습된 코사인 유사도를 그대로 "같은 화자 정도"로 썼다.
# 그런데 남은 오류는 0.5~2초 발화에 몰려 있고, 이 길이에서 코사인은 크게 흔들린다.
# 그래서 Train 통화 안의 발화 쌍으로 "두 발화가 같은 화자인가"를 직접 학습한다.
# 입력은 두 모델의 코사인, 통화 평균을 뺀 코사인, 통화 안에서의 상대 순위, 두 발화의
# 길이, 회선 소리 차이다. 판별기가 길이에 따라 코사인을 얼마나 믿을지 스스로 배운다.
#
# 쌍 특징은 모두 (i, j)와 (j, i)에서 같은 값이 되게 만들어(최소·최대·절댓값·평균),
# 판별기 출력도 대칭이다. 발화 순서는 어디에도 들어가지 않는다.
# ---------------------------------------------------------------------------

PAIR_COLUMNS = (
    "cos_first", "cos_first_centered", "cos_first_relative",
    "cos_second", "cos_second_centered", "cos_second_relative",
    "rank_mean", "rank_max", "min_log_duration", "max_log_duration", "log_count",
    "channel_cos", "channel_distance",
) + tuple(f"channel_gap_{index}" for index in range(len(CHANNEL_COLUMNS)))
PAIR_STAGE2_COLUMNS = (
    "pair_vote", "pair_vote_confident", "pair_field_sum", "pair_to_caller", "pair_to_operator",
    "pair_difference", "pair_anchor_caller", "pair_anchor_operator", "pair_anchor_difference",
    "pair_neighbor_vote", "pair_same_mean",
)


def _cosine_parts(unit: np.ndarray, others: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """코사인, 통화 평균을 뺀 코사인, 두 발화의 평균 유사도를 뺀 상대 코사인."""
    similarity = unit @ unit.T
    centered_unit = normalize(unit - unit.mean(axis=0, keepdims=True)).astype(np.float64)
    centered = centered_unit @ centered_unit.T
    mean = (similarity * others).sum(axis=1) / max(count - 1, 1)
    relative = similarity - (mean[:, None] + mean[None, :]) / 2.0
    return similarity, centered, relative


def pair_setup(split: int | None, projection: dict[str, np.ndarray] | None = None) -> dict:
    """쌍 특징 설정. split은 이어 붙인 임베딩에서 첫 번째 지문(ECAPA)의 차원 수다(None이면
    지문이 하나). projection이 있으면 첫 번째 지문을 PCA로 줄인 뒤 차원별 비교를 더한다(EXP-008)."""
    return {"split": split, "projection": projection}


def pair_width(setup: dict) -> int:
    projection = setup.get("projection")
    return len(PAIR_COLUMNS) + (2 * len(projection["components"]) if projection is not None else 0)


def pair_feature_cube(
    embeddings: np.ndarray, aux: np.ndarray | None, channel_scale: dict[str, np.ndarray] | None,
    setup: dict,
) -> np.ndarray:
    """통화 하나의 모든 발화 쌍 특징 (n, n, pair_width(setup)). 대각선은 쓰지 않는다.

    앞의 len(PAIR_COLUMNS)열은 EXP-007과 같다. projection이 있으면 그 뒤에 PCA로 줄인
    ECAPA 지문(통화 평균을 뺀 값)의 차원별 곱과 차이의 절댓값을 붙인다. 코사인은 모든
    차원을 똑같이 더한 요약이라, 이 데이터에서 화자를 가르는 차원이 따로 있으면 놓친다.
    """
    split = setup.get("split")
    projection = setup.get("projection")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    count = len(embeddings)
    cube = np.zeros((count, count, pair_width(setup)), dtype=np.float32)
    others = ~np.eye(count, dtype=bool)
    first = embeddings if split is None else embeddings[:, :split]
    unit_first = normalize(first).astype(np.float64)
    for offset, unit in ((0, unit_first), (3, None if split is None else normalize(embeddings[:, split:]).astype(np.float64))):
        if unit is None:
            continue
        similarity, centered, relative = _cosine_parts(unit, others, count)
        cube[:, :, offset] = similarity
        cube[:, :, offset + 1] = centered
        cube[:, :, offset + 2] = relative
    # 통화 안 순위: i가 보기에 j보다 더 닮은 다른 발화가 몇 개인가. 값 비교만 하므로
    # 동점도 차례 없이 같은 순위가 되어 발화 순서와 무관하다.
    unit_all = normalize(embeddings).astype(np.float64)
    similarity = np.where(others, unit_all @ unit_all.T, -np.inf)
    rank = (similarity[:, None, :] > similarity[:, :, None]).sum(axis=2).astype(np.float64)
    rank /= max(count - 2, 1)
    cube[:, :, 6] = (rank + rank.T) / 2.0
    cube[:, :, 7] = np.maximum(rank, rank.T)
    cube[:, :, 10] = np.log(count)
    if aux is not None:
        duration = aux[:, -1].astype(np.float64)
        cube[:, :, 8] = np.minimum(duration[:, None], duration[None, :])
        cube[:, :, 9] = np.maximum(duration[:, None], duration[None, :])
        if channel_scale is not None:
            channel = (aux[:, :-1] - channel_scale["mean"]) / channel_scale["std"]
            # 통화 평균을 빼 두 화자가 함께 겪는 회선 상태를 지운다(2단계 v2와 같은 처리).
            channel = (channel - channel.mean(axis=0, keepdims=True)).astype(np.float64)
            unit_channel = normalize(channel).astype(np.float64)
            cube[:, :, 11] = unit_channel @ unit_channel.T
            gap = np.abs(channel[:, None, :] - channel[None, :, :])
            cube[:, :, 12] = np.sqrt((gap ** 2).sum(axis=2))
            cube[:, :, 13:len(PAIR_COLUMNS)] = gap
    if projection is not None:
        reduced = project(first, projection).astype(np.float64)
        reduced -= reduced.mean(axis=0, keepdims=True)
        dims = reduced.shape[1]
        cube[:, :, len(PAIR_COLUMNS):len(PAIR_COLUMNS) + dims] = reduced[:, None, :] * reduced[None, :, :]
        cube[:, :, len(PAIR_COLUMNS) + dims:] = np.abs(reduced[:, None, :] - reduced[None, :, :])
    return cube


def sample_pair_rows(count: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """학습용으로 통화 하나에서 쌍 (i < j)을 최대 limit개 고른다."""
    left, right = np.triu_indices(count, 1)
    if len(left) > limit:
        chosen = np.sort(rng.choice(len(left), size=limit, replace=False))
        left, right = left[chosen], right[chosen]
    return left, right


def build_pair_training(
    embeddings: np.ndarray, labels: np.ndarray, counts: np.ndarray, calls: np.ndarray,
    aux: np.ndarray | None, channel_scale: dict[str, np.ndarray] | None, setup: dict,
    limit: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """판별기 학습 데이터. 정답은 두 발화의 speaker가 같은지다(speaker는 정답으로만 쓴다).

    돌려주는 세 번째 값은 쌍이 속한 통화 번호다. 조기 종료용 분할을 통화 단위로 하기 위해 쓴다.
    """
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rng = np.random.default_rng(seed)
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    call_blocks: list[np.ndarray] = []
    for call in calls:
        start, size = int(starts[call]), int(counts[call])
        if size < 2:
            continue
        rows = slice(start, start + size)
        left, right = sample_pair_rows(size, limit, rng)
        cube = pair_feature_cube(embeddings[rows], None if aux is None else aux[rows], channel_scale, setup)
        feature_blocks.append(cube[left, right])
        label_blocks.append((labels[rows][left] == labels[rows][right]).astype(np.int64))
        call_blocks.append(np.full(len(left), call, dtype=np.int64))
    return np.vstack(feature_blocks), np.concatenate(label_blocks), np.concatenate(call_blocks)


def same_speaker_matrix(
    model, embeddings: np.ndarray, aux: np.ndarray | None, channel_scale: dict[str, np.ndarray] | None,
    setup: dict,
) -> np.ndarray:
    """통화 하나의 '같은 화자일 확률' 행렬. 위 삼각만 예측해 대칭으로 채운다."""
    count = len(embeddings)
    same = np.eye(count, dtype=np.float32)
    if count < 2:
        return same
    left, right = np.triu_indices(count, 1)
    cube = pair_feature_cube(embeddings, aux, channel_scale, setup)
    values = positive_probability(model, cube[left, right]).astype(np.float32)
    same[left, right] = values
    same[right, left] = values
    return same


def same_speaker_matrices(
    model, embeddings: np.ndarray, counts: np.ndarray, calls: np.ndarray,
    aux: np.ndarray | None, channel_scale: dict[str, np.ndarray] | None, setup: dict,
    chunk_pairs: int = 1_000_000,
) -> dict[int, np.ndarray]:
    """여러 통화의 행렬을 만든다. 예측은 쌍을 모아 한꺼번에 해 LightGBM 호출 횟수를 줄인다."""
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    output: dict[int, np.ndarray] = {}
    pending: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    pending_pairs = 0

    def flush() -> None:
        nonlocal pending, pending_pairs
        if not pending:
            return
        values = positive_probability(model, np.vstack([features for _, _, _, features in pending])).astype(np.float32)
        position = 0
        for call, left, right, _ in pending:
            same = np.eye(int(counts[call]), dtype=np.float32)
            part = values[position:position + len(left)]
            same[left, right] = part
            same[right, left] = part
            output[call] = same
            position += len(left)
        pending, pending_pairs = [], 0

    for call in calls:
        start, size = int(starts[call]), int(counts[call])
        if size < 2:
            output[int(call)] = np.eye(size, dtype=np.float32)
            continue
        rows = slice(start, start + size)
        left, right = np.triu_indices(size, 1)
        cube = pair_feature_cube(embeddings[rows], None if aux is None else aux[rows], channel_scale, setup)
        pending.append((int(call), left, right, cube[left, right]))
        pending_pairs += len(left)
        if pending_pairs >= chunk_pairs:
            flush()
    flush()
    return output


# 2단계 v4 (EXP-009): 통화 하나를 통째로 보고 정하는 특징.
# v3의 pair_* 특징은 발화마다 "다른 발화들이 1단계에서 받은 확률"을 한 번 모을 뿐이라, 다른
# 발화의 확률이 틀려 있으면 그 오류를 그대로 받는다. 여기서는 통화 안 모든 발화의 판정을
# 같은 화자 확률 S로 서로 맞춰 가며 동시에 정한다. 군집처럼 통화를 딱 자르지 않고(EXP-002
# 실패), 결과를 2단계 분류기의 입력 특징으로만 준다.
JOINT_COLUMNS = ("field_weak", "field_strong", "spectral_score", "spectral_gap", "spectral_agreement")
FIELD_STRENGTHS = (1.0, 3.0)
FIELD_STEPS = 20


def joint_call_features(same: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """통화 단위 동시 판정 특징.

    - field_*: 평균장(mean-field) 갱신. 내 확률 q_i를 "1단계 판정(사전) + 나와 같은 화자일
      발화들은 같은 쪽, 다른 화자일 발화들은 반대쪽"이 되게 모든 발화를 한꺼번에 반복
      갱신한다. 이웃 영향의 세기를 약/강 두 가지로 낸다. 모든 발화를 동시에 갱신하므로
      발화 순서와 무관하다.
    - spectral_score: 결합 행렬 (2S-1)의 가장 큰 고유벡터. 화자가 둘이면 이 벡터의 부호가
      두 화자를 가른다. 고유벡터의 부호는 임의이므로 1단계 판정과 같은 방향으로 맞춘다.
    - spectral_gap: 첫째·둘째 고유값 차이의 비율. 통화가 두 덩어리로 또렷이 갈리는 정도.
    - spectral_agreement: 고유벡터와 1단계 판정의 일치 정도(통화 단위).
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-4, 1 - 1e-4)
    count = len(probabilities)
    output = np.zeros((count, len(JOINT_COLUMNS)), dtype=np.float32)
    if count < 2:
        output[:, 0] = probabilities
        output[:, 1] = probabilities
        return output
    coupling = 2.0 * np.asarray(same, dtype=np.float64) - 1.0
    np.fill_diagonal(coupling, 0.0)
    prior = np.log(probabilities / (1.0 - probabilities))
    for column, strength in enumerate(FIELD_STRENGTHS):
        belief = probabilities.copy()
        for _ in range(FIELD_STEPS):
            field = coupling @ (2.0 * belief - 1.0) / (count - 1)
            # 절반만 움직여(감쇠) 두 화자가 번갈아 뒤집히는 진동을 막는다.
            belief = 0.5 * belief + 0.5 / (1.0 + np.exp(-(prior + strength * field)))
        output[:, column] = belief
    values, vectors = np.linalg.eigh(coupling)
    leading = vectors[:, -1]
    direction = 2.0 * probabilities - 1.0
    if float(leading @ direction) < 0:
        leading = -leading
    output[:, 2] = leading * np.sqrt(count)
    output[:, 3] = (values[-1] - values[-2]) / (abs(values[-1]) + abs(values[-2]) + 1e-8)
    output[:, 4] = abs(float(leading @ direction)) / (np.linalg.norm(direction) + 1e-8)
    return output


def pair_stage2_features(same: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """판별기의 같은 화자 확률 S로 다른 발화의 1단계 확률 p를 모은다.

    - pair_vote: 같은 화자면 상대의 판정을, 다른 화자면 그 반대를 따른 평균.
      S·p + (1-S)·(1-p)의 평균이다.
    - pair_vote_confident: 위 값을 S와 p가 확실할수록 크게 가중한 평균.
    - pair_field_sum: Σ(2S-1)(2p-1). 평균이 아닌 합이라 근거의 양(통화 길이)도 담는다.
    - pair_to_*, pair_anchor_*, pair_neighbor_vote: v1/v2의 같은 이름 특징을 코사인 대신 S로 잰 값.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(probabilities)
    output = np.full((count, len(PAIR_STAGE2_COLUMNS)), 0.5, dtype=np.float32)
    if count < 2:
        output[:, 2] = 0.0
        output[:, 9] = probabilities
        return output
    same = np.asarray(same, dtype=np.float64)
    others = ~np.eye(count, dtype=bool)
    agree = same * probabilities[None, :] + (1.0 - same) * (1.0 - probabilities[None, :])
    output[:, 0] = (agree * others).sum(axis=1) / (count - 1)
    weight = np.abs(2.0 * same - 1.0) * np.abs(2.0 * probabilities[None, :] - 1.0) * others
    total = weight.sum(axis=1)
    output[:, 1] = np.where(total > 1e-8, (agree * weight).sum(axis=1) / np.maximum(total, 1e-8), 0.5)
    output[:, 2] = ((2.0 * same - 1.0) * (2.0 * probabilities[None, :] - 1.0) * others).sum(axis=1)
    to_caller, to_operator = weighted_to_sides(same, probabilities, others)
    output[:, 3] = to_caller
    output[:, 4] = to_operator
    output[:, 5] = to_caller - to_operator
    neutral = (same * others).sum(axis=1) / (count - 1)
    output[:, 6] = top_mean(same, others & (probabilities >= ANCHOR)[None, :], neutral)
    output[:, 7] = top_mean(same, others & (probabilities <= 1.0 - ANCHOR)[None, :], neutral)
    output[:, 8] = output[:, 6] - output[:, 7]
    # 가장 같은 화자 같은 k개(k번째 값 이상 전부, 동점을 차례로 가르지 않는다)의 평균 확률.
    k = min(NEIGHBORS, count - 1)
    for row in range(count):
        values = same[row, others[row]]
        threshold = np.partition(values, count - 1 - k)[count - 1 - k]
        output[row, 9] = probabilities[others[row]][values >= threshold].mean()
    output[:, 10] = neutral
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
        same = None
        if version >= 3:
            # 같은 화자 확률은 1단계 확률과 무관하므로 회차마다 다시 계산하지 않는다.
            same = same_speaker_matrix(checkpoint["pair_model"], embeddings, aux,
                                       checkpoint.get("channel_scale"),
                                       pair_setup(checkpoint.get("embed_split"), checkpoint.get("pair_projection")))
        for model in models:
            stage2 = stage2_features(embeddings, probabilities, aux, checkpoint.get("channel_scale"), version, same)
            probabilities = positive_probability(model, stage2)
    return (probabilities >= 0.5).astype(np.int64)
