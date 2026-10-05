"""Mission 2 화자 임베딩 + 통화 내 비교 학습/검증 스크립트.

발화마다 ECAPA 화자 임베딩을 뽑아 `--embed_cache_dir`에 shard로 저장하고,
두 단계 분류기(speaker_embed.py 참고)를 학습한다. `--mfcc_cache_dir`를 주면
EXP-001/002에서 만든 MFCC 특징 캐시를 읽어 1단계 특징에 함께 넣는다.

MFCC 캐시는 읽기만 한다. 설정이 다르거나 통화가 모자라면 손대지 않고 멈춘다.
재추출에 5시간이 걸리는 캐시이기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from common import (
    SAMPLE_RATE, SPEAKER_TO_INDEX, decode_annotation, decode_waveform, detect_timestamp_unit,
    get_utterances, slice_utterance,
)
from speaker_embed import (
    BANK_COLUMNS, ECAPA_SOURCE, EMBED_DIM, MAX_SAMPLES, MIN_SAMPLES, PAIR_COLUMNS, PAIR_STAGE2_COLUMNS, STAGE2_COLUMNS,
    STAGE2_EXTRA_COLUMNS, WAVLM_SOURCE, BoosterClassifier, EcapaEncoder, WavLMEncoder, bank_features,
    build_pair_training, build_stage1, build_stage2, combine_embeddings, fit_bank, fit_channel_scale, fit_projection,
    positive_probability, project, same_speaker_matrices, stage2_aux,
)
from train_mfcc_rf import build_features, collect_pairs, iter_prefetched, load_shards, shard_path


ENCODER_DIRNAME = "ecapa_voxceleb"
SECOND_ENCODER_DIRNAME = "wavlm_base_plus_sv"
TARGET_NAMES = ["119대원(0)", "신고자(1)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2 화자 임베딩 + 통화 내 비교 학습/검증")
    parser.add_argument("--data_root", type=Path, required=True, help="001~012 part 폴더 경로")
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission2_speaker.ckpt"))
    parser.add_argument("--embed_cache_dir", type=Path, required=True,
                        help="임베딩 shard 저장 폴더. MFCC 캐시와 다른 폴더를 권장한다.")
    parser.add_argument("--mfcc_cache_dir", type=Path, default=None,
                        help="EXP-001 MFCC 캐시. 주면 1단계 특징에 MFCC와 그 통화 상대값을 더한다.")
    parser.add_argument("--encoder_dir", type=Path, default=Path("pretrained/ecapa_voxceleb"),
                        help="ECAPA 파일을 받아 둘 폴더")
    parser.add_argument("--device", default=None, help="cuda 또는 cpu. 비우면 GPU가 있으면 GPU를 쓴다.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shard_calls", type=int, default=2000)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--prefetch_depth", type=int, default=32)
    parser.add_argument("--n_mfcc", type=int, default=20, help="MFCC 캐시를 만들 때 쓴 값과 같아야 한다.")
    parser.add_argument("--timestamp_unit", choices=("auto", "seconds", "milliseconds"), default="auto")
    parser.add_argument("--folds", type=int, default=4, help="2단계 학습용 out-of-fold 확률을 만들 fold 수")
    parser.add_argument("--max_iter", type=int, default=300, help="1단계 부스팅 반복 수 상한")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="1단계 부스팅 학습률")
    parser.add_argument("--bank_clusters", type=int, default=0,
                        help="Train 대원/신고자 목소리 사전의 군집 수. 0이면 사용하지 않는다(EXP-003).")
    parser.add_argument("--stage2", choices=("auto", "on", "off"), default="auto",
                        help="2단계 사용 여부. auto는 Validation 정확도가 가장 높은 회차까지 저장한다.")
    parser.add_argument("--stage2_version", type=int, choices=(1, 2, 3), default=2,
                        help="1: EXP-003/004 입력 6개. 2: 발화 길이·확실한 발화와의 비교·회선 비교 추가. "
                             "3: 같은 화자 판별기로 잰 비교 추가(EXP-007)")
    parser.add_argument("--pair_samples", type=int, default=120,
                        help="2단계 v3 판별기 학습에 통화마다 쓰는 발화 쌍 수 상한")
    parser.add_argument("--pair_max_iter", type=int, default=500, help="2단계 v3 판별기 부스팅 반복 수 상한")
    parser.add_argument("--rounds", type=int, default=2, help="2단계 반복 회차 수 (EXP-003/004는 1)")
    parser.add_argument("--stage1_cache", type=Path, default=None,
                        help="1단계 모델과 확률을 저장/재사용할 파일. 설정이 같으면 1단계 학습(약 40분)을 건너뛴다.")
    parser.add_argument("--second_encoder", choices=("none", "wavlm"), default="none",
                        help="두 번째 목소리 지문. wavlm이면 WavLM 화자 모델 임베딩을 ECAPA와 함께 쓴다(EXP-006).")
    parser.add_argument("--second_cache_dir", type=Path, default=None, help="두 번째 지문 shard 저장 폴더")
    parser.add_argument("--second_encoder_dir", type=Path, default=Path("pretrained/wavlm_base_plus_sv"),
                        help="WavLM 파일을 받아 둘 폴더")
    parser.add_argument("--second_dims", type=int, default=128, help="두 번째 지문을 PCA로 줄일 차원")
    parser.add_argument("--extract_only", action="store_true",
                        help="임베딩 캐시만 만들고 끝낸다. 추출은 GPU, 학습은 CPU 런타임에서 따로 돌릴 때 쓴다.")
    parser.add_argument("--feature_fraction", type=float, default=1.0,
                        help="1단계 나무마다 쓰는 특징 비율. 1보다 작으면 학습이 빨라진다.")
    parser.add_argument("--bagging_fraction", type=float, default=1.0,
                        help="1단계 나무마다 쓰는 행 비율. 1보다 작으면 학습이 빨라진다.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def elapsed(start: float) -> str:
    seconds = int(time.perf_counter() - start)
    return f"{seconds // 3600}시간 {seconds % 3600 // 60}분 {seconds % 60}초"


def build_embeddings(
    pairs: list[tuple[Path, Path]], encoder, timestamp_unit: str, desc: str, split: str,
    cache_dir: Path, shard_calls: int, prefetch: int, prefetch_depth: int, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """발화 임베딩을 통화 순서대로 뽑는다. 행과 통화 경계는 MFCC 캐시와 같은 규칙으로 쌓는다."""
    signature = {
        "encoder": encoder.source, "timestamp_unit": timestamp_unit, "pairs": len(pairs),
        "min_samples": MIN_SAMPLES, "max_samples": MAX_SAMPLES,
    }
    blocks, label_blocks, count_blocks, done, skipped, _ = load_shards(cache_dir, split, signature)
    if done:
        print(f"[{desc}] 캐시에서 통화 {done:,}/{len(pairs):,}개 복원")
    pending_vectors: list[np.ndarray] = []
    pending_labels: list[int] = []
    pending_counts: list[int] = []
    pending_calls = pending_skipped = 0

    def flush() -> None:
        nonlocal pending_vectors, pending_labels, pending_counts, pending_calls, pending_skipped
        if pending_calls == 0:
            return
        x = np.vstack(pending_vectors).astype(np.float16) if pending_vectors else np.zeros((0, encoder.dim), dtype=np.float16)
        y = np.asarray(pending_labels, dtype=np.int64)
        counts = np.asarray(pending_counts, dtype=np.int64)
        temporary = cache_dir / f".{split}_partial.npz"
        np.savez(temporary, x=x, y=y, counts=counts, calls=pending_calls, skipped=pending_skipped)
        temporary.replace(shard_path(cache_dir, split, len(blocks)))
        blocks.append(x)
        label_blocks.append(y)
        count_blocks.append(counts)
        pending_vectors, pending_labels, pending_counts = [], [], []
        pending_calls = pending_skipped = 0

    for audio_path, payload in tqdm(iter_prefetched(pairs[done:], prefetch, prefetch_depth),
                                    desc=desc, initial=done, total=len(pairs)):
        segments: list[np.ndarray] = []
        labels: list[int] = []
        try:
            if isinstance(payload, OSError):
                raise payload
            audio_bytes, label_bytes = payload
            annotation = decode_annotation(label_bytes)
            waveform = decode_waveform(audio_bytes)
            unit = detect_timestamp_unit(annotation, len(waveform) / SAMPLE_RATE) if timestamp_unit == "auto" else timestamp_unit
            for utterance in get_utterances(annotation):
                # speaker는 학습 정답으로만 읽는다. text 및 순서는 전혀 사용하지 않는다.
                speaker = str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()
                if speaker not in SPEAKER_TO_INDEX:
                    skipped += 1
                    pending_skipped += 1
                    continue
                segments.append(slice_utterance(waveform, utterance, unit))
                labels.append(SPEAKER_TO_INDEX[speaker])
        except (OSError, RuntimeError, ValueError) as error:
            # MFCC 캐시와 같게, 실패 전까지 자른 발화는 남기고 통화를 끝낸다.
            skipped += 1
            pending_skipped += 1
            print(f"[warn] {audio_path.name}: {error}")
        if segments:
            pending_vectors.append(encoder.embed(segments, batch_size))
            pending_labels.extend(labels)
        pending_counts.append(len(segments))
        pending_calls += 1
        if pending_calls >= shard_calls:
            flush()
    flush()
    filled = [block for block in blocks if block.size]
    if not filled:
        raise ValueError(f"{desc}에서 유효한 발화를 만들지 못했습니다.")
    vectors = np.vstack(filled).astype(np.float32)
    labels = np.concatenate([block for block in label_blocks if block.size])
    counts = np.concatenate([block for block in count_blocks if block.size])
    if int(counts.sum()) != len(vectors):
        raise ValueError(f"[{desc}] 통화 경계 합({int(counts.sum()):,})이 발화 수({len(vectors):,})와 다릅니다.")
    print(f"[{desc}] 발화: {len(vectors):,}, 통화: {len(counts):,}, 건너뜀: {skipped:,}")
    return vectors, labels, counts


def check_mfcc_cache(cache_dir: Path, split: str, n_mfcc: int, timestamp_unit: str, calls: int) -> None:
    """MFCC 캐시가 그대로 쓸 수 있는 상태인지 읽기만 해서 확인한다.

    build_features는 설정이 다르면 shard를 지우고, 통화가 모자라면 나머지를 추출한다.
    여기서 걸러 두면 그 두 경우 모두 캐시를 건드리기 전에 멈춘다.
    """
    meta_path = cache_dir / f"{split}_meta.json"
    expected = {"n_mfcc": n_mfcc, "timestamp_unit": timestamp_unit, "pairs": calls}
    if not meta_path.is_file():
        raise FileNotFoundError(f"MFCC 캐시 설정 파일이 없습니다: {meta_path}")
    found = json.loads(meta_path.read_text(encoding="utf-8"))
    if found != expected:
        raise ValueError(f"MFCC 캐시 설정이 다릅니다. 캐시를 건드리지 않고 멈춥니다: {found} != {expected}")
    cached = 0
    for index in range(10_000):
        path = shard_path(cache_dir, split, index)
        if not path.is_file():
            break
        with np.load(path) as data:
            cached += int(data["calls"])
    if cached != calls:
        raise ValueError(f"MFCC 캐시에 통화 {cached:,}/{calls:,}개만 있습니다. 캐시를 건드리지 않고 멈춥니다.")


def load_mfcc(
    args: argparse.Namespace, pairs: list[tuple[Path, Path]], split: str, desc: str,
    labels: np.ndarray, counts: np.ndarray,
) -> np.ndarray:
    check_mfcc_cache(args.mfcc_cache_dir, split, args.n_mfcc, args.timestamp_unit, len(pairs))
    features, mfcc_labels, mfcc_counts = build_features(
        pairs, args.n_mfcc, args.timestamp_unit, desc, split, args.mfcc_cache_dir,
        args.shard_calls, args.prefetch, args.prefetch_depth,
    )
    # 두 캐시의 행이 같은 발화를 가리키는지 확인한다. 어긋나면 특징이 엉뚱한 발화와 묶인다.
    if not np.array_equal(mfcc_counts, counts) or not np.array_equal(mfcc_labels, labels):
        raise ValueError(f"[{desc}] MFCC 캐시와 임베딩 캐시의 발화 정렬이 다릅니다. --mfcc_cache_dir 없이 실행하세요.")
    return features


def make_classifier(max_iter: int, seed: int, learning_rate: float = 0.1) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=63, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
    )


def report(name: str, labels: np.ndarray, probabilities: np.ndarray) -> float:
    prediction = (probabilities >= 0.5).astype(np.int64)
    accuracy = accuracy_score(labels, prediction)
    print(f"Validation Accuracy ({name}): {accuracy:.4f}")
    print(classification_report(labels, prediction, labels=[0, 1], target_names=TARGET_NAMES, digits=4, zero_division=0))
    return float(accuracy)


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or args.folds < 2 or args.max_iter < 1 or args.batch_size < 1 or args.rounds < 0:
        raise ValueError("data_root, folds(2 이상), max_iter, batch_size, rounds를 확인하세요.")
    if args.mfcc_cache_dir is not None and args.mfcc_cache_dir.resolve() == args.embed_cache_dir.resolve():
        raise ValueError("임베딩 캐시는 MFCC 캐시와 다른 폴더를 쓰세요.")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    total_start = time.perf_counter()
    train_pairs = collect_pairs(args.data_root, "Training")
    val_pairs = collect_pairs(args.data_root, "Validation")
    if args.mfcc_cache_dir is not None:
        # 임베딩을 몇 시간 뽑은 뒤에 MFCC 캐시 문제로 멈추지 않도록 먼저 확인한다.
        check_mfcc_cache(args.mfcc_cache_dir, "train", args.n_mfcc, args.timestamp_unit, len(train_pairs))
        check_mfcc_cache(args.mfcc_cache_dir, "val", args.n_mfcc, args.timestamp_unit, len(val_pairs))

    start = time.perf_counter()
    encoder = EcapaEncoder(args.encoder_dir, device)
    loader = (args.shard_calls, args.prefetch, args.prefetch_depth, args.batch_size)
    emb_train, y_train, train_counts = build_embeddings(
        train_pairs, encoder, args.timestamp_unit, "Train 임베딩", "embed_train", args.embed_cache_dir, *loader)
    emb_val, y_val, val_counts = build_embeddings(
        val_pairs, encoder, args.timestamp_unit, "Validation 임베딩", "embed_val", args.embed_cache_dir, *loader)
    del encoder
    projection = None
    if args.second_encoder == "wavlm":
        second = WavLMEncoder(args.second_encoder_dir, device)
        second_dir = args.second_cache_dir or args.embed_cache_dir
        prefix = second.cache_prefix
        second_train, second_y, second_counts = build_embeddings(
            train_pairs, second, args.timestamp_unit, "Train WavLM", f"{prefix}_train", second_dir, *loader)
        second_val, second_val_y, second_val_counts = build_embeddings(
            val_pairs, second, args.timestamp_unit, "Validation WavLM", f"{prefix}_val", second_dir, *loader)
        del second
        # 두 지문의 행이 같은 발화를 가리켜야 이어 붙일 수 있다.
        for name, (a, b) in {"Train 정답": (second_y, y_train), "Train 통화 경계": (second_counts, train_counts),
                             "Validation 정답": (second_val_y, y_val), "Validation 통화 경계": (second_val_counts, val_counts)}.items():
            if not np.array_equal(a, b):
                raise ValueError(f"ECAPA와 WavLM 캐시의 {name}이 다릅니다.")
        projection = fit_projection(second_train, args.second_dims, args.seed)
        emb_train = combine_embeddings(emb_train, project(second_train, projection))
        emb_val = combine_embeddings(emb_val, project(second_val, projection))
        del second_train, second_val
        print(f"두 번째 지문: WavLM 512 → PCA {args.second_dims}, ECAPA와 이어 붙여 {emb_train.shape[1]}차원")
    print(f"임베딩 단계 소요: {elapsed(start)}")
    if args.extract_only:
        print("--extract_only: 임베딩 캐시만 만들고 끝냅니다.")
        return

    mfcc_train = mfcc_val = None
    if args.mfcc_cache_dir is not None:
        start = time.perf_counter()
        mfcc_train = load_mfcc(args, train_pairs, "train", "Train MFCC", y_train, train_counts)
        mfcc_val = load_mfcc(args, val_pairs, "val", "Validation MFCC", y_val, val_counts)
        print(f"MFCC 캐시 로딩 소요: {elapsed(start)}")

    duration_val = np.expm1(mfcc_val[:, -1]) if mfcc_val is not None else None
    aux_train, aux_val = stage2_aux(mfcc_train), stage2_aux(mfcc_val)
    channel_scale = fit_channel_scale(aux_train) if aux_train is not None else None
    groups = np.repeat(np.arange(len(train_counts)), train_counts)
    folds = list(GroupKFold(n_splits=args.folds).split(emb_train, y_train, groups))
    print(f"Train 발화: {len(y_train):,}, Validation 발화: {len(y_val):,}")
    print(f"다수 클래스 기준선: {float(np.bincount(y_val, minlength=2).max() / len(y_val)):.4f}")

    stage1_signature = {
        "max_iter": args.max_iter, "learning_rate": args.learning_rate, "bank_clusters": args.bank_clusters,
        "folds": args.folds, "seed": args.seed, "use_mfcc": args.mfcc_cache_dir is not None,
        "train_rows": len(y_train), "val_rows": len(y_val), "learner": "lightgbm",
        "second_encoder": args.second_encoder, "second_dims": args.second_dims if projection is not None else 0,
        "feature_fraction": args.feature_fraction, "bagging_fraction": args.bagging_fraction,
    }
    cached = None
    if args.stage1_cache is not None and args.stage1_cache.is_file():
        cached = joblib.load(args.stage1_cache)
        # EXP-005 캐시는 EXP-006에서 더한 설정 키가 없다. 없는 키는 그때의 기본값으로 본다.
        saved_signature = {**STAGE1_SIGNATURE_DEFAULTS, **cached.get("signature", {})}
        if saved_signature != stage1_signature:
            print(f"1단계 캐시 설정이 달라 다시 학습합니다: {cached.get('signature')}")
            cached = None
    if cached is not None:
        # 2단계만 바꾸는 실험은 1단계(약 40분)를 다시 학습하지 않는다.
        stage1, bank = cached["stage1"], cached["bank"]
        out_of_fold, val_stage1 = cached["out_of_fold"], cached["val_stage1"]
        print(f"1단계 캐시 사용: {args.stage1_cache}")
        del mfcc_train, mfcc_val
        stage1_accuracy = report("1단계, 발화별", y_val, val_stage1)
    else:
        x_train, x_val, bank = build_stage1_inputs(
            args, emb_train, y_train, train_counts, mfcc_train, emb_val, val_counts, mfcc_val, folds)
        # 1단계 행렬을 만든 뒤에는 MFCC 원본이 필요 없다. 학습 중 메모리 정점을 낮추려 먼저 지운다.
        del mfcc_train, mfcc_val
        print(f"[메모리] 1단계 행렬 완성: {memory_status()}")
        # EXP-005 첫 세 시도는 첫 fold 시작 직후 메모리 부족(^C)으로 죽었다. scikit-learn
        # HistGradientBoosting이 입력을 float64로 복사하고, 조기 종료용 분할에서 그 사본을 한 번
        # 더 복사하기 때문이다. LightGBM은 float32를 그대로 받아 전체를 한 번만 구간화하고,
        # fold와 조기 종료 분할은 그 구간화 결과를 행 번호로만 나눠 쓴다.
        # 원본 행렬은 로컬 디스크에 두고 필요한 행만 읽는다.
        spill_path = Path(tempfile.gettempdir()) / "mission2_stage1_x_train.npy"
        np.save(spill_path, x_train)
        del x_train
        x_train = np.load(spill_path, mmap_mode="r")
        start = time.perf_counter()
        dataset = make_dataset(x_train, y_train)
        print(f"[메모리] LightGBM 구간화 완료 ({elapsed(start)}): {memory_status()}")
        out_of_fold, best_rounds = fold_probabilities(args, x_train, y_train, folds, groups, dataset, stage1_signature)
        # 전체 모델은 조기 종료할 검증 데이터가 따로 없으므로 fold에서 찾은 반복 수의 평균을 쓴다.
        final_rounds = max(1, int(round(float(np.mean(best_rounds)))))
        stage1 = train_booster(args, dataset, None, None, final_rounds)
        del dataset, x_train
        try:
            spill_path.unlink(missing_ok=True)
        except PermissionError:  # Windows에서 매핑이 늦게 풀리는 경우. 임시 폴더라 남아도 무해하다.
            pass
        val_stage1 = positive_probability(stage1, x_val)
        del x_val
        print(f"1단계 학습 소요: {elapsed(start)} (fold 반복 {best_rounds} → 전체 모델 반복 {final_rounds})")
        stage1_accuracy = report("1단계, 발화별", y_val, val_stage1)
        if args.stage1_cache is not None:
            args.stage1_cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"signature": stage1_signature, "stage1": stage1, "bank": bank,
                         "out_of_fold": out_of_fold, "val_stage1": val_stage1}, args.stage1_cache)
            print(f"1단계 캐시 저장: {args.stage1_cache}")
            for index in range(1, args.folds + 1):
                fold_cache_path(args.stage1_cache, index).unlink(missing_ok=True)

    stage1_oof_accuracy = float(accuracy_score(y_train, out_of_fold >= 0.5))
    print(f"Train out-of-fold Accuracy (1단계): {stage1_oof_accuracy:.4f}")
    embed_split = EMBED_DIM if projection is not None else None
    pair_model = same_train = same_val = None
    if args.stage2_version >= 3:
        if aux_train is None:
            raise ValueError("2단계 v3는 발화 길이와 회선 특징이 필요합니다. --mfcc_cache_dir를 주세요.")
        pair_model, same_train, same_val = train_pair_model(
            args, emb_train, y_train, train_counts, aux_train, emb_val, val_counts, aux_val,
            channel_scale, embed_split, folds, groups)

    # 2단계를 여러 회차 반복한다. 회차마다 바로 앞 회차의 확률로 통화 내 비교를 다시 한다.
    # Train 입력 확률은 매번 통화 단위 fold의 out-of-fold 값이라 과신하지 않는다.
    start = time.perf_counter()
    columns = (STAGE2_COLUMNS + (STAGE2_EXTRA_COLUMNS if args.stage2_version >= 2 else ())
               + (PAIR_STAGE2_COLUMNS if args.stage2_version >= 3 else ()))
    print(f"2단계 v{args.stage2_version} 입력: {', '.join(columns)}")
    train_probability, val_probability = out_of_fold, val_stage1
    stage2_models: list[HistGradientBoostingClassifier] = []
    round_accuracy = [stage1_accuracy]
    round_oof_accuracy = [stage1_oof_accuracy]
    round_probability = [val_stage1]
    for round_index in range(1, args.rounds + 1):
        x2_train = build_stage2(emb_train, train_probability, train_counts, aux_train, channel_scale,
                                args.stage2_version, same_train)
        x2_val = build_stage2(emb_val, val_probability, val_counts, aux_val, channel_scale, args.stage2_version, same_val)
        model = make_classifier(200, args.seed).fit(x2_train, y_train)
        val_probability = positive_probability(model, x2_val)
        stage2_models.append(model)
        round_probability.append(val_probability)
        round_accuracy.append(report(f"2단계 {round_index}회차, 통화 내 비교", y_val, val_probability))
        # 회차마다 Train out-of-fold 정확도를 잰다. 설정 선택은 Validation이 아니라 이 값으로 한다.
        next_probability = np.zeros(len(y_train), dtype=np.float64)
        for fit_rows, held_rows in folds:
            fold_model = make_classifier(200, args.seed).fit(x2_train[fit_rows], y_train[fit_rows])
            next_probability[held_rows] = positive_probability(fold_model, x2_train[held_rows])
        train_probability = next_probability
        round_oof_accuracy.append(float(accuracy_score(y_train, next_probability >= 0.5)))
        print(f"Train out-of-fold Accuracy (2단계 {round_index}회차): {round_oof_accuracy[-1]:.4f}")
    print(f"2단계 학습 소요: {elapsed(start)} ({args.rounds}회차)")

    if args.stage2 == "off":
        rounds_used = 0
    elif args.stage2 == "on":
        rounds_used = args.rounds
    else:
        # Validation으로 고르면 그 점수가 낙관적으로 부풀므로 Train out-of-fold로 고른다(EXP-007부터).
        rounds_used = int(np.argmax(round_oof_accuracy))
    selected = round_accuracy[rounds_used]
    stage2_accuracy = max(round_accuracy[1:]) if args.rounds else stage1_accuracy
    print(f"회차별 Validation Accuracy: {' → '.join(f'{value:.4f}' for value in round_accuracy)}")
    print(f"회차별 Train out-of-fold Accuracy: {' → '.join(f'{value:.4f}' for value in round_oof_accuracy)}")
    print(f"선택한 설정: 2단계 {rounds_used}회차 (--stage2 {args.stage2}) → Accuracy {selected:.4f}")

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 오류 분석용. 어떤 통화·길이에서 틀리는지 다음 실험을 고르는 데 쓴다.
    predictions_path = args.ckpt_path.with_name(args.ckpt_path.stem + "_val_predictions.npz")
    np.savez_compressed(
        predictions_path, labels=y_val, counts=val_counts, stage1=val_stage1,
        stage2=round_probability[max(rounds_used, 1)] if args.rounds else val_stage1,
        rounds=np.vstack(round_probability),
        duration=duration_val if duration_val is not None else np.zeros(0),
    )
    print(f"Validation 예측 저장: {predictions_path}")
    # 추론이 인터넷 없이 돌도록 ECAPA 파일을 체크포인트 옆에 함께 둔다.
    EcapaEncoder.bundle_from(args.encoder_dir, args.ckpt_path.parent / ENCODER_DIRNAME)
    if projection is not None:
        EcapaEncoder.bundle_from(args.second_encoder_dir, args.ckpt_path.parent / SECOND_ENCODER_DIRNAME)
    joblib.dump({
        "model_type": "mission2_speaker_embedding",
        "stage1": stage1,
        "stage2": stage2_models[0] if stage2_models else None,
        "stage2_models": stage2_models[:rounds_used],
        "stage2_version": args.stage2_version,
        "channel_scale": channel_scale,
        "use_stage2": rounds_used > 0,
        "use_mfcc": args.mfcc_cache_dir is not None,
        "bank": bank,
        "n_mfcc": args.n_mfcc,
        "timestamp_unit": args.timestamp_unit,
        "encoder_dir": ENCODER_DIRNAME,
        "encoder_source": ECAPA_SOURCE,
        "second_encoder": args.second_encoder if projection is not None else None,
        "second_encoder_dir": SECOND_ENCODER_DIRNAME if projection is not None else None,
        "second_encoder_source": WAVLM_SOURCE if projection is not None else None,
        "projection": projection,
        "pair_model": pair_model,
        "embed_split": embed_split,
        "batch_size": args.batch_size,
        "fallback_speaker": int(np.bincount(y_train, minlength=2).argmax()),
        "seed": args.seed,
        "validation_accuracy": float(selected),
        "validation_accuracy_stage1": stage1_accuracy,
        "validation_accuracy_stage2": stage2_accuracy,
        "validation_accuracy_rounds": round_accuracy,
        "train_oof_accuracy_rounds": round_oof_accuracy,
    }, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path} (+ {args.ckpt_path.parent / ENCODER_DIRNAME})")
    print(f"전체 소요: {elapsed(total_start)}")


# EXP-006에서 1단계 캐시 설정에 더한 키와 그 전의 값. 예전 캐시(EXP-005)를 그대로 쓰기 위해 둔다.
STAGE1_SIGNATURE_DEFAULTS = {"second_encoder": "none", "second_dims": 0, "feature_fraction": 1.0, "bagging_fraction": 1.0}


def train_pair_model(args, emb_train, y_train, train_counts, aux_train, emb_val, val_counts, aux_val,
                     channel_scale, embed_split, folds, groups):
    """2단계 v3의 같은 화자 판별기를 학습하고, Train/Validation의 같은 화자 확률 행렬을 만든다.

    Train 통화의 행렬은 그 통화를 보지 않은 fold 판별기로 만든다(1단계 out-of-fold와 같은
    fold). 자기 통화로 학습한 판별기는 그 통화의 쌍을 지나치게 잘 맞혀 2단계가 과신한다.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    start = time.perf_counter()
    all_calls = np.arange(len(train_counts))
    x_pair, y_pair, pair_calls = build_pair_training(
        emb_train, y_train, train_counts, all_calls, aux_train, channel_scale, embed_split, args.pair_samples, args.seed)
    print(f"같은 화자 판별기 학습 쌍: {len(y_pair):,} (같은 화자 {y_pair.mean():.3f}), 특징 {x_pair.shape[1]}개 "
          f"({elapsed(start)}), {memory_status()}")
    params = {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50,
        "max_bin": 255, "seed": args.seed, "deterministic": True, "force_row_wise": True, "verbose": -1,
    }
    dataset = lgb.Dataset(x_pair, label=y_pair, params={"max_bin": 255, "verbose": -1}, free_raw_data=False).construct()

    def fit(rows: np.ndarray, rounds: int, stop_seed: int | None) -> BoosterClassifier:
        """rows로 학습한다. stop_seed가 있으면 그 통화 10%를 떼어 조기 종료 판단에 쓴다."""
        if stop_seed is None:
            return BoosterClassifier(lgb.train(params, dataset.subset(np.sort(rows)), num_boost_round=rounds))
        calls = np.unique(pair_calls[rows])
        held = np.random.default_rng(stop_seed).choice(calls, size=max(1, len(calls) // 10), replace=False)
        is_stop = np.isin(pair_calls[rows], held)
        booster = lgb.train(params, dataset.subset(np.sort(rows[~is_stop])), num_boost_round=rounds,
                            valid_sets=[dataset.subset(np.sort(rows[is_stop]))],
                            callbacks=[lgb.early_stopping(20, verbose=False)])
        return BoosterClassifier(lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration)))

    same_train: list[np.ndarray | None] = [None] * len(train_counts)
    pair_oof = np.zeros(len(y_pair), dtype=np.float64)
    best_rounds: list[int] = []
    for fold, (fit_rows, held_rows) in enumerate(folds, 1):
        held_calls = np.unique(groups[held_rows])
        in_fit = ~np.isin(pair_calls, held_calls)
        model = fit(np.flatnonzero(in_fit), args.pair_max_iter, args.seed + fold)
        best_rounds.append(model.n_iter_)
        pair_oof[~in_fit] = positive_probability(model, x_pair[~in_fit])
        for call, matrix in same_speaker_matrices(model, emb_train, train_counts, held_calls, aux_train,
                                                  channel_scale, embed_split).items():
            same_train[call] = matrix
        held_accuracy = accuracy_score(y_pair[~in_fit], pair_oof[~in_fit] >= 0.5)
        print(f"[판별기 fold {fold}/{len(folds)}] 반복 {model.n_iter_}, held-out 쌍 Accuracy {held_accuracy:.4f} "
              f"({elapsed(start)}), {memory_status()}")
    # 판별기가 코사인보다 나은지 바로 보이도록 같은 쌍에서 AUC를 비교한다.
    short = x_pair[:, PAIR_COLUMNS.index("min_log_duration")] < np.log1p(1.0)
    for name, rows in (("전체", np.ones(len(y_pair), dtype=bool)), ("짧은 쪽 1초 미만", short)):
        if rows.sum() == 0 or len(np.unique(y_pair[rows])) < 2:
            continue
        cosine_auc = roc_auc_score(y_pair[rows], x_pair[rows, PAIR_COLUMNS.index("cos_first")])
        print(f"같은 화자 판별 out-of-fold ({name}, 쌍 {rows.sum():,}): 판별기 AUC "
              f"{roc_auc_score(y_pair[rows], pair_oof[rows]):.4f} / ECAPA 코사인 AUC {cosine_auc:.4f} / "
              f"판별기 Accuracy {accuracy_score(y_pair[rows], pair_oof[rows] >= 0.5):.4f}")
    final_rounds = max(1, int(round(float(np.mean(best_rounds)))))
    pair_model = fit(np.arange(len(y_pair)), final_rounds, None)
    del dataset, x_pair
    found = same_speaker_matrices(pair_model, emb_val, val_counts, np.arange(len(val_counts)), aux_val,
                                  channel_scale, embed_split)
    same_val = [found[call] for call in range(len(val_counts))]
    print(f"같은 화자 판별기 소요: {elapsed(start)} (fold 반복 {best_rounds} → 전체 {final_rounds}), {memory_status()}")
    return pair_model, same_train, same_val


def fold_cache_path(stage1_cache: Path, index: int) -> Path:
    return stage1_cache.with_name(f"{stage1_cache.stem}.fold{index}.npz")


def build_stage1_inputs(args, emb_train, y_train, train_counts, mfcc_train, emb_val, val_counts, mfcc_val, folds):
    """1단계 입력 행렬을 만든다. 목소리 사전을 쓰면 그 유사도 열도 채운다."""
    extra = len(BANK_COLUMNS) if args.bank_clusters > 0 else 0
    x_train = build_stage1(emb_train, train_counts, mfcc_train, extra)
    x_val = build_stage1(emb_val, val_counts, mfcc_val, extra)
    bank = None
    if extra:
        # Train 발화의 사전 특징은 그 발화가 속한 fold를 빼고 만든 사전으로 잰다. 자기 통화의
        # 목소리가 사전에 들어 있으면 유사도가 Validation보다 높게 나와 모델이 과신한다.
        start = time.perf_counter()
        for fit_rows, held_rows in folds:
            fold_bank = fit_bank(emb_train[fit_rows], y_train[fit_rows], args.bank_clusters, args.seed)
            x_train[held_rows, -extra:] = bank_features(emb_train[held_rows], fold_bank)
        bank = fit_bank(emb_train, y_train, args.bank_clusters, args.seed)
        x_val[:, -extra:] = bank_features(emb_val, bank)
        print(f"목소리 사전: 대원/신고자 각 {args.bank_clusters}개 중심 ({elapsed(start)})")
    print(f"1단계 특징: {x_train.shape[1]}차원 (MFCC {'포함' if args.mfcc_cache_dir else '제외'}, "
          f"목소리 사전 {'포함' if bank is not None else '제외'})")
    return x_train, x_val, bank


def memory_status() -> str:
    """Colab에서 메모리 부족으로 죽을 때 어디까지 갔는지 보이도록 사용량을 찍는다."""
    try:
        import psutil
    except ImportError:
        return "psutil 없음"
    memory = psutil.virtual_memory()
    gigabyte = 1024 ** 3
    return f"사용 {memory.used / gigabyte:.1f}GB / 전체 {memory.total / gigabyte:.1f}GB (여유 {memory.available / gigabyte:.1f}GB)"


def stage1_params(args) -> dict:
    """EXP-003/004의 HistGradientBoosting(잎 63개, 학습률 0.1, 잎 최소 20행)과 같은 크기로 맞춘다."""
    return {
        "objective": "binary", "learning_rate": args.learning_rate, "num_leaves": 63,
        "min_data_in_leaf": 20, "lambda_l2": 0.0, "max_bin": 255, "seed": args.seed,
        "deterministic": True, "force_row_wise": True, "verbose": -1,
        # 나무마다 특징·행 일부만 보면 반복당 시간이 줄어든다(EXP-006부터 1단계 3.5시간 단축용).
        "feature_fraction": args.feature_fraction, "feature_fraction_seed": args.seed,
        "bagging_fraction": args.bagging_fraction, "bagging_freq": 1 if args.bagging_fraction < 1.0 else 0,
        "bagging_seed": args.seed,
    }


def make_dataset(x_train: np.ndarray, y_train: np.ndarray):
    """Train 전체를 한 번만 구간화한다. 각 fold는 이 결과를 행 번호로 나눠 쓴다."""
    import lightgbm as lgb

    return lgb.Dataset(x_train, label=y_train, params={"max_bin": 255, "verbose": -1}, free_raw_data=True).construct()


def train_booster(args, dataset, train_rows, stop_rows, rounds: int) -> BoosterClassifier:
    """train_rows로 학습한다. stop_rows가 있으면 그 성능이 20회 연속 나아지지 않을 때 멈춘다."""
    import lightgbm as lgb

    train_set = dataset if train_rows is None else dataset.subset(np.sort(train_rows))
    if stop_rows is None:
        return BoosterClassifier(lgb.train(stage1_params(args), train_set, num_boost_round=rounds))
    stop_set = dataset.subset(np.sort(stop_rows))
    booster = lgb.train(stage1_params(args), train_set, num_boost_round=rounds, valid_sets=[stop_set],
                        callbacks=[lgb.early_stopping(20, verbose=False)])
    # 조기 종료 지점 이후의 나무는 버린다. 저장과 예측 모두 가장 좋았던 반복까지만 쓴다.
    best = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    return BoosterClassifier(best)


def stopping_split(fit_rows: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """fit 행의 통화 10%를 조기 종료 판단용으로 떼어 낸다. 같은 통화의 발화는 한쪽에만 둔다."""
    calls = np.unique(groups[fit_rows])
    rng = np.random.default_rng(seed)
    held_calls = rng.choice(calls, size=max(1, len(calls) // 10), replace=False)
    is_stop = np.isin(groups[fit_rows], held_calls)
    return fit_rows[~is_stop], fit_rows[is_stop]


def fold_probabilities(args, x_train, y_train, folds, groups, dataset, signature: dict) -> tuple[np.ndarray, list[int]]:
    """통화 단위 fold로 Train out-of-fold 확률을 만든다.

    2단계는 "다른 발화가 받은 1단계 확률"을 입력으로 쓴다. Train에서 그 확률을 자기
    자신으로 학습한 모델로 내면 지나치게 정확해 2단계가 과신한다. 그래서 각 통화의
    확률은 그 통화를 보지 않은 모델에서 얻는다.

    `--stage1_cache`를 주면 fold가 끝날 때마다 그 확률을 저장한다. Colab이 중간에
    실행을 멈춰도 다음 실행은 끝난 fold를 건너뛴다.
    """
    start = time.perf_counter()
    out_of_fold = np.zeros(len(y_train), dtype=np.float64)
    best_rounds: list[int] = []
    signature_text = json.dumps(signature, sort_keys=True)
    for fold, (fit_rows, held_rows) in enumerate(folds, 1):
        saved_path = fold_cache_path(args.stage1_cache, fold) if args.stage1_cache is not None else None
        if saved_path is not None and saved_path.is_file():
            with np.load(saved_path) as saved:
                if str(saved["signature"]) == signature_text and len(saved["probabilities"]) == len(held_rows):
                    out_of_fold[held_rows] = saved["probabilities"]
                    best_rounds.append(int(saved["rounds"]))
                    print(f"[fold {fold}/{args.folds}] 저장된 결과 사용: {saved_path.name}")
                    continue
        print(f"[fold {fold}/{args.folds}] 학습 시작 ({elapsed(start)}), {memory_status()}")
        train_rows, stop_rows = stopping_split(fit_rows, groups, args.seed + fold)
        model = train_booster(args, dataset, train_rows, stop_rows, args.max_iter)
        out_of_fold[held_rows] = positive_probability(model, x_train[held_rows])
        best_rounds.append(model.n_iter_)
        held_accuracy = accuracy_score(y_train[held_rows], out_of_fold[held_rows] >= 0.5)
        print(f"[fold {fold}/{args.folds}] 반복 {model.n_iter_}, held-out Accuracy {held_accuracy:.4f} ({elapsed(start)})")
        if saved_path is not None:
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(saved_path, probabilities=out_of_fold[held_rows], rounds=model.n_iter_, signature=signature_text)
    return out_of_fold, best_rounds

if __name__ == "__main__":
    main()
