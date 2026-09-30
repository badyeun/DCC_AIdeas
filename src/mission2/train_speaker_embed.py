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
    ECAPA_SOURCE, EMBED_DIM, MAX_SAMPLES, MIN_SAMPLES, STAGE2_COLUMNS, build_stage1, build_stage2,
    bundle_encoder, embed_segments, load_encoder, positive_probability,
)
from train_mfcc_rf import build_features, collect_pairs, iter_prefetched, load_shards, shard_path


ENCODER_DIRNAME = "ecapa_voxceleb"
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
    parser.add_argument("--stage2", choices=("auto", "on", "off"), default="auto",
                        help="2단계 사용 여부. auto는 Validation 정확도가 더 높은 쪽을 저장한다.")
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
        "encoder": ECAPA_SOURCE, "timestamp_unit": timestamp_unit, "pairs": len(pairs),
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
        x = np.vstack(pending_vectors).astype(np.float16) if pending_vectors else np.zeros((0, EMBED_DIM), dtype=np.float16)
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
            pending_vectors.append(embed_segments(encoder, segments, batch_size))
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


def make_classifier(max_iter: int, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
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
    if not args.data_root.is_dir() or args.folds < 2 or args.max_iter < 1 or args.batch_size < 1:
        raise ValueError("data_root, folds(2 이상), max_iter, batch_size를 확인하세요.")
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
    encoder = load_encoder(args.encoder_dir, device)
    common_args = (args.embed_cache_dir, args.shard_calls, args.prefetch, args.prefetch_depth, args.batch_size)
    emb_train, y_train, train_counts = build_embeddings(train_pairs, encoder, args.timestamp_unit, "Train 임베딩", "embed_train", *common_args)
    emb_val, y_val, val_counts = build_embeddings(val_pairs, encoder, args.timestamp_unit, "Validation 임베딩", "embed_val", *common_args)
    print(f"임베딩 단계 소요: {elapsed(start)}")

    mfcc_train = mfcc_val = None
    if args.mfcc_cache_dir is not None:
        start = time.perf_counter()
        mfcc_train = load_mfcc(args, train_pairs, "train", "Train MFCC", y_train, train_counts)
        mfcc_val = load_mfcc(args, val_pairs, "val", "Validation MFCC", y_val, val_counts)
        print(f"MFCC 캐시 로딩 소요: {elapsed(start)}")

    x_train = build_stage1(emb_train, train_counts, mfcc_train)
    x_val = build_stage1(emb_val, val_counts, mfcc_val)
    del mfcc_train, mfcc_val
    print(f"1단계 특징: {x_train.shape[1]}차원 (MFCC {'포함' if args.mfcc_cache_dir else '제외'})")
    print(f"Train 발화: {len(y_train):,}, Validation 발화: {len(y_val):,}")
    print(f"다수 클래스 기준선: {float(np.bincount(y_val, minlength=2).max() / len(y_val)):.4f}")

    # 2단계는 "다른 발화가 받은 1단계 확률"을 입력으로 쓴다. Train에서 그 확률을 자기
    # 자신으로 학습한 모델로 내면 지나치게 정확해 2단계가 과신한다. 그래서 통화 단위
    # fold로 나눠, 각 통화의 확률은 그 통화를 보지 않은 모델에서 얻는다.
    start = time.perf_counter()
    groups = np.repeat(np.arange(len(train_counts)), train_counts)
    out_of_fold = np.zeros(len(y_train), dtype=np.float64)
    for fold, (fit_rows, held_rows) in enumerate(GroupKFold(n_splits=args.folds).split(x_train, y_train, groups), 1):
        model = make_classifier(args.max_iter, args.seed).fit(x_train[fit_rows], y_train[fit_rows])
        out_of_fold[held_rows] = positive_probability(model, x_train[held_rows])
        held_accuracy = accuracy_score(y_train[held_rows], out_of_fold[held_rows] >= 0.5)
        print(f"[fold {fold}/{args.folds}] 반복 {model.n_iter_}, held-out Accuracy {held_accuracy:.4f} ({elapsed(start)})")
    stage1 = make_classifier(args.max_iter, args.seed).fit(x_train, y_train)
    val_stage1 = positive_probability(stage1, x_val)
    print(f"1단계 학습 소요: {elapsed(start)} (전체 모델 반복 {stage1.n_iter_})")
    stage1_accuracy = report("1단계, 발화별", y_val, val_stage1)

    start = time.perf_counter()
    stage2 = make_classifier(200, args.seed).fit(build_stage2(emb_train, out_of_fold, train_counts), y_train)
    val_stage2 = positive_probability(stage2, build_stage2(emb_val, val_stage1, val_counts))
    print(f"2단계 학습 소요: {elapsed(start)} (입력: {', '.join(STAGE2_COLUMNS)})")
    stage2_accuracy = report("2단계, 통화 내 비교", y_val, val_stage2)

    use_stage2 = args.stage2 == "on" or (args.stage2 == "auto" and stage2_accuracy > stage1_accuracy)
    selected = stage2_accuracy if use_stage2 else stage1_accuracy
    print(f"선택한 설정: stage2={use_stage2} (--stage2 {args.stage2}) → Accuracy {selected:.4f}")

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 추론이 인터넷 없이 돌도록 ECAPA 파일을 체크포인트 옆에 함께 둔다.
    bundle_encoder(args.encoder_dir, args.ckpt_path.parent / ENCODER_DIRNAME)
    joblib.dump({
        "model_type": "mission2_speaker_embedding",
        "stage1": stage1,
        "stage2": stage2,
        "use_stage2": bool(use_stage2),
        "use_mfcc": args.mfcc_cache_dir is not None,
        "n_mfcc": args.n_mfcc,
        "timestamp_unit": args.timestamp_unit,
        "encoder_dir": ENCODER_DIRNAME,
        "encoder_source": ECAPA_SOURCE,
        "batch_size": args.batch_size,
        "fallback_speaker": int(np.bincount(y_train, minlength=2).argmax()),
        "seed": args.seed,
        "validation_accuracy": float(selected),
        "validation_accuracy_stage1": stage1_accuracy,
        "validation_accuracy_stage2": stage2_accuracy,
    }, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path} (+ {args.ckpt_path.parent / ENCODER_DIRNAME})")
    print(f"전체 소요: {elapsed(total_start)}")


if __name__ == "__main__":
    main()
