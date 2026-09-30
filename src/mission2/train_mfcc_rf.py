"""Mission 2 MFCC + RandomForest baseline 학습/검증.

발화 단위로 학습하되 Training/Validation은 원본 통화 파일 단위로 분리된
주최 측 폴더를 사용하므로 같은 통화의 발화가 두 split에 섞이지 않습니다.
`--cache_dir`를 주면 특징을 통화 묶음(shard)마다 저장해, 중단되어도 다음
실행에서 마지막 shard 다음부터 이어받습니다.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from common import (
    CALL_CONTEXT_CHOICES, SAMPLE_RATE, SPEAKER_TO_INDEX, add_call_context, build_annotation_index,
    call_context_width, cluster_speakers, decode_annotation, decode_waveform,
    detect_timestamp_unit, find_audio_files, get_utterances, slice_utterance,
    speaker_probability, waveform_to_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2 MFCC + RandomForest 학습/검증")
    parser.add_argument("--data_root", type=Path, required=True, help="001~012 part 폴더 경로")
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission2_mfcc_rf.joblib"))
    parser.add_argument("--cache_dir", type=Path, default=None, help="특징 shard 저장 폴더. 주면 중단 후 이어받는다.")
    parser.add_argument("--shard_calls", type=int, default=2000, help="shard 하나에 담을 통화 수")
    parser.add_argument("--prefetch", type=int, default=8, help="파일을 미리 읽는 스레드 수. 0이면 사용하지 않는다.")
    parser.add_argument("--prefetch_depth", type=int, default=32, help="미리 읽어 둘 통화 수")
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--timestamp_unit", choices=("auto", "seconds", "milliseconds"), default="auto")
    parser.add_argument("--call_context", choices=CALL_CONTEXT_CHOICES, default="center",
                        help="통화 안의 상대값을 특징에 더한다. none이면 발화 조각만 본다.")
    parser.add_argument("--cluster", choices=("auto", "on", "off"), default="auto",
                        help="통화 안 두 군집 묶기. auto는 Validation 정확도가 더 높은 쪽을 저장한다.")
    parser.add_argument("--no_cluster", dest="cluster", action="store_const", const="off",
                        help="--cluster off와 같다.")
    parser.add_argument("--cluster_min_utterances", type=int, default=6,
                        help="이보다 발화가 적은 통화는 묶지 않는다.")
    parser.add_argument("--cluster_margin", type=float, default=0.10,
                        help="두 군집의 평균 확률 차이가 이보다 작으면 묶지 않는다.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    folders = ("TS_서울_구급", "TL_서울_구급") if split == "Training" else ("VS_서울_구급", "VL_서울_구급")
    parts = sorted(path for path in data_root.iterdir() if path.is_dir())
    # 분할 다운로드로 WAV와 JSON이 서로 다른 part에 나뉘어 들어갈 수 있어 전체를 한 번에 색인한다.
    index: dict[str, Path] = {}
    for part in parts:
        label_dir = part / "대학부 데이터" / split / "2.라벨링데이터" / folders[1]
        if label_dir.is_dir():
            for stem, label_path in build_annotation_index(label_dir).items():
                index.setdefault(stem, label_path)
    pairs: list[tuple[Path, Path]] = []
    for part in parts:
        audio_dir = part / "대학부 데이터" / split / "1.원천데이터" / folders[0]
        if not audio_dir.is_dir():
            continue
        for audio_path in find_audio_files(audio_dir):
            label_path = index.get(audio_path.stem)
            if label_path is None:
                raise FileNotFoundError(f"JSON을 찾지 못했습니다: {audio_path}")
            pairs.append((audio_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"{split} WAV/JSON pair를 찾지 못했습니다: {data_root}")
    print(f"[{split}] 통화 pair: {len(pairs):,}")
    return pairs


def shard_path(cache_dir: Path, split: str, index: int) -> Path:
    return cache_dir / f"{split}_{index:05d}.npz"


def counts_path(cache_dir: Path, split: str) -> Path:
    """예전 캐시에서 복원한 통화 경계를 담아 두는 파일. 다음 실행은 JSON을 다시 읽지 않는다."""
    return cache_dir / f"{split}_counts.npy"


def load_shards(
    cache_dir: Path, split: str, signature: dict[str, object]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int, int, bool]:
    """설정이 같을 때만 기존 shard를 이어받고, 다르면 지우고 처음부터 시작한다.

    통화 경계(`counts`)가 없는 예전 shard도 그대로 이어받는다. 이 경우
    `complete`가 False로 나오고, 호출한 쪽이 JSON만 다시 읽어 경계를 복원한다.
    특징은 그대로 쓰므로 WAV 재추출은 일어나지 않는다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{split}_meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) != signature:
        print(f"[{split}] 캐시 설정이 달라 기존 shard를 삭제합니다.")
        for path in cache_dir.glob(f"{split}_*.npz"):
            path.unlink()
        counts_path(cache_dir, split).unlink(missing_ok=True)
    meta_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
    blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []
    done = skipped = 0
    complete = True
    for index in count():
        path = shard_path(cache_dir, split, index)
        if not path.is_file():
            break
        with np.load(path) as data:
            blocks.append(data["x"])
            label_blocks.append(data["y"])
            if "counts" in data.files:
                count_blocks.append(data["counts"])
            else:
                complete = False
            done += int(data["calls"])
            skipped += int(data["skipped"])
    return blocks, label_blocks, count_blocks, done, skipped, complete


def count_labeled_utterances(label_path: Path) -> int:
    """JSON만 읽어 특징이 만들어졌을 발화 수를 센다. WAV는 건드리지 않는다."""
    annotation = decode_annotation(label_path.read_bytes())
    total = 0
    for utterance in get_utterances(annotation):
        speaker = str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()
        if speaker in SPEAKER_TO_INDEX:
            total += 1
    return total


def reconstruct_counts(pairs: list[tuple[Path, Path]], desc: str, workers: int) -> np.ndarray:
    """예전 캐시에 없는 통화 경계를 JSON에서 복원한다.

    통화 순서는 `collect_pairs`의 정렬로 정해져 항상 같고, 캐시의 행 순서도
    같은 순서로 쌓였다. 그래서 JSON의 발화 수만 세면 경계가 그대로 복원된다.
    """
    labels = [label_path for _, label_path in pairs]
    if workers < 1:
        values = [count_labeled_utterances(path) for path in tqdm(labels, desc=desc)]
    else:
        with ThreadPoolExecutor(workers) as pool:
            values = list(tqdm(pool.map(count_labeled_utterances, labels), total=len(labels), desc=desc))
    return np.asarray(values, dtype=np.int64)


def apply_call_context(features: np.ndarray, counts: np.ndarray, mode: str) -> np.ndarray:
    """통화 단위로 묶어 상대값 특징을 붙인다."""
    if mode == "none":
        return features
    width = call_context_width(features.shape[1], mode)
    output = np.empty((features.shape[0], width), dtype=np.float32)
    start = 0
    for value in counts:
        size = int(value)
        if size <= 0:
            continue
        output[start:start + size] = add_call_context(features[start:start + size], mode)
        start += size
    if start != len(features):
        raise ValueError(f"통화 경계 합({start:,})이 발화 수({len(features):,})와 다릅니다.")
    return output


def cluster_predictions(
    base_features: np.ndarray, probabilities: np.ndarray, counts: np.ndarray,
    min_utterances: int, margin: float, seed: int,
) -> np.ndarray:
    """통화마다 두 군집으로 묶어 발화별 예측을 다시 정한다."""
    predictions = (probabilities >= 0.5).astype(np.int64)
    start = 0
    for value in tqdm(counts, desc="통화 단위 군집 묶기"):
        size = int(value)
        if size <= 0:
            continue
        predictions[start:start + size] = cluster_speakers(
            base_features[start:start + size], probabilities[start:start + size],
            min_utterances, margin, seed,
        )
        start += size
    if start != len(predictions):
        raise ValueError(f"통화 경계 합({start:,})이 발화 수({len(predictions):,})와 다릅니다.")
    return predictions


def read_pair(audio_path: Path, label_path: Path) -> tuple[bytes, bytes] | OSError:
    """읽기 실패를 예외 대신 값으로 돌려준다. 미리 읽는 스레드에서 터지면 전체가 멈춘다."""
    try:
        return audio_path.read_bytes(), label_path.read_bytes()
    except OSError as error:
        return error


def iter_prefetched(pairs: list[tuple[Path, Path]], workers: int, depth: int):
    """WAV/JSON을 미리 읽어 두고 원래 순서대로 돌려준다.

    원격 파일시스템에서는 읽기 대기가 특징 계산보다 오래 걸린다. 다음 통화의
    파일을 미리 요청해 두면 그 대기가 계산 시간과 겹쳐 사라진다.
    """
    if workers < 1:
        for audio_path, label_path in pairs:
            yield audio_path, read_pair(audio_path, label_path)
        return
    with ThreadPoolExecutor(workers) as pool:
        queued: deque = deque()
        position = 0
        while position < len(pairs) and len(queued) < max(depth, 1):
            queued.append((pairs[position], pool.submit(read_pair, *pairs[position])))
            position += 1
        while queued:
            (audio_path, _), future = queued.popleft()
            if position < len(pairs):
                queued.append((pairs[position], pool.submit(read_pair, *pairs[position])))
                position += 1
            yield audio_path, future.result()


def build_features(
    pairs: list[tuple[Path, Path]], n_mfcc: int, timestamp_unit: str, desc: str,
    split: str | None = None, cache_dir: Path | None = None, shard_calls: int = 2000,
    prefetch: int = 8, prefetch_depth: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    count_blocks: list[np.ndarray] = []
    done = skipped = 0
    if cache_dir is not None:
        signature = {"n_mfcc": n_mfcc, "timestamp_unit": timestamp_unit, "pairs": len(pairs)}
        blocks, label_blocks, count_blocks, done, skipped, complete = load_shards(cache_dir, split, signature)
        if done:
            print(f"[{desc}] 캐시에서 통화 {done:,}/{len(pairs):,}개 복원")
        if done and not complete:
            # 통화 경계를 저장하기 전에 만든 캐시다. JSON만 다시 읽어 복원한다.
            restored = sum(int(block.shape[0]) for block in blocks)
            saved_path = counts_path(cache_dir, split)
            saved = np.load(saved_path) if saved_path.is_file() else None
            # 저장해 둔 경계는 통화 수와 발화 합이 모두 캐시와 맞을 때만 믿는다.
            if saved is not None and len(saved) == done and int(saved.sum()) == restored:
                count_blocks = [saved.astype(np.int64)]
                print(f"[{desc}] 저장된 통화 경계 사용: 발화 {restored:,}개가 캐시와 일치")
            else:
                count_blocks = [reconstruct_counts(pairs[:done], f"[{desc}] 통화 경계 복원", prefetch)]
                rebuilt = int(count_blocks[0].sum())
                if rebuilt != restored:
                    raise ValueError(
                        f"[{desc}] 통화 경계 복원 실패: JSON 발화 {rebuilt:,}개 vs 캐시 행 {restored:,}개. "
                        f"캐시를 지우고 다시 추출하세요: {cache_dir}"
                    )
                np.save(saved_path, count_blocks[0])
                print(f"[{desc}] 통화 경계 복원 완료: 발화 {rebuilt:,}개가 캐시와 일치")
    pending_features: list[np.ndarray] = []
    pending_labels: list[int] = []
    pending_counts: list[int] = []
    pending_calls = pending_skipped = 0

    def flush() -> None:
        nonlocal pending_features, pending_labels, pending_counts, pending_calls, pending_skipped
        if cache_dir is None or pending_calls == 0:
            return
        width = pending_features[0].size if pending_features else 0
        x = np.vstack(pending_features) if pending_features else np.zeros((0, width), dtype=np.float32)
        y = np.asarray(pending_labels, dtype=np.int64)
        counts = np.asarray(pending_counts, dtype=np.int64)
        # 쓰는 도중에 죽어도 깨진 shard가 남지 않도록 임시 파일에 쓴 뒤 교체한다.
        temporary = cache_dir / f".{split}_partial.npz"
        np.savez(temporary, x=x, y=y, counts=counts, calls=pending_calls, skipped=pending_skipped)
        temporary.replace(shard_path(cache_dir, split, len(blocks)))
        blocks.append(x)
        label_blocks.append(y)
        count_blocks.append(counts)
        pending_features, pending_labels, pending_counts = [], [], []
        pending_calls = pending_skipped = 0

    remaining = pairs[done:]
    stream = iter_prefetched(remaining, prefetch, prefetch_depth)
    for audio_path, payload in tqdm(stream, desc=desc, initial=done, total=len(pairs)):
        added = 0
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
                pending_features.append(waveform_to_feature(slice_utterance(waveform, utterance, unit), n_mfcc))
                pending_labels.append(SPEAKER_TO_INDEX[speaker])
                added += 1
        except (OSError, RuntimeError, ValueError) as error:
            skipped += 1
            pending_skipped += 1
            print(f"[warn] {audio_path.name}: {error}")
        # 통화 중간에 실패해도 실제로 쌓인 행 수를 그대로 적어 경계가 어긋나지 않게 한다.
        pending_counts.append(added)
        pending_calls += 1
        if pending_calls >= shard_calls:
            flush()

    flush()
    if cache_dir is None:
        blocks = [np.vstack(pending_features)] if pending_features else []
        label_blocks = [np.asarray(pending_labels, dtype=np.int64)]
        count_blocks = [np.asarray(pending_counts, dtype=np.int64)]
    filled = [block for block in blocks if block.size]
    if not filled:
        raise ValueError(f"{desc}에서 유효한 발화를 만들지 못했습니다.")
    features = np.vstack(filled)
    labels = np.concatenate([block for block in label_blocks if block.size])
    counts = np.concatenate([block for block in count_blocks if block.size]) if count_blocks else np.zeros(0, dtype=np.int64)
    if int(counts.sum()) != len(features):
        raise ValueError(
            f"[{desc}] 통화 경계 합({int(counts.sum()):,})이 발화 수({len(features):,})와 다릅니다."
        )
    print(f"[{desc}] 발화: {len(features):,}, 통화: {len(counts):,}, 건너뜀: {skipped:,}")
    return features, labels, counts


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or args.n_mfcc < 1 or args.n_estimators < 1:
        raise ValueError("data_root와 양수 n_mfcc/n_estimators를 확인하세요.")
    if args.shard_calls < 1 or args.prefetch_depth < 1:
        raise ValueError("shard_calls와 prefetch_depth는 1 이상이어야 합니다.")
    if args.cluster_min_utterances < 2 or not 0.0 <= args.cluster_margin <= 1.0:
        raise ValueError("cluster_min_utterances는 2 이상, cluster_margin은 0~1이어야 합니다.")
    train_pairs = collect_pairs(args.data_root, "Training")
    val_pairs = collect_pairs(args.data_root, "Validation")
    x_train, y_train, train_counts = build_features(train_pairs, args.n_mfcc, args.timestamp_unit, "Train 특징 추출", "train", args.cache_dir, args.shard_calls, args.prefetch, args.prefetch_depth)
    x_val, y_val, val_counts = build_features(val_pairs, args.n_mfcc, args.timestamp_unit, "Validation 특징 추출", "val", args.cache_dir, args.shard_calls, args.prefetch, args.prefetch_depth)
    print(f"통화 맥락: {args.call_context} (특징 {x_train.shape[1]} → {call_context_width(x_train.shape[1], args.call_context)}차원)")
    # 군집 묶기는 원본 특징의 프레임 평균만 쓰므로 확장 전 배열을 따로 남겨 둔다.
    x_val_base = x_val
    x_train = apply_call_context(x_train, train_counts, args.call_context)
    x_val = apply_call_context(x_val, val_counts, args.call_context)
    classifier = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    classifier.fit(x_train, y_train)
    probabilities = speaker_probability(classifier, x_val)
    prediction = (probabilities >= 0.5).astype(np.int64)
    accuracy = accuracy_score(y_val, prediction)
    majority = float(np.bincount(y_val, minlength=2).max() / len(y_val))
    print(f"Train 발화: {len(y_train):,}, Validation 발화: {len(y_val):,}")
    print(f"다수 클래스 기준선: {majority:.4f}")
    print(f"Validation Accuracy (발화별): {accuracy:.4f}")
    print(classification_report(y_val, prediction, labels=[0, 1], target_names=["119대원(0)", "신고자(1)"], digits=4, zero_division=0))
    # 군집 묶기는 끄고 켠 결과를 모두 보여 준다. 무엇이 기여했는지 구분해서 기록하기 위해서다.
    clustered = cluster_predictions(x_val_base, probabilities, val_counts, args.cluster_min_utterances, args.cluster_margin, args.seed)
    cluster_accuracy = accuracy_score(y_val, clustered)
    print(f"Validation Accuracy (통화 군집): {cluster_accuracy:.4f}")
    print(classification_report(y_val, clustered, labels=[0, 1], target_names=["119대원(0)", "신고자(1)"], digits=4, zero_division=0))
    # EXP-002에서 군집 묶기를 무조건 켰다가 0.8537이 0.6676으로 떨어진 채 저장됐다.
    # auto는 두 결과 중 나은 쪽만 체크포인트에 남겨 추론이 같은 사고를 겪지 않게 한다.
    use_cluster = args.cluster == "on" or (args.cluster == "auto" and cluster_accuracy > accuracy)
    selected = cluster_accuracy if use_cluster else accuracy
    print(f"선택한 설정: call_context={args.call_context}, cluster={use_cluster} (--cluster {args.cluster}) → Accuracy {selected:.4f}")
    fallback_speaker = int(np.bincount(y_train, minlength=2).argmax())
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_type": "mission2_mfcc_random_forest",
        "classifier": classifier,
        "n_mfcc": args.n_mfcc,
        "timestamp_unit": args.timestamp_unit,
        "fallback_speaker": fallback_speaker,
        "call_context": args.call_context,
        "cluster": bool(use_cluster),
        "cluster_min_utterances": args.cluster_min_utterances,
        "cluster_margin": args.cluster_margin,
        "seed": args.seed,
        "validation_accuracy": float(selected),
        "validation_accuracy_utterance": float(accuracy),
        "validation_accuracy_cluster": float(cluster_accuracy),
    }, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
