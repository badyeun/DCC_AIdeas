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
    SAMPLE_RATE, SPEAKER_TO_INDEX, build_annotation_index, decode_annotation, decode_waveform,
    detect_timestamp_unit, find_audio_files, get_utterances,
    slice_utterance, waveform_to_feature,
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


def load_shards(
    cache_dir: Path, split: str, signature: dict[str, object]
) -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    """설정이 같을 때만 기존 shard를 이어받고, 다르면 지우고 처음부터 시작한다."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{split}_meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text(encoding="utf-8")) != signature:
        print(f"[{split}] 캐시 설정이 달라 기존 shard를 삭제합니다.")
        for path in cache_dir.glob(f"{split}_*.npz"):
            path.unlink()
    meta_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
    blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    done = skipped = 0
    for index in count():
        path = shard_path(cache_dir, split, index)
        if not path.is_file():
            break
        with np.load(path) as data:
            blocks.append(data["x"])
            label_blocks.append(data["y"])
            done += int(data["calls"])
            skipped += int(data["skipped"])
    return blocks, label_blocks, done, skipped


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
) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    done = skipped = 0
    if cache_dir is not None:
        signature = {"n_mfcc": n_mfcc, "timestamp_unit": timestamp_unit, "pairs": len(pairs)}
        blocks, label_blocks, done, skipped = load_shards(cache_dir, split, signature)
        if done:
            print(f"[{desc}] 캐시에서 통화 {done:,}/{len(pairs):,}개 복원")
    pending_features: list[np.ndarray] = []
    pending_labels: list[int] = []
    pending_calls = pending_skipped = 0

    def flush() -> None:
        nonlocal pending_features, pending_labels, pending_calls, pending_skipped
        if cache_dir is None or pending_calls == 0:
            return
        width = pending_features[0].size if pending_features else 0
        x = np.vstack(pending_features) if pending_features else np.zeros((0, width), dtype=np.float32)
        y = np.asarray(pending_labels, dtype=np.int64)
        # 쓰는 도중에 죽어도 깨진 shard가 남지 않도록 임시 파일에 쓴 뒤 교체한다.
        temporary = cache_dir / f".{split}_partial.npz"
        np.savez(temporary, x=x, y=y, calls=pending_calls, skipped=pending_skipped)
        temporary.replace(shard_path(cache_dir, split, len(blocks)))
        blocks.append(x)
        label_blocks.append(y)
        pending_features, pending_labels = [], []
        pending_calls = pending_skipped = 0

    remaining = pairs[done:]
    stream = iter_prefetched(remaining, prefetch, prefetch_depth)
    for audio_path, payload in tqdm(stream, desc=desc, initial=done, total=len(pairs)):
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
        except (OSError, RuntimeError, ValueError) as error:
            skipped += 1
            pending_skipped += 1
            print(f"[warn] {audio_path.name}: {error}")
        pending_calls += 1
        if pending_calls >= shard_calls:
            flush()

    flush()
    if cache_dir is None:
        blocks = [np.vstack(pending_features)] if pending_features else []
        label_blocks = [np.asarray(pending_labels, dtype=np.int64)]
    filled = [block for block in blocks if block.size]
    if not filled:
        raise ValueError(f"{desc}에서 유효한 발화를 만들지 못했습니다.")
    features = np.vstack(filled)
    labels = np.concatenate([block for block in label_blocks if block.size])
    print(f"[{desc}] 발화: {len(features):,}, 건너뜀: {skipped:,}")
    return features, labels


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or args.n_mfcc < 1 or args.n_estimators < 1:
        raise ValueError("data_root와 양수 n_mfcc/n_estimators를 확인하세요.")
    if args.shard_calls < 1 or args.prefetch_depth < 1:
        raise ValueError("shard_calls와 prefetch_depth는 1 이상이어야 합니다.")
    train_pairs = collect_pairs(args.data_root, "Training")
    val_pairs = collect_pairs(args.data_root, "Validation")
    x_train, y_train = build_features(train_pairs, args.n_mfcc, args.timestamp_unit, "Train 특징 추출", "train", args.cache_dir, args.shard_calls, args.prefetch, args.prefetch_depth)
    x_val, y_val = build_features(val_pairs, args.n_mfcc, args.timestamp_unit, "Validation 특징 추출", "val", args.cache_dir, args.shard_calls, args.prefetch, args.prefetch_depth)
    classifier = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, prediction)
    majority = float(np.bincount(y_val, minlength=2).max() / len(y_val))
    print(f"Train 발화: {len(y_train):,}, Validation 발화: {len(y_val):,}")
    print(f"다수 클래스 기준선: {majority:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(classification_report(y_val, prediction, labels=[0, 1], target_names=["119대원(0)", "신고자(1)"], digits=4, zero_division=0))
    fallback_speaker = int(np.bincount(y_train, minlength=2).argmax())
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_type": "mission2_mfcc_random_forest", "classifier": classifier, "n_mfcc": args.n_mfcc, "timestamp_unit": args.timestamp_unit, "fallback_speaker": fallback_speaker, "validation_accuracy": float(accuracy)}, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
