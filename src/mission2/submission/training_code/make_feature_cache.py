"""1단계: 발화별 특징 캐시 만들기 (Colab GPU, Training 약 5시간 · Validation 약 40분).

python make_feature_cache.py --data_root <part 폴더들이 있는 폴더> --encoder_dir <ECAPA 파일 폴더> \
    --mfcc_cache_dir mission2_cache --embed_cache_dir mission2_embed_cache

서울 Training(TS/TL_서울_구급)과 Validation(VS/VL_서울_구급)의 통화마다 발화를 startAt~endAt으로 자르고
- MFCC 통계 261차원 (features.mfcc_features) → `{mfcc_cache_dir}/{train|val}_00000.npz` ... (float32)
- ECAPA 임베딩 192차원 (features.EcapaEncoder) → `{embed_cache_dir}/embed_{train|val}_00000.npz` ... (float16)
을 통화 2,000개 단위 shard로 저장한다. shard마다 x(특징), y(speaker 정답), counts(통화별 발화 수)가 들어 있다.
Drive 연결이 끊겨도 다시 실행하면 마지막 shard 다음부터 이어 만든다.

annotation에서는 startAt, endAt(자르기)과 speaker(학습 정답 y로만 저장)만 읽는다. text와 발화 순서는 쓰지 않는다.
Validation 캐시는 정확도 확인용이며 학습에는 쓰지 않는다.

원래 실행: 같은 처리를 MFCC는 EXP-001 스크립트, ECAPA는 EXP-003 스크립트의 캐시 단계로 따로 만들었다.
이 파일은 그 두 캐시 단계만 남겨 한 번에 만들도록 정리한 것이다(같은 features.py 함수, 같은 shard 형식).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 제출 폴더의 features.py
from features import (  # noqa: E402
    SAMPLE_RATE, EcapaEncoder, build_annotation_index, detect_timestamp_unit, find_audio_files, get_utterances,
    load_annotation, load_waveform, mfcc_features, slice_utterance,
)

FOLDERS = {"train": ("Training", "TS_서울_구급", "TL_서울_구급"), "val": ("Validation", "VS_서울_구급", "VL_서울_구급")}
SPEAKER_TO_INDEX = {"0": 0, "1": 1}


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """서울 구급 WAV와 같은 이름의 JSON 짝. 분할 다운로드로 WAV와 JSON이 다른 part에 있을 수 있어 전체를 색인한다."""
    name, audio_folder, label_folder = FOLDERS[split]
    parts = sorted(path for path in data_root.iterdir() if path.is_dir())
    index: dict[str, Path] = {}
    for part in parts:
        label_dir = part / "대학부 데이터" / name / "2.라벨링데이터" / label_folder
        if label_dir.is_dir():
            for stem, path in build_annotation_index(label_dir).items():
                index.setdefault(stem, path)
    pairs = []
    for part in parts:
        audio_dir = part / "대학부 데이터" / name / "1.원천데이터" / audio_folder
        if audio_dir.is_dir():
            for audio_path in find_audio_files(audio_dir):
                if audio_path.stem not in index:
                    raise FileNotFoundError(f"JSON을 찾지 못했습니다: {audio_path}")
                pairs.append((audio_path, index[audio_path.stem]))
    print(f"[{split}] 통화 {len(pairs):,}개")
    return pairs


def prefetched(pairs: list[tuple[Path, Path]], workers: int = 8, depth: int = 32):
    """다음 통화의 WAV를 미리 읽어 둔다. Drive에서는 읽기 대기가 특징 계산보다 길어 이것으로 시간을 겹친다."""
    def read(pair):
        try:
            return load_waveform(pair[0]), load_annotation(pair[1])
        except (OSError, RuntimeError, ValueError) as error:
            return error

    with ThreadPoolExecutor(workers) as pool:
        queue: deque = deque()
        for pair in pairs:
            queue.append((pair, pool.submit(read, pair)))
            if len(queue) >= depth:  # 원래 순서대로 하나씩 꺼낸다
                done_pair, future = queue.popleft()
                yield done_pair, future.result()
        while queue:
            done_pair, future = queue.popleft()
            yield done_pair, future.result()


def call_features(waveform: np.ndarray, annotation: dict, encoder: EcapaEncoder, batch_size: int):
    """통화 하나 → (MFCC (n, 261), ECAPA (n, 192), 정답 (n,)). 잘못된 구간이 나오면 그 앞까지만 쓴다."""
    unit = detect_timestamp_unit(annotation, len(waveform) / SAMPLE_RATE)
    segments, mfcc, labels = [], [], []
    try:
        for utterance in get_utterances(annotation):
            speaker = str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()  # 학습 정답으로만 쓴다
            if speaker not in SPEAKER_TO_INDEX:
                continue
            segment = slice_utterance(waveform, utterance, unit)
            mfcc.append(mfcc_features(segment))
            segments.append(segment)
            labels.append(SPEAKER_TO_INDEX[speaker])
    except ValueError as error:
        print(f"[warn] {error}")
    embeddings = encoder.embed(segments, batch_size) if segments else np.zeros((0, 192), dtype=np.float32)
    return np.asarray(mfcc, dtype=np.float32).reshape(-1, 261), embeddings, np.asarray(labels, dtype=np.int64)


def build_split(split: str, pairs, encoder, mfcc_dir: Path, embed_dir: Path, shard_calls: int, batch_size: int) -> None:
    signature = {"split": split, "pairs": len(pairs), "n_mfcc": 20, "timestamp_unit": "auto"}
    for folder, prefix in ((mfcc_dir, split), (embed_dir, f"embed_{split}")):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{prefix}_meta.json").write_text(json.dumps(signature), encoding="utf-8")
    start = time.time()
    for shard, begin in enumerate(range(0, len(pairs), shard_calls)):
        mfcc_path, embed_path = mfcc_dir / f"{split}_{shard:05d}.npz", embed_dir / f"embed_{split}_{shard:05d}.npz"
        if mfcc_path.is_file() and embed_path.is_file():
            continue  # 이미 만든 shard는 건너뛴다(이어 만들기)
        blocks, skipped = [], 0
        chosen = pairs[begin:begin + shard_calls]
        for (audio_path, _), payload in prefetched(chosen):
            if isinstance(payload, Exception):
                print(f"[warn] {audio_path.name}: {payload}")
                blocks.append((np.zeros((0, 261), np.float32), np.zeros((0, 192), np.float32), np.zeros(0, np.int64)))
                skipped += 1
                continue
            blocks.append(call_features(*payload, encoder, batch_size))
        counts = np.array([len(block[2]) for block in blocks], dtype=np.int64)
        y = np.concatenate([block[2] for block in blocks])
        np.savez(mfcc_path, x=np.vstack([block[0] for block in blocks]), y=y, counts=counts, calls=len(blocks), skipped=skipped)
        np.savez(embed_path, x=np.vstack([block[1] for block in blocks]).astype(np.float16), y=y, counts=counts,
                 calls=len(blocks), skipped=skipped)
        print(f"[{split}] shard {shard}: 통화 {begin + len(blocks):,}/{len(pairs):,}, 발화 {len(y):,} ({(time.time() - start) / 60:.0f}분)")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Mission 2 발화별 MFCC·ECAPA 특징 캐시")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--encoder_dir", type=Path, required=True, help="speechbrain/spkrec-ecapa-voxceleb 파일 폴더")
    parser.add_argument("--mfcc_cache_dir", type=Path, required=True)
    parser.add_argument("--embed_cache_dir", type=Path, required=True)
    parser.add_argument("--shard_calls", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    encoder = EcapaEncoder(args.encoder_dir, "cuda" if torch.cuda.is_available() else "cpu")
    for split in ("train", "val"):
        build_split(split, collect_pairs(args.data_root, split), encoder, args.mfcc_cache_dir, args.embed_cache_dir,
                    args.shard_calls, args.batch_size)


if __name__ == "__main__":
    main()
