"""EXP-010 시범: 녹음 환경 프로파일(common.channel_profile)이 1단계에 도움이 되는지 작게 잰다.

본 실험은 Train 전체 WAV를 다시 읽고(1.5시간 이상) 1단계를 다시 학습(3.5시간)해야 한다.
그 전에 Train 통화 일부만으로 "기존 1단계 특징" 대 "기존 + 프로파일"의 통화 단위
out-of-fold 정확도를 비교해, 효과가 있을 때만 본 실험을 한다.

기존 캐시(ECAPA 임베딩, MFCC)는 shard 파일을 직접 읽기만 한다. load_shards는 설정이
다르면 shard를 지우므로 쓰지 않는다. Validation은 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from common import (
    SAMPLE_RATE, SPEAKER_TO_INDEX, add_call_context, channel_profile, decode_annotation, decode_waveform,
    detect_timestamp_unit, get_utterances, slice_utterance,
)
from speaker_embed import build_stage1, iter_calls
from train_mfcc_rf import collect_pairs, iter_prefetched, shard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-010 시범: 녹음 환경 프로파일 효과 측정")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--embed_cache_dir", type=Path, required=True)
    parser.add_argument("--mfcc_cache_dir", type=Path, required=True)
    parser.add_argument("--every", type=int, default=7, help="Train 통화를 몇 개마다 하나씩 쓸지")
    parser.add_argument("--max_iter", type=int, default=600)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_cache(cache_dir: Path, split: str, counts_file: Path | None = None):
    """shard를 읽기만 해서 (x, y, counts)를 돌려준다."""
    blocks, labels, counts = [], [], []
    index = 0
    while shard_path(cache_dir, split, index).is_file():
        with np.load(shard_path(cache_dir, split, index)) as data:
            blocks.append(data["x"])
            labels.append(data["y"])
            if "counts" in data.files:
                counts.append(data["counts"])
        index += 1
    x = np.vstack([block for block in blocks if block.size])
    y = np.concatenate([block for block in labels if block.size])
    if counts:
        boundary = np.concatenate(counts)
    else:  # 예전 MFCC 캐시는 통화 경계를 따로 저장해 두었다.
        boundary = np.load(counts_file)
    if int(boundary.sum()) != len(x):
        raise ValueError(f"{cache_dir} {split}: 통화 경계 합과 행 수가 다릅니다.")
    return x, y, boundary.astype(np.int64)


def oof_accuracy(x: np.ndarray, y: np.ndarray, groups: np.ndarray, args) -> np.ndarray:
    """1단계와 같은 LightGBM 설정으로 통화 단위 4-fold out-of-fold 확률을 낸다."""
    import lightgbm as lgb

    params = {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 20,
        "max_bin": 255, "seed": args.seed, "deterministic": True, "force_row_wise": True, "verbose": -1,
        "feature_fraction": 0.5, "feature_fraction_seed": args.seed,
    }
    dataset = lgb.Dataset(x, label=y, params={"max_bin": 255, "verbose": -1}, free_raw_data=False).construct()
    output = np.zeros(len(y))
    for fold, (fit_rows, held_rows) in enumerate(GroupKFold(n_splits=4).split(x, y, groups), 1):
        calls = np.unique(groups[fit_rows])
        stop_calls = np.random.default_rng(args.seed + fold).choice(calls, size=len(calls) // 10, replace=False)
        is_stop = np.isin(groups[fit_rows], stop_calls)
        booster = lgb.train(params, dataset.subset(np.sort(fit_rows[~is_stop])), num_boost_round=args.max_iter,
                            valid_sets=[dataset.subset(np.sort(fit_rows[is_stop]))],
                            callbacks=[lgb.early_stopping(20, verbose=False)])
        output[held_rows] = booster.predict(x[held_rows], num_iteration=booster.best_iteration)
        print(f"  fold {fold}: 반복 {booster.best_iteration}, held-out {accuracy_score(y[held_rows], output[held_rows] >= 0.5):.4f}")
    return output


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    pairs = collect_pairs(args.data_root, "Training")
    emb, y_emb, counts = read_cache(args.embed_cache_dir, "embed_train")
    mfcc, y_mfcc, mfcc_counts = read_cache(args.mfcc_cache_dir, "train", args.mfcc_cache_dir / "train_counts.npy")
    if len(counts) != len(pairs) or not np.array_equal(counts, mfcc_counts) or not np.array_equal(y_emb, y_mfcc):
        raise ValueError("ECAPA·MFCC 캐시와 통화 목록이 맞지 않습니다.")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    chosen = np.arange(0, len(pairs), args.every)
    rows = np.concatenate([np.arange(starts[call], starts[call] + counts[call]) for call in chosen])
    sub_counts = counts[chosen]
    print(f"시범 통화 {len(chosen):,}개, 발화 {len(rows):,}개 (Train의 1/{args.every})")

    profiles: list[np.ndarray] = []
    for position, (audio_path, payload) in enumerate(tqdm(iter_prefetched([pairs[c] for c in chosen], args.prefetch, 32),
                                                          total=len(chosen), desc="프로파일 추출")):
        if isinstance(payload, OSError):
            raise payload
        audio_bytes, label_bytes = payload
        annotation = decode_annotation(label_bytes)
        waveform = decode_waveform(audio_bytes)
        unit = detect_timestamp_unit(annotation, len(waveform) / SAMPLE_RATE)
        made = 0
        for utterance in get_utterances(annotation):
            # 캐시와 같은 규칙으로 행을 만든다. speaker는 행을 맞추고 정답을 확인하는 데만 쓴다.
            speaker = str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()
            if speaker not in SPEAKER_TO_INDEX:
                continue
            profiles.append(channel_profile(slice_utterance(waveform, utterance, unit)))
            made += 1
        if made != sub_counts[position]:
            raise ValueError(f"{audio_path.name}: 발화 수 {made} != 캐시 {sub_counts[position]}")
    profile = np.vstack(profiles)
    print(f"프로파일 {profile.shape} 추출 완료 ({time.perf_counter() - start:.0f}초)")

    y = y_emb[rows]
    groups = np.repeat(np.arange(len(chosen)), sub_counts)
    base = build_stage1(emb[rows].astype(np.float32), sub_counts, mfcc[rows].astype(np.float32))
    relative = np.zeros((len(rows), profile.shape[1] * 2), dtype=np.float32)
    for begin, end in iter_calls(sub_counts):
        relative[begin:end] = add_call_context(profile[begin:end], "center")
    duration = np.expm1(mfcc[rows, -1])
    results = {}
    for name, x in (("기존 1단계 특징", base), ("기존 + 프로파일", np.hstack((base, relative)))):
        print(f"[{name}] {x.shape[1]}차원 학습")
        results[name] = oof_accuracy(x, y, groups, args) >= 0.5
    print("\n=== 시범 결과 (Train 통화 단위 out-of-fold, 1단계) ===")
    for label, mask in (("전체", np.ones(len(y), dtype=bool)), ("1초 미만", duration < 1),
                        ("1~2초", (duration >= 1) & (duration < 2)), ("2초 이상", duration >= 2)):
        a, b = (float(np.mean(results[key][mask] == y[mask])) for key in results)
        print(f"{label:>6} (발화 {mask.sum():,}): 기존 {a:.4f} → 프로파일 추가 {b:.4f} ({(b - a) * 100:+.2f}%p)")
    print(f"전체 소요 {time.perf_counter() - start:.0f}초")


if __name__ == "__main__":
    main()
