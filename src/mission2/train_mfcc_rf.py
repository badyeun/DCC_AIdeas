"""Mission 2 MFCC + RandomForest baseline 학습/검증.

발화 단위로 학습하되 Training/Validation은 원본 통화 파일 단위로 분리된
주최 측 폴더를 사용하므로 같은 통화의 발화가 두 split에 섞이지 않습니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from common import (
    SPEAKER_TO_INDEX, detect_timestamp_unit, find_audio_files, get_utterances,
    load_annotation, load_waveform, resolve_annotation_path, build_annotation_index,
    slice_utterance, waveform_to_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2 MFCC + RandomForest 학습/검증")
    parser.add_argument("--data_root", type=Path, required=True, help="001~012 part 폴더 경로")
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission2_mfcc_rf.joblib"))
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--timestamp_unit", choices=("auto", "seconds", "milliseconds"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    folders = ("TS_서울_구급", "TL_서울_구급") if split == "Training" else ("VS_서울_구급", "VL_서울_구급")
    pairs: list[tuple[Path, Path]] = []
    for part in sorted(path for path in data_root.iterdir() if path.is_dir()):
        base = part / "대학부 데이터" / split
        audio_dir = base / "1.원천데이터" / folders[0]
        label_dir = base / "2.라벨링데이터" / folders[1]
        if not audio_dir.is_dir() or not label_dir.is_dir():
            continue
        index = build_annotation_index(label_dir)
        for audio_path in find_audio_files(audio_dir):
            label_path = resolve_annotation_path(audio_path, audio_dir, label_dir, index)
            if label_path is None:
                raise FileNotFoundError(f"JSON을 찾지 못했습니다: {audio_path}")
            pairs.append((audio_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"{split} WAV/JSON pair를 찾지 못했습니다: {data_root}")
    print(f"[{split}] 통화 pair: {len(pairs):,}")
    return pairs


def build_features(pairs: list[tuple[Path, Path]], n_mfcc: int, timestamp_unit: str, desc: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0
    for audio_path, label_path in tqdm(pairs, desc=desc):
        try:
            annotation = load_annotation(label_path)
            waveform = load_waveform(audio_path)
            unit = detect_timestamp_unit(annotation, len(waveform) / 16_000) if timestamp_unit == "auto" else timestamp_unit
            for utterance in get_utterances(annotation):
                # speaker는 학습 정답으로만 읽는다. text 및 순서는 전혀 사용하지 않는다.
                speaker = str(utterance.get("speaker", utterance.get("Speaker", ""))).strip()
                if speaker not in SPEAKER_TO_INDEX:
                    skipped += 1
                    continue
                features.append(waveform_to_feature(slice_utterance(waveform, utterance, unit), n_mfcc))
                labels.append(SPEAKER_TO_INDEX[speaker])
        except (OSError, ValueError) as error:
            skipped += 1
            print(f"[warn] {audio_path.name}: {error}")
    if not features:
        raise ValueError(f"{desc}에서 유효한 발화를 만들지 못했습니다.")
    print(f"[{desc}] 발화: {len(features):,}, 건너뜀: {skipped:,}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or args.n_mfcc < 1 or args.n_estimators < 1:
        raise ValueError("data_root와 양수 n_mfcc/n_estimators를 확인하세요.")
    train_pairs = collect_pairs(args.data_root, "Training")
    val_pairs = collect_pairs(args.data_root, "Validation")
    x_train, y_train = build_features(train_pairs, args.n_mfcc, args.timestamp_unit, "Train 특징 추출")
    x_val, y_val = build_features(val_pairs, args.n_mfcc, args.timestamp_unit, "Validation 특징 추출")
    classifier = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, prediction)
    print(f"Train 발화: {len(y_train):,}, Validation 발화: {len(y_val):,}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(classification_report(y_val, prediction, labels=[0, 1], target_names=["119대원(0)", "신고자(1)"], digits=4, zero_division=0))
    fallback_speaker = int(np.bincount(y_train, minlength=2).argmax())
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_type": "mission2_mfcc_random_forest", "classifier": classifier, "n_mfcc": args.n_mfcc, "timestamp_unit": args.timestamp_unit, "fallback_speaker": fallback_speaker, "validation_accuracy": float(accuracy)}, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
