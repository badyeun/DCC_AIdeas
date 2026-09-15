"""Mission 1 MFCC(+F0) + RandomForest baseline.

Expected data root structure:
  data(inho)/
    <part-001>/대학부 데이터/Training/...
    <part-001>/대학부 데이터/Validation/...
    ...

All part folders under --data_root are discovered automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from common import (
    GENDER_TO_INDEX,
    INDEX_TO_GENDER,
    load_annotation,
    load_caller_audio,
    waveform_to_mfcc_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 1 MFCC + RandomForest 학습/검증")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="001~012 part 폴더가 들어 있는 data(inho) 경로",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("saved_models/mission1_mfcc_rf.joblib"),
    )
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument(
        "--timestamp_unit",
        choices=("auto", "seconds", "milliseconds"),
        default="auto",
    )
    parser.add_argument(
        "--no_f0",
        action="store_true",
        help="F0 특징을 제외합니다. MFCC-only 비교 실험용입니다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_split_pairs(data_root: Path, split: str):
    if split not in {"Training", "Validation"}:
        raise ValueError(f"지원하지 않는 split: {split}")

    parts = sorted([
        p for p in data_root.iterdir()
        if p.is_dir()
    ])

    if split == "Training":
        audio_folder = "TS_서울_구급"
        label_folder = "TL_서울_구급"
    else:
        audio_folder = "VS_서울_구급"
        label_folder = "VL_서울_구급"

    audio_files = []
    label_files = []

    # 001~012 전체에서 WAV / JSON 각각 수집
    for part in parts:
        base = part / "대학부 데이터"

        audio_dir = (
            base
            / split
            / "1.원천데이터"
            / audio_folder
        )

        label_dir = (
            base
            / split
            / "2.라벨링데이터"
            / label_folder
        )

        if audio_dir.exists():
            audio_files.extend(audio_dir.rglob("*.wav"))

        if label_dir.exists():
            label_files.extend(label_dir.rglob("*.json"))

    # JSON을 파일 stem 기준으로 매핑
    label_map = {
        p.stem: p
        for p in label_files
    }

    pairs = []
    missing = []

    for audio_path in audio_files:
        json_path = label_map.get(audio_path.stem)

        if json_path is None:
            missing.append(audio_path)
            continue

        pairs.append((audio_path, json_path))

    print(f"[{split}] WAV 개수       : {len(audio_files)}")
    print(f"[{split}] JSON 개수      : {len(label_files)}")
    print(f"[{split}] 매칭된 pair 개수: {len(pairs)}")
    print(f"[{split}] JSON 없는 WAV   : {len(missing)}")

    if missing:
        print("\n매칭 실패 예시:")
        for p in missing[:5]:
            print(p)

        raise FileNotFoundError(
            f"{split}에서 JSON과 매칭되지 않은 WAV가 "
            f"{len(missing)}개 있습니다."
        )

    return sorted(
        pairs,
        key=lambda x: x[0].stem
    )


def build_features(
    pairs: list[tuple[Path, Path]],
    n_mfcc: int,
    use_f0: bool,
    timestamp_unit: str,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for audio_path, json_path in tqdm(pairs, desc=desc):
        annotation = load_annotation(json_path)
        gender = str(annotation.get("gender", "")).strip().upper()
        if gender not in GENDER_TO_INDEX:
            raise ValueError(f"성별 레이블이 올바르지 않습니다: {json_path} -> {gender!r}")

        # common.py의 load_caller_audio가 utterances 중 speaker == 1 구간만 사용합니다.
        waveform = load_caller_audio(
            audio_path,
            annotation,
            timestamp_unit=timestamp_unit,
        )
        feature = waveform_to_mfcc_feature(waveform, n_mfcc, use_f0=use_f0)

        features.append(feature)
        labels.append(GENDER_TO_INDEX[gender])

    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()
    if args.n_mfcc < 1 or args.n_estimators < 1:
        raise ValueError("n_mfcc와 n_estimators는 양수여야 합니다.")
    if not args.data_root.is_dir():
        raise ValueError(f"data_root가 존재하지 않습니다: {args.data_root}")

    train_pairs = collect_split_pairs(args.data_root, "Training")
    val_pairs = collect_split_pairs(args.data_root, "Validation")

    print(f"Training pairs  : {len(train_pairs):,}")
    print(f"Validation pairs: {len(val_pairs):,}")

    use_f0 = not args.no_f0
    x_train, y_train = build_features(
        train_pairs,
        args.n_mfcc,
        use_f0,
        args.timestamp_unit,
        desc="Train feature extraction",
    )
    x_val, y_val = build_features(
        val_pairs,
        args.n_mfcc,
        use_f0,
        args.timestamp_unit,
        desc="Val feature extraction",
    )

    print(f"Train feature shape: {x_train.shape}")
    print(f"Val feature shape  : {x_val.shape}")
    print(f"F0 사용: {use_f0}")

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    print("RandomForest 학습 시작...")
    classifier.fit(x_train, y_train)

    train_acc = classifier.score(x_train, y_train)
    val_pred = classifier.predict(x_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    print(f"[참고] Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy : {val_acc:.4f}")
    print(f"Validation macro F1 : {val_f1:.4f}")

    target_names = [INDEX_TO_GENDER[i] for i in range(len(INDEX_TO_GENDER))]
    print("\nValidation classification report")
    print(
        classification_report(
            y_val,
            val_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    majority_index = int(
        np.bincount(y_train, minlength=len(GENDER_TO_INDEX)).argmax()
    )
    fallback_gender = INDEX_TO_GENDER[majority_index]
    majority_ratio = float((y_train == majority_index).mean())
    print(f"Train 다수 클래스 baseline: {fallback_gender} ({majority_ratio:.4f})")

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "mfcc_random_forest",
            "classifier": classifier,
            "n_mfcc": args.n_mfcc,
            "use_f0": use_f0,
            "timestamp_unit": args.timestamp_unit,
            "gender_to_index": GENDER_TO_INDEX,
            "fallback_gender": fallback_gender,
            "validation_accuracy": float(val_acc),
            "validation_macro_f1": float(val_f1),
        },
        args.ckpt_path,
    )
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
