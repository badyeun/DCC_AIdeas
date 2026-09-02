"""서울 학습 데이터로 MFCC-랜덤 포레스트 성별 베이스라인을 학습합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import (
    GENDER_TO_INDEX,
    annotation_path_for,
    load_annotation,
    load_caller_audio,
    waveform_to_mfcc_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 1 MFCC-랜덤 포레스트 학습")
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission1_mfcc_rf.joblib"))
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument(
        "--timestamp_unit", choices=("auto", "seconds", "milliseconds"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_mfcc < 1 or args.n_estimators < 1:
        raise ValueError("n_mfcc와 n_estimators는 양수여야 합니다.")

    features: list[np.ndarray] = []
    labels: list[int] = []
    audio_files = sorted(args.audio_dir.rglob("*.wav"))
    if not audio_files:
        raise ValueError(f"WAV 파일을 찾을 수 없습니다: {args.audio_dir}")

    for audio_path in audio_files:
        json_path = annotation_path_for(audio_path, args.audio_dir, args.label_dir)
        if not json_path.is_file():
            raise FileNotFoundError(f"대응하는 JSON 파일이 없습니다: {json_path}")
        annotation = load_annotation(json_path)
        gender = str(annotation.get("gender", "")).strip().upper()
        if gender not in GENDER_TO_INDEX:
            raise ValueError(f"성별 레이블이 올바르지 않습니다: {json_path}")
        waveform = load_caller_audio(
            audio_path, annotation, timestamp_unit=args.timestamp_unit
        )
        features.append(waveform_to_mfcc_feature(waveform, args.n_mfcc))
        labels.append(GENDER_TO_INDEX[gender])

    feature_matrix = np.vstack(features)
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(feature_matrix, labels)
    print(f"학습 정확도: {classifier.score(feature_matrix, labels):.4f}")

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "mfcc_random_forest",
            "classifier": classifier,
            "n_mfcc": args.n_mfcc,
            "timestamp_unit": args.timestamp_unit,
            "gender_to_index": GENDER_TO_INDEX,
        },
        args.ckpt_path,
    )
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
