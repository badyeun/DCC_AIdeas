"""출제 규격의 명령 한 번으로 Mission 3 CSV를 생성합니다."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from common import (
    TARGET_SYMPTOMS,
    dialogue_text,
    find_label_files,
    format_symptom_cell,
    load_annotation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 3 symptom inference.")
    # 주최 측 공통 실행 명령과 맞추기 위해 받지만, Mission 3에서 음성은 읽지 않습니다.
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = joblib.load(path)
    required = {"vectorizer", "classifier", "thresholds", "target_symptoms"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint was not created by src/mission3/train.py")
    if tuple(checkpoint["target_symptoms"]) != TARGET_SYMPTOMS:
        raise ValueError("Checkpoint symptom order does not match this inference code")
    return checkpoint


def main() -> None:
    args = parse_args()
    if not args.label_dir.is_dir():
        raise NotADirectoryError(f"label_dir does not exist: {args.label_dir}")

    checkpoint = load_checkpoint(args.ckpt_path)
    label_files = find_label_files(args.label_dir)

    # 추론 시 라벨 JSON에서는 오직 utterances[].text만 모델 입력으로 사용합니다.
    texts = [dialogue_text(load_annotation(path)) for path in label_files]
    features = checkpoint["vectorizer"].transform(texts)
    probabilities = checkpoint["classifier"].predict_proba(features)
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    predictions = probabilities >= thresholds

    rows: list[tuple[str, str]] = []
    for path, flags in zip(label_files, predictions):
        symptoms = [
            symptom for symptom, is_present in zip(TARGET_SYMPTOMS, flags) if is_present
        ]
        rows.append((path.name, format_symptom_cell(symptoms)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["label file name", "symptom"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
