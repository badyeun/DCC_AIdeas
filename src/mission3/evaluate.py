"""Validation을 학습하지 않고 Mission 3 CSV의 Macro F1만 계산합니다."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from common import TARGET_SYMPTOMS, find_label_files, load_annotation, target_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Mission 3 prediction CSV.")
    parser.add_argument("--label_dir", type=Path, required=True, help="Validation JSON folder")
    parser.add_argument("--prediction", type=Path, required=True, help="mission3.csv")
    return parser.parse_args()


def parse_symptoms(value: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid symptom cell: {value!r}") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"symptom must be a string representation of a list: {value!r}")
    unknown = set(parsed) - set(TARGET_SYMPTOMS)
    if unknown:
        raise ValueError(f"Unknown target symptoms in prediction: {sorted(unknown)}")
    return set(parsed)


def load_predictions(path: Path) -> dict[str, set[str]]:
    predictions: dict[str, set[str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["label file name", "symptom"]:
            raise ValueError(
                "CSV header must be exactly: label file name,symptom "
                f"(received {reader.fieldnames})"
            )
        for row in reader:
            name = row["label file name"]
            if name in predictions:
                raise ValueError(f"Duplicate label file name in CSV: {name}")
            predictions[name] = parse_symptoms(row["symptom"])
    return predictions


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.prediction)
    truths: list[list[int]] = []
    predicted: list[list[int]] = []

    label_files = find_label_files(args.label_dir)
    expected_names = {path.name for path in label_files}
    if set(predictions) != expected_names:
        missing = sorted(expected_names - set(predictions))[:5]
        extra = sorted(set(predictions) - expected_names)[:5]
        raise ValueError(f"CSV/JSON file mismatch. missing={missing}, extra={extra}")

    for path in label_files:
        truths.append(target_vector(load_annotation(path)))
        predicted.append(
            [int(symptom in predictions[path.name]) for symptom in TARGET_SYMPTOMS]
        )

    scores = f1_score(
        np.asarray(truths), np.asarray(predicted), average=None, zero_division=0
    )
    for symptom, score in zip(TARGET_SYMPTOMS, scores):
        print(f"{symptom:<8} F1={score:.4f}")
    print(f"Macro F1 = {scores.mean():.4f}")


if __name__ == "__main__":
    main()
