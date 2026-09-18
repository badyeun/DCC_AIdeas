"""한 번의 명령으로 Mission 3 제출 CSV를 생성한다."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
from scipy.sparse import hstack

from common import TARGET_SYMPTOMS, annotation_to_text, find_json_files, load_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 3 symptom inference.")
    # Mission 공통 실행 규격을 유지한다. Mission 3에서는 audio_dir를 읽지 않는다.
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.label_dir.is_dir() or not args.ckpt_path.is_file():
        raise FileNotFoundError("label_dir와 ckpt_path를 확인하세요.")
    checkpoint = joblib.load(args.ckpt_path)
    expected = tuple(checkpoint.get("target_symptoms", ()))
    if checkpoint.get("model_type") != "mission3_tfidf_logistic_regression" or expected != TARGET_SYMPTOMS:
        raise ValueError("Mission 3 TF-IDF-LogisticRegression 체크포인트가 아닙니다.")
    word_vectorizer, char_vectorizer = checkpoint["word_vectorizer"], checkpoint["char_vectorizer"]
    models, thresholds = checkpoint["models"], checkpoint["thresholds"]
    rows = []
    for json_path in find_json_files(args.label_dir):
        predicted: list[str] = []
        try:
            text = annotation_to_text(load_annotation(json_path))
            features = hstack((word_vectorizer.transform([text]), char_vectorizer.transform([text]))).tocsr()
            for symptom, model, threshold in zip(TARGET_SYMPTOMS, models, thresholds):
                if model.predict_proba(features)[0, 1] >= threshold:
                    predicted.append(symptom)
        except Exception as error:
            print(f"[warn] {json_path.name}: {type(error).__name__}: {error} -> 빈 라벨", file=sys.stderr)
        # 요구 형식은 string이며 0개 라벨도 허용한다. Python list repr은 string으로 기록된다.
        rows.append((json_path.name, str(predicted)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label file name", "symptom"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
