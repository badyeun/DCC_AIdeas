"""Training 서울 데이터로 Mission 3 TF-IDF 베이스라인을 학습합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from common import TARGET_SYMPTOMS, dialogue_text, find_label_files, load_annotation, target_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mission 3 symptom baseline.")
    parser.add_argument("--label_dir", type=Path, required=True, help="Training JSON folder")
    parser.add_argument(
        "--ckpt_path", type=Path, default=Path("saved_models/mission3_model.ckpt")
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=100_000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--c", type=float, default=2.0, help="Logistic Regression C")
    parser.add_argument("--max_iter", type=int, default=1_000)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def load_training_data(label_dir: Path) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    targets: list[list[int]] = []
    for path in find_label_files(label_dir):
        annotation = load_annotation(path)
        texts.append(dialogue_text(annotation))
        targets.append(target_vector(annotation))
    return texts, np.asarray(targets, dtype=np.int8)


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    # 한국어 형태소 분석기 없이도 오타와 활용형에 강한 문자 단위 TF-IDF입니다.
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_classifier(args: argparse.Namespace) -> OneVsRestClassifier:
    # 9개 증상에 대해 "있음/없음" 이진 분류기를 하나씩 학습합니다.
    estimator = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        penalty="l2",
        solver="liblinear",
        random_state=args.seed,
    )
    return OneVsRestClassifier(estimator, n_jobs=args.n_jobs)


def tune_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """내부 확인용 데이터에서 증상별 최적 기준값을 고릅니다."""
    thresholds = np.full(len(TARGET_SYMPTOMS), 0.5, dtype=np.float32)
    candidates = np.arange(0.10, 0.91, 0.05)
    for index in range(len(TARGET_SYMPTOMS)):
        truth = targets[:, index]
        if truth.min() == truth.max():
            continue
        best_score = -1.0
        for threshold in candidates:
            prediction = (probabilities[:, index] >= threshold).astype(np.int8)
            score = f1_score(truth, prediction, zero_division=0)
            if score > best_score:
                best_score = score
                thresholds[index] = threshold
    return thresholds


def print_scores(targets: np.ndarray, predictions: np.ndarray, thresholds: np.ndarray) -> None:
    scores = f1_score(targets, predictions, average=None, zero_division=0)
    print("\nInternal holdout result (Training data split only)")
    print("-" * 55)
    for symptom, score, threshold in zip(TARGET_SYMPTOMS, scores, thresholds):
        print(f"{symptom:<8} F1={score:.4f}  threshold={threshold:.2f}")
    print("-" * 55)
    print(f"Macro F1 = {scores.mean():.4f}\n")


def main() -> None:
    args = parse_args()
    if not args.label_dir.is_dir():
        raise NotADirectoryError(f"label_dir does not exist: {args.label_dir}")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if args.min_df < 1 or args.max_features < 1 or args.max_iter < 1 or args.c <= 0:
        raise ValueError("min_df, max_features, max_iter and C must be positive")

    texts, targets = load_training_data(args.label_dir)
    if len(texts) < 2:
        raise ValueError("At least two training samples are required")
    print(f"Loaded {len(texts)} Training conversations.")

    train_texts, holdout_texts, train_targets, holdout_targets = train_test_split(
        texts,
        targets,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    check_vectorizer = make_vectorizer(args)
    train_features = check_vectorizer.fit_transform(train_texts)
    holdout_features = check_vectorizer.transform(holdout_texts)
    check_classifier = make_classifier(args)
    check_classifier.fit(train_features, train_targets)
    probabilities = check_classifier.predict_proba(holdout_features)
    thresholds = tune_thresholds(holdout_targets, probabilities)
    predictions = (probabilities >= thresholds).astype(np.int8)
    print_scores(holdout_targets, predictions, thresholds)

    # 내부 확인이 끝나면 Validation이 아닌 Training 전체로 최종 모델을 다시 학습합니다.
    print("Refitting the final model with all Training conversations...")
    vectorizer = make_vectorizer(args)
    features = vectorizer.fit_transform(texts)
    classifier = make_classifier(args)
    classifier.fit(features, targets)

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "thresholds": thresholds,
            "target_symptoms": TARGET_SYMPTOMS,
            "model_name": "char TF-IDF + One-vs-Rest Logistic Regression",
        },
        args.ckpt_path,
    )
    print(f"Saved checkpoint: {args.ckpt_path}")


if __name__ == "__main__":
    main()
