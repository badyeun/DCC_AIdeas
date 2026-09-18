"""Mission 3 TF-IDF + Logistic Regression 멀티라벨 baseline 학습/검증."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

from common import ConstantBinaryClassifier, TARGET_SYMPTOMS, annotation_to_text, collect_split, load_annotation, target_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 3 TF-IDF + Logistic Regression 학습/검증")
    parser.add_argument("--data_root", type=Path, required=True, help="001~012 part 폴더 경로")
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission3_tfidf_lr.joblib"))
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=120_000)
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_examples(paths: list[Path], description: str) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[np.ndarray] = []
    skipped = 0
    for path in tqdm(paths, desc=description):
        try:
            annotation = load_annotation(path)
            texts.append(annotation_to_text(annotation))
            labels.append(target_vector(annotation))  # 9개 외 라벨만 제거하고 sample은 보존한다.
        except (OSError, ValueError) as error:
            skipped += 1
            print(f"[warn] {path}: {error}")
    if not texts:
        raise ValueError(f"{description}에서 유효한 sample을 만들지 못했습니다.")
    print(f"[{description}] sample: {len(texts):,}, skipped: {skipped:,}")
    return texts, np.vstack(labels)


def make_features(word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer, texts: list[str], *, fit: bool):
    if fit:
        return hstack((word_vectorizer.fit_transform(texts), char_vectorizer.fit_transform(texts))).tocsr()
    return hstack((word_vectorizer.transform(texts), char_vectorizer.transform(texts))).tocsr()


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.10, 0.91, 0.02):
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t, best_f1


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir() or args.min_df < 1 or args.C <= 0:
        raise ValueError("data_root, min_df, C를 확인하세요.")
    train_texts, y_train = read_examples(collect_split(args.data_root, "Training"), "Train")
    val_texts, y_val = read_examples(collect_split(args.data_root, "Validation"), "Validation")
    # 두 vectorizer 모두 Training text에서만 fit한다. Validation은 transform만 수행한다.
    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=args.min_df, max_features=args.max_features, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=args.min_df, max_features=args.max_features, sublinear_tf=True)
    x_train = make_features(word_vectorizer, char_vectorizer, train_texts, fit=True)
    x_val = make_features(word_vectorizer, char_vectorizer, val_texts, fit=False)
    models, probabilities, thresholds, class_f1 = [], np.zeros_like(y_val, dtype=np.float64), [], []
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        if np.unique(y_train[:, index]).size < 2:
            model = ConstantBinaryClassifier(int(y_train[0, index]))
        else:
            model = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1_000, random_state=args.seed)
            model.fit(x_train, y_train[:, index])
        models.append(model)
        probabilities[:, index] = model.predict_proba(x_val)[:, 1]
        threshold, score = best_threshold(y_val[:, index], probabilities[:, index])
        thresholds.append(threshold)
        class_f1.append(score)
        print(f"{symptom:6s} positive={int(y_train[:, index].sum()):4d} threshold={threshold:.2f} F1={score:.4f}")
    prediction = probabilities >= np.asarray(thresholds)
    macro_f1 = f1_score(y_val, prediction, average="macro", zero_division=0)
    print(f"Validation Macro F1: {macro_f1:.4f}")
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_type": "mission3_tfidf_logistic_regression", "target_symptoms": TARGET_SYMPTOMS, "word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer, "models": models, "thresholds": thresholds, "validation_macro_f1": float(macro_f1), "validation_class_f1": class_f1}, args.ckpt_path)
    print(f"저장 완료: {args.ckpt_path}")


if __name__ == "__main__":
    main()
