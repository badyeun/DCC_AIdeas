"""Mission 3: 모델 입력은 양측 발화의 text만, symptom은 학습/평가 정답만."""
from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support

TARGET_SYMPTOMS = (
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
)


def dialogue_text(annotation: dict) -> str:
    """입력/정답과 무관하게 동일하게 전처리하며 발화 순서를 유지한다."""
    utterances = annotation.get("utterances")
    if not isinstance(utterances, list):
        utterances = annotation.get("utterences", [])
    if not isinstance(utterances, list):
        raise ValueError("utterances/utterences must be a list")
    texts = []
    for utterance in utterances:
        if isinstance(utterance, dict) and isinstance(utterance.get("text"), str):
            text = " ".join(unicodedata.normalize("NFC", utterance["text"]).split())
            if text:
                texts.append(text)
    return "\n".join(texts)


def target_vector(annotation: dict) -> list[int]:
    if "symptom" not in annotation:
        raise ValueError("Training/evaluation JSON has no symptom target")
    raw = annotation["symptom"]
    if raw is None:
        raw = []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
        raise ValueError("symptom must be a string or a list of strings")
    found = {x.strip() for x in raw}
    return [int(name in found) for name in TARGET_SYMPTOMS]


def text_group(text: str) -> str:
    # 파일명, 날짜, speaker 등은 모델 특징에도 중복 판단에도 사용하지 않는다.
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_dataset(label_dir: Path, include_targets: bool = False, workers: int = 8):
    paths = sorted(Path(label_dir).rglob("*.json"))
    if not paths:
        raise ValueError(f"JSON 파일이 없습니다: {label_dir}")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("동일한 JSON 파일명이 중복됩니다. 제출 CSV의 식별자가 모호합니다.")

    def read(path):
        with path.open(encoding="utf-8-sig") as handle:
            annotation = json.load(handle)
        if not isinstance(annotation, dict):
            raise ValueError(f"JSON object가 아닙니다: {path}")
        # 추론 경로에서는 target_vector를 호출하거나 symptom에 접근하지 않는다.
        return dialogue_text(annotation), target_vector(annotation) if include_targets else None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(read, paths))
    texts = [row[0] for row in rows]
    targets = np.asarray([row[1] for row in rows], dtype=np.int8) if include_targets else None
    return paths, texts, targets


def grouped_targets(texts, targets):
    groups = np.asarray([text_group(text) for text in texts])
    unique, inverse = np.unique(groups, return_inverse=True)
    aggregate = np.zeros((len(unique), len(TARGET_SYMPTOMS)), dtype=np.int8)
    np.maximum.at(aggregate, inverse, targets)
    return groups, inverse, aggregate


def make_splits(texts, targets, calibration_size: float, n_folds: int, seed: int):
    """전체 Training의 보정 집합은 모든 모델 fit/epoch 선택에서 분리한다."""
    if not 0 < calibration_size < 0.5 or n_folds < 2:
        raise ValueError("calibration_size must be in (0, .5); n_folds >= 2")
    groups, inverse, aggregate = grouped_targets(texts, targets)
    group_ids = np.arange(len(aggregate))
    outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=calibration_size, random_state=seed)
    dev_groups, cal_groups = next(outer.split(group_ids, aggregate))
    development = np.flatnonzero(np.isin(inverse, dev_groups))
    calibration = np.flatnonzero(np.isin(inverse, cal_groups))
    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
    folds = []
    for train_g, stop_g in splitter.split(dev_groups, aggregate[dev_groups]):
        train_indices = np.flatnonzero(np.isin(inverse, dev_groups[train_g]))
        stop_indices = np.flatnonzero(np.isin(inverse, dev_groups[stop_g]))
        folds.append((train_indices, stop_indices))
    for name, indices in [("calibration", calibration), ("development", development),
                          *[(f"fold{i}/{part}", ids) for i, pair in enumerate(folds)
                            for part, ids in zip(("fit", "stop"), pair)]]:
        counts = targets[indices].sum(axis=0)
        if len(indices) == 0 or np.any(counts == 0) or np.any(counts == len(indices)):
            raise ValueError(f"{name}: 각 증상의 양성/음성이 필요합니다. 데이터/분할 수를 확인하세요.")
    return development, calibration, folds, groups


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_predictions(paths, probabilities, thresholds, output: Path):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(paths), len(TARGET_SYMPTOMS)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction shape or non-finite probabilities")
    predicted = probabilities >= np.asarray(thresholds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label file name", "symptom"])
        for path, flags in zip(paths, predicted):
            # 기존 제출 CSV와 같은 Python string-list 표현. 빈 list도 허용한다.
            writer.writerow([Path(path).name, str([s for s, flag in zip(TARGET_SYMPTOMS, flags) if flag])])
    return predicted.astype(np.int8)


def score_predictions(targets, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, zero_division=0)
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "samples": int(len(targets)),
        "per_symptom": {name: {"f1": float(f1[i]), "precision": float(precision[i]),
                                "recall": float(recall[i]), "support": int(support[i])}
                        for i, name in enumerate(TARGET_SYMPTOMS)},
    }
