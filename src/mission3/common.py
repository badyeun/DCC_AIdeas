"""Mission 3 텍스트/라벨 처리 공통 코드."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SYMPTOMS = (
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
)
UTTERANCE_KEYS = ("utterances", "utterences")


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        annotation = json.load(file)
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation must be an object: {path}")
    return annotation


def get_utterances(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in UTTERANCE_KEYS:
        items = annotation.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def annotation_to_text(annotation: dict[str, Any]) -> str:
    """발화 전사문만 연결한다. speaker·시간·다른 라벨은 사용하지 않는다."""
    texts = [item.get("text", "").strip() for item in get_utterances(annotation)]
    return " ".join(text for text in texts if isinstance(text, str) and text)


def normalize_symptom_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,|;]", value) if part.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.extend(normalize_symptom_list(item))
            elif isinstance(item, dict):
                for key in ("name", "label", "symptom", "value"):
                    if key in item:
                        result.extend(normalize_symptom_list(item[key]))
                        break
        return result
    return []


def target_vector(annotation: dict[str, Any]) -> np.ndarray:
    labels = set(normalize_symptom_list(annotation.get("symptom")))
    return np.asarray([int(symptom in labels) for symptom in TARGET_SYMPTOMS], dtype=np.int8)


def find_json_files(label_dir: Path) -> list[Path]:
    return sorted(path for path in label_dir.rglob("*.json") if path.is_file())


def collect_split(data_root: Path, split: str) -> list[Path]:
    """각 part의 서울 라벨 JSON을 찾는다. Training/Validation만 명시적으로 분리한다."""
    folder = "TL_서울_구급" if split == "Training" else "VL_서울_구급"
    paths: list[Path] = []
    for part in sorted(path for path in data_root.iterdir() if path.is_dir()):
        label_dir = part / "대학부 데이터" / split / "2.라벨링데이터" / folder
        if label_dir.is_dir():
            paths.extend(find_json_files(label_dir))
    if not paths:
        raise FileNotFoundError(f"{split} 서울 라벨 JSON을 찾지 못했습니다: {data_root}")
    return paths


class ConstantBinaryClassifier:
    """학습 split에 양성 또는 음성만 있을 때를 위한 직렬화 가능한 분류기."""

    def __init__(self, value: int):
        self.value = int(value)

    def predict_proba(self, features: Any) -> np.ndarray:
        n_samples = features.shape[0]
        positive = np.full(n_samples, self.value, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))
