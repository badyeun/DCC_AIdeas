"""Mission 3의 데이터 처리와 출력 형식을 공유합니다.

Mission 3의 모델 입력은 annotation JSON의 발화 ``text``만 사용합니다.
``symptom``은 학습 정답이나 평가 정답으로만 사용합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TARGET_SYMPTOMS = (
    "고열",
    "구토",
    "두통",
    "복통",
    "어지러움",
    "열상",
    "오심",
    "전신쇠약",
    "호흡곤란",
)
UTTERANCE_KEYS = ("utterances", "utterences")


def find_label_files(label_dir: Path) -> list[Path]:
    """label_dir 아래의 JSON을 항상 같은 순서로 반환합니다."""
    files = sorted(label_dir.rglob("*.json"))
    if not files:
        raise ValueError(f"No .json files found in {label_dir}")
    return files


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation must be a JSON object: {path}")
    return annotation


def dialogue_text(annotation: dict[str, Any]) -> str:
    """대화에서 전사 text만 꺼내어 하나의 문서로 만듭니다.

    speaker, startAt, endAt 및 기타 annotation은 모델 입력에
    포함하지 않습니다.
    """
    utterances: list[Any] = []
    for key in UTTERANCE_KEYS:
        value = annotation.get(key)
        if isinstance(value, list):
            utterances = value
            break

    texts: list[str] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        text = utterance.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def target_symptoms(annotation: dict[str, Any]) -> list[str]:
    """학습/평가 정답에서 출제 대상 9개 증상만 남깁니다."""
    raw_symptoms = annotation.get("symptom", [])
    if raw_symptoms is None:
        raw_symptoms = []
    elif isinstance(raw_symptoms, str):
        raw_symptoms = [raw_symptoms]
    if not isinstance(raw_symptoms, list):
        raise ValueError("symptom must be a list of strings")

    found = {str(value).strip() for value in raw_symptoms}
    return [symptom for symptom in TARGET_SYMPTOMS if symptom in found]


def target_vector(annotation: dict[str, Any]) -> list[int]:
    """9개 증상을 고정된 순서의 0/1 정답으로 변환합니다."""
    symptoms = set(target_symptoms(annotation))
    return [int(symptom in symptoms) for symptom in TARGET_SYMPTOMS]


def format_symptom_cell(symptoms: list[str]) -> str:
    """CSV의 symptom 칸을 ``['두통', '복통']`` 모양의 문자열로 만듭니다."""
    return str(symptoms)
