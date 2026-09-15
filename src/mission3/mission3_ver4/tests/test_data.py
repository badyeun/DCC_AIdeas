"""Competition input boundaries and leakage-safe dataset/submission behavior."""
from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "mission3_ver4_test_data_module", Path(__file__).resolve().parents[1] / "data.py"
)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def write_annotation(path: Path, annotation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8-sig")


def test_transcript_uses_both_sides_text_only_and_preserves_array_order():
    original = {
        "symptom": ["구토"],
        "address": "서울",
        "gender": "F",
        "startAt": 0,
        "endAt": 10,
        "utterances": [
            {"text": "  어디가\t아프세요? ", "speaker": 0, "startAt": 8, "endAt": 10},
            {"text": "토했어요.\n배도 아파요.", "speaker": 1, "startAt": 0, "endAt": 5},
        ],
    }
    expected = "어디가 아프세요?\n토했어요. 배도 아파요."
    assert data.dialogue_text(original) == expected

    changed = copy.deepcopy(original)
    changed.update(symptom={"deliberately": "invalid target"}, address="부산", gender="M")
    changed["startAt"], changed["endAt"] = "invalid", None
    for index, utterance in enumerate(changed["utterances"]):
        utterance.update(speaker=1 - index, startAt=1000 - index, endAt=None)
    assert data.dialogue_text(changed) == expected

    text_only = {"utterances": [{"text": u["text"]} for u in original["utterances"]]}
    assert data.dialogue_text(text_only) == expected


def test_inference_ignores_missing_or_malformed_targets(tmp_path):
    documents = [
        {"utterances": [{"text": "구급차 보내주세요."}]},
        {"utterances": [{"text": "머리가 아파요."}], "symptom": {"invalid": [42]}},
    ]
    for index, document in enumerate(documents):
        write_annotation(tmp_path / f"call{index}.json", document)

    paths, texts, targets = data.load_dataset(tmp_path, include_targets=False, workers=2)
    assert [p.name for p in paths] == ["call0.json", "call1.json"]
    assert texts == ["구급차 보내주세요.", "머리가 아파요."]
    assert targets is None
    with pytest.raises(ValueError, match="symptom"):
        data.load_dataset(tmp_path, include_targets=True, workers=2)


def test_target_filtering_keeps_empty_and_all_nine_class_rows(tmp_path):
    raw_targets = [
        [], None, ["찰과상", "기타"], "복통",
        [" 구토 ", "구토", "찰과상"], [*data.TARGET_SYMPTOMS, "찰과상"],
    ]
    for index, symptom in enumerate(raw_targets):
        write_annotation(tmp_path / f"{index}.json", {
            "utterances": [{"text": f"대화 {index}"}], "symptom": symptom,
        })

    paths, texts, targets = data.load_dataset(tmp_path, include_targets=True, workers=2)
    assert len(paths) == len(texts) == len(raw_targets)
    assert targets.shape == (6, 9)
    np.testing.assert_array_equal(targets[:3], np.zeros((3, 9), dtype=np.int8))
    assert [data.TARGET_SYMPTOMS[i] for i in np.flatnonzero(targets[3])] == ["복통"]
    assert [data.TARGET_SYMPTOMS[i] for i in np.flatnonzero(targets[4])] == ["구토"]
    np.testing.assert_array_equal(targets[5], np.ones(9, dtype=np.int8))


def test_sorted_submission_has_one_row_per_file_and_round_trips_labels(tmp_path):
    folder = tmp_path / "labels"
    names = ["z.json", "a.json", "한글,통화.json"]
    for name in names:
        write_annotation(folder / name, {"utterances": [{"text": name}]})
    paths, _, _ = data.load_dataset(folder, workers=2)
    probabilities = np.zeros((len(paths), 9))
    probabilities[1, [1, 3]] = [0.5, 0.9]
    probabilities[2, :] = 1.0
    output = tmp_path / "nested" / "mission3.csv"

    prediction = data.write_predictions(paths, probabilities, np.full(9, 0.5), output)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["label file name", "symptom"]
        rows = list(reader)
    assert [row["label file name"] for row in rows] == sorted(names)
    assert len(rows) == len(set(row["label file name"] for row in rows)) == len(names)
    assert [ast.literal_eval(row["symptom"]) for row in rows] == [
        [], ["구토", "복통"], list(data.TARGET_SYMPTOMS),
    ]
    assert prediction.shape == (3, 9)
    assert prediction[0].sum() == 0


def test_duplicate_submission_basenames_are_rejected(tmp_path):
    annotation = {"utterances": [{"text": "환자가 아파요."}]}
    write_annotation(tmp_path / "first" / "same.json", annotation)
    write_annotation(tmp_path / "second" / "same.json", annotation)
    with pytest.raises(ValueError, match="중복"):
        data.load_dataset(tmp_path, workers=2)


@pytest.mark.parametrize("invalid", [
    np.zeros((1, 8)), np.full((1, 9), np.nan), np.full((1, 9), np.inf),
])
def test_invalid_predictions_never_create_a_submission(tmp_path, invalid):
    output = tmp_path / "mission3.csv"
    with pytest.raises(ValueError, match="prediction"):
        data.write_predictions([Path("call.json")], invalid, np.full(9, 0.5), output)
    assert not output.exists()


def test_duplicate_texts_stay_together_across_every_split():
    rng = np.random.default_rng(318)
    texts = []
    targets = []
    for group in range(240):
        label = rng.binomial(1, 0.3, size=9).astype(np.int8)
        for variant in range(2):
            # Whitespace variants and contradictory targets still identify one call group.
            texts.append(f"통화 {group}\n머리가 아파요." if variant == 0
                         else f"  통화 {group}   머리가 아파요.  ")
            target = label.copy()
            if variant:
                target[group % 9] = 1 - target[group % 9]
            targets.append(target)
    targets = np.asarray(targets)
    development, calibration, folds, groups = data.make_splits(
        texts, targets, calibration_size=0.2, n_folds=3, seed=42,
    )

    all_indices = set(range(len(texts)))
    dev_set, cal_set = set(development), set(calibration)
    assert dev_set | cal_set == all_indices
    assert not dev_set & cal_set
    assert not set(groups[development]) & set(groups[calibration])
    for group in range(240):
        assert groups[2 * group] == groups[2 * group + 1]
        assert ((2 * group in dev_set) == (2 * group + 1 in dev_set))

    stop_indices = []
    for fit, stop in folds:
        fit_set, stop_set = set(fit), set(stop)
        assert fit_set | stop_set == dev_set
        assert not fit_set & stop_set
        assert not (fit_set | stop_set) & cal_set
        assert not set(groups[fit]) & set(groups[stop])
        stop_indices.extend(stop)
    assert sorted(stop_indices) == sorted(development.tolist())

    repeated = data.make_splits(texts, targets, calibration_size=0.2, n_folds=3, seed=42)
    np.testing.assert_array_equal(development, repeated[0])
    np.testing.assert_array_equal(calibration, repeated[1])
    for previous, current in zip(folds, repeated[2]):
        for previous_indices, current_indices in zip(previous, current):
            np.testing.assert_array_equal(previous_indices, current_indices)
