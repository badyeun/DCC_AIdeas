"""Text-only Mission 3 data, grouped Training holdout, and discrete predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

TARGET_SYMPTOMS = (
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
)
LEXICAL_PATTERNS = {
    "nausea": r"오심|메스꺼|메스껍|메슥|매스꺼|매스껍|매슥|울렁|구역|미식|느글|토할",
    "question": r"\?|나요|까요|습니까|신가|세요",
    "negation": r"아니|없|않|못|안\s|아닌",
}
LEXICAL_COUNTS_NOTE = (
    "Regex matches are review candidates only: questions and negation may appear elsewhere "
    "in the dialogue. These flags do not establish annotation errors or modify labels/predictions."
)


def dialogue_text(annotation: dict) -> str:
    """Read only utterance text, keeping turn boundaries and chronological order."""
    key = "utterances" if "utterances" in annotation else "utterences"
    utterances = annotation.get(key)
    if not isinstance(utterances, list):
        raise ValueError("utterances/utterences must be a list")
    texts = []
    for utterance in utterances:
        if not isinstance(utterance, dict) or not isinstance(utterance.get("text"), str):
            raise ValueError("Each utterance must be an object with string text")
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
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError("symptom must be a string, a list of strings, or null")
    found = {item.strip() for item in raw}
    return [int(name in found) for name in TARGET_SYMPTOMS]


def text_group(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_dataset(label_dir, include_targets: bool = False):
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise ValueError(f"Label directory does not exist: {label_dir}")
    paths = sorted(path for path in label_dir.rglob("*.json") if path.is_file())
    if not paths:
        raise ValueError(f"No label JSON files: {label_dir}")
    if len({path.name.casefold() for path in paths}) != len(paths):
        raise ValueError("Duplicate JSON filenames make submission identifiers ambiguous")

    def read(path):
        with path.open(encoding="utf-8-sig") as handle:
            annotation = json.load(handle)
        if not isinstance(annotation, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        # Inference never inspects symptom, even when a supplied value is malformed.
        return dialogue_text(annotation), target_vector(annotation) if include_targets else None

    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(read, paths))
    texts = [row[0] for row in rows]
    targets = np.asarray([row[1] for row in rows], dtype=np.int8) if include_targets else None
    return paths, texts, targets


def _binary(values, rows=None, name="predictions"):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != len(TARGET_SYMPTOMS)
            or (rows is not None and values.shape[0] != rows)
            or not np.isin(values, [0, 1]).all()):
        raise ValueError(f"{name} must have shape (samples, 9) and contain only discrete 0/1 values")
    return values.astype(np.int8)


def _greedy_groups(features, sizes, holdout_size, seed):
    """Dependency-free iterative stratification of indivisible transcript groups.

    This balances label presence and absence, then row counts. Group constraints
    mean neither this fallback nor iterstrat can guarantee exact proportions.
    """
    rng = np.random.default_rng(seed)
    active = np.ones(len(features), dtype=bool)
    assignment = np.full(len(features), -1, dtype=np.int8)
    remaining = features.sum(axis=0).astype(float)
    proportions = np.asarray([1.0 - holdout_size, holdout_size])
    desired_labels = proportions[:, None] * remaining[None, :]
    desired_rows = proportions * sizes.sum()
    coverage = np.zeros((2, features.shape[1]), dtype=bool)
    while active.any():
        available = np.flatnonzero(remaining > 0)
        if not len(available):
            candidates = np.flatnonzero(active)
            rng.shuffle(candidates)
            label = None
        else:
            rare = available[remaining[available] == remaining[available].min()]
            label = int(rng.choice(rare))
            candidates = np.flatnonzero(active & features[:, label].astype(bool))
            rng.shuffle(candidates)
        for group in candidates:
            deficits = desired_rows if label is None else desired_labels[:, label]
            sides = np.flatnonzero(deficits == deficits.max())
            if len(sides) > 1:
                sides = sides[desired_rows[sides] == desired_rows[sides].max()]
            present = np.flatnonzero(features[group])
            final_occurrences = present[remaining[present] == 1]
            required = np.unique(np.where(~coverage[:, final_occurrences])[0])
            if len(required) == 1:
                # Reserve the last group for a side still missing this target.
                sides = required
            side = int(rng.choice(sides))
            assignment[group] = side
            desired_rows[side] -= sizes[group]
            desired_labels[side] -= features[group]
            remaining -= features[group]
            coverage[side] |= features[group].astype(bool)
            active[group] = False
    return np.flatnonzero(assignment == 0), np.flatnonzero(assignment == 1)


def make_split(texts, targets, holdout_size: float = .2, seed: int = 42):
    """Split Training only; normalized duplicate transcripts cannot cross sides."""
    if not 0 < holdout_size < .5:
        raise ValueError("holdout_size must be in (0, .5)")
    targets = _binary(targets, len(texts), "targets")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain strings only")
    groups = np.asarray([text_group(text) for text in texts])
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    presence = np.zeros((len(unique), len(TARGET_SYMPTOMS)), dtype=np.int8)
    absence = np.zeros_like(presence)
    np.maximum.at(presence, inverse, targets)
    np.maximum.at(absence, inverse, 1 - targets)
    features = np.concatenate([presence, absence], axis=1)
    if len(unique) < 2 or np.any(features.sum(axis=0) < 2):
        raise ValueError("Each symptom needs positive and negative examples in at least two text groups")
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        MultilabelStratifiedShuffleSplit = None
    best = None
    # Retry only group allocation, never inspecting official Validation.
    for attempt in range(8):
        current_seed = seed + attempt
        if MultilabelStratifiedShuffleSplit is None:
            fit_groups, holdout_groups = _greedy_groups(features, sizes, holdout_size, current_seed)
        else:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=holdout_size, random_state=current_seed)
            fit_groups, holdout_groups = next(splitter.split(np.arange(len(unique)), features))
        fit = np.flatnonzero(np.isin(inverse, fit_groups))
        holdout = np.flatnonzero(np.isin(inverse, holdout_groups))
        valid = all(len(ids) and np.all(targets[ids].sum(axis=0) > 0)
                    and np.all(targets[ids].sum(axis=0) < len(ids)) for ids in (fit, holdout))
        if not valid and MultilabelStratifiedShuffleSplit is not None:
            fit_groups, holdout_groups = _greedy_groups(features, sizes, holdout_size, current_seed)
            fit = np.flatnonzero(np.isin(inverse, fit_groups))
            holdout = np.flatnonzero(np.isin(inverse, holdout_groups))
            valid = all(len(ids) and np.all(targets[ids].sum(axis=0) > 0)
                        and np.all(targets[ids].sum(axis=0) < len(ids)) for ids in (fit, holdout))
        if valid:
            ratio = targets[holdout].sum(axis=0) / targets.sum(axis=0)
            discrepancy = float(np.abs(ratio - holdout_size).mean()
                                + abs(len(holdout) / len(targets) - holdout_size))
            if best is None or discrepancy < best[0]:
                best = (discrepancy, fit, holdout)
    if best is None:
        raise ValueError("Cannot split text groups with positive and negative examples for every symptom")
    return best[1], best[2]


def _atomic_text(path, write, encoding="utf-8-sig"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="",
                                         dir=path.parent, prefix=".write_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    _atomic_text(path, lambda handle: json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False),
                 encoding="utf-8")


def write_predictions(paths, predictions, output):
    """Write already discrete model decisions; this API has no score or threshold."""
    predictions = _binary(predictions, len(paths))
    if len({Path(path).name.casefold() for path in paths}) != len(paths):
        raise ValueError("Duplicate prediction identifiers")

    def write(handle):
        writer = csv.writer(handle)
        writer.writerow(["label file name", "symptom"])
        for path, flags in zip(paths, predictions):
            writer.writerow([Path(path).name, str([name for name, flag in zip(TARGET_SYMPTOMS, flags) if flag])])
    _atomic_text(output, write)
    return predictions


def score_predictions(targets, predictions):
    targets = _binary(targets, name="targets")
    predictions = _binary(predictions, len(targets))
    if not len(targets):
        raise ValueError("Cannot score an empty dataset")
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, zero_division=0)
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "samples": int(len(targets)),
        "per_symptom": {name: {"f1": float(f1[i]), "precision": float(precision[i]),
                                "recall": float(recall[i]), "support": int(support[i]),
                                "false_positive": int(((targets[:, i] == 0) & (predictions[:, i] == 1)).sum()),
                                "false_negative": int(((targets[:, i] == 1) & (predictions[:, i] == 0)).sum())}
                        for i, name in enumerate(TARGET_SYMPTOMS)},
    }


def training_diagnostics(texts, targets):
    """Return aggregate counts only, including suspected nausea annotation ambiguity."""
    targets = _binary(targets, len(texts), "targets")
    groups = {}
    for index, text in enumerate(texts):
        groups.setdefault(text_group(text), []).append(index)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    conflicts = [indices for indices in duplicates if len(np.unique(targets[indices], axis=0)) > 1]
    nausea = targets[:, TARGET_SYMPTOMS.index("오심")].astype(bool)
    vomiting = targets[:, TARGET_SYMPTOMS.index("구토")].astype(bool)
    signals = {key: np.asarray([bool(re.search(pattern, text)) for text in texts])
               for key, pattern in LEXICAL_PATTERNS.items()}
    lexical = {key: int(flags.sum()) for key, flags in signals.items()}
    for key in ("question", "negation"):
        cooccurrence = signals["nausea"] & signals[key]
        lexical[f"nausea_and_{key}"] = int(cooccurrence.sum())
        lexical[f"nausea_and_{key}_label_positive"] = int((cooccurrence & nausea).sum())
        lexical[f"nausea_and_{key}_label_negative"] = int((cooccurrence & ~nausea).sum())
    lexical["nausea_mention_label_positive"] = int((signals["nausea"] & nausea).sum())
    lexical["nausea_mention_label_negative"] = int((signals["nausea"] & ~nausea).sum())
    return {
        "samples": len(texts), "all_zero_samples": int((targets.sum(axis=1) == 0).sum()),
        "empty_text_samples": sum(not text.strip() for text in texts),
        "positive_counts": {name: int(targets[:, i].sum()) for i, name in enumerate(TARGET_SYMPTOMS)},
        "text_groups": len(groups), "duplicate_groups": len(duplicates),
        "duplicate_extra_samples": sum(len(indices) - 1 for indices in duplicates),
        "conflicting_label_groups": len(conflicts),
        "conflicting_label_samples": sum(len(indices) for indices in conflicts),
        "nausea_vomiting_contingency": {
            "neither": int((~nausea & ~vomiting).sum()), "nausea_only": int((nausea & ~vomiting).sum()),
            "vomiting_only": int((~nausea & vomiting).sum()), "both": int((nausea & vomiting).sum())},
        "lexical_cooccurrence_counts": lexical,
        "lexical_patterns": LEXICAL_PATTERNS,
        "lexical_counts_note": LEXICAL_COUNTS_NOTE,
    }


def export_errors(paths, texts, targets, predictions, output_dir, max_per_label: int = 20):
    """Save full text locally for manual review; never print sensitive dialogue."""
    targets = _binary(targets, len(texts), "targets")
    predictions = _binary(predictions, len(texts))
    if len(paths) != len(texts) or type(max_per_label) is not int or max_per_label < 1:
        raise ValueError("paths/texts must align and max_per_label must be a positive integer")
    output_dir = Path(output_dir)
    outputs = []
    signals = {key: np.asarray([bool(re.search(pattern, text)) for text in texts])
               for key, pattern in LEXICAL_PATTERNS.items()}
    header = ["label file name", "symptom", "error_type", "true_symptoms", "predicted_symptoms", "text",
              "review_category", "review_note", "has_nausea_expression", "has_question_expression",
              "has_negation_expression"]

    def review_row(index, symptom, category):
        actual = [name for name, flag in zip(TARGET_SYMPTOMS, targets[index]) if flag]
        predicted = [name for name, flag in zip(TARGET_SYMPTOMS, predictions[index]) if flag]
        return [Path(paths[index]).name, symptom, category, str(actual), str(predicted), texts[index], "", "",
                int(signals["nausea"][index]), int(signals["question"][index]), int(signals["negation"][index])]

    for label, name in enumerate(TARGET_SYMPTOMS):
        fn = np.flatnonzero((targets[:, label] == 1) & (predictions[:, label] == 0)).tolist()
        fp = np.flatnonzero((targets[:, label] == 0) & (predictions[:, label] == 1)).tolist()
        indices = []
        while (fn or fp) and len(indices) < max_per_label:
            for pending in (fn, fp):
                if pending and len(indices) < max_per_label:
                    indices.append(pending.pop(0))
        output = output_dir / f"errors_{name}.csv"

        def write(handle):
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in indices:
                writer.writerow(review_row(index, name, "FN" if targets[index, label] else "FP"))
        _atomic_text(output, write)
        outputs.append(output)

    nausea_index, vomiting_index = TARGET_SYMPTOMS.index("오심"), TARGET_SYMPTOMS.index("구토")
    nausea, vomiting = targets[:, nausea_index].astype(bool), targets[:, vomiting_index].astype(bool)
    combinations = {"neither": ~nausea & ~vomiting, "nausea_only": nausea & ~vomiting,
                    "vomiting_only": ~nausea & vomiting, "both": nausea & vomiting}
    group_counts = {}
    for name, mask in combinations.items():
        row = {"n": int(mask.sum())}
        for symptom, column in (("nausea", nausea_index), ("vomiting", vomiting_index)):
            actual, predicted = targets[:, column].astype(bool), predictions[:, column].astype(bool)
            row[f"predicted_{symptom}"] = int((mask & predicted).sum())
            row[f"{symptom}_tp"] = int((mask & actual & predicted).sum())
            row[f"{symptom}_fp"] = int((mask & ~actual & predicted).sum())
            row[f"{symptom}_fn"] = int((mask & actual & ~predicted).sum())
            row[f"{symptom}_tn"] = int((mask & ~actual & ~predicted).sum())
        group_counts[name] = row
    summary_path = output_dir / "nausea_vomiting_error_groups.json"
    write_json(summary_path, group_counts)
    outputs.append(summary_path)

    predicted_nausea = predictions[:, nausea_index].astype(bool)
    controls = {"TP": nausea & predicted_nausea, "FP": ~nausea & predicted_nausea,
                "FN": nausea & ~predicted_nausea, "TN": ~nausea & ~predicted_nausea}
    rng = np.random.default_rng(42)
    sampled = []
    for category, mask in controls.items():
        candidates = np.flatnonzero(mask)
        chosen = rng.choice(candidates, size=min(10, len(candidates)), replace=False)
        sampled.extend((int(index), category) for index in chosen)
    # Mix categories for human review while preserving reproducibility.
    rng.shuffle(sampled)
    review_path = output_dir / "review_오심.csv"

    def write_controls(handle):
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, category in sampled:
            writer.writerow(review_row(index, "오심", category))
    _atomic_text(review_path, write_controls)
    outputs.append(review_path)
    return outputs
