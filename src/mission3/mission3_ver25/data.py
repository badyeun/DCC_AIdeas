"""Text-only data and label-balanced, indivisible Training transcript groups."""
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
        texts.append(utterance["text"])
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


def _assign_groups(counts, sizes, proportions, seed):
    """Iteratively allocate the rarest remaining group label, weighted by rows.

    Positive and negative counts are balanced together. A transcript group is
    indivisible, including when its original rows have conflicting labels.
    """
    rng = np.random.default_rng(seed)
    presence = counts > 0
    remaining = presence.sum(axis=0)
    label_deficits = proportions[:, None] * counts.sum(axis=0)
    row_deficits = proportions * sizes.sum()
    active = np.ones(len(sizes), dtype=bool)
    assignment = np.full(len(sizes), -1, dtype=np.int8)
    coverage = np.zeros((len(proportions), counts.shape[1]), dtype=bool)
    assigned_counts = np.zeros(len(proportions), dtype=int)
    unassigned_count = len(sizes)
    while unassigned_count:
        available = np.flatnonzero(remaining > 0)
        rare = available[remaining[available] == remaining[available].min()]
        label = int(rng.choice(rare))
        candidates = np.flatnonzero(active & presence[:, label])
        # Large indivisible groups are placed while there is still capacity.
        rng.shuffle(candidates)
        candidates = candidates[np.argsort(-sizes[candidates], kind="stable")]
        for group in candidates:
            sides = np.flatnonzero(row_deficits > 0)
            if not len(sides):
                sides = np.arange(len(proportions))
            missing = np.flatnonzero(~coverage[:, label])
            if len(missing) and remaining[label] <= len(missing):
                sides = missing
            # Every split must contain a group, even for very small datasets.
            empty = np.flatnonzero(assigned_counts == 0)
            if unassigned_count == len(empty):
                sides = empty
            sides = sides[label_deficits[sides, label] == label_deficits[sides, label].max()]
            sides = sides[row_deficits[sides] == row_deficits[sides].max()]
            side = int(rng.choice(sides))
            assignment[group] = side
            row_deficits[side] -= sizes[group]
            label_deficits[side] -= counts[group]
            remaining -= presence[group]
            coverage[side] |= presence[group]
            active[group] = False
            assigned_counts[side] += 1
            unassigned_count -= 1
    return assignment


def dataset_digest(texts, targets):
    """Hash ordered exact model inputs and original nine-column targets."""
    targets = _binary(targets, len(texts), "targets")
    digest = hashlib.sha256()
    digest.update(json.dumps(TARGET_SYMPTOMS, ensure_ascii=False).encode("utf-8"))
    for text, row in zip(texts, targets):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(row.tobytes())
    return digest.hexdigest()


def three_way_split(texts, targets, seed: int = 42):
    """Return fit/development/confirmation indices and an aggregate manifest.

    Eight deterministic allocation attempts minimize only split label/row
    imbalance. Model predictions and official Validation never participate.
    No row or original label is dropped, masked, or changed.
    """
    targets = _binary(targets, len(texts), "targets")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain strings only")
    groups = np.asarray([text_group(text) for text in texts])
    unique, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < 3:
        raise ValueError("Three-way split requires at least three distinct normalized text groups")
    positive = np.zeros((len(unique), len(TARGET_SYMPTOMS)), dtype=np.int64)
    np.add.at(positive, inverse, targets)
    counts = np.concatenate([positive, sizes[:, None] - positive], axis=1)
    proportions = np.asarray([.70, .15, .15])
    totals = counts.sum(axis=0)
    best = None
    for attempt in range(8):
        assignment = _assign_groups(counts, sizes, proportions, seed + attempt)
        splits = tuple(np.flatnonzero(assignment[inverse] == side) for side in range(3))
        observed = np.asarray([counts[assignment == side].sum(axis=0) for side in range(3)])
        missing = int(((observed == 0) & (totals > 0)).sum())
        imbalance = float(np.abs(observed / np.maximum(totals, 1) - proportions[:, None]).mean()
                          + np.abs(np.asarray([len(ids) for ids in splits]) / len(texts) - proportions).mean())
        rank = (missing, imbalance)
        if best is None or rank < best[0]:
            best = (rank, splits, assignment, attempt)
    _, splits, assignment, attempt = best
    data_hash = dataset_digest(texts, targets)
    split_hash = hashlib.sha256(data_hash.encode("ascii") + assignment[inverse].tobytes()).hexdigest()
    report = {
        "method": "iterative multilabel stratification of indivisible normalized text groups",
        "group_normalization": "NFC and whitespace collapse; model input text is preserved",
        "seed": seed, "allocation_attempt": attempt, "label_order": list(TARGET_SYMPTOMS),
        "data_digest": data_hash, "split_digest": split_hash,
        "samples": len(texts), "text_groups": len(unique),
        "duplicate_groups": int((sizes > 1).sum()),
        "duplicate_extra_samples": int((sizes - 1).sum()),
        "conflicting_label_groups": int(np.any((positive > 0) & (positive < sizes[:, None]), axis=1).sum()),
        "all_rows_preserved": True, "cross_split_text_groups": 0,
        "positive_counts": dict(zip(TARGET_SYMPTOMS, targets.sum(axis=0).astype(int).tolist())),
        "splits": {}, "coverage_warnings": [],
        "limitations": "Group constraints may prevent exact fractions or label coverage. Near duplicates are not checked. "
                       "Training has prior experiment exposure; this is not a new external test set.",
    }
    for side, (name, ids) in enumerate(zip(("fit", "development", "confirmation"), splits)):
        positives = targets[ids].sum(axis=0)
        report["splits"][name] = {
            "samples": len(ids), "text_groups": int((assignment == side).sum()),
            "requested_fraction": float(proportions[side]), "actual_fraction": len(ids) / len(texts),
            "indices_digest": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "group_digest": hashlib.sha256("".join(unique[assignment == side]).encode("ascii")).hexdigest(),
            "positive_counts": dict(zip(TARGET_SYMPTOMS, positives.astype(int).tolist())),
            "negative_counts": dict(zip(TARGET_SYMPTOMS, (len(ids) - positives).astype(int).tolist())),
            "positive_prevalence": dict(zip(TARGET_SYMPTOMS, (positives / len(ids)).tolist())),
        }
        for index, symptom in enumerate(TARGET_SYMPTOMS):
            if positives[index] in (0, len(ids)):
                report["coverage_warnings"].append({"split": name, "symptom": symptom,
                    "missing_class": "positive" if positives[index] == 0 else "negative"})
    return *splits, report


def _subset_groups_at_capacity(assignment, sizes, capacity, seed):
    """Keep the preferred groups first, while reaching the largest feasible size."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sizes))
    order = order[np.argsort(assignment[order], kind="stable")]
    # Bit i indicates whether i rows can be formed from whole remaining groups.
    # Suffix reachability lets us preserve preferred groups without underfilling
    # a feasible request merely because one large group was selected first.
    mask = (1 << (capacity + 1)) - 1
    reachable = [0] * (len(order) + 1)
    reachable[-1] = 1
    for position in range(len(order) - 1, -1, -1):
        tail = reachable[position + 1]
        reachable[position] = (tail | (tail << int(sizes[order[position]]))) & mask
    remaining = reachable[0].bit_length() - 1
    selected = np.zeros(len(sizes), dtype=bool)
    for position, group in enumerate(order):
        after = remaining - int(sizes[group])
        if after >= 0 and (reachable[position + 1] >> after) & 1:
            selected[group] = True
            remaining = after
    return selected


def sample_fit_subset(texts, targets, fit, sample_count: int = 4000, seed: int = 42):
    """Select a reproducible, label-balanced pilot from fit groups only.

    Original texts, targets, and the existing three-way split are unchanged.
    If indivisible duplicates prevent the exact request, return the largest
    feasible count below it and record the shortfall. No model predictions or
    development/confirmation labels influence the allocation.
    """
    targets = _binary(targets, len(texts), "targets")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain strings only")
    fit = np.asarray(fit)
    if (fit.ndim != 1 or not np.issubdtype(fit.dtype, np.integer)
            or len(np.unique(fit)) != len(fit)
            or np.any(fit < 0) or np.any(fit >= len(texts))):
        raise ValueError("fit must contain unique valid integer indices")
    if len(fit) < sample_count:
        raise ValueError("sample_count exceeds the available fit samples")
    fit = np.sort(fit.astype(np.int64))
    groups = np.asarray([text_group(text) for text in texts])
    all_groups, all_inverse, all_sizes = np.unique(groups, return_inverse=True, return_counts=True)
    fit_sizes = np.bincount(all_inverse[fit], minlength=len(all_groups))
    if np.any((fit_sizes > 0) & (fit_sizes != all_sizes)):
        raise ValueError("fit must contain complete normalized text groups to prevent split leakage")
    unique, inverse, sizes = np.unique(groups[fit], return_inverse=True, return_counts=True)
    positive = np.zeros((len(unique), len(TARGET_SYMPTOMS)), dtype=np.int64)
    np.add.at(positive, inverse, targets[fit])
    counts = np.concatenate([positive, sizes[:, None] - positive], axis=1)
    parent_positive = targets[fit].sum(axis=0)
    parent_prevalence = parent_positive / len(fit)
    parent_counts = counts.sum(axis=0)
    proportions = np.asarray([sample_count / len(fit), 1 - sample_count / len(fit)])
    best = None
    for attempt in range(8):
        if sample_count == len(fit):
            preferred = np.zeros(len(sizes), dtype=np.int8)
        else:
            preferred = _assign_groups(counts, sizes, proportions, seed + attempt)
        selected = _subset_groups_at_capacity(preferred, sizes, sample_count, seed + attempt)
        indices = fit[selected[inverse]]
        if not len(indices):
            raise ValueError("No complete fit text group fits within sample_count")
        observed = counts[selected].sum(axis=0)
        missing = int(((observed == 0) & (parent_counts > 0)).sum())
        deviation = float(np.abs(targets[indices].mean(axis=0) - parent_prevalence).mean())
        rank = (missing, deviation)
        if best is None or rank < best[0]:
            best = (rank, indices, selected, attempt)
    _, indices, selected, attempt = best
    data_hash = dataset_digest(texts, targets)
    parent_hash = hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest()
    indices_hash = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    subset_hash = hashlib.sha256((data_hash + parent_hash + indices_hash).encode("ascii")).hexdigest()
    observed_positive = targets[indices].sum(axis=0)
    observed_prevalence = observed_positive / len(indices)
    report = {
        "method": "iterative multilabel stratification within fit, with exact whole-group capacity",
        "group_normalization": "NFC and whitespace collapse; model input text is preserved",
        "seed": seed, "allocation_attempt": attempt, "label_order": list(TARGET_SYMPTOMS),
        "requested_samples": sample_count, "actual_samples": len(indices),
        "parent_fit_samples": len(fit), "unfilled_samples": sample_count - len(indices),
        "exact_sample_count": len(indices) == sample_count,
        "data_digest": data_hash, "parent_fit_digest": parent_hash,
        "indices_digest": indices_hash, "subset_digest": subset_hash,
        "parent_fit_text_groups": len(unique), "selected_text_groups": int(selected.sum()),
        "selected_duplicate_groups": int(((sizes > 1) & selected).sum()),
        "selected_conflicting_label_groups": int((np.any(
            (positive > 0) & (positive < sizes[:, None]), axis=1) & selected).sum()),
        "outside_fit_samples": 0, "original_targets_preserved": True,
        "positive_counts": dict(zip(TARGET_SYMPTOMS, observed_positive.astype(int).tolist())),
        "negative_counts": dict(zip(TARGET_SYMPTOMS, (len(indices) - observed_positive).astype(int).tolist())),
        "positive_prevalence": dict(zip(TARGET_SYMPTOMS, observed_prevalence.tolist())),
        "parent_positive_counts": dict(zip(TARGET_SYMPTOMS, parent_positive.astype(int).tolist())),
        "parent_positive_prevalence": dict(zip(TARGET_SYMPTOMS, parent_prevalence.tolist())),
        "prevalence_deviation": dict(zip(TARGET_SYMPTOMS,
                                         (observed_prevalence - parent_prevalence).tolist())),
        "coverage": {}, "coverage_warnings": [],
        "limitations": "Indivisible groups may prevent exact sample counts or class coverage. "
                       "This pilot subset does not establish full-data performance.",
    }
    for index, symptom in enumerate(TARGET_SYMPTOMS):
        positive_present = bool(observed_positive[index] > 0)
        negative_present = bool(observed_positive[index] < len(indices))
        report["coverage"][symptom] = {"positive": positive_present, "negative": negative_present}
        if not positive_present and parent_positive[index] > 0:
            report["coverage_warnings"].append({"symptom": symptom, "missing_class": "positive"})
        if not negative_present and parent_positive[index] < len(fit):
            report["coverage_warnings"].append({"symptom": symptom, "missing_class": "negative"})
    return indices, report


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


