"""Training-only review candidates without transcript or source identifier export."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

import numpy as np


TARGET_SYMPTOMS = (
    "고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란",
)
STRONG_CUE_PATTERNS = {
    "nausea_spellings": r"메스꺼|메스껍|메슥|매스꺼|매스껍|매슥",
    "fluttering": r"울렁",
    "retching": r"구역질",
    "colloquial_nausea": r"미식|느글",
    "prospective_vomiting": r"토할\s*(?:것|거|꺼)\s*같|토할\s*(?:라고|려고)|토가\s*나올",
}
QUESTION_PATTERN = r"\?|나요|까요|습니까|신가|세요"
NEGATION_PATTERN = r"아니|없|않|못|안\s|아닌"
REVIEW_STATUS = "unreviewed_candidate"
CSV_FIELDS = (
    "original_index", "split", "labels", "cue_groups", "candidate_reason",
    "has_question_expression", "has_negation_expression",
    "has_nonquestion_nonnegation_cue", "review_status", "review_note",
)


def _split_indices(values, rows: int, name: str) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer index array")
    if np.any(indices < 0) or np.any(indices >= rows):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices")
    return indices.astype(np.int64, copy=False)


def write_audit(paths, texts, targets, fit, holdout, output_dir) -> dict:
    """Audit a caller-supplied Training partition without changing its gold targets.

    ``paths`` is used only to validate row alignment; files and filenames are never
    read or exported. Callers must supply Training rows, never official Validation.
    Regex flags identify review candidates, not symptom presence or label errors.
    Each call creates a new child directory and cannot overwrite an earlier audit.
    """
    texts = list(texts)
    rows = len(texts)
    if not rows or len(paths) != rows or any(not isinstance(text, str) for text in texts):
        raise ValueError("paths and string texts must align and contain at least one row")
    gold = np.asarray(targets)
    if (gold.shape != (rows, len(TARGET_SYMPTOMS)) or gold.dtype.kind not in "biu"
            or not np.isin(gold, [0, 1]).all()):
        raise ValueError("targets must be a binary array with shape (rows, 9)")
    fit = _split_indices(fit, rows, "fit")
    holdout = _split_indices(holdout, rows, "holdout")
    if np.intersect1d(fit, holdout).size:
        raise ValueError("fit and holdout must be disjoint")
    if not np.array_equal(np.sort(np.concatenate((fit, holdout))), np.arange(rows)):
        raise ValueError("fit and holdout must partition every Training row")

    cue_regexes = {name: re.compile(pattern) for name, pattern in STRONG_CUE_PATTERNS.items()}
    question_regex, negation_regex = re.compile(QUESTION_PATTERN), re.compile(NEGATION_PATTERN)
    cue_flags = np.zeros((rows, len(cue_regexes)), dtype=bool)
    unqualified_flags = np.zeros_like(cue_flags)
    questions, negations = np.zeros(rows, dtype=bool), np.zeros(rows, dtype=bool)
    for index, text in enumerate(texts):
        lines = text.splitlines()
        questions[index] = bool(question_regex.search(text))
        negations[index] = bool(negation_regex.search(text))
        for column, regex in enumerate(cue_regexes.values()):
            cue_flags[index, column] = any(regex.search(line) for line in lines)
            unqualified_flags[index, column] = any(
                regex.search(line) and not question_regex.search(line) and not negation_regex.search(line)
                for line in lines
            )

    nausea = gold[:, TARGET_SYMPTOMS.index("오심")].astype(bool)
    cue, unqualified = cue_flags.any(axis=1), unqualified_flags.any(axis=1)
    candidates = cue | nausea
    split_name = np.empty(rows, dtype=object)
    split_name[fit], split_name[holdout] = "fit", "holdout"

    def counts(indices):
        local_nausea, local_cue = nausea[indices], cue[indices]
        local_unqualified = unqualified[indices]
        return {
            "samples": int(len(indices)),
            "nausea_positive": int(local_nausea.sum()),
            "strong_cue_samples": int(local_cue.sum()),
            "strong_cue_label_positive": int((local_cue & local_nausea).sum()),
            "strong_cue_label_negative": int((local_cue & ~local_nausea).sum()),
            "no_strong_cue_label_positive": int((~local_cue & local_nausea).sum()),
            "nonquestion_nonnegation_cue_samples": int(local_unqualified.sum()),
            "nonquestion_nonnegation_cue_label_positive": int((local_unqualified & local_nausea).sum()),
            "nonquestion_nonnegation_cue_label_negative": int((local_unqualified & ~local_nausea).sum()),
            "review_candidates": int(candidates[indices].sum()),
            "cue_groups": {
                name: {
                    "samples": int(cue_flags[indices, column].sum()),
                    "label_positive": int((cue_flags[indices, column] & local_nausea).sum()),
                    "label_negative": int((cue_flags[indices, column] & ~local_nausea).sum()),
                    "nonquestion_nonnegation_samples": int(unqualified_flags[indices, column].sum()),
                    "nonquestion_nonnegation_label_positive": int(
                        (unqualified_flags[indices, column] & local_nausea).sum()),
                    "nonquestion_nonnegation_label_negative": int(
                        (unqualified_flags[indices, column] & ~local_nausea).sum()),
                }
                for column, name in enumerate(cue_regexes)
            },
        }

    summary = {
        "format": "mission3_training_nausea_audit_v1",
        "source_policy": "Training only; official Validation must not be supplied",
        "review_status": REVIEW_STATUS,
        "targets_changed": False,
        "rows_removed": 0,
        "index_convention": "0-based original_index in the caller's sorted Training paths",
        "strong_cue_patterns": STRONG_CUE_PATTERNS.copy(),
        "question_pattern": QUESTION_PATTERN,
        "negation_pattern": NEGATION_PATTERN,
        "interpretation": (
            "Regex matches and nonquestion/nonnegation flags are review candidates, not human semantic "
            "judgments. A match does not establish nausea, and absence does not establish a label error. "
            "Question/negation matches can concern another symptom; 'not feeling well' can describe a "
            "positive symptom. Gold targets are preserved, with no automatic relabeling or removal."
        ),
        "counts": {"training": counts(np.arange(rows)), "fit": counts(fit), "holdout": counts(holdout)},
    }
    parent = Path(output_dir)
    child = "audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex
    run_dir = parent / child
    run_dir.mkdir(parents=True, exist_ok=False)
    summary["output_dir"] = str(run_dir.resolve())
    summary["files"] = {"summary": "audit_summary.json", "candidates": "review_candidates.csv"}
    with (run_dir / "review_candidates.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index in np.flatnonzero(candidates):
            reason = ("strong_cue_label_positive" if nausea[index] else "strong_cue_label_negative")
            if not cue[index]:
                reason = "no_strong_cue_label_positive"
            writer.writerow({
                "original_index": int(index), "split": split_name[index],
                "labels": json.dumps([name for name, flag in zip(TARGET_SYMPTOMS, gold[index]) if flag],
                                     ensure_ascii=False),
                "cue_groups": json.dumps([name for column, name in enumerate(cue_regexes)
                                          if cue_flags[index, column]]),
                "candidate_reason": reason,
                "has_question_expression": int(questions[index]),
                "has_negation_expression": int(negations[index]),
                "has_nonquestion_nonnegation_cue": int(unqualified[index]),
                "review_status": REVIEW_STATUS, "review_note": "",
            })
    with (run_dir / "audit_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary
