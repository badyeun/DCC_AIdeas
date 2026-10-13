"""Frozen Training-only error groups; never used by training or inference."""
import re

import numpy as np


# These definitions predate ver19 and are diagnostic text matches only.
# They cannot determine affirmation, patient identity, or label correctness.
PATTERNS = {
    "direct": r"(?:메스꺼|메스껍|메슥|매스꺼|매스껍|매슥)|(?:울렁)|(?:구역질)|(?:미식|느글)|(?:토할\s*(?:것|거|꺼)\s*같|토할\s*(?:라고|려고)|토가\s*나올)|오심|구역감|메식|매식|미슥|미싯|메싯|매슥|니글|니긋|메슥|울렁|토[가를]?\s*(?:나올|나오려|나려고)|토할\s*(?:거|것|꺼|듯)|속[이가은는]?\s*(?:막\s*|너무\s*|좀\s*)?(?:매스|메스|미식|미슥|느글|니글)",
    "vomiting": r"구토|토했|토하|토해|토를|토하고|게워|구역질",
    "dizziness": r"어지러|어지럼|어질|빙빙",
    "question": r"\?|나요|까요|습니까|신가|세요",
    "negation": r"아니|없|않|못|안\s|아닌",
}
COMPILED = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}


def _binary(values, rows):
    values = np.asarray(values)
    if values.shape != (rows, 9) or values.dtype.kind not in "biu" or not np.isin(values, [0, 1]).all():
        raise ValueError("Diagnostic targets and predictions must be aligned binary (rows, 9) arrays")
    return values


def _counts(target, prediction):
    tp = int(((target == 1) & (prediction == 1)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    tn = int(((target == 0) & (prediction == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def compare_nausea_groups(texts, targets, baseline, challenger):
    """Call with Training holdout rows only, after both candidates are fixed."""
    texts = list(texts)
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("Diagnostic texts must be a nonempty collection of strings")
    targets, baseline, challenger = [_binary(value, len(texts)) for value in (targets, baseline, challenger)]
    flags = {key: np.array([bool(pattern.search(text)) for text in texts])
             for key, pattern in COMPILED.items() if key in ("direct", "vomiting", "dizziness")}
    near_question, near_negation = [], []
    for text in texts:
        turns = text.splitlines()
        neighbors = set()
        for index, turn in enumerate(turns):
            if COMPILED["direct"].search(turn):
                neighbors.update(range(max(0, index - 1), min(len(turns), index + 2)))
        window = "\n".join(turns[index] for index in sorted(neighbors))
        near_question.append(bool(COMPILED["question"].search(window)))
        near_negation.append(bool(COMPILED["negation"].search(window)))
    q, n = np.asarray(near_question), np.asarray(near_negation)
    y = targets[:, 6]
    masks = {
        "all": np.ones(len(texts), dtype=bool),
        "direct_expression_match": flags["direct"],
        "no_direct_expression_match": ~flags["direct"],
        "direct_near_question_expression": flags["direct"] & q,
        "direct_near_negation_expression": flags["direct"] & n,
        "direct_without_near_question_or_negation_expression": flags["direct"] & ~q & ~n,
        "no_direct_with_vomiting_expression": ~flags["direct"] & flags["vomiting"],
        "no_direct_with_dizziness_expression": ~flags["direct"] & flags["dizziness"],
        "gold_vomiting_without_nausea": (targets[:, 1] == 1) & (y == 0),
        "gold_dizziness_without_nausea": (targets[:, 4] == 1) & (y == 0),
        "gold_vomiting_or_dizziness_without_nausea": ((targets[:, 1] == 1) | (targets[:, 4] == 1)) & (y == 0),
    }
    result = {"scope": "Training internal holdout only; diagnostic after candidate selection",
              "affects_training_or_selection": False, "definitions": PATTERNS,
              "limits": ["Groups overlap and must not be summed.",
                         "Text matches, questions and negations are review flags, not semantic labels.",
                         "No expression match does not establish missing evidence or incorrect annotation.",
                         "No source text, sample identifiers, labels to replace, or fitted rules are exported."],
              "groups": {}}
    for name, mask in masks.items():
        truth, before, after = y[mask], baseline[mask, 6], challenger[mask, 6]
        result["groups"][name] = {
            "samples": int(mask.sum()), "positive_targets": int(truth.sum()),
            "baseline_ce": _counts(truth, before), "nausea_gce": _counts(truth, after),
            "fn_to_tp": int(((truth == 1) & (before == 0) & (after == 1)).sum()),
            "tp_to_fn": int(((truth == 1) & (before == 1) & (after == 0)).sum()),
            "fp_to_tn": int(((truth == 0) & (before == 1) & (after == 0)).sum()),
            "tn_to_fp": int(((truth == 0) & (before == 0) & (after == 1)).sum()),
        }
    return result
