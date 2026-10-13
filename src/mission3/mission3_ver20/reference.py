"""Immutable, fit-only historical Training holdout comparison for ver20.

The embedded reference contains discrete predictions and aggregate provenance,
not text, identifiers, Validation labels, probabilities, or a trained model.
It is a descriptive benchmark; repeated use of this holdout limits generality.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from . import data

REFERENCE_PATH = Path(__file__).with_name("training_reference.json")
# Set from the independently rebuilt, verified historical artifact.
REFERENCE_FILE_SHA256 = "6979dfc0d92e7441ebd50eb1fc531b9cd11f91eab3bb9aae3912784b6e9a5581"


def array_digest(values, dtype):
    """Hash shape and canonical bytes, independent of host integer byte order."""
    values = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256(json.dumps(list(values.shape)).encode("ascii") + b"\0")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _indices(values, total, name):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
            or (values < 0).any() or (values >= total).any()
            or len(np.unique(values)) != len(values)):
        raise ValueError(f"Historical reference: invalid {name} indices")
    return values.astype("<i8")


def _require_equal(actual, expected, name):
    if actual != expected:
        raise ValueError(f"Historical reference mismatch: {name}")


def _metrics_match(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_metrics_match(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(actual, (int, float)) and np.isfinite(actual)
                and np.isfinite(expected) and abs(actual - expected) <= 1e-12)
    return actual == expected


def _decode_predictions(payload, rows):
    _require_equal(payload.get("encoding"), "zlib+base64/int8-c-order", "prediction encoding")
    _require_equal(payload.get("shape"), [rows, len(data.TARGET_SYMPTOMS)], "prediction shape")
    try:
        compressed = base64.b64decode(payload["data"], validate=True)
        decoder = zlib.decompressobj()
        expected_size = rows * len(data.TARGET_SYMPTOMS)
        raw = decoder.decompress(compressed, expected_size + 1)
        if (len(raw) != expected_size or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail):
            raise ValueError("Historical reference: invalid compressed prediction size")
        predictions = np.frombuffer(raw, dtype=np.int8).reshape(rows, len(data.TARGET_SYMPTOMS)).copy()
    except (KeyError, TypeError, ValueError, zlib.error) as error:
        raise ValueError("Historical reference: corrupt prediction payload") from error
    predictions = data._binary(predictions, rows)
    _require_equal(array_digest(predictions, "i1"), payload.get("sha256"), "prediction checksum")
    return predictions


def verify_training_reference(training_digest_value, targets, fit, holdout, *, path=None,
                              split_seed=42, holdout_size=0.2):
    """Fail closed unless the historical rows, targets, split, and payload match.

    ``training_digest_value`` must be the runtime's digest of the current,
    canonically sorted Training texts and targets. No Validation is accessed.
    The returned ``predictions`` ndarray is for comparison, not JSON output.
    """
    path = REFERENCE_PATH if path is None else Path(path)
    raw = path.read_bytes()
    _require_equal(hashlib.sha256(raw).hexdigest(), REFERENCE_FILE_SHA256, "reference file checksum")
    reference = json.loads(raw.decode("utf-8"))
    _require_equal(reference.get("format"), "mission3_training_reference_v1", "format")
    _require_equal(reference.get("labels"), list(data.TARGET_SYMPTOMS), "label order")
    _require_equal(reference.get("training_scope"), "fit_only", "historical training scope")
    _require_equal(reference.get("decision_rule"), "categorical predict/argmax", "decision rule")
    _require_equal(training_digest_value, reference.get("training_digest"), "Training data digest")
    _require_equal(split_seed, reference.get("split_seed"), "split seed")
    _require_equal(holdout_size, reference.get("holdout_size"), "holdout proportion")
    targets = data._binary(targets, reference["training_samples"], "targets")
    fit = _indices(fit, len(targets), "fit")
    holdout = _indices(holdout, len(targets), "holdout")
    if (len(fit) + len(holdout) != len(targets)
            or len(np.union1d(fit, holdout)) != len(targets)):
        raise ValueError("Historical reference: fit/holdout must partition Training")
    _require_equal(len(fit), reference["fit_samples"], "fit sample count")
    _require_equal(len(holdout), reference["holdout_samples"], "holdout sample count")
    hashes = reference["array_hashes"]
    _require_equal(array_digest(fit, "<i8"), hashes["fit_indices"], "fit order")
    _require_equal(array_digest(holdout, "<i8"), hashes["holdout_indices"], "holdout order")
    _require_equal(array_digest(targets, "i1"), hashes["training_targets"], "Training targets")
    _require_equal(array_digest(targets[holdout], "i1"), hashes["holdout_targets"], "holdout targets")
    predictions = _decode_predictions(reference.pop("predictions_payload"), len(holdout))
    _require_equal(array_digest(predictions, "i1"), hashes["predictions"], "reference prediction hash")
    metrics = data.score_predictions(targets[holdout], predictions)
    if not _metrics_match(metrics, reference["metrics"]):
        raise ValueError("Historical reference mismatch: recomputed metrics")
    reference["predictions"] = predictions
    reference["verified_reference_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return reference


def reference_summary(reference):
    """Return verified aggregate metadata suitable for manifests and JSON."""
    return {key: value for key, value in reference.items() if key != "predictions"}


def compare_reference(reference, new_predictions, targets, holdout):
    """Compare raw argmax decisions on the exact same Training holdout rows."""
    targets = data._binary(targets, reference["training_samples"], "targets")
    holdout = _indices(holdout, len(targets), "holdout")
    hashes = reference["array_hashes"]
    _require_equal(array_digest(holdout, "<i8"), hashes["holdout_indices"], "comparison holdout order")
    _require_equal(array_digest(targets, "i1"), hashes["training_targets"], "comparison Training targets")
    truth = targets[holdout]
    old = data._binary(reference["predictions"], len(holdout))
    _require_equal(array_digest(old, "i1"), hashes["predictions"], "comparison reference predictions")
    new = data._binary(new_predictions, len(holdout))
    old_metrics = data.score_predictions(truth, old)
    if not _metrics_match(old_metrics, reference["metrics"]):
        raise ValueError("Historical reference mismatch: comparison reference metrics")
    new_metrics = data.score_predictions(truth, new)
    nausea = "오심"
    macro_delta = new_metrics["macro_f1"] - old_metrics["macro_f1"]
    nausea_delta = new_metrics["per_symptom"][nausea]["f1"] - old_metrics["per_symptom"][nausea]["f1"]
    per_symptom = {}
    for index, label in enumerate(data.TARGET_SYMPTOMS):
        y, before, after = truth[:, index].astype(bool), old[:, index].astype(bool), new[:, index].astype(bool)
        per_symptom[label] = {
            "delta": {key: new_metrics["per_symptom"][label][key] - old_metrics["per_symptom"][label][key]
                      for key in ("f1", "precision", "recall", "false_positive", "false_negative")},
            "paired": {
                "fn_to_tp": int((y & ~before & after).sum()),
                "tp_to_fn": int((y & before & ~after).sum()),
                "fp_to_tn": int((~y & before & ~after).sum()),
                "tn_to_fp": int((~y & ~before & after).sum()),
                "persistent_fn": int((y & ~before & ~after).sum()),
                "persistent_fp": int((~y & before & after).sum()),
            },
        }
    return {
        "available": True, "reference_version": reference["version"],
        "reference_candidate": reference["candidate_id"], "comparison_scope": "Training internal holdout only",
        "decision_rule": reference["decision_rule"], "samples": len(holdout),
        "reference_metrics": old_metrics, "new_metrics": new_metrics,
        "macro_f1_delta": macro_delta, "nausea_f1_delta": nausea_delta,
        "both_macro_and_nausea_improved": bool(macro_delta > 0 and nausea_delta > 0),
        "status": ("both_improved" if macro_delta > 0 and nausea_delta > 0 else
                   "macro_only_improved" if macro_delta > 0 else
                   "nausea_only_improved" if nausea_delta > 0 else "neither_improved"),
        "per_symptom": per_symptom,
        "limitations": reference["limitations"],
    }
