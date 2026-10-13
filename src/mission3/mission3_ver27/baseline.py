"""Verify the completed ver26 reference and compare a controlled LR trial."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import analysis, data

MATCH_SETTINGS = (
    "pilot_samples", "seed", "fit_fraction", "development_fraction", "confirmation_fraction",
    "qwen_max_length", "control_max_length", "max_epochs", "effective_batch_size",
    "warmup_ratio", "control_lr", "weight_decay", "lora_r", "lora_alpha", "lora_dropout", "gradient_clip",
)
KINDS = ("qwen", "control", "ensemble")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _pin():
    return _read_json(Path(__file__).with_name("baseline_reference.json"))


def _check(budget, stage):
    if budget is not None:
        budget.check(stage, training=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _require(condition, message):
    if not condition:
        raise ValueError("ver26 baseline: " + message)


def load_reference(path, texts, targets, split, subset, settings, output_dir, budget=None):
    """Read verified small artifacts only; never load or modify old weights.

    A copied/moved reference is supported if its completion marker and required
    files exactly match the versioned pin. No latest-run heuristic is used.
    """
    _check(budget, "baseline verification")
    _require(isinstance(path, (str, Path)) and bool(str(path).strip()), "baseline_run is required")
    root = Path(path).expanduser().resolve()
    _require(root.is_dir(), "baseline_run folder is missing; restore the completed ver26 result folder")
    pin = _pin()
    completion = root / "screening_complete.json"
    _require(completion.is_file() and not completion.is_symlink(), "screening_complete.json is missing")
    _require(_sha(completion) == pin["completion_sha256"], "completion marker differs from the pinned run")
    marker = _read_json(completion)
    _require(marker.get("version") == "ver26" and marker.get("status") == "screening_complete",
             "reference is not a completed ver26 screening")
    _require(marker.get("training_scope") == "training_subset" and marker.get("submission_ready") is False,
             "reference training scope is invalid")
    _require(marker.get("confirmation_evaluated") is False and marker.get("validation_evaluated") is False,
             "reference evaluated a held-out gate")
    for name, digest in pin["required_files"].items():
        _check(budget, "baseline artifact hash")
        local = root / name
        _require(Path(name).name == name and local.resolve().is_relative_to(root) and not local.is_symlink(),
                 "unsafe reference artifact path")
        _require(local.is_file(), "required artifact is missing: " + name)
        _require(marker.get("hashes", {}).get(name) == digest and _sha(local) == digest,
                 "artifact integrity check failed: " + name)
    original = _read_json(root / "effective_settings.json")
    for key in MATCH_SETTINGS:
        _require(settings.get(key) == original.get(key), "comparison setting changed: " + key)
    for key in ("lora_lr", "head_lr"):
        expected = float(original[key]) * pin["learning_rate_multiplier"]
        _require(np.isclose(settings.get(key, -1), expected, rtol=1e-12, atol=0),
                 "learning rate must be exactly the planned 0.3 multiple: " + key)
    current_specs = _read_json(Path(__file__).with_name("model_specs.json"))
    previous_specs = _read_json(root / "model_specs.json")
    for kind in ("qwen", "control"):
        for key in ("model_id", "revision"):
            _require(current_specs[kind][key] == previous_specs[kind][key], "pretrained source changed")
    _require(len(texts) == len(targets) == pin["training_rows"], "training row count differs")
    digest = data.dataset_digest(texts, targets)
    _require(digest == pin["training_dataset_digest"] == _read_json(root / "training_data.json")["digest"],
             "original Training dataset digest differs")
    previous_split = _arrays(root / "split.npz")
    _require(set(split) == set(previous_split) == {"fit", "development", "confirmation"}, "split keys differ")
    _require(all(np.array_equal(split[key], previous_split[key]) for key in split), "split rows or order differ")
    _require(np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(targets))),
             "splits do not partition Training exactly once")
    old_subset = _arrays(root / "subset.npz")["indices"]
    _require(len(subset) == pin["subset_rows"] and np.array_equal(subset, old_subset), "subset rows or order differ")
    _require(len(np.unique(subset)) == len(subset) and np.isin(subset, split["fit"]).all(), "subset leaks outside fit")
    packed = _arrays(root / "development_predictions.npz")
    _require(set(packed) == {"indices", "targets", "qwen_logits", "control_logits", "ensemble_logits"},
             "prediction fields differ")
    _require(len(packed["indices"]) == pin["development_rows"]
             and np.array_equal(packed["indices"], split["development"]), "development row alignment differs")
    _require(np.array_equal(packed["targets"], targets[split["development"]]), "development labels differ")
    logits = {kind: packed[kind + "_logits"] for kind in KINDS}
    scores = {kind: analysis.metrics(packed["targets"], values) for kind, values in logits.items()}
    stored = _read_json(root / "development_comparison.json")["metrics"]
    _require(scores == stored, "stored reference metrics do not match saved predictions")
    mixed = analysis.ensemble_logits(logits["control"], logits["qwen"])
    _require(np.allclose(mixed, logits["ensemble"], rtol=1e-12, atol=1e-12)
             and np.array_equal(mixed >= 0, logits["ensemble"] >= 0), "reference ensemble is not a fixed 1:1 mean")
    selection = _read_json(root / "screening_selection.json")
    _require(selection["training_dataset_digest"] == digest, "selection dataset differs")
    _require(selection["selected_candidate"] == "ensemble", "pinned reference selection differs")
    for kind in ("qwen", "control"):
        history = _read_json(root / (kind + "_history.json"))
        _require([r["selected_epochs"] for r in history] == [1, 2, 3], "reference history is incomplete")
        best = max(history, key=lambda row: row["macro_f1"])
        _require(best["selected_epochs"] == selection["selected_epochs"][kind]
                 and best["metrics"] == scores[kind], "best reference epoch does not match selection")
    verification = _read_json(root / "screening_portable_verification.json")
    _require(all(verification[k]["decisions_equal"] is True and verification[k]["samples"] == 16
                 for k in ("qwen", "control")), "reference reload verification is incomplete")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = output / "baseline_ver26"
    _require(not copied.exists(), "reference output already exists")
    copied.mkdir()
    for name in ["screening_complete.json", *pin["required_files"]]:
        _check(budget, "copy baseline comparison artifacts")
        shutil.copyfile(root / name, copied / name)
        _require(_sha(copied / name) == _sha(root / name), "reference copy changed")
    evidence = {"version": "ver26", "run_id": pin["run_id"], "source_path": str(root),
                "completion_sha256": pin["completion_sha256"], "training_dataset_digest": digest,
                "required_artifact_hashes_verified": len(pin["required_files"]),
                "split_subset_and_original_targets_verified": True, "scores_recomputed": True,
                "weight_files_required_or_copied": False, "selected_epochs": selection["selected_epochs"],
                "macro_f1": {k: scores[k]["macro_f1"] for k in KINDS}}
    data.write_json(output / "baseline_reference_verified.json", evidence)
    return {"evidence": evidence, "indices": packed["indices"], "targets": packed["targets"],
            "logits": logits, "metrics": scores, "settings": original, "pin": pin}


def compare_reference(reference, new_logits, current_scores, current_records, output_dir, budget=None):
    """Use a fixed historical control to isolate the new Qwen's contribution.

    Fresh KLUE comparisons remain diagnostic. Historical control weights are
    not imported into a deployable ensemble; these are saved-logit comparisons.
    """
    output = Path(output_dir)
    truth = reference["targets"]
    for kind in ("qwen", "control"):
        _require(analysis.metrics(truth, new_logits[kind]) == current_scores[kind], "new metrics or rows differ")
    fresh_mixed = analysis.ensemble_logits(new_logits["control"], new_logits["qwen"])
    _require(analysis.metrics(truth, fresh_mixed) == current_scores["ensemble"], "new ensemble metrics differ")
    matched = analysis.ensemble_logits(reference["logits"]["control"], new_logits["qwen"])
    matched_scores = analysis.metrics(truth, matched)
    pairs = {
        "new_qwen_minus_ver26_qwen": (reference["logits"]["qwen"], new_logits["qwen"]),
        "same_control_ensemble_minus_ver26_ensemble": (reference["logits"]["ensemble"], matched),
        "fresh_ensemble_minus_ver26_ensemble": (reference["logits"]["ensemble"], fresh_mixed),
        "fresh_control_minus_ver26_control": (reference["logits"]["control"], new_logits["control"]),
    }
    comparisons = {}
    for name, (old, new) in pairs.items():
        _check(budget, "historical paired comparison")
        old_metrics, new_metrics = analysis.metrics(truth, old), analysis.metrics(truth, new)
        comparisons[name] = {
            "macro_f1_delta": new_metrics["macro_f1"] - old_metrics["macro_f1"],
            "macro_ap_delta": new_metrics["macro_average_precision"] - old_metrics["macro_average_precision"],
            "nausea_f1_delta": new_metrics["nausea_f1"] - old_metrics["nausea_f1"],
            "per_symptom_f1_delta": {s: new_metrics["per_symptom"][s]["f1"] - old_metrics["per_symptom"][s]["f1"]
                                     for s in data.TARGET_SYMPTOMS},
            "paired_errors": analysis.paired_error_changes(truth, old, new),
            "paired_ci": analysis.paired_bootstrap_ci(truth, old, new),
        }
    primary = comparisons["same_control_ensemble_minus_ver26_ensemble"]
    minimum_gain = reference["pin"]["promising_macro_gain"]
    promising = primary["macro_f1_delta"] >= minimum_gain and primary["paired_ci"]["macro_f1_delta"]["lower"] > 0
    effective = _read_json(output / "effective_settings.json")
    batch_fields = ("qwen_batch_size", "qwen_gradient_accumulation", "control_batch_size", "control_gradient_accumulation")
    batch_matches = {key: effective[key] == reference["settings"][key] for key in batch_fields}
    gate = {
        "status": "promising_for_larger_fit" if promising else "no_clear_gain",
        "primary_comparison": "same_control_ensemble_minus_ver26_ensemble",
        "rule": "Predeclared Macro F1 gain >= 0.005 and descriptive paired 95% lower bound > 0",
        "reference_macro_f1": reference["metrics"]["ensemble"]["macro_f1"],
        "candidate_macro_f1": matched_scores["macro_f1"], "macro_f1_delta": primary["macro_f1_delta"],
        "paired_ci": primary["paired_ci"]["macro_f1_delta"], "promising_macro_gain": minimum_gain,
        "promising_for_larger_fit": bool(promising), "automatic_full_training_recommended": False,
        "full_training_authorized": False, "submission_ready": False, "official_validation_evaluated": False,
        "official_validation_macro_f1": None, "target_0_7_achieved_on_official_validation": None,
        "selected_epoch_is_not_full_data_optimum": True, "batch_settings_match_reference": batch_matches,
        "matched_control_is_saved_predictions_only": True,
        "next_step": "Confirm recipe and epochs on a larger fit subset within the time budget."
                     if promising else "Review label-specific representation next; do not repeat longer training automatically.",
        "limitation": "Development was reused for epoch/candidate selection. Bootstrap excludes selection bias, seed variance, "
                      "and environment changes; this is a screening gate, not proof of generalization or a 0.7 score.",
    }
    report = {"reference": reference["evidence"], "baseline_metrics": reference["metrics"],
              "new_metrics": current_scores, "new_selected_epochs": {k: current_records[k]["selected_epochs"] for k in ("qwen", "control")},
              "new_qwen_with_ver26_control_metrics": matched_scores, "comparisons": comparisons,
              "decision": "fixed sigmoid >= 0.5", "blend": "fixed 1:1 probabilities; no fitted weights",
              "batch_settings_match_reference": batch_matches, "limitation": gate["limitation"]}
    data.write_json(output / "historical_comparison.json", report)
    data.write_json(output / "baseline_gate.json", gate)
    with (output / "historical_comparison_predictions.npz").open("wb") as handle:
        np.savez_compressed(handle, indices=reference["indices"], targets=truth,
                            old_qwen_logits=reference["logits"]["qwen"], old_control_logits=reference["logits"]["control"],
                            old_ensemble_logits=reference["logits"]["ensemble"], new_qwen_logits=new_logits["qwen"],
                            same_control_ensemble_logits=matched, fresh_ensemble_logits=fresh_mixed)
    return gate
