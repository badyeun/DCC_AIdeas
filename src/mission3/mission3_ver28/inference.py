"""Offline, text-only inference from a completed ver28 run directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import analysis, data, trainer
from .runtime import FORMAT, find_run, read_json


def predict(label_dir, run_dir, output, device="cpu", batch_size=8):
    root = find_run(run_dir)
    if read_json(root / "run_state.json")["status"] != "complete":
        raise ValueError("Inference requires a completed full-training run")
    manifest = read_json(root / "submission_manifest.json")
    if (manifest.get("format") != FORMAT or manifest.get("label_order") != list(data.TARGET_SYMPTOMS)
            or manifest.get("decision_threshold") != .5):
        raise ValueError("Submission format, label order, or decision policy mismatch")
    members = manifest.get("members", {})
    if not 1 <= len(members) <= 2:
        raise ValueError("Invalid submission ensemble")
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predictions = []
    for relative in members.values():
        checkpoint = (root / relative).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError("Model path escapes the submission folder")
        predictions.append(trainer.predict_checkpoint(checkpoint, texts, device=device, batch_size=batch_size))
    logits = analysis.ensemble_logits(*predictions) if len(predictions) == 2 else predictions[0]
    data.write_predictions(paths, (logits >= 0).astype(np.int8), output)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    print(predict(args.label_dir, args.run_dir, args.output, args.device, args.batch_size))


if __name__ == "__main__":
    main()
