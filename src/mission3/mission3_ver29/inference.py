"""Offline, text-only inference from a completed ver29 run directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Also support the competition's direct `python inference.py` entrypoint.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

import numpy as np

from . import analysis, data, trainer
from .runtime import FORMAT, RECIPES, find_run, read_json


def predict(label_dir, run_dir, output, device="cpu", batch_size=8):
    checkpoint = Path(run_dir).expanduser().resolve()
    if checkpoint.is_file():
        if checkpoint.name != 'submission_manifest.json':
            raise ValueError("ckpt_path must be the complete run folder or its submission_manifest.json")
        run_dir = checkpoint.parent
    root = find_run(run_dir)
    if read_json(root / "run_state.json")["status"] != "complete":
        raise ValueError("Inference requires a completed full-training run")
    manifest = read_json(root / "submission_manifest.json")
    if (manifest.get("format") != FORMAT or manifest.get("label_order") != list(data.TARGET_SYMPTOMS)
            or manifest.get("decision_threshold") != .5):
        raise ValueError("Submission format, label order, or decision policy mismatch")
    members = manifest.get("members", {})
    if not isinstance(members, dict) or list(members) not in RECIPES.values():
        raise ValueError("Invalid submission ensemble")
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predictions = []
    for relative in members.values():
        checkpoint = (root / relative).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError("Model path escapes the submission folder")
        predictions.append(trainer.predict_checkpoint(checkpoint, texts, device=device, batch_size=batch_size))
    logits = analysis.equal_mean_logits(predictions)
    data.write_predictions(paths, (logits >= 0).astype(np.int8), output)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", "--label_dir", dest="label_dir", required=True)
    parser.add_argument("--run-dir", "--ckpt_path", dest="run_dir", required=True)
    parser.add_argument("--audio_dir", help="Accepted for the competition interface; audio is not read")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(predict(args.label_dir, args.run_dir, args.output, device, args.batch_size))


if __name__ == "__main__":
    main()
