"""Training-only worker; use run.py/the notebook for the shared hard deadline."""
import argparse
import json
from pathlib import Path
import torch
if __package__:
    from .pipeline import train_pipeline
else:
    from pipeline import train_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_label_dir", "base_dir", "pretrained_dir", "output_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config_json")
    parser.add_argument("--start_epoch", type=float, required=True)
    parser.add_argument("--deadline_epoch", type=float, required=True)
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--attempt_marker", required=True, help="Local receipt for this run's verified portable model")
    args = parser.parse_args()
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    if args.auto_batch:
        identity = Path(args.output_dir) / "run_identity.json"
        if identity.is_file():
            saved = json.loads(identity.read_text())["config"]
            for key in ("batch_size", "window_batch_size", "gradient_accumulation_steps", "inference_batch_size"):
                config.setdefault(key, saved[key])
        elif torch.cuda.is_available():
            memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            batch = 8 if memory >= 35 else 4 if memory >= 22 else 2
            config.setdefault("batch_size", batch)
            config.setdefault("window_batch_size", batch)
            config.setdefault("gradient_accumulation_steps", 32 // batch)
            config.setdefault("inference_batch_size", batch)
    report = train_pipeline(args.train_label_dir, args.base_dir, args.output_dir, config, args.pretrained_dir,
                            args.start_epoch, args.deadline_epoch, args.attempt_marker)
    print("Training-only selection:", report["selected_name"], flush=True)


if __name__ == "__main__":
    main()
