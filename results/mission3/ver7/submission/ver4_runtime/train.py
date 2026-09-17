"""Run all Training-only fitting and calibration; never accepts Validation."""
import argparse
import json
from pathlib import Path

if __package__:
    from .pipeline import train_pipeline
else:
    from pipeline import train_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional JSON overrides of DEFAULT_CONFIG")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    train_pipeline(args.label_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
