"""Score frozen CSV predictions after inference, without changing the model."""
import argparse
import json
from pathlib import Path

if __package__:
    from .data import write_json
    from .pipeline import evaluate_csv
else:
    from data import write_json
    from pipeline import evaluate_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label_dir", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_csv(args.label_dir, args.prediction)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
