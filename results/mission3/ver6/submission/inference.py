"""One-command Mission 3 inference; only transcript text is a model input."""
import argparse
import json
from pathlib import Path

if __package__:
    from .pipeline import evaluate_csv, predict_pipeline
    from .ver4_runtime.data import write_json
else:
    from pipeline import evaluate_csv, predict_pipeline
    from ver4_runtime.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", default="", help="Accepted for the competition interface; text task does not read audio.")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--cache_dir", help="Optional persistent raw prediction cache shared across experiments.")
    parser.add_argument("--base_dir", help="Relocated ver4 folder for an external-base ver6 experiment.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only after CSV is saved and all targets exist.")
    args = parser.parse_args()
    paths, _, _ = predict_pipeline(args.ckpt_path, args.label_dir, args.output, args.device, args.batch_size,
                                   cache_dir=args.cache_dir, base_dir=args.base_dir)
    if args.evaluate:
        target_count = sum("symptom" in json.loads(path.read_text(encoding="utf-8-sig")) for path in paths)
        if target_count == len(paths):
            report = evaluate_csv(args.label_dir, args.output)
            write_json(Path(args.output).with_name("validation_metrics.json"), report)
            print(f"Final evaluation Macro F1: {report['macro_f1']:.6f}", flush=True)
            for name, values in report["per_symptom"].items():
                print(f"  {name}: {values['f1']:.6f}", flush=True)
        else:
            print(f"정답 포함 JSON {target_count}/{len(paths)}: 예측 완료, 평가는 생략합니다.", flush=True)


if __name__ == "__main__":
    main()
