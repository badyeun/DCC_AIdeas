"""Competition interface: text-only inference, then optional read-only scoring."""
import argparse
import json
from pathlib import Path
if __package__:
    from .pipeline import predict_pipeline, evaluate_csv, write_json
else:
    from pipeline import predict_pipeline, evaluate_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", default="", help="Accepted for the required interface; audio is not read")
    for name in ("label_dir", "ckpt_path", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_dir")
    parser.add_argument("--candidate_output")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    paths, _, _ = predict_pipeline(args.ckpt_path, args.label_dir, args.output, args.device,
                                   args.batch_size, args.cache_dir, args.candidate_output)
    if args.evaluate:
        if all("symptom" in json.loads(p.read_text(encoding="utf-8-sig")) for p in paths):
            for csv_path, report_name in ((args.output, "validation_metrics.json"),
                                           (args.candidate_output, "candidate_validation_metrics.json")):
                if csv_path and Path(csv_path).is_file():
                    report = evaluate_csv(args.label_dir, csv_path)
                    write_json(Path(csv_path).with_name(report_name), report)
                    print(f"{report_name}: Macro F1 = {report['macro_f1']:.6f}", flush=True)
        else:
            print("CSV saved; some labels have no targets, so scoring is skipped.", flush=True)


if __name__ == "__main__":
    main()
