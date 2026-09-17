"""Train the complementary model and select calibration using Training only."""
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
    parser.add_argument("--train_label_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--pretrained_dir", required=True, help="Prepared public RoBERTa-large encoder/tokenizer directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_json")
    parser.add_argument("--auto_batch", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    if args.auto_batch:
        run_path = Path(args.output_dir) / "run.json"
        if run_path.is_file():
            # Keep the original batch schedule. Incomplete neural training also
            # verifies the GPU/precision identity before reusing epoch states.
            saved = json.loads(run_path.read_text(encoding="utf-8"))["config"]
            for key in ("batch_size", "window_batch_size", "gradient_accumulation_steps", "tapt_batch_size", "tapt_accumulation_steps"):
                config.setdefault(key, saved[key])
            print("이전 실행의 batch 설정을 유지하여 재개합니다.", flush=True)
        elif torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            batch = 8 if memory_gb >= 35 else 4 if memory_gb >= 22 else 2
            config.setdefault("batch_size", batch)
            config.setdefault("window_batch_size", batch)
            config.setdefault("gradient_accumulation_steps", max(1, 32 // config["batch_size"]))
            config.setdefault("tapt_batch_size", batch)
            config.setdefault("tapt_accumulation_steps", max(1, 32 // config["tapt_batch_size"]))
    print("ver8 추가 학습 설정:", json.dumps(config, ensure_ascii=False), flush=True)
    checkpoint, report = train_pipeline(args.train_label_dir, args.base_dir, args.output_dir, config,
                                        pretrained_dir=args.pretrained_dir)
    print("Checkpoint:", checkpoint, flush=True)
    print("선택:", report["selected_name"], flush=True)


if __name__ == "__main__":
    main()
