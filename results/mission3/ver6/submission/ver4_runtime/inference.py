"""Official one-command inference. Mission 3 accepts but does not read audio_dir."""
import argparse
from pathlib import Path

if __package__:
    from .pipeline import predict_pipeline
else:
    from pipeline import predict_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", required=True, help="Organizer interface; unused text-only task")
    parser.add_argument("--label_dir", required=True, type=Path)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    predict_pipeline(args.ckpt_path, args.label_dir, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
