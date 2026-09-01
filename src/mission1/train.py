"""서울 학습 데이터만으로 Mission 1 신고자 성별 베이스라인을 학습합니다."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    GENDER_TO_INDEX,
    MAX_FRAMES,
    SimpleCNN,
    annotation_path_for,
    load_annotation,
    load_caller_audio,
    waveform_to_mel,
)


class CallerGenderDataset(Dataset):
    def __init__(
        self,
        audio_dir: Path,
        label_dir: Path,
        max_frames: int,
        timestamp_unit: str,
    ) -> None:
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.max_frames = max_frames
        self.timestamp_unit = timestamp_unit
        self.samples: list[tuple[Path, Path, int]] = []

        for audio_path in sorted(audio_dir.rglob("*.wav")):
            json_path = annotation_path_for(audio_path, audio_dir, label_dir)
            if not json_path.is_file():
                raise FileNotFoundError(f"Missing annotation for {audio_path}: {json_path}")
            gender = str(load_annotation(json_path).get("gender", "")).strip().upper()
            if gender not in GENDER_TO_INDEX:
                raise ValueError(f"Invalid gender label in {json_path}: {gender!r}")
            self.samples.append((audio_path, json_path, GENDER_TO_INDEX[gender]))

        if not self.samples:
            raise ValueError(f"No .wav files found in {audio_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, json_path, gender = self.samples[index]
        waveform = load_caller_audio(
            audio_path, load_annotation(json_path), timestamp_unit=self.timestamp_unit
        )
        return waveform_to_mel(waveform, self.max_frames), torch.tensor(gender, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mission 1 gender baseline.")
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, default=Path("saved_models/mission1_model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument(
        "--timestamp_unit", choices=("auto", "seconds", "milliseconds"), default="auto"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_frames < 1:
        raise ValueError("epochs, batch_size, and max_frames must be positive")
    set_seed(args.seed)

    dataset = CallerGenderDataset(
        args.audio_dir, args.label_dir, args.max_frames, args.timestamp_unit
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training on {len(dataset)} calls using {device}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        print(
            f"epoch={epoch} loss={total_loss / len(dataset):.4f} "
            f"accuracy={correct / len(dataset):.4f}"
        )

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "max_frames": args.max_frames,
            "timestamp_unit": args.timestamp_unit,
            "gender_to_index": GENDER_TO_INDEX,
        },
        args.ckpt_path,
    )
    print(f"Saved checkpoint: {args.ckpt_path}")


if __name__ == "__main__":
    main()
