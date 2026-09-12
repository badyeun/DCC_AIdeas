"""Mission 1 Log-Mel + SimpleCNN baseline.

- 001~012 part 폴더를 --data_root 아래에서 자동 탐색합니다.
- Training에서는 긴 신고자 음성의 mel window 중 하나를 epoch마다 무작위 선택합니다.
- Validation에서는 한 통화의 모든 window logits를 평균하여 통화 단위로 평가합니다.
- validation accuracy가 가장 높은 epoch를 checkpoint로 저장합니다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    GENDER_TO_INDEX,
    INDEX_TO_GENDER,
    MAX_FRAMES,
    SimpleCNN,
    load_annotation,
    load_caller_audio,
    waveform_to_mel_windows,
)


def collect_split_pairs(data_root: Path, split: str):
    if split not in {"Training", "Validation"}:
        raise ValueError(f"지원하지 않는 split: {split}")

    parts = sorted([
        p for p in data_root.iterdir()
        if p.is_dir()
    ])

    if split == "Training":
        audio_folder = "TS_서울_구급"
        label_folder = "TL_서울_구급"
    else:
        audio_folder = "VS_서울_구급"
        label_folder = "VL_서울_구급"

    audio_files = []
    label_files = []

    for part in parts:
        base = part / "대학부 데이터"

        audio_dir = (
            base
            / split
            / "1.원천데이터"
            / audio_folder
        )

        label_dir = (
            base
            / split
            / "2.라벨링데이터"
            / label_folder
        )

        if audio_dir.exists():
            audio_files.extend(audio_dir.rglob("*.wav"))

        if label_dir.exists():
            label_files.extend(label_dir.rglob("*.json"))

    label_map = {
        p.stem: p
        for p in label_files
    }

    pairs = []
    missing = []

    for audio_path in audio_files:
        json_path = label_map.get(audio_path.stem)

        if json_path is None:
            missing.append(audio_path)
            continue

        # JSON에서 gender 읽기
        annotation = load_annotation(json_path)

        gender = str(
            annotation.get("gender", "")
        ).strip().upper()

        if gender not in GENDER_TO_INDEX:
            raise ValueError(
                f"잘못된 gender 값입니다: {gender!r}\n"
                f"JSON: {json_path}"
            )

        pairs.append(
            (
                audio_path,
                json_path,
                GENDER_TO_INDEX[gender],
            )
        )

    print(f"[{split}] WAV 개수       : {len(audio_files)}")
    print(f"[{split}] JSON 개수      : {len(label_files)}")
    print(f"[{split}] 매칭된 pair 개수: {len(pairs)}")
    print(f"[{split}] JSON 없는 WAV   : {len(missing)}")

    if missing:
        raise FileNotFoundError(
            f"{split}에서 JSON과 매칭되지 않은 WAV가 "
            f"{len(missing)}개 있습니다."
        )

    return sorted(
        pairs,
        key=lambda x: x[0].stem
    )

class CallerGenderDataset(Dataset):
    """Training용 Dataset: 통화마다 임의의 mel window 하나를 반환합니다."""

    def __init__(
        self,
        samples: list[tuple[Path, Path, int]],
        max_frames: int,
        window_hop_frames: int,
        timestamp_unit: str,
    ) -> None:
        self.samples = samples
        self.max_frames = max_frames
        self.window_hop_frames = window_hop_frames
        self.timestamp_unit = timestamp_unit

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, json_path, gender = self.samples[index]
        annotation = load_annotation(json_path)

        # common.py에서 speaker == 1(신고자) 발화 구간을 이어 붙인 waveform을 생성합니다.
        waveform = load_caller_audio(
            audio_path,
            annotation,
            timestamp_unit=self.timestamp_unit,
        )
        windows = waveform_to_mel_windows(
            waveform,
            self.max_frames,
            self.window_hop_frames,
        )

        # 긴 통화가 window 개수만큼 과도하게 가중되지 않도록
        # epoch마다 통화당 window 하나를 무작위 선택합니다.
        selected_window = windows[torch.randint(len(windows), (1,)).item()]
        return selected_window, torch.tensor(gender, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 1 Log-Mel + CNN 학습/검증")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="001~012 part 폴더가 들어 있는 data(inho) 경로",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("saved_models/mission1_model.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--window_hop_frames", type=int, default=150)
    parser.add_argument(
        "--timestamp_unit",
        choices=("auto", "seconds", "milliseconds"),
        default="auto",
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


def evaluate(
    model: nn.Module,
    samples: list[tuple[Path, Path, int]],
    device: torch.device,
    max_frames: int,
    window_hop_frames: int,
    timestamp_unit: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validation: 통화별 모든 window의 logits를 평균해 최종 성별을 예측합니다."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for audio_path, json_path, gender in tqdm(samples, desc="Validation"):
            annotation = load_annotation(json_path)
            waveform = load_caller_audio(
                audio_path,
                annotation,
                timestamp_unit=timestamp_unit,
            )
            windows = waveform_to_mel_windows(
                waveform,
                max_frames,
                window_hop_frames,
            )

            # windows: [num_windows, ...]. 한 통화의 모든 window를 한 번에 추론합니다.
            windows = windows.to(device)
            logits = model(windows)
            mean_logits = logits.mean(dim=0)
            pred = int(mean_logits.argmax().item())

            y_true.append(gender)
            y_pred.append(pred)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    accuracy = float((y_true_arr == y_pred_arr).mean())
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    return accuracy, macro_f1, y_true_arr, y_pred_arr


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.max_frames < 1
        or args.window_hop_frames < 1
    ):
        raise ValueError("학습 횟수, 배치 크기, 창 길이, 창 간격은 모두 양수여야 합니다.")
    if not args.data_root.is_dir():
        raise ValueError(f"data_root가 존재하지 않습니다: {args.data_root}")

    set_seed(args.seed)

    train_samples = collect_split_pairs(args.data_root, "Training")
    val_samples = collect_split_pairs(args.data_root, "Validation")

    print(f"Training samples  : {len(train_samples):,}")
    print(f"Validation samples: {len(val_samples):,}")

    train_dataset = CallerGenderDataset(
        train_samples,
        args.max_frames,
        args.window_hop_frames,
        args.timestamp_unit,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    label_array = np.asarray([gender for _, _, gender in train_samples])
    majority_index = int(
        np.bincount(label_array, minlength=len(GENDER_TO_INDEX)).argmax()
    )
    fallback_gender = INDEX_TO_GENDER[majority_index]
    majority_ratio = float((label_array == majority_index).mean())

    print(f"Device: {device}")
    print(f"Train 다수 클래스 baseline: {fallback_gender} ({majority_ratio:.4f})")

    best_val_acc = -1.0
    best_val_f1 = -1.0
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        ):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss = total_loss / len(train_dataset)
        train_acc = correct / len(train_dataset)

        val_acc, val_f1, y_true, y_pred = evaluate(
            model,
            val_samples,
            device,
            args.max_frames,
            args.window_hop_frames,
            args.timestamp_unit,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "max_frames": args.max_frames,
                    "window_hop_frames": args.window_hop_frames,
                    "timestamp_unit": args.timestamp_unit,
                    "gender_to_index": GENDER_TO_INDEX,
                    "fallback_gender": fallback_gender,
                    "best_validation_accuracy": float(best_val_acc),
                    "best_validation_macro_f1": float(best_val_f1),
                    "best_epoch": epoch,
                },
                args.ckpt_path,
            )
            print(f"Best checkpoint 저장: {args.ckpt_path}")

    # 최종 보고서는 저장된 best checkpoint 기준으로 다시 평가합니다.
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_acc, val_f1, y_true, y_pred = evaluate(
        model,
        val_samples,
        device,
        args.max_frames,
        args.window_hop_frames,
        args.timestamp_unit,
    )

    target_names = [INDEX_TO_GENDER[i] for i in range(len(INDEX_TO_GENDER))]
    print("\n=== Best validation result ===")
    print(f"Best epoch              : {checkpoint['best_epoch']}")
    print(f"Validation accuracy     : {val_acc:.4f}")
    print(f"Validation macro F1     : {val_f1:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print(f"최종 checkpoint: {args.ckpt_path}")


if __name__ == "__main__":
    main()
