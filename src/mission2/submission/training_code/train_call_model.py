"""2단계: 통화 단위 트랜스포머 학습 (모델 코드는 제출 폴더의 model.py).

python train_call_model.py --embed_cache_dir mission2_embed_cache --mfcc_cache_dir mission2_cache \
    --work_dir work --out A0_base.pt --epochs 20 --threads 14 [--seed 7] [--pair_weight 0.5] [--holdout 0] ...

make_feature_cache.py가 만든 발화별 캐시(ECAPA 192 + MFCC 261)만 읽는다. WAV를 읽지 않아 GPU 없이 CPU에서 돈다
(로컬 PC CPU 16코어 기준 모델당 1~2시간).

- 학습 데이터: 서울 Training 29,200통화. 그중 10%(--holdout)를 떼어 epoch 선택(조기 종료)에 쓴다.
- Validation 3,640통화는 학습이 끝난 뒤 정확도를 한 번 계산하는 데만 쓴다(역전파·모델 선택에 쓰지 않음).
- 정답(speaker)은 손실 계산에만 쓴다. 입력에는 발화 음성 특징만 들어가고 발화 순서 정보는 없다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 제출 폴더의 model.py
from model import ECAPA_DIM, INPUT_DIM, MFCC_DIM, build_model, call_inputs, predict_calls  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2 통화 단위 트랜스포머 학습")
    parser.add_argument("--embed_cache_dir", type=Path, required=True, help="ECAPA 캐시 (embed_train_*, embed_val_*)")
    parser.add_argument("--mfcc_cache_dir", type=Path, required=True, help="MFCC 캐시 (train_*, val_*)")
    parser.add_argument("--work_dir", type=Path, required=True, help="캐시를 옮겨 둘 디스크 배열(memmap) 폴더")
    parser.add_argument("--out", type=Path, required=True, help="모델 저장 경로 (.pt)")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20, help="학습률 스케줄 길이(최대 epoch)")
    parser.add_argument("--run_epochs", type=int, default=None,
                        help="앞의 몇 epoch만 실제로 돌린다(스케줄은 --epochs 그대로). 학습 코드 동작 확인용")
    parser.add_argument("--batch_calls", type=int, default=32, help="한 번에 넣는 통화 수")
    parser.add_argument("--max_call_len", type=int, default=96, help="학습 때 이보다 긴 통화는 발화를 무작위로 골라 쓴다")
    parser.add_argument("--drop_utterances", type=float, default=0.2, help="학습 때 통화마다 최대 이 비율의 발화를 무작위로 뺀다")
    parser.add_argument("--holdout", type=float, default=0.1,
                        help="epoch 선택용 Train 통화 비율. 0이면 Train 전체로 --epochs만큼 학습하고 마지막 모델을 쓴다")
    parser.add_argument("--patience", type=int, default=4, help="holdout이 이 횟수만큼 연속으로 나아지지 않으면 멈춘다")
    parser.add_argument("--pair_weight", type=float, default=0.0, help="같은 화자 보조 과제 손실 가중치. 0이면 끈다")
    parser.add_argument("--pair_dim", type=int, default=64, help="같은 화자 보조 과제의 투영 차원")
    parser.add_argument("--pair_scale", type=float, default=10.0, help="코사인 유사도에 곱하는 배율(logit 크기)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def elapsed(start: float) -> str:
    seconds = int(time.perf_counter() - start)
    return f"{seconds // 3600}시간 {seconds % 3600 // 60}분 {seconds % 60}초"


# ---------------------------------------------------------------- 캐시 읽기

def read_cache(cache_dir: Path, split: str, width: int, target: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """shard를 읽어 디스크 배열(float16)로 옮긴다. (특징, 정답, 통화별 발화 수)를 돌려준다.

    특징을 RAM에 다 올리지 않고 memmap에서 필요한 통화만 읽어, 메모리가 적은 PC에서도 돈다.
    """
    if not (cache_dir / f"{split}_meta.json").is_file():
        raise FileNotFoundError(f"캐시 설정 파일이 없습니다: {cache_dir / f'{split}_meta.json'}")
    paths = []
    while (cache_dir / f"{split}_{len(paths):05d}.npz").is_file():
        paths.append(cache_dir / f"{split}_{len(paths):05d}.npz")
    if not paths:
        raise FileNotFoundError(f"shard가 없습니다: {cache_dir}/{split}_00000.npz")
    sizes, labels, counts, calls = [], [], [], 0
    for path in paths:
        with np.load(path) as data:
            sizes.append(len(data["y"]))
            labels.append(data["y"].astype(np.int64))
            calls += int(data["calls"])
            if "counts" in data.files:
                counts.append(data["counts"].astype(np.int64))
    # 초기 MFCC 캐시는 shard에 통화 경계가 없어 따로 저장한 `{split}_counts.npy`를 쓴다.
    boundary = np.concatenate(counts) if len(counts) == len(paths) else np.load(cache_dir / f"{split}_counts.npy").astype(np.int64)
    rows = int(sum(sizes))
    if len(boundary) != calls or int(boundary.sum()) != rows:
        raise ValueError(f"[{split}] 통화 경계가 캐시와 맞지 않습니다: 통화 {len(boundary)} vs {calls}, 발화 {int(boundary.sum())} vs {rows}")
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(rows, width))
    start = 0
    for path, size in zip(paths, sizes):
        with np.load(path) as data:
            if data["x"].shape[1] != width:
                raise ValueError(f"{path.name}: 차원 {data['x'].shape[1]} != {width}")
            output[start:start + size] = data["x"].astype(np.float16)
        start += size
    output.flush()
    del output
    return np.load(target, mmap_mode="r"), np.concatenate(labels), boundary


def load_split(args, split: str) -> dict:
    """ECAPA와 MFCC 캐시를 같은 발화 순서로 맞춰 읽는다. 정답이나 통화 경계가 어긋나면 멈춘다."""
    ecapa, labels, counts = read_cache(args.embed_cache_dir, f"embed_{split}", ECAPA_DIM, args.work_dir / f"{split}_ecapa.npy")
    mfcc, mfcc_labels, mfcc_counts = read_cache(args.mfcc_cache_dir, split, MFCC_DIM, args.work_dir / f"{split}_mfcc.npy")
    if not np.array_equal(labels, mfcc_labels) or not np.array_equal(counts, mfcc_counts):
        raise ValueError(f"[{split}] ECAPA와 MFCC 캐시의 발화 정렬이 다릅니다.")
    return {"ecapa": ecapa, "mfcc": mfcc, "labels": labels, "counts": counts, "offsets": np.concatenate([[0], np.cumsum(counts)])}


def mfcc_scale(data: dict, calls: np.ndarray) -> dict[str, np.ndarray]:
    """학습 통화의 MFCC 평균·표준편차 (입력 표준화용, 체크포인트에 저장된다)."""
    total = np.zeros(MFCC_DIM, dtype=np.float64)
    square = np.zeros(MFCC_DIM, dtype=np.float64)
    rows = 0
    for call in calls:
        block = np.asarray(data["mfcc"][call_rows(data, call)], dtype=np.float64)
        total += block.sum(axis=0)
        square += (block ** 2).sum(axis=0)
        rows += len(block)
    mean = total / rows
    std = np.sqrt(np.maximum(square / rows - mean ** 2, 1e-12))
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


def call_rows(data: dict, call: int) -> slice:
    return slice(int(data["offsets"][call]), int(data["offsets"][call + 1]))


def make_call(data: dict, call: int, scale: dict, keep: np.ndarray | None = None) -> np.ndarray:
    """통화 하나의 입력 행렬 (model.call_inputs). keep: 학습 때 무작위로 고른 발화 위치."""
    rows = call_rows(data, call)
    ecapa = np.asarray(data["ecapa"][rows], dtype=np.float32)
    mfcc = np.asarray(data["mfcc"][rows], dtype=np.float32)
    if keep is not None:
        ecapa, mfcc = ecapa[keep], mfcc[keep]
    return call_inputs(ecapa, mfcc, scale)


def evaluate(model, data: dict, calls: np.ndarray, scale: dict, device: str) -> tuple[float, np.ndarray]:
    """주어진 통화 전체의 정확도와 발화별 확률(calls 순서로 이어 붙임). 메모리를 아끼려고 500통화씩 나눈다."""
    pieces: list[np.ndarray] = []
    for begin in range(0, len(calls), 500):
        pieces.extend(predict_calls(model, [make_call(data, call, scale) for call in calls[begin:begin + 500]], device))
    probabilities = np.concatenate(pieces)
    labels = np.concatenate([data["labels"][call_rows(data, call)] for call in calls])
    return float(((probabilities >= 0.5) == labels).mean()), probabilities


# ---------------------------------------------------------------- 학습

def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    total_start = start = time.perf_counter()
    train = load_split(args, "train")
    val = load_split(args, "val")
    print(f"캐시 로딩: Train 발화 {len(train['labels']):,} / 통화 {len(train['counts']):,}, "
          f"Validation 발화 {len(val['labels']):,} / 통화 {len(val['counts']):,} ({elapsed(start)})")

    # Train 통화를 섞어 앞 10%를 holdout(epoch 선택용)으로 뗀다.
    calls = rng.permutation(len(train["counts"]))
    held = np.sort(calls[:int(len(calls) * args.holdout)])
    fit = np.sort(calls[int(len(calls) * args.holdout):])
    fit = fit[train["counts"][fit] > 0]
    scale = mfcc_scale(train, fit)
    config = {"in_dim": INPUT_DIM, "d_model": args.d_model, "layers": args.layers,
              "heads": args.heads, "ff": args.ff, "dropout": args.dropout}
    if args.pair_weight > 0:
        config["pair_dim"] = args.pair_dim
    model = build_model(config).to(device)
    print(f"모델: {config}, 파라미터 {sum(p.numel() for p in model.parameters()):,}개, 장치 {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 첫 epoch 동안 선형 warmup 후 cosine으로 줄인다.
    steps_per_epoch = int(np.ceil(len(fit) / args.batch_calls))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / steps_per_epoch) * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)))
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def save(path: Path, state: dict, epoch: int, holdout: float | None, **extra) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_type": "mission2_call_transformer", "config": config, "state_dict": state, "scale": scale,
                    "best_epoch": epoch, "holdout_accuracy": holdout, **extra}, path)

    best_accuracy, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, (args.run_epochs or args.epochs) + 1):
        start = time.perf_counter()
        model.train()
        order = rng.permutation(fit)
        running = 0.0
        for begin in range(0, len(order), args.batch_calls):
            batch_inputs, batch_labels = [], []
            for call in order[begin:begin + args.batch_calls]:
                size = int(train["counts"][call])
                keep = np.arange(size)
                # 증강: 발화 일부를 무작위로 빼고, 긴 통화는 무작위로 골라 줄이고, 차례를 섞는다.
                # 통화 크기에 덜 민감하게 만들기 위한 것이며 순서 정보를 만들지 않는다.
                if size > 2 and args.drop_utterances > 0:
                    keep = rng.choice(size, size=max(2, size - int(rng.integers(0, int(size * args.drop_utterances) + 1))), replace=False)
                if len(keep) > args.max_call_len:
                    keep = rng.choice(keep, size=args.max_call_len, replace=False)
                keep = rng.permutation(keep)
                batch_inputs.append(make_call(train, call, scale, keep))
                batch_labels.append(train["labels"][call_rows(train, call)][keep])
            # 길이가 다른 통화를 padding으로 맞춰 한 batch로 만든다(padding 자리는 attention·손실에서 빠진다).
            longest = max(len(item) for item in batch_inputs)
            x = np.zeros((len(batch_inputs), longest, INPUT_DIM), dtype=np.float32)
            y = np.zeros((len(batch_inputs), longest), dtype=np.float32)
            padding = np.ones((len(batch_inputs), longest), dtype=bool)
            for row, (item, label) in enumerate(zip(batch_inputs, batch_labels)):
                x[row, :len(item)] = item
                y[row, :len(item)] = label
                padding[row, :len(item)] = False
            x_t, y_t, pad_t = (torch.from_numpy(a).to(device) for a in (x, y, padding))
            logits, hidden = model(x_t, pad_t, return_hidden=True)
            valid = (~pad_t).float()
            loss = (loss_fn(logits, y_t) * valid).sum() / valid.sum()
            if args.pair_weight > 0:
                # 같은 화자 보조 과제: 통화 안 모든 발화 쌍에 대해 "같은 사람인가"를 맞힌다.
                # 통화에는 화자가 둘뿐이므로, 은닉 표현이 두 목소리를 가르도록 직접 학습시킨다. 쌍은 대칭이라 순서와 무관하다.
                unit = torch.nn.functional.normalize(model.pair(hidden), dim=-1)
                pair_logits = args.pair_scale * unit @ unit.transpose(1, 2)
                same = (y_t[:, :, None] == y_t[:, None, :]).float()
                pair_mask = valid[:, :, None] * valid[:, None, :]
                pair_mask = pair_mask * (1 - torch.eye(longest, device=pair_mask.device))[None]
                pair_loss = (loss_fn(pair_logits, same) * pair_mask).sum() / pair_mask.sum().clamp(min=1)
                loss = loss + args.pair_weight * pair_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(batch_inputs)
        if len(held) == 0:
            # holdout이 없으면 고를 기준이 없으므로 마지막 epoch 상태를 쓴다.
            best_accuracy, best_epoch = float("nan"), epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save(args.out.with_suffix(".best.pt"), best_state, epoch, None)
            print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, holdout 없음 (Train 전체 학습) ({elapsed(start)})")
            continue
        held_accuracy, _ = evaluate(model, train, held, scale, device)
        marker = ""
        if held_accuracy > best_accuracy:
            best_accuracy, best_epoch, waited = held_accuracy, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = " ← 최고"
            # 학습이 중간에 끊겨도 그때까지 가장 좋았던 모델이 남도록 매번 저장한다.
            save(args.out.with_suffix(".best.pt"), best_state, epoch, held_accuracy)
        else:
            waited += 1
        print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, Train holdout Accuracy {held_accuracy:.4f}{marker} ({elapsed(start)})")
        if waited >= args.patience:
            print(f"holdout이 {args.patience}회 연속 나아지지 않아 멈춥니다.")
            break

    # 학습이 끝난 뒤 Validation 정확도를 한 번만 계산한다.
    model.load_state_dict(best_state)
    val_accuracy, val_probability = evaluate(model, val, np.arange(len(val["counts"])), scale, device)
    print(f"최고 epoch {best_epoch}: Train holdout Accuracy {best_accuracy:.4f}")
    print(f"Validation Accuracy (트랜스포머): {val_accuracy:.4f}")
    for name, value in (("119대원(0)", 0), ("신고자(1)", 1)):
        mask = val["labels"] == value
        print(f"  {name} recall {((val_probability >= 0.5)[mask] == value).mean():.4f}")
    save(args.out, best_state, best_epoch, best_accuracy, validation_accuracy=val_accuracy,
         args={key: str(value) for key, value in vars(args).items()})
    np.save(args.out.with_suffix(".val_probability.npy"), val_probability)
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump({"holdout": best_accuracy, "validation": val_accuracy, "best_epoch": best_epoch, "config": config},
                  file, ensure_ascii=False, indent=1)
    print(f"저장: {args.out} / 전체 소요 {elapsed(total_start)}")


if __name__ == "__main__":
    main()
