"""Mission 1 통화 단위 성별 트랜스포머 학습 (gender_call_model.py 참고). 캐시만 읽어 CPU로 돈다.

모델 선택(early stopping)은 예은과 같은 내부 holdout 2,920통화로 한다. 그래야 예은 모델의
holdout 확률과 같은 통화로 섞어 볼 수 있다. Validation은 학습에 쓰지 않고 마지막에 한 번 판정만 한다.
`--holdout_mode none`이면 Train 29,200통화 전체로 `--epochs`만큼 학습하고 마지막 상태를 쓴다
(epoch 수는 holdout 실험으로 정한다).

결과: `{out}.pt`(모델), `{out}.npz`(holdout·Validation 통화별 stem과 F 확률), `{out}.json`(요약).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from gender_call_model import INPUT_DIM, MFCC_DIM, batch_arrays, build_model, call_inputs, predict_calls
from gender_data import load_split, yeeun_holdout_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 1 통화 단위 성별 트랜스포머")
    parser.add_argument("--m2_local", type=Path, required=True, help="mission2_local 폴더 (캐시·memmap 읽기 전용)")
    parser.add_argument("--export_dir", type=Path, required=True, help="labels_train/val.npz와 yeeun/ 폴더")
    parser.add_argument("--out", type=Path, required=True, help="결과 경로 접두어 (.pt/.npz/.json)")
    parser.add_argument("--holdout_mode", choices=("yeeun", "none"), default="yeeun")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_calls", type=int, default=32)
    parser.add_argument("--max_call_len", type=int, default=96)
    parser.add_argument("--drop_utterances", type=float, default=0.2)
    parser.add_argument("--token_weight", type=float, default=0.3, help="신고자 발화별 성별 보조 손실 가중치")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def elapsed(start: float) -> str:
    seconds = int(time.perf_counter() - start)
    return f"{seconds // 60}분 {seconds % 60}초"


def mfcc_scale(data: dict, calls: np.ndarray) -> dict[str, np.ndarray]:
    total = np.zeros(MFCC_DIM, dtype=np.float64)
    square = np.zeros(MFCC_DIM, dtype=np.float64)
    rows = 0
    for call in calls:
        block = np.asarray(data["mfcc"][data["offsets"][call]:data["offsets"][call + 1]], dtype=np.float64)
        total += block.sum(axis=0)
        square += (block ** 2).sum(axis=0)
        rows += len(block)
    mean = total / rows
    std = np.sqrt(np.maximum(square / rows - mean ** 2, 1e-12))
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


def make_call(data: dict, call: int, scale: dict, keep: np.ndarray | None = None) -> np.ndarray:
    span = slice(int(data["offsets"][call]), int(data["offsets"][call + 1]))
    ecapa, mfcc, role = data["ecapa"][span], data["mfcc"][span], data["speaker"][span]
    if keep is not None:
        ecapa, mfcc, role = ecapa[keep], mfcc[keep], role[keep]
    return call_inputs(ecapa, mfcc, role, scale)


def augment_keep(role: np.ndarray, rng: np.random.Generator, drop: float, longest: int) -> np.ndarray:
    """발화 일부를 무작위로 빼고 긴 통화는 줄인다. 신고자 발화는 최소 1개 남긴다. 순서는 섞는다."""
    size = len(role)
    keep = np.arange(size)
    if size > 2 and drop > 0:
        keep = rng.choice(size, size=max(2, size - int(rng.integers(0, int(size * drop) + 1))), replace=False)
    if len(keep) > longest:
        keep = rng.choice(keep, size=longest, replace=False)
    if not (role[keep] == 1).any():
        keep = np.append(keep[1:], rng.choice(np.flatnonzero(role == 1)))
    return rng.permutation(keep)


def evaluate(model, data: dict, calls: np.ndarray, scale: dict) -> tuple[float, np.ndarray]:
    probabilities = np.zeros(len(calls), dtype=np.float64)
    for begin in range(0, len(calls), 500):
        chunk = calls[begin:begin + 500]
        probabilities[begin:begin + len(chunk)] = predict_calls(model, [make_call(data, c, scale) for c in chunk])
    return float(((probabilities >= 0.5) == data["gender"][calls]).mean()), probabilities


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    total_start = time.perf_counter()
    train = load_split(args.m2_local, args.export_dir, "train")
    val = load_split(args.m2_local, args.export_dir, "val")
    held_mask = yeeun_holdout_mask(train["stems"], train["gender"], args.export_dir)
    if args.holdout_mode == "none":
        held_mask[:] = False
    held = np.flatnonzero(held_mask)
    fit = np.flatnonzero(~held_mask)
    print(f"Train 학습 {len(fit):,}통화 / holdout {len(held):,}통화 / Validation {len(val['counts']):,}통화")
    scale = mfcc_scale(train, fit)
    config = {"in_dim": INPUT_DIM, "d_model": args.d_model, "layers": args.layers,
              "heads": args.heads, "ff": args.ff, "dropout": args.dropout}
    model = build_model(config)
    print(f"모델: {config}, 파라미터 {sum(p.numel() for p in model.parameters()):,}개")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(fit) / args.batch_calls))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / steps_per_epoch)
        * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)))
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def checkpoint(state, epoch, held_accuracy) -> dict:
        return {"model_type": "mission1_gender_call_transformer", "config": config, "state_dict": state,
                "scale": scale, "best_epoch": epoch, "holdout_accuracy": held_accuracy}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy, best_loss, best_state, best_epoch, waited = -1.0, np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        order = rng.permutation(fit)
        running = 0.0
        for begin in range(0, len(order), args.batch_calls):
            inputs = []
            for call in order[begin:begin + args.batch_calls]:
                role = train["speaker"][train["offsets"][call]:train["offsets"][call + 1]]
                keep = augment_keep(role, rng, args.drop_utterances, args.max_call_len)
                inputs.append(make_call(train, call, scale, keep))
            x, padding, caller = batch_arrays(inputs)
            target = torch.from_numpy(train["gender"][order[begin:begin + args.batch_calls]].astype(np.float32))
            x_t, pad_t, caller_t = (torch.from_numpy(a) for a in (x, padding, caller))
            logits, token_logits = model(x_t, pad_t, caller_t)
            loss = loss_fn(logits, target).mean()
            if args.token_weight > 0:
                weight = caller_t.float()
                token_loss = (loss_fn(token_logits, target[:, None].expand_as(token_logits)) * weight).sum() / weight.sum()
                loss = loss + args.token_weight * token_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(inputs)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if len(held) == 0:
            best_state, best_epoch = state, epoch
            torch.save(checkpoint(state, epoch, None), args.out.with_suffix(".best.pt"))
            print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, holdout 없음 ({elapsed(start)})", flush=True)
            continue
        held_accuracy, held_probability = evaluate(model, train, held, scale)
        clipped = np.clip(held_probability, 1e-6, 1 - 1e-6)
        target = train["gender"][held]
        held_loss = float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean())
        marker = ""
        # 정확도는 2,920통화라 계단이 크다(1통화 0.034%p). 동률이면 log loss가 낮은 쪽을 고른다.
        if held_accuracy > best_accuracy or (held_accuracy == best_accuracy and held_loss < best_loss):
            best_accuracy, best_loss, best_state, best_epoch, waited = held_accuracy, held_loss, state, epoch, 0
            marker = " ← 최고"
            torch.save(checkpoint(state, epoch, held_accuracy), args.out.with_suffix(".best.pt"))
        else:
            waited += 1
        print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, holdout Acc {held_accuracy:.4f} "
              f"(오류 {int(round((1 - held_accuracy) * len(held)))}) logloss {held_loss:.4f}{marker} ({elapsed(start)})", flush=True)
        if waited >= args.patience:
            print(f"holdout이 {args.patience}회 연속 나아지지 않아 멈춥니다.")
            break

    model.load_state_dict(best_state)
    held_probability = evaluate(model, train, held, scale)[1] if len(held) else np.zeros(0)
    val_calls = np.arange(len(val["counts"]))
    val_accuracy, val_probability = evaluate(model, val, val_calls, scale)
    print(f"최고 epoch {best_epoch}: holdout Acc {best_accuracy:.4f}")
    print(f"Validation Acc {val_accuracy:.4f} (오류 {int(round((1 - val_accuracy) * len(val_calls)))}/{len(val_calls)})")
    final = checkpoint(best_state, best_epoch, best_accuracy if len(held) else None)
    final.update(validation_accuracy=val_accuracy, args={k: str(v) for k, v in vars(args).items()})
    torch.save(final, args.out.with_suffix(".pt"))
    np.savez(args.out.with_suffix(".npz"),
             holdout_stems=train["stems"][held], holdout_prob=held_probability, holdout_y=train["gender"][held],
             val_stems=val["stems"], val_prob=val_probability, val_y=val["gender"])
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump({"holdout": best_accuracy if len(held) else None, "validation": val_accuracy,
                   "best_epoch": best_epoch, "config": config,
                   "args": {k: str(v) for k, v in vars(args).items()}}, file, ensure_ascii=False, indent=1)
    print(f"저장: {args.out}.* / 전체 소요 {elapsed(total_start)}")


if __name__ == "__main__":
    main()
