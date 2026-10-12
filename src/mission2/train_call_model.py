"""Mission 2 통화 단위 트랜스포머 학습/검증 (call_model.py 참고).

이미 만든 캐시만 읽는다. WAV는 읽지 않으므로 Colab GPU 없이 CPU에서도 돌아간다.
- ECAPA 임베딩 캐시 (`embed_train_*`, `embed_val_*`)
- MFCC 특징 캐시 (`train_*`, `val_*`, 예전 캐시면 `{split}_counts.npy`)
- (선택) 1단계 캐시 joblib의 out-of-fold / Validation 1단계 확률 → 입력에 쌓기
- (선택) 기존 파이프라인의 Validation 예측 npz → 같은 가중치(0.5)로 섞은 정확도

캐시는 **읽기만** 한다. train_mfcc_rf.load_shards는 설정이 다르면 shard를 지우므로 쓰지 않는다.
특징은 `--work_dir`의 디스크 배열(memmap)에 옮겨 두고 필요한 통화만 읽어, RAM이 적어도 돈다.

모델 선택은 Train 통화 10%를 떼어 낸 holdout으로 한다. Validation은 마지막에 한 번만 본다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

from call_model import ECAPA_DIM, MFCC_DIM, build_model, call_inputs, input_width, predict_calls

SECOND_DIM = 512  # WavLM-Base-Plus-SV x-vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission 2 통화 단위 트랜스포머")
    parser.add_argument("--embed_cache_dir", type=Path, required=True)
    parser.add_argument("--mfcc_cache_dir", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True, help="디스크 배열을 둘 폴더")
    parser.add_argument("--out", type=Path, required=True, help="모델 저장 경로 (.pt)")
    parser.add_argument("--stage1_cache", type=Path, default=None, help="1단계 캐시 joblib. 주면 1단계 확률을 입력에 쌓는다.")
    parser.add_argument("--second_cache_dir", type=Path, default=None,
                        help="WavLM 임베딩 캐시(wavlm_train_*, wavlm_val_*). 주면 두 번째 목소리 지문을 입력에 더한다.")
    parser.add_argument("--base_val_predictions", type=Path, default=None,
                        help="기존 파이프라인 *_val_predictions.npz. 주면 0.5:0.5로 섞은 정확도도 낸다.")
    parser.add_argument("--kind", choices=("transformer", "mlp"), default="transformer",
                        help="mlp: 발화마다 따로 판정하는 MLP (발화 사이에 정보가 오가지 않음)")
    parser.add_argument("--no_call_context", dest="call_context", action="store_false",
                        help="통화 평균을 뺀 값을 입력에 넣지 않는다. --kind mlp와 함께 쓰면 각 발화를 완전히 독립적으로 판정한다")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_calls", type=int, default=32)
    parser.add_argument("--max_call_len", type=int, default=96, help="학습 때 이보다 긴 통화는 발화를 무작위로 골라 쓴다")
    parser.add_argument("--drop_utterances", type=float, default=0.2, help="학습 때 통화마다 최대 이 비율의 발화를 무작위로 뺀다")
    parser.add_argument("--holdout", type=float, default=0.1,
                        help="모델 선택용으로 떼어 낼 Train 통화 비율. 0이면 Train 전체로 --epochs만큼 학습하고 마지막 모델을 쓴다"
                             " (epoch 수를 holdout 실험으로 정한 뒤 데이터를 다 쓰는 용도)")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pair_weight", type=float, default=0.0,
                        help="같은 화자 보조 과제 손실 가중치. 0이면 끈다(EXP-011 A0~A3, F0)")
    parser.add_argument("--pair_dim", type=int, default=64, help="같은 화자 보조 과제의 투영 차원")
    parser.add_argument("--pair_scale", type=float, default=10.0, help="코사인 유사도에 곱하는 배율(logit 크기)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def elapsed(start: float) -> str:
    seconds = int(time.perf_counter() - start)
    return f"{seconds // 3600}시간 {seconds % 3600 // 60}분 {seconds % 60}초"


def read_cache(cache_dir: Path, split: str, width: int, target: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """shard를 읽기만 해서 디스크 배열(float16)로 옮긴다. 정답과 통화 경계를 함께 돌려준다."""
    meta = cache_dir / f"{split}_meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"캐시 설정 파일이 없습니다: {meta}")
    paths = []
    for index in range(10_000):
        path = cache_dir / f"{split}_{index:05d}.npz"
        if not path.is_file():
            break
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"shard가 없습니다: {cache_dir}/{split}_00000.npz")
    sizes, labels, counts, calls, complete = [], [], [], 0, True
    for path in paths:
        with np.load(path) as data:
            sizes.append(len(data["y"]))
            labels.append(data["y"].astype(np.int64))
            calls += int(data["calls"])
            if "counts" in data.files:
                counts.append(data["counts"].astype(np.int64))
            else:
                complete = False
    rows = int(sum(sizes))
    if complete:
        boundary = np.concatenate(counts)
    else:
        saved = cache_dir / f"{split}_counts.npy"
        if not saved.is_file():
            raise FileNotFoundError(f"예전 캐시인데 통화 경계 파일이 없습니다: {saved} (train_mfcc_rf.py를 한 번 돌리면 생깁니다)")
        boundary = np.load(saved).astype(np.int64)
    if len(boundary) != calls or int(boundary.sum()) != rows:
        raise ValueError(f"[{split}] 통화 경계가 캐시와 맞지 않습니다: 통화 {len(boundary)} vs {calls}, 발화 {int(boundary.sum())} vs {rows}")
    target.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(rows, width))
    start = 0
    for path, size in zip(paths, sizes):
        with np.load(path) as data:
            block = data["x"]
            if block.shape[1] != width:
                raise ValueError(f"{path.name}: 차원 {block.shape[1]} != {width}")
            output[start:start + size] = block.astype(np.float16)
        start += size
    output.flush()
    del output
    return np.load(target, mmap_mode="r"), np.concatenate(labels), boundary


def load_split(args, split: str) -> dict:
    """ECAPA와 MFCC를 같은 발화 순서로 맞춰 읽는다. 어긋나면 멈춘다."""
    ecapa, labels, counts = read_cache(args.embed_cache_dir, f"embed_{split}", ECAPA_DIM, args.work_dir / f"{split}_ecapa.npy")
    mfcc, mfcc_labels, mfcc_counts = read_cache(args.mfcc_cache_dir, split, MFCC_DIM, args.work_dir / f"{split}_mfcc.npy")
    if not np.array_equal(labels, mfcc_labels) or not np.array_equal(counts, mfcc_counts):
        raise ValueError(f"[{split}] ECAPA와 MFCC 캐시의 발화 정렬이 다릅니다.")
    offsets = np.concatenate([[0], np.cumsum(counts)])
    second = None
    if args.second_cache_dir is not None:
        second, second_labels, second_counts = read_cache(
            args.second_cache_dir, f"wavlm_{split}", SECOND_DIM, args.work_dir / f"{split}_wavlm.npy")
        if not np.array_equal(labels, second_labels) or not np.array_equal(counts, second_counts):
            raise ValueError(f"[{split}] ECAPA와 WavLM 캐시의 발화 정렬이 다릅니다.")
    return {"ecapa": ecapa, "mfcc": mfcc, "second": second, "labels": labels, "counts": counts, "offsets": offsets}


def mfcc_scale(data: dict, calls: np.ndarray) -> dict[str, np.ndarray]:
    """학습 통화의 MFCC 평균·표준편차. 조각씩 읽어 메모리를 아낀다."""
    total = np.zeros(MFCC_DIM, dtype=np.float64)
    square = np.zeros(MFCC_DIM, dtype=np.float64)
    rows = 0
    for begin in range(0, len(calls), 2000):
        for call in calls[begin:begin + 2000]:
            block = np.asarray(data["mfcc"][data["offsets"][call]:data["offsets"][call + 1]], dtype=np.float64)
            total += block.sum(axis=0)
            square += (block ** 2).sum(axis=0)
            rows += len(block)
    mean = total / rows
    std = np.sqrt(np.maximum(square / rows - mean ** 2, 1e-12))
    return {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}


def call_rows(data: dict, call: int) -> slice:
    return slice(int(data["offsets"][call]), int(data["offsets"][call + 1]))


def make_call(data: dict, call: int, scale: dict, stage1: np.ndarray | None, keep: np.ndarray | None = None,
              context: bool = True) -> np.ndarray:
    rows = call_rows(data, call)
    ecapa = np.asarray(data["ecapa"][rows], dtype=np.float32)
    mfcc = np.asarray(data["mfcc"][rows], dtype=np.float32)
    second = None if data.get("second") is None else np.asarray(data["second"][rows], dtype=np.float32)
    probability = None if stage1 is None else stage1[rows]
    if keep is not None:
        ecapa, mfcc = ecapa[keep], mfcc[keep]
        second = None if second is None else second[keep]
        probability = None if probability is None else probability[keep]
    return call_inputs(ecapa, mfcc, scale, probability, second, context)


def evaluate(model, data: dict, calls: np.ndarray, scale: dict, stage1, device: str,
             context: bool = True) -> tuple[float, np.ndarray]:
    """주어진 통화 전체를 판정해 정확도와 발화별 확률을 낸다 (확률은 calls 순서로 이어 붙임).

    입력을 한꺼번에 만들면 Validation만 약 400MB라, 통화 500개씩 나눠 만든다."""
    pieces: list[np.ndarray] = []
    for begin in range(0, len(calls), 500):
        inputs = [make_call(data, call, scale, stage1, None, context) for call in calls[begin:begin + 500]]
        pieces.extend(predict_calls(model, inputs, device))
    probabilities = np.concatenate(pieces)
    labels = np.concatenate([data["labels"][call_rows(data, call)] for call in calls])
    return float(((probabilities >= 0.5) == labels).mean()), probabilities


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    total_start = time.perf_counter()
    start = time.perf_counter()
    train = load_split(args, "train")
    val = load_split(args, "val")
    print(f"캐시 로딩: Train 발화 {len(train['labels']):,} / 통화 {len(train['counts']):,}, "
          f"Validation 발화 {len(val['labels']):,} / 통화 {len(val['counts']):,} ({elapsed(start)})")

    stage1_train = stage1_val = None
    if args.stage1_cache is not None:
        cached = joblib.load(args.stage1_cache)
        stage1_train = np.asarray(cached["out_of_fold"], dtype=np.float64)
        stage1_val = np.asarray(cached["val_stage1"], dtype=np.float64)
        if len(stage1_train) != len(train["labels"]) or len(stage1_val) != len(val["labels"]):
            raise ValueError("1단계 캐시의 행 수가 임베딩 캐시와 다릅니다.")
        print(f"1단계 확률을 입력에 쌓음: Train OOF {((stage1_train >= 0.5) == train['labels']).mean():.4f}, "
              f"Validation {((stage1_val >= 0.5) == val['labels']).mean():.4f}")

    calls = rng.permutation(len(train["counts"]))
    held = np.sort(calls[:int(len(calls) * args.holdout)])
    fit = np.sort(calls[int(len(calls) * args.holdout):])
    fit = fit[train["counts"][fit] > 0]
    scale = mfcc_scale(train, fit)
    config = {
        "in_dim": input_width(stage1_train is not None, SECOND_DIM if train["second"] is not None else 0, args.call_context),
        "kind": args.kind, "context": args.call_context,
        "d_model": args.d_model, "layers": args.layers,
        "heads": args.heads, "ff": args.ff, "dropout": args.dropout,
    }
    if args.pair_weight > 0:
        config["pair_dim"] = args.pair_dim
    model = build_model(config).to(device)
    print(f"모델: {config}, 파라미터 {sum(p.numel() for p in model.parameters()):,}개, 장치 {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(fit) / args.batch_calls))
    total_steps = steps_per_epoch * args.epochs
    warmup = steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)))
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    best_accuracy, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        order = rng.permutation(fit)
        running = 0.0
        for begin in range(0, len(order), args.batch_calls):
            batch_inputs, batch_labels = [], []
            for call in order[begin:begin + args.batch_calls]:
                size = int(train["counts"][call])
                keep = np.arange(size)
                # 발화 일부를 빼거나 긴 통화를 줄이는 것은 통화 크기에 덜 민감하게 만들기 위한 증강이다.
                # 고르는 발화는 무작위이며 순서 정보를 만들지 않는다.
                if size > 2 and args.drop_utterances > 0:
                    keep = rng.choice(size, size=max(2, size - int(rng.integers(0, int(size * args.drop_utterances) + 1))), replace=False)
                if len(keep) > args.max_call_len:
                    keep = rng.choice(keep, size=args.max_call_len, replace=False)
                keep = rng.permutation(keep)
                batch_inputs.append(make_call(train, call, scale, stage1_train, keep, args.call_context))
                batch_labels.append(train["labels"][call_rows(train, call)][keep])
            longest = max(len(item) for item in batch_inputs)
            x = np.zeros((len(batch_inputs), longest, config["in_dim"]), dtype=np.float32)
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
                # 통화에는 화자가 둘뿐이므로, 은닉 표현이 통화 안에서 두 목소리를 가르도록 직접 학습시킨다.
                # 쌍은 대칭이고 순서와 무관하다. 정답(speaker)은 학습에만 쓴다.
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
            # holdout이 없으면 고를 기준이 없으므로 매 epoch 마지막 상태를 최고로 둔다.
            held_accuracy = float("nan")
            best_accuracy, best_epoch = held_accuracy, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_type": "mission2_call_transformer", "config": config, "state_dict": best_state,
                        "scale": scale, "use_stage1": stage1_train is not None, "use_second": train["second"] is not None, "context": args.call_context, "best_epoch": epoch,
                        "holdout_accuracy": None}, args.out.with_suffix(".best.pt"))
            print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, holdout 없음 (Train 전체 학습) ({elapsed(start)})")
            continue
        held_accuracy, _ = evaluate(model, train, held, scale, stage1_train, device, args.call_context)
        marker = ""
        if held_accuracy > best_accuracy:
            best_accuracy, best_epoch, waited = held_accuracy, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = " ← 최고"
            # 학습이 중간에 끊겨도 그때까지 가장 좋았던 모델이 남도록 매번 저장한다.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_type": "mission2_call_transformer", "config": config, "state_dict": best_state,
                        "scale": scale, "use_stage1": stage1_train is not None, "use_second": train["second"] is not None, "context": args.call_context, "best_epoch": epoch,
                        "holdout_accuracy": held_accuracy}, args.out.with_suffix(".best.pt"))
        else:
            waited += 1
        print(f"[epoch {epoch:2d}] loss {running / len(fit):.4f}, Train holdout Accuracy {held_accuracy:.4f}{marker} ({elapsed(start)})")
        if waited >= args.patience:
            print(f"holdout이 {args.patience}회 연속 나아지지 않아 멈춥니다.")
            break

    model.load_state_dict(best_state)
    val_calls = np.arange(len(val["counts"]))
    val_accuracy, val_probability = evaluate(model, val, val_calls, scale, stage1_val, device, args.call_context)
    print(f"최고 epoch {best_epoch}: Train holdout Accuracy {best_accuracy:.4f}")
    if stage1_train is not None and len(held):
        held_rows = np.concatenate([np.arange(*call_rows(train, call).indices(len(train["labels"]))) for call in held])
        print(f"  같은 holdout의 1단계 LightGBM Accuracy: {((stage1_train[held_rows] >= 0.5) == train['labels'][held_rows]).mean():.4f}")
    print(f"Validation Accuracy (트랜스포머): {val_accuracy:.4f}")
    labels = val["labels"]
    for name, value in (("119대원(0)", 0), ("신고자(1)", 1)):
        mask = labels == value
        print(f"  {name} recall {((val_probability >= 0.5)[mask] == value).mean():.4f}")
    blend_accuracy = None
    if args.base_val_predictions is not None:
        with np.load(args.base_val_predictions) as base:
            if not np.array_equal(base["labels"], labels) or not np.array_equal(base["counts"], val["counts"]):
                raise ValueError("기존 파이프라인 예측의 발화 정렬이 다릅니다.")
            base_probability = base["stage2"]
        base_accuracy = float(((base_probability >= 0.5) == labels).mean())
        blend = 0.5 * base_probability + 0.5 * val_probability
        blend_accuracy = float(((blend >= 0.5) == labels).mean())
        print(f"Validation Accuracy (기존 파이프라인): {base_accuracy:.4f}")
        print(f"Validation Accuracy (0.5:0.5 평균): {blend_accuracy:.4f}  ({(blend_accuracy - base_accuracy) * 100:+.2f}%p)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "mission2_call_transformer", "config": config, "state_dict": best_state,
        "scale": scale, "use_stage1": stage1_train is not None, "use_second": train["second"] is not None, "context": args.call_context, "best_epoch": best_epoch,
        "holdout_accuracy": best_accuracy, "validation_accuracy": val_accuracy, "blend_accuracy": blend_accuracy,
        "args": {key: str(value) for key, value in vars(args).items()},
    }, args.out)
    np.save(args.out.with_suffix(".val_probability.npy"), val_probability)
    print(f"저장: {args.out} / 전체 소요 {elapsed(total_start)}")
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump({"holdout": best_accuracy, "validation": val_accuracy, "blend": blend_accuracy,
                   "best_epoch": best_epoch, "config": config}, file, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
