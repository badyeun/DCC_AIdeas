"""예은 3모델 앙상블(40:40:20)과 새 모델의 확률을 같은 통화끼리 섞어 본다.

- 예은 holdout 확률(npz의 `val_idx`)은 stem 오름차순 Train 인덱스다 → stem으로 바꿔 맞춘다.
- 예은 공식 Validation 확률은 `filenames`(…wav)로 맞춘다.
- 섞는 비율은 **holdout으로만** 고르고, Validation은 그 비율에서 한 번 확인한다
  (참고용으로 Validation 전체 곡선도 출력하지만 선택에는 쓰지 않는다).
- 정답이 예은 파일과 우리 라벨에서 같은지 확인한다(정렬 검증).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

YEEUN_WEIGHTS = (0.4, 0.4, 0.2)  # Attention+F0, Stats+F0, Stats(No F0)


def load_yeeun(export_dir: Path) -> dict:
    folder = export_dir / "yeeun"
    with np.load(export_dir / "labels_train.npz") as data:
        sorted_stems = np.sort(data["stems"].astype(str))
    held = {}
    names = ("internal_attention_f0_seed_42_result.npz", "internal_stats_f0_attention_seed42_result.npz",
             "internal_stats_attention_seed42_result.npz")
    probs, index, truth = [], None, None
    for name in names:
        with np.load(folder / name) as data:
            if index is None:
                index, truth = data["val_idx"].copy(), data["y_true"].astype(int)
            elif not np.array_equal(index, data["val_idx"]):
                raise ValueError(f"{name}: holdout 인덱스 불일치")
            probs.append(data["probs"][:, 1].astype(np.float64))
    held["stems"] = sorted_stems[index]
    held["y"] = truth
    held["models"] = np.stack(probs)
    with np.load(folder / "official_validation_stats_results.npz") as data:
        val_names = np.asarray([Path(str(n)).stem for n in data["filenames"]])
        val_y = data["y_true"].astype(int)
        stats_f0 = data["stats_f0_probs"][:, 1].astype(np.float64)
        stats = data["stats_probs"][:, 1].astype(np.float64)
    with np.load(folder / "official_validation_f0_comparison.npz") as data:
        f0_names = np.asarray([Path(str(n)).stem for n in data["filenames"]])
        lookup = {name: i for i, name in enumerate(f0_names)}
        attention_f0 = data["new_f0_probs"][:, 1].astype(np.float64)[[lookup[n] for n in val_names]]
    val = {"stems": val_names, "y": val_y, "models": np.stack([attention_f0, stats_f0, stats])}
    for part in (held, val):
        part["ensemble"] = np.tensordot(YEEUN_WEIGHTS, part["models"], axes=1)
    return {"holdout": held, "val": val}


def align(stems: np.ndarray, source_stems: np.ndarray, values: np.ndarray) -> np.ndarray:
    lookup = {str(name): i for i, name in enumerate(source_stems)}
    missing = [s for s in stems if s not in lookup]
    if missing:
        raise ValueError(f"새 모델 결과에 없는 통화 {len(missing)}개 (예: {missing[:3]})")
    return values[[lookup[s] for s in stems]]


def errors(probability: np.ndarray, y: np.ndarray) -> int:
    return int(((probability >= 0.5) != y).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="새 모델 결과 npz들 (여러 개면 평균)")
    args = parser.parse_args()
    yeeun = load_yeeun(args.export_dir)
    new = {}
    for split in ("holdout", "val"):
        part = yeeun[split]
        columns = []
        for run in args.runs:
            with np.load(run) as data:
                truth = align(part["stems"], data[f"{split}_stems"].astype(str), data[f"{split}_y"])
                if not np.array_equal(truth, part["y"]):
                    raise ValueError(f"{run.name} {split}: 정답이 예은 파일과 다릅니다 (정렬 오류)")
                columns.append(align(part["stems"], data[f"{split}_stems"].astype(str), data[f"{split}_prob"]))
        new[split] = np.mean(columns, axis=0)
        print(f"[{split}] 통화 {len(part['y'])}: 예은 앙상블 오류 {errors(part['ensemble'], part['y'])}, "
              f"예은 개별 {[errors(m, part['y']) for m in part['models']]}, "
              f"새 모델 오류 {errors(new[split], part['y'])} (개별 {[errors(c, part['y']) for c in columns]})")
        wrong_y = (part["ensemble"] >= 0.5) != part["y"]
        wrong_n = (new[split] >= 0.5) != part["y"]
        print(f"   겹침: 둘 다 오류 {int((wrong_y & wrong_n).sum())}, 예은만 {int((wrong_y & ~wrong_n).sum())}, "
              f"새 모델만 {int((~wrong_y & wrong_n).sum())}")
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    print("\n새 모델 비율 w: (1-w)·예은 + w·새 모델")
    print(" w     holdout오류  Val오류")
    rows = []
    for w in grid:
        h = errors((1 - w) * yeeun["holdout"]["ensemble"] + w * new["holdout"], yeeun["holdout"]["y"])
        v = errors((1 - w) * yeeun["val"]["ensemble"] + w * new["val"], yeeun["val"]["y"])
        rows.append((w, h, v))
        print(f" {w:.2f}  {h:6d}      {v:6d}")
    # holdout 오류 최소 구간의 가운데를 고른다(동률이 여러 개면 한쪽 끝을 고르는 우연을 줄이려고).
    best = min(r[1] for r in rows)
    tied = [r for r in rows if r[1] == best]
    chosen = tied[len(tied) // 2]
    print(f"\nholdout으로 고른 w={chosen[0]:.2f}: holdout 오류 {chosen[1]} (예은 {rows[0][1]}), "
          f"Validation 오류 {chosen[2]} (예은 {rows[0][2]}) → {chosen[2] - rows[0][2]:+d}개, "
          f"Val Acc {1 - chosen[2] / len(yeeun['val']['y']):.4f}")


if __name__ == "__main__":
    main()
