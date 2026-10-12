"""Colab에서 한 번 실행: 로컬 PC 실험에 필요한 Mission 1 라벨과 예은 모델 확률만 모은다.

왜 필요한가
- Mission 2 때 만든 발화별 ECAPA·MFCC 캐시(로컬 PC에 있음)를 Mission 1 성별 분류에 재사용하려 한다.
  캐시에는 통화 경계(`counts`)와 speaker 라벨만 있고 통화의 파일 이름과 `gender`가 없다.
- 캐시의 통화 순서는 Mission 2 `train_mfcc_rf.collect_pairs`(part 폴더 정렬 → part 안 WAV 경로 정렬)다.
  같은 순서로 JSON만 읽어 (파일 stem, gender, 발화 수)를 저장한다. WAV는 읽지 않는다.
- 발화 수는 Mission 2 캐시를 만들 때와 같은 규칙(speaker가 "0"/"1"인 발화만)으로 센다. 로컬에서
  캐시의 `counts`와 한 통화도 빠짐없이 같은지 비교해 정렬을 검증한다.
- 예은 모델의 내부 holdout·공식 Validation 확률 npz도 같이 모아 로컬에서 앙상블을 시험한다.

`gender`는 학습 정답과 채점에만 쓴다. `text` 등 다른 필드는 읽지 않는다.

사용 예 (Colab):
    python export_labels_for_local.py --data_root "/content/drive/MyDrive/data(inho)" \
        --search_root /content/drive/MyDrive --out_dir /content/mission1_export
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

GENDER_TO_INDEX = {"M": 0, "F": 1}  # 예은 코드와 같은 값
SPEAKERS = {"0", "1"}
UTTERANCE_KEYS = ("utterances", "utterences")

# 예은의 holdout 선택(노트북 셀 204~206)과 공식 Validation 평가에 쓰인 파일들
YEEUN_FILES = (
    "internal_attention_f0_seed_42_result.npz",
    "internal_stats_f0_attention_seed42_result.npz",
    "internal_stats_attention_seed42_result.npz",
    "official_validation_stats_results.npz",
    "official_validation_f0_comparison.npz",
    "internal_split_seed42.npz",
    "wavlm_train_names_restored.npz",
)


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Mission 2 `train_mfcc_rf.collect_pairs`와 같은 순서. 바꾸면 캐시와 정렬이 깨진다."""
    folders = ("TS_서울_구급", "TL_서울_구급") if split == "Training" else ("VS_서울_구급", "VL_서울_구급")
    parts = sorted(path for path in data_root.iterdir() if path.is_dir())
    index: dict[str, Path] = {}
    for part in parts:
        label_dir = part / "대학부 데이터" / split / "2.라벨링데이터" / folders[1]
        if label_dir.is_dir():
            for path in sorted(label_dir.rglob("*.json")):
                index.setdefault(path.stem, path)
    pairs: list[tuple[Path, Path]] = []
    for part in parts:
        audio_dir = part / "대학부 데이터" / split / "1.원천데이터" / folders[0]
        if not audio_dir.is_dir():
            continue
        audio = sorted(p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
        for audio_path in audio:
            label_path = index.get(audio_path.stem)
            if label_path is None:
                raise FileNotFoundError(f"JSON을 찾지 못했습니다: {audio_path}")
            pairs.append((audio_path, label_path))
    print(f"[{split}] 통화 pair: {len(pairs):,}")
    return pairs


def read_label(label_path: Path) -> tuple[int, int]:
    annotation = json.loads(label_path.read_bytes().decode("utf-8"))
    gender = str(annotation.get("gender", "")).strip().upper()
    if gender not in GENDER_TO_INDEX:
        raise ValueError(f"잘못된 gender {gender!r}: {label_path}")
    utterances: list = []
    for key in UTTERANCE_KEYS:
        if isinstance(annotation.get(key), list):
            utterances = [item for item in annotation[key] if isinstance(item, dict)]
            break
    count = sum(
        1 for item in utterances
        if str(item.get("speaker", item.get("Speaker", ""))).strip() in SPEAKERS
    )
    return GENDER_TO_INDEX[gender], count


def export_split(data_root: Path, split: str, out_dir: Path, workers: int) -> None:
    pairs = collect_pairs(data_root, split)
    labels = [label for _, label in pairs]
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(read_label, labels))
    name = "train" if split == "Training" else "val"
    stems = np.asarray([audio.stem for audio, _ in pairs])
    gender = np.asarray([row[0] for row in rows], dtype=np.int64)
    counts = np.asarray([row[1] for row in rows], dtype=np.int64)
    np.savez(out_dir / f"labels_{name}.npz", stems=stems, gender=gender, counts=counts)
    print(f"[{split}] 저장: {len(stems):,}통화, 발화 {int(counts.sum()):,}, F 비율 {gender.mean():.4f}")


def copy_yeeun_files(search_roots: list[Path], out_dir: Path) -> None:
    """같은 이름이 여러 곳에 있으면 모두 가져온다(경로를 이름에 붙여 구분)."""
    wanted = set(YEEUN_FILES)
    found: dict[str, list[Path]] = {name: [] for name in YEEUN_FILES}
    for root in search_roots:
        if not root.is_dir():
            print(f"[warn] 검색 폴더 없음: {root}")
            continue
        for current, _dirs, files in os.walk(root):
            for file in files:
                if file in wanted:
                    found[file].append(Path(current) / file)
    target = out_dir / "yeeun"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, paths in found.items():
        print(f"{'OK ' if paths else '없음'} {name}: {len(paths)}곳")
        for number, path in enumerate(paths):
            copied = target / (name if number == 0 else f"{path.stem}__{number}.npz")
            shutil.copy2(path, copied)
            manifest[copied.name] = str(path)
            print(f"     {path}")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--search_root", type=Path, nargs="+", required=True,
                        help="예은의 mission1 산출물을 찾을 폴더들 (예: /content/drive/MyDrive/mission1)")
    parser.add_argument("--out_dir", type=Path, default=Path("/content/mission1_export"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Validation"):
        export_split(args.data_root, split, args.out_dir, args.workers)
    copy_yeeun_files(args.search_root, args.out_dir)
    print("완료:", args.out_dir)


if __name__ == "__main__":
    main()
