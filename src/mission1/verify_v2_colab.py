"""Colab 실데이터 검증: 제출 폴더 v2를 공식 명령 그대로 Validation에 돌리고, 학습 때 값과 비교한다.

1. Validation WAV/JSON 폴더(part별로 나뉨)를 찾아 `python inference.py --audio_dir ... --label_dir ...
   --ckpt_path ... --output ...`를 그대로 실행한다(`--save_probs`만 추가).
   `--limit N`이면 정렬 순서 앞 N개 통화만 링크해 빠르게 시험한다.
2. 채점: JSON의 gender로 정확도·오류 수.
3. 비교(reference npz): 예은 3모델 확률이 예은이 저장한 공식 Validation 확률과, ECAPA 확률이 로컬 캐시로
   계산한 값과, 최종 판정이 로컬에서 계산한 판정과 같은지.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def find_dirs(data_root: Path) -> tuple[list[Path], list[Path]]:
    audio = sorted(data_root.glob("*/대학부 데이터/Validation/1.원천데이터/VS_서울_구급"))
    labels = sorted(data_root.glob("*/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급"))
    if not audio or not labels:
        raise FileNotFoundError(f"Validation 폴더를 찾지 못했습니다: {data_root}")
    return audio, labels


def make_subset(audio_dirs: list[Path], label_dirs: list[Path], limit: int, target: Path) -> tuple[list[Path], list[Path]]:
    wavs = sorted(path for folder in audio_dirs for path in folder.rglob("*.wav"))[:limit]
    jsons = {path.stem: path for folder in label_dirs for path in folder.rglob("*.json")}
    (target / "wav").mkdir(parents=True, exist_ok=True)
    (target / "json").mkdir(parents=True, exist_ok=True)
    for wav in wavs:
        (target / "wav" / wav.name).symlink_to(wav)
        (target / "json" / f"{wav.stem}.json").symlink_to(jsons[wav.stem])
    return [target / "wav"], [target / "json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True, help="mission1_submission_v2 폴더")
    parser.add_argument("--reference", type=Path, required=True, help="v2_reference_val.npz")
    parser.add_argument("--work", type=Path, default=Path("/content/v2_check"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    audio_dirs, label_dirs = find_dirs(args.data_root)
    print(f"Validation 폴더: 음성 {len(audio_dirs)}개, 라벨 {len(label_dirs)}개")
    tag = f"limit{args.limit}" if args.limit else "full"
    if args.limit:
        audio_dirs, label_dirs = make_subset(audio_dirs, label_dirs, args.limit, args.work / f"subset_{int(time.time())}")
    output = args.work / tag / "mission1.csv"
    probs = args.work / tag / "probs.npz"
    command = [sys.executable, "inference.py",
               "--audio_dir", *map(str, audio_dirs), "--label_dir", *map(str, label_dirs),
               "--ckpt_path", "./ecapa_gender/ecapa_lr.npz", "--output", str(output), "--save_probs", str(probs)]
    print("실행:", " ".join(command[1:3]), "...")
    start = time.time()
    subprocess.run(command, cwd=args.submission, check=True)
    print(f"추론 소요 {int(time.time() - start) // 60}분 {int(time.time() - start) % 60}초")

    with output.open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if rows[0] != ["audio file name", "gender"]:
        raise ValueError(f"CSV 헤더가 다릅니다: {rows[0]}")
    predicted = {Path(name).stem: gender for name, gender in rows[1:]}
    truth = {}
    for folder in label_dirs:
        for path in folder.rglob("*.json"):
            if path.stem in predicted:
                truth[path.stem] = str(json.loads(path.read_text(encoding="utf-8")).get("gender", "")).strip().upper()
    wrong = sum(predicted[stem] != truth[stem] for stem in predicted)
    print(f"\n[채점] CSV {len(rows) - 1}행, 정확도 {1 - wrong / len(predicted):.4f} (오류 {wrong}/{len(predicted)})")

    with np.load(args.reference) as ref, np.load(probs) as got:
        lookup = {str(stem): i for i, stem in enumerate(ref["stems"])}
        index = np.array([lookup[Path(str(name)).stem] for name in got["names"]])
        yeeun_diff = np.nanmax(np.abs(got["yeeun"] - ref["yeeun"][index]), axis=0)
        ecapa_diff = np.nanmax(np.abs(got["ecapa"] - ref["ecapa"][index]))
        final_ref = ref["final"][index]
        changed = int(((got["final"] >= 0.5) != (final_ref >= 0.5)).sum())
        ref_wrong = int(((final_ref >= 0.5) != ref["y"][index]).sum())
        print(f"[비교] 예은 3모델 확률 최대 차이 (Att+F0, Stats+F0, Stats): {np.round(yeeun_diff, 6)}")
        print(f"[비교] 예은 3모델 앙상블 판정이 저장값과 다른 통화: "
              f"{int(((got['yeeun'] @ np.array([.4, .4, .2]) >= .5) != (ref['yeeun'][index] @ np.array([.4, .4, .2]) >= .5)).sum())}")
        print(f"[비교] ECAPA 확률 최대 차이: {ecapa_diff:.6f}")
        print(f"[비교] 최종 판정이 로컬 계산과 다른 통화: {changed} (로컬 기대 오류 {ref_wrong})")
        print(f"[비교] 실패(NaN): 예은 {int(np.isnan(got['yeeun'][:, 0]).sum())}, ECAPA {int(np.isnan(got['ecapa']).sum())}")


if __name__ == "__main__":
    main()
