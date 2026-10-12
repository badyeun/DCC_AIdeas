"""Mission 2 제출 폴더 조립 (Mission 1 `build_submission_v2.py`와 같은 형식).

추론 파일(`inference.py`, `common.py`, `speaker_embed.py`, `call_model.py`)은 실데이터로 검증한 것과
같은 파일을 그대로 복사하고 sha256을 출력한다. 체크포인트는 `pack_call_model.py`로 묶은 한 파일
(트랜스포머 10개 + ECAPA 인코더)이며 Colab에서 `inference.py`로 검증한 그 파일을 넣는다.

사용:
    python build_submission.py --ckpt .../mission2_final10.ckpt --runs .../mission2_local/runs \
        --notebook .../Mission2_training.ipynb --out .../mission2_submission_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

INFERENCE_FILES = ("inference.py", "common.py", "speaker_embed.py", "call_model.py")
TRAINING_FILES = ("train_call_model.py", "call_model.py", "pack_call_model.py", "train_speaker_embed.py",
                  "train_mfcc_rf.py", "speaker_embed.py", "common.py", "build_submission.py",
                  "make_training_notebook.py", "experiment_log.md")
# 최종 10개 (목소리 사전을 쓰는 B0는 제외). 이름 → 학습 명령에 붙인 인자.
FINAL_MODELS = ("A0_base", "A1_seed7", "A3_seed11", "F0_full", "A2_wide",
                "C0_pair", "C1_pair_seed7", "C2_pair_w1", "C3_pair_seed11", "F1_full_pair")
REQUIREMENTS = """numpy
torch
torchaudio
speechbrain==1.1.1
ruamel.yaml==0.18.6
librosa
soundfile
scikit-learn
joblib
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_commands(runs: Path) -> str:
    """sweep.jsonl에 남은 실제 인자로 최종 10개 학습 명령을 적는다."""
    rows = {json.loads(line)["name"]: json.loads(line) for line in (runs / "sweep.jsonl").read_text(encoding="utf-8").splitlines()}
    lines = ["# 최종 10개 트랜스포머 학습 명령 (실제 실행 인자)", "",
             "공통: `python train_call_model.py --embed_cache_dir <ECAPA 캐시> --mfcc_cache_dir <MFCC 캐시> "
             "--work_dir <작업 폴더> --out <이름>.pt --epochs 20 --threads 14` 에 아래 인자를 더한다.", "",
             "| 이름 | 추가 인자 | 최고 epoch | Train holdout | Validation |", "| --- | --- | ---: | ---: | ---: |"]
    for name in FINAL_MODELS:
        row = rows[name]
        extra = row["args"].replace("--epochs 20 --threads 14", "").strip() or "(없음)"
        holdout = "—" if row["holdout"] != row["holdout"] else f"{row['holdout']:.4f}"
        lines.append(f"| {name} | `{extra}` | {row['best_epoch']} | {holdout} | {row['validation']:.4f} |")
    lines += ["", "묶기: `python pack_call_model.py --models <10개 .pt> --base_ckpt <ECAPA 인코더가 든 체크포인트> "
              "--blend 0 --out mission2_final10.ckpt`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Colab에서 검증한 mission2_final10.ckpt")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parent / "submission" / "README.md")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"이미 있습니다. 지우고 다시 만드세요: {args.out}")
    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True)
    for name in INFERENCE_FILES:
        shutil.copy2(here / name, args.out / name)
        print(f"추론 {name:<22} sha256 {sha256(args.out / name)}")
    shutil.copy2(args.ckpt, args.out / "mission2_final10.ckpt")
    print(f"가중치 mission2_final10.ckpt   sha256 {sha256(args.out / 'mission2_final10.ckpt')}")
    training = args.out / "training_code"
    training.mkdir()
    for name in TRAINING_FILES:
        shutil.copy2(here / name, training / name)
    (training / "TRAINING_COMMANDS.md").write_text(training_commands(args.runs), encoding="utf-8")
    shutil.copy2(args.notebook, args.out / args.notebook.name)
    shutil.copy2(args.readme, args.out / "README.md")
    (args.out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    print(f"완료: {args.out}")


if __name__ == "__main__":
    main()
