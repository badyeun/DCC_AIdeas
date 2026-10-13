"""Mission 2 제출 폴더 조립.

python build_submission.py --ckpt mission2_final10.ckpt --verified verify_sha256.txt --out mission2_submission_final

추론 파일(inference.py, features.py, model.py)과 체크포인트가 Colab 실데이터 검증 때 쓴 파일과
sha256이 같은지 확인하고 복사한다. 다르면 멈춘다. requirements.txt는 검증 환경의 실제 설치 버전으로 고정한 것을 복사한다. 학습 노트북은 조립 뒤 make_training_notebook.py로 폴더 안에 만든다.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent          # training_code/
SOURCE = HERE.parent                            # 제출 폴더 원본 (inference.py 등)
VERIFIED_FILES = ("inference.py", "features.py", "model.py", "mission2_final10.ckpt")
TRAINING_FILES = ("make_feature_cache.py", "train_call_model.py", "pack_model.py", "make_training_notebook.py",
                  "build_submission.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 2 제출 폴더 조립")
    parser.add_argument("--ckpt", type=Path, required=True, help="Colab에서 검증한 mission2_final10.ckpt")
    parser.add_argument("--verified", type=Path, required=True, help="검증 때 쓴 파일의 sha256 목록 (sha256sum 출력)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"이미 있습니다. 지우고 다시 만드세요: {args.out}")
    expected = {line.split()[1].lstrip("*"): line.split()[0] for line in args.verified.read_text(encoding="utf-8").splitlines() if line.strip()}
    args.out.mkdir(parents=True)
    for name in VERIFIED_FILES:
        shutil.copy2(args.ckpt if name.endswith(".ckpt") else SOURCE / name, args.out / name)
        actual = sha256(args.out / name)
        if actual != expected.get(name):
            raise ValueError(f"{name}: 검증한 파일과 다릅니다 ({actual[:12]} vs {str(expected.get(name))[:12]}). Colab 재검증이 필요합니다.")
        print(f"검증본과 같음  {name:<24} {actual}")
    for name in ("requirements.txt", "README.md"):
        shutil.copy2(SOURCE / name, args.out / name)
    (args.out / "training_code").mkdir()
    for name in TRAINING_FILES:
        shutil.copy2(HERE / name, args.out / "training_code" / name)
    print(f"완료: {args.out} (다음: make_training_notebook.py로 노트북 생성)")


if __name__ == "__main__":
    main()
