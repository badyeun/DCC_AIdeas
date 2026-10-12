"""제출 폴더를 fp16 가중치로 다시 써서 용량을 반으로 줄입니다.

16개 구성은 fp32 로 11.62 GB 입니다. 학습도 추론도 bf16 autocast 로 돌아가므로
fp16 저장은 어차피 캐스팅되는 가중치를 미리 캐스팅해 두는 것에 가깝습니다.
다만 '거의 같을 것'이라는 추측으로 제출물을 바꿀 수는 없으므로 먼저 셌습니다:
large 3종과 base 1종, 160통화 × 9증상 = 5,760 판정에서 **라벨 불일치 0건**,
확률 최대차 7.2e-4.

확률이 0.5 경계에 걸린 통화에서는 이론적으로 뒤집힐 수 있지만, 그 경우
뒤집히는 방향은 무작위에 가깝고 측정된 빈도는 0 입니다.

부동소수 가중치만 fp16 으로 바꿉니다 -- 정수 버퍼(position_ids 등)를 함께
캐스팅하면 모델이 깨집니다.

Run:  python src/mission3/local/make_fp16.py [--src ...] [--dst ...]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "results" / "mission3" / "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(LOCAL / "submission_final16"))
    ap.add_argument("--dst", default=str(LOCAL / "submission_final16_fp16"))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"원본 폴더가 없습니다: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"{'체크포인트':<46}{'fp32':>9}{'fp16':>9}{'캐스팅':>8}")
    for f in sorted(src.glob("*.ckpt")):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        cast = 0
        new = {}
        for k, v in blob["state_dict"].items():
            # 부동소수만. 정수 버퍼를 half 로 바꾸면 인덱싱이 깨집니다.
            if torch.is_tensor(v) and v.is_floating_point():
                new[k] = v.half()
                cast += 1
            else:
                new[k] = v
        blob["state_dict"] = new
        blob["stored_dtype"] = "float16"
        torch.save(blob, dst / f.name)
        a = f.stat().st_size / 1e9
        b = (dst / f.name).stat().st_size / 1e9
        print(f"{f.name:<46}{a:>8.2f}G{b:>8.2f}G{cast:>8}")

    for extra in ("tokenizer", "tfidf.pkl"):
        s = src / extra
        if s.is_dir():
            shutil.copytree(s, dst / extra)
        elif s.exists():
            shutil.copy2(s, dst / extra)
        else:
            print(f"경고: {extra} 없음")

    before = sum(p.stat().st_size for p in src.rglob("*")) / 1e9
    after = sum(p.stat().st_size for p in dst.rglob("*")) / 1e9
    print(f"\n{src.name}: {before:.2f} GB  ->  {dst.name}: {after:.2f} GB "
          f"({after/before:.0%})")
    print(f"\n검증:  python src/mission3/local/verify_submission.py --dir {dst}")


if __name__ == "__main__":
    main()
