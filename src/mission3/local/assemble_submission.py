"""고른 체크포인트만 담은 제출 폴더를 만듭니다.

inference.py 는 --ckpt_path 폴더 안의 모든 .ckpt 를 앙상블합니다. 학습을
거듭하면서 submission_models/ 에는 쓰지 않을 체크포인트도 쌓였으므로,
제출용으로는 고른 것만 복사한 폴더를 따로 만듭니다.

구성 근거: 크기 곡선이 10개에서 포화하므로 멤버 수가 아니라 **계열 수**가
남은 지렛대입니다. 무작위 조합 200개 중 200개에서 이질 구성이 동질 구성을
이겼습니다(+0.0016).

Run:  python src/mission3/local/assemble_submission.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "mission3" / "local" / "submission_models"
DST = ROOT / "results" / "mission3" / "local" / "submission_final"

# 계열: (설명, 파일명 패턴, 몇 개)
PLAN = [
    ("base mean lr2e-5", ["roberta-base_raw_s42_FINAL.ckpt",
                          "roberta-base_raw_s1_FINAL.ckpt"]),
    ("base attn lr2e-5", ["roberta-base_attn_raw_s42_FINAL.ckpt",
                          "roberta-base_attn_raw_s1_FINAL.ckpt"]),
    ("base mean lr3e-5", ["roberta-base_raw_s11_D3.ckpt"]),
    ("base attn lr3e-5", ["roberta-base_attn_raw_s13_D4.ckpt"]),
    # 파일명에는 전처리 이름이 들어가고, 도메인 적응 모델은 학습에 쓴
    # 디렉터리 이름이 들어갑니다. 추측하지 말고 실제로 생성된 이름을 씁니다.
    ("정규화 텍스트",      ["roberta-base_synonym+filler_s15_D5.ckpt",
                          "roberta-base_synonym+filler_s16_D5.ckpt"]),
    ("도메인 적응",        ["domain_adapted_full_raw_s17_D6.ckpt",
                          "domain_adapted_full_attn_raw_s18_D6.ckpt"]),
    ("CLS 풀링",          ["roberta-base_cls_raw_s19_D7.ckpt"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing, chosen = [], []
    print(f"{'계열':<20}{'체크포인트':<46}{'상태':>8}")
    for family, files in PLAN:
        for i, f in enumerate(files):
            p = SRC / f
            ok = p.exists()
            (chosen if ok else missing).append(f)
            print(f"{family if i == 0 else '':<20}{f:<46}{'있음' if ok else '없음':>8}")
    print(f"\n선택 {len(chosen)}개 · 누락 {len(missing)}개")
    if missing:
        print("누락분은 재학습이 끝나면 생깁니다. 지금 조립하면 그만큼 빠집니다.")
    if args.dry_run:
        return

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    # The four reused checkpoints predate the self-contained format: they have
    # no encoder config, so inference would fetch 440MB of pretrained weights
    # for each one and immediately overwrite them. Backfill the config while
    # copying, and the whole folder loads offline.
    from transformers import AutoConfig
    upgraded = 0
    for f in chosen:
        blob = torch.load(SRC / f, map_location="cpu", weights_only=False)
        if "encoder_config" not in blob:
            blob["encoder_config"] = AutoConfig.from_pretrained(blob["model_name"]).to_dict()
            blob.setdefault("normalize", "raw")
            upgraded += 1
            torch.save(blob, DST / f)
        else:
            shutil.copy2(SRC / f, DST / f)
    if upgraded:
        print()
        print(f"구형 체크포인트 {upgraded}개에 인코더 설정을 채워 넣었습니다 "
              f"(추론 시 다운로드 불필요)")
    for extra in ("tokenizer", "tfidf.pkl"):
        s = SRC / extra
        if s.is_dir():
            shutil.copytree(s, DST / extra)
        elif s.exists():
            shutil.copy2(s, DST / extra)
        else:
            print(f"경고: {extra} 없음")

    print(f"\n조립 완료: {DST}")
    print(f"{'파일':<46}{'헤드':>16}{'전처리':>16}{'아키텍처':>22}")
    for p in sorted(DST.glob("*.ckpt")):
        b = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{p.name:<46}{b.get('head','mean'):>16}"
              f"{b.get('normalize','raw'):>16}{b.get('model_name',''):>22}")
    size = sum(f.stat().st_size for f in DST.rglob("*")) / 1e9
    print(f"\n총 {len(list(DST.glob('*.ckpt')))}개 체크포인트 · {size:.2f} GB")
    print(f"\n실행:  python src/mission3/local/inference.py --audio_dir {{wav}} \\")
    print(f"           --label_dir {{json}} --ckpt_path {DST} --output ./outputs/mission3.csv")


if __name__ == "__main__":
    main()
