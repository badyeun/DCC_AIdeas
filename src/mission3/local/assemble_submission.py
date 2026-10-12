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

TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "mission3" / "local" / "submission_models"
# The eleven-member folder stays where it is. It is the configuration that
# measured 0.65658 on the official Validation set, so it remains the fallback
# until the sixteen-member one is verified end to end.
DST = ROOT / "results" / "mission3" / "local" / "submission_final16"

# 계열: (설명, 체크포인트 파일명)
#
# 열한 개 base 에 large 급 다섯 개를 더합니다. 홀드아웃 0.64666 -> 0.65038,
# 반분 240회 전승. 고른 게 아니라 **large 급 후보 전부**이므로 선택 편향이
# 없고, 중첩 검증(A반 결정/B반 측정)에서 300회 중 269회 이 구성이 뽑혔습니다.
#
# large 를 다섯 개에서 멈추는 이유: 여섯 번째로 넣어 본 large 레시피 세 개가
# -0.00096 ~ +0.00014 로 전부 기여가 없었습니다. 예측 불일치는 기존과 비슷한데
# (0.0334~0.0410 vs 0.0364) 단독 품질이 0.002~0.006 낮고, 품질/다양성
# 손익분기(단독 -0.004)를 넘지 못합니다.
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
    # large: _LGE2 는 3-epoch 스케줄의 epoch 2 가중치입니다. 같은 이름의 _LGF
    # 파일은 2-epoch 스케줄로 돌린 구버전이고 앙상블 이득의 절반을 잃으므로
    # (-0.00172) 쓰지 않습니다.
    ("large mean",        ["roberta-large_raw_s1_LGE2.ckpt",
                          "roberta-large_raw_s7_LGE2.ckpt"]),
    ("large attn",        ["roberta-large_attn_raw_s42_LGE2.ckpt"]),
    ("large 정규화",       ["roberta-large_synonym+filler_s42_LGF.ckpt"]),
    ("large TAPT",        ["klue-roberta-large-tapt_raw_s42_LGF.ckpt"]),
]

# 제출 폴더에 들어가면 안 되는 것을 구조적으로 막습니다. 오늘 1% 데이터로
# 돌린 maxlen 64 스모크 체크포인트가 이 폴더에 남아 있었습니다 -- 삭제
# 구문이 실제 파일명과 다른 이름을 지우려 했기 때문입니다. 이름을 맞게
# 고쳐도, 다음에 또 무언가 남을 수 있으므로 내용으로 검사합니다.
EXPECT_TRAINED_ON = 29_200
EXPECT_MAX_LENGTH = 512


def audit(blob, name):
    """출하 자격을 내용으로 확인합니다. 하나라도 어긋나면 조립을 멈춥니다."""
    problems = []
    if blob.get("trained_on") != EXPECT_TRAINED_ON:
        problems.append(f"학습 통화 수 {blob.get('trained_on'):,} "
                        f"(기대 {EXPECT_TRAINED_ON:,})")
    if blob.get("max_length") != EXPECT_MAX_LENGTH:
        problems.append(f"max_length {blob.get('max_length')} "
                        f"(기대 {EXPECT_MAX_LENGTH})")
    if blob.get("threshold") != 0.5:
        problems.append(f"임계값 {blob.get('threshold')} (기대 0.5)")
    if list(blob.get("target_symptoms", [])) != list(TARGETS):
        problems.append("증상 목록이 다릅니다")
    return problems


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
    # 내용 검사. 계획에 적힌 이름이 맞다는 것만으로는 부족합니다 -- 이름이
    # 맞아도 1% 데이터로 돌린 모델일 수 있습니다. 오늘 그런 체크포인트가
    # 실제로 이 폴더에 남아 있었습니다.
    print()
    print(f"{'체크포인트':<46}{'학습 통화':>11}{'maxlen':>8}{'판정':>7}")
    failed = {}
    for f in chosen:
        blob = torch.load(SRC / f, map_location="cpu", weights_only=False)
        bad = audit(blob, f)
        if bad:
            failed[f] = bad
        print(f"{f:<46}{blob.get('trained_on', 0):>11,}"
              f"{str(blob.get('max_length')):>8}{'문제' if bad else 'OK':>7}")
    if failed:
        print()
        print("조립을 멈춥니다 -- 출하 자격을 못 갖춘 체크포인트가 있습니다:")
        for f, bad in failed.items():
            print(f"  {f}")
            for b in bad:
                print(f"    - {b}")
        raise SystemExit(1)

    if args.dry_run:
        print()
        print("--dry-run: 복사는 하지 않았습니다.")
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
