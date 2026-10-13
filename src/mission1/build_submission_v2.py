"""Mission 1 제출 폴더 v2 조립: 예은 제출 폴더(원본은 건드리지 않음) + ECAPA 성별 모델.

예은 폴더에서는 3모델 추론에 실제로 쓰는 파일만 복사한다(예전 백업 스크립트·RF 체크포인트 제외).
ECAPA 쪽은 인코더 파일, 로지스틱 회귀 가중치(.pt, holdout으로 고른 blend 비율 포함)를 넣는다.
학습 노트북은 이 폴더를 만든 뒤 make_ecapa_notebook.py로 실행·합쳐서 넣는다.

사용:
    python build_submission_v2.py --yeeun_dir .../mission1_submission --lr_model .../pooled_caller_ecapa.model.npz \
        --encoder_dir .../models/ecapa_voxceleb --blend 0.75 --out .../mission1_submission_v2
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch

# 예은 학습 결과물 중 최종 추론에 쓰는 가중치와 사전학습 모델만 복사한다.
YEEUN_FILES = (
    "attention_f0/attention_f0_seed_42.pt", "stats_f0/stats_f0_seed42.pt", "stats_no_f0/stats_attention_seed42.pt",
    "pretrained/wavlm-base-plus/config.json", "pretrained/wavlm-base-plus/preprocessor_config.json",
    "pretrained/wavlm-base-plus/model.safetensors",
)
# 추론 코드 (WavLM 쪽은 예은 코드에서 최종 추론에 쓰는 부분만 그대로 옮긴 wavlm_3model.py, common.py)
OUR_FILES = ("submission_v2/inference.py", "submission_v2/inference_ecapa_blend.py", "submission_v2/ecapa_gender.py",
             "submission_v2/wavlm_3model.py", "submission_v2/common.py", "submission_v2/README.md")
# 제출물 "학습 코드"용: ECAPA 쪽 학습 코드와, 그 입력인 발화별 특징 캐시를 만든 Mission 2 코드
TRAINING_CODE = {
    "training_code": ("gender_data.py", "train_gender_pooled.py", "blend_with_yeeun.py",
                      "export_labels_for_local.py", "build_submission_v2.py", "make_ecapa_notebook.py"),
    "training_code/mission2_ecapa_cache": ("../mission2/common.py", "../mission2/speaker_embed.py",
                                           "../mission2/train_speaker_embed.py", "../mission2/train_mfcc_rf.py"),
}
# ruamel.yaml 0.19 이상은 speechbrain(hyperpyyaml)과 충돌한 적이 있어 고정한다.
EXTRA_REQUIREMENTS = ["speechbrain==1.1.1", "ruamel.yaml==0.18.6"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yeeun_dir", type=Path, required=True)
    parser.add_argument("--lr_model", type=Path, required=True)
    parser.add_argument("--encoder_dir", type=Path, required=True)
    parser.add_argument("--blend", type=float, required=True, help="ECAPA 비율 (holdout으로 고른 값)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"이미 있습니다. 지우고 다시 만드세요: {args.out}")
    here = Path(__file__).resolve().parent
    for name in YEEUN_FILES:
        source = args.yeeun_dir / name
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"예은 {name} ({digest(target)})")
    for name in OUR_FILES:
        shutil.copy2(here / name, args.out / Path(name).name)
        print(f"추가 {Path(name).name}")
    for folder, names in TRAINING_CODE.items():
        (args.out / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(here / name, args.out / folder / Path(name).name)
    model_dir = args.out / "ecapa_gender"
    shutil.copytree(args.encoder_dir, model_dir / "encoder")
    with np.load(args.lr_model) as data:
        if list(data["blocks"]) != ["caller_ecapa"]:
            raise ValueError(f"제출 코드는 caller_ecapa 특징만 계산합니다: {list(data['blocks'])}")
        # 제출 규정의 가중치 형식(.pt)으로 저장. 텐서와 숫자만 담아 weights_only=True로 읽힌다.
        torch.save({
            "scaler_mean": torch.from_numpy(np.asarray(data["scaler_mean"], dtype=np.float64)),
            "scaler_scale": torch.from_numpy(np.asarray(data["scaler_scale"], dtype=np.float64)),
            "coef": torch.from_numpy(np.asarray(data["coef"], dtype=np.float64)),
            "intercept": float(data["intercept"][0]), "C": float(data["C"]), "blend": float(args.blend),
        }, model_dir / "ecapa_lr.pt")
    requirements = (args.yeeun_dir / "requirements.txt").read_text(encoding="utf-8").strip().splitlines()
    (args.out / "requirements.txt").write_text("\n".join(requirements + EXTRA_REQUIREMENTS) + "\n", encoding="utf-8")
    print(f"완료: {args.out}")


if __name__ == "__main__":
    main()
