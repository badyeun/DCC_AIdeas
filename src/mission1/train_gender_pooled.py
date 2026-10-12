"""Mission 1 기준선: 통화 요약 벡터(gender_data.pooled_features) + 로지스틱 회귀.

ECAPA 화자 임베딩만으로 성별을 얼마나 맞히는지, 예은 WavLM 모델과 어디서 다르게 틀리는지
빠르게(수 분) 보기 위한 용도다. C는 예은 내부 holdout으로 고르고 Validation은 마지막에 한 번 본다.
결과 npz 형식은 train_gender_call_model.py와 같다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gender_data import ECAPA_DIM, MFCC_DIM, load_split, yeeun_holdout_mask

# pooled_features 열 구성: 신고자 ECAPA 평균·표준편차, 신고자 MFCC 평균·표준편차, 대원 ECAPA·MFCC 평균, MFCC 차, 개수 3
BLOCKS = {
    "caller_ecapa": (0, ECAPA_DIM * 2),
    "caller_mfcc": (ECAPA_DIM * 2, ECAPA_DIM * 2 + MFCC_DIM * 2),
    "agent": (ECAPA_DIM * 2 + MFCC_DIM * 2, ECAPA_DIM * 3 + MFCC_DIM * 4),
    "counts": (ECAPA_DIM * 3 + MFCC_DIM * 4, ECAPA_DIM * 3 + MFCC_DIM * 4 + 3),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2_local", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--pooled_dir", type=Path, required=True, help="pooled_{train,val}.npy 폴더")
    parser.add_argument("--blocks", nargs="+", default=list(BLOCKS), choices=list(BLOCKS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train = load_split(args.m2_local, args.export_dir, "train")
    val = load_split(args.m2_local, args.export_dir, "val")
    held = yeeun_holdout_mask(train["stems"], train["gender"], args.export_dir)
    columns = np.concatenate([np.arange(*BLOCKS[name]) for name in args.blocks])
    x_train = np.load(args.pooled_dir / "pooled_train.npy")[:, columns]
    x_val = np.load(args.pooled_dir / "pooled_val.npy")[:, columns]
    scaler = StandardScaler().fit(x_train[~held])
    x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
    y = train["gender"]

    best = None
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = LogisticRegression(C=c, max_iter=3000).fit(x_train[~held], y[~held])
        probability = model.predict_proba(x_train[held])[:, 1]
        accuracy = float(((probability >= 0.5) == y[held]).mean())
        logloss = float(-np.mean(y[held] * np.log(np.clip(probability, 1e-7, 1)) + (1 - y[held]) * np.log(np.clip(1 - probability, 1e-7, 1))))
        print(f"C={c}: holdout Acc {accuracy:.4f} (오류 {int((1 - accuracy) * held.sum() + 0.5)}), logloss {logloss:.4f}")
        if best is None or (accuracy, -logloss) > (best[0], -best[1]):
            best = (accuracy, logloss, c, model, probability)
    accuracy, _, c, model, held_probability = best
    val_probability = model.predict_proba(x_val)[:, 1]
    val_accuracy = float(((val_probability >= 0.5) == val["gender"]).mean())
    print(f"선택 C={c}: holdout {accuracy:.4f} / Validation {val_accuracy:.4f} (오류 {int((1 - val_accuracy) * len(x_val) + 0.5)})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix(".npz"),
             holdout_stems=train["stems"][held], holdout_prob=held_probability, holdout_y=y[held],
             val_stems=val["stems"], val_prob=val_probability, val_y=val["gender"])
    # 제출용: sklearn 버전과 무관하게 numpy만으로 다시 계산할 수 있도록 숫자만 저장한다.
    np.savez(args.out.with_suffix(".model.npz"), columns=columns, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             coef=model.coef_[0], intercept=model.intercept_, C=c, blocks=np.asarray(args.blocks))
    args.out.with_suffix(".json").write_text(json.dumps(
        {"holdout": accuracy, "validation": val_accuracy, "C": c, "blocks": args.blocks}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
