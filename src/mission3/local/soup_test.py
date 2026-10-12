"""가중치 평균(model soup)이 앙상블 성능을 보존하는가.

모델 수를 실제로 줄이는 방법은 둘뿐이다 -- 가중치를 합치거나(soup), 앙상블을
한 모델에 증류하거나. 조합을 '고르는' 길은 전부 선택 편향으로 막혔다.

여기서는 같은 레시피(lr 2e-5, 3 epoch, batch 16x2, mean, raw)로 seed 만 바꿔
홀드아웃 분할에 학습한 세 모델을 쓴다. 전체 데이터 모델로는 판정할 수 없다 --
채점할 데이터가 남지 않으므로.

비교:
  단일 3개 각각        <- 하한
  확률 평균 앙상블 3개   <- 현재 방식
  가중치 평균 soup 1개  <- 모델 1개로 같은 점수가 나오는가
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_local import TARGETS, TRAIN_DIR, load, macro_f1, split            # noqa: E402

R = HERE.parents[2] / "results" / "mission3" / "local"
SRC = R / "submission_models"
PARTS = ["roberta-base_raw_s42_SOUP.ckpt", "roberta-base_raw_s1_SOUP.ckpt",
         "roberta-base_raw_s7_SOUP.ckpt"]


def make_soup(paths, out):
    """부동소수 텐서만 산술 평균. 정수 버퍼는 첫 체크포인트 값을 쓴다."""
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = set(blobs[0]["state_dict"])
    for b in blobs[1:]:
        assert set(b["state_dict"]) == keys, "구조가 다른 체크포인트는 섞을 수 없습니다"
    sd = {}
    for k in blobs[0]["state_dict"]:
        vs = [b["state_dict"][k] for b in blobs]
        if torch.is_tensor(vs[0]) and vs[0].is_floating_point():
            sd[k] = torch.stack([v.float() for v in vs]).mean(0)
        else:
            sd[k] = vs[0]
    blob = dict(blobs[0])
    blob["state_dict"] = sd
    blob["soup_of"] = [Path(p).name for p in paths]
    blob["seed"] = -1
    torch.save(blob, out)
    return out


@torch.no_grad()
def probs_for(ckpt, texts, device):
    import inference
    return inference.probabilities(Path(ckpt), texts, device, 32)


def main():
    _, texts, Y = load(TRAIN_DIR, "train")
    _, te = split(texts, Y, seed=2026)
    hold = [texts[i] for i in te]
    Yh = Y[te]
    f1 = lambda P: macro_f1(Yh, (P >= .5).astype(np.int8))
    T = np.load(R / "tfidf_holdout_probs.npy")
    blend = lambda P: f1(.9 * P + .1 * T)

    print("=== 재료 세 개 (각각 단독) ===")
    ind = []
    for p in PARTS:
        q = np.load(R / f"probs_{Path(p).stem}.npy")
        ind.append(q)
        print(f"  {Path(p).stem:<34}{f1(q):.5f}   혼합 {blend(q):.5f}")
    ens = np.mean(ind, axis=0)
    print(f"\n  확률 평균 3개 앙상블{'':<16}{f1(ens):.5f}   혼합 {blend(ens):.5f}")

    out = SRC / "roberta-base_raw_SOUPMIX.ckpt"
    print(f"\n가중치 평균 생성 중... ({len(PARTS)}개 → 1개)")
    make_soup([SRC / p for p in PARTS], out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = probs_for(out, hold, device)
    np.save(R / "probs_roberta-base_raw_SOUPMIX.npy", sp.astype(np.float32))
    print(f"\n  가중치 평균 soup 1개{'':<16}{f1(sp):.5f}   혼합 {blend(sp):.5f}")

    print(f"\n{'=' * 58}")
    print(f"{'비교':<34}{'Macro F1':>11}{'앙상블 대비':>13}")
    base = blend(ens)
    print(f"{'단일 평균 (3개 각각의 평균)':<34}{np.mean([blend(q) for q in ind]):>11.5f}"
          f"{np.mean([blend(q) for q in ind])-base:>+13.5f}")
    print(f"{'확률 평균 앙상블 (모델 3개)':<34}{base:>11.5f}{0:>+13.5f}")
    print(f"{'가중치 평균 soup (모델 1개)':<34}{blend(sp):>11.5f}{blend(sp)-base:>+13.5f}")
    keep = (blend(sp) - np.mean([blend(q) for q in ind])) / \
           max(base - np.mean([blend(q) for q in ind]), 1e-9)
    print(f"\n앙상블 이득 중 soup 이 보존한 비율: {keep:.0%}")


if __name__ == "__main__":
    main()
