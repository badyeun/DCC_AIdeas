"""앙상블 구성을 AP로 다시 고릅니다.

1라운드에서는 F1@0.5(seed 편차 0.0048)로 구성을 비교했는데, 대부분의 차이가
그 편차보다 작아 사실상 동전던지기였습니다. AP의 편차는 0.0008이라 같은
질문에 제대로 답할 수 있습니다.

노이즈 마이닝을 피하려고 조합은 **미리 정한 그룹**만 봅니다. 모든 부분집합을
훑어 최고를 고르면 그 최고값은 홀드아웃에 맞춰진 값입니다.
"""
import sys, numpy as np
from itertools import combinations
from pathlib import Path
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")

POOL = {
    "mean lr3e-5": ["probs_roberta-base_raw_s42.npy", "probs_roberta-base_raw_s1.npy",
                    "probs_roberta-base_raw_s7.npy"],
    "attn lr3e-5": ["probs_roberta-base_attn_raw_s42.npy", "probs_roberta-base_attn_raw_s1.npy",
                    "probs_roberta-base_attn_raw_s7.npy"],
    "mean lr2e-5": ["probs_roberta-base_raw_s42_lr2e-5e3.npy",
                    "probs_roberta-base_raw_s1_lr2e-5e3.npy",
                    "probs_roberta-base_raw_s7_lr2e-5e3.npy"],
}
load = {k: [np.load(R / f) for f in v] for k, v in POOL.items()}


def ap(P):
    return np.mean([average_precision_score(Y[:, i], P[:, i]) for i in range(9)])


def f1(P):
    return macro_f1(Y, (P >= .5).astype(np.int8))


def show(label, members, w=0.0):
    P = np.mean(members, axis=0)
    if w:
        P = (1 - w) * P + w * T
    print(f"{label:<34}{len(members):>4}{f1(P):>11.5f}{ap(P):>11.5f}")
    return ap(P)


print("미리 정한 그룹만 비교 (부분집합 전수 탐색 안 함)")
print(f"{'구성':<34}{'모델수':>4}{'F1@0.5':>11}{'평균 AP':>11}")
for k, v in load.items():
    show(k, v)
show("mean lr3e-5 + attn lr3e-5", load["mean lr3e-5"] + load["attn lr3e-5"])
show("mean lr2e-5 + attn lr3e-5", load["mean lr2e-5"] + load["attn lr3e-5"])
show("mean lr3e-5 + mean lr2e-5", load["mean lr3e-5"] + load["mean lr2e-5"])
allm = load["mean lr3e-5"] + load["attn lr3e-5"] + load["mean lr2e-5"]
show("세 그룹 전부", allm)
print()
print("TF-IDF 0.10 혼합 후:")
for k, v in load.items():
    show(k + " + TFIDF", v, 0.10)
show("mean lr2e-5 + attn lr3e-5 + TFIDF", load["mean lr2e-5"] + load["attn lr3e-5"], 0.10)
show("세 그룹 전부 + TFIDF", allm, 0.10)

print("\n현 제출 구성(mean+attn, lr 2e-5)에 해당하는 홀드아웃 근사:")
print("  attn은 lr3e-5로만 학습된 것이 홀드아웃에 있어 정확히 같지는 않습니다.")

print("\n같은 크기에서 그룹 섞기가 도움이 되는가 (크기 6 고정):")
pairs = [("mean lr3e-5 + attn lr3e-5", load["mean lr3e-5"] + load["attn lr3e-5"]),
         ("mean lr2e-5 + attn lr3e-5", load["mean lr2e-5"] + load["attn lr3e-5"]),
         ("mean lr3e-5 + mean lr2e-5", load["mean lr3e-5"] + load["mean lr2e-5"])]
for name, mem in pairs:
    print(f"  {name:<32} AP {ap(np.mean(mem, axis=0)):.5f}")
