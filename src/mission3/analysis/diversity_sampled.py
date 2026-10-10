"""이질 구성이 정말 나은가 — 손으로 고른 조합 하나가 아니라 분포로.

앞선 비교는 제가 고른 이질 6개가 동질 6개를 +0.0023 이겼습니다(부트스트랩
94% 승, 95% 구간은 0 포함). 그 조합을 제가 골랐다는 게 문제입니다. 여기서는
계열마다 무작위로 하나씩 뽑아 이질 6개를 만들고, 같은 수의 동질 6개와
비교합니다. 조합 선택의 운이 평균에서 사라집니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")

FAM = {
    "mean2e5": [f"roberta-base_raw_s{s}_lr2e-5e3" for s in (42, 1, 7)],
    "attn2e5": [f"roberta-base_attn_raw_s{s}_lr2e5_LS" for s in (42, 1, 7)],
    "mean3e5": [f"roberta-base_raw_s{s}" for s in (42, 1, 7)],
    "attn3e5": [f"roberta-base_attn_raw_s{s}" for s in (42, 1, 7)],
    "norm":    [f"roberta-base_synonym+filler_s{s}" for s in (42, 1, 7)],
    "dapt":    [f"domain_adapted_raw_s{s}_DAPT_LS" for s in (42, 1, 7)]
               + ["domain_adapted_attn_raw_s42_DAPT_LS"],
    "large":   ["roberta-large_raw_s42", "roberta-large_synonym+filler_s42"],
}
P = {k: [L(n) for n in v] for k, v in FAM.items()}


def f1(mem, w=0.10):
    m = np.mean(mem, axis=0)
    return (macro_f1(Y, (m >= .5).astype(np.int8)),
            macro_f1(Y, ((1 - w) * m + w * T >= .5).astype(np.int8)))


rng = np.random.default_rng(13)
N = 200
homo_pool = P["mean2e5"] + P["attn2e5"]          # 현 제출의 레시피
others = [k for k in FAM if k not in ("mean2e5", "attn2e5")]

res = {"동질 6개 (제출 레시피)": [], "이질 6개 (계열마다 하나)": [],
       "반반 (제출 3 + 다른 계열 3)": []}
for _ in range(N):
    idx = rng.choice(len(homo_pool), 6, replace=False)
    res["동질 6개 (제출 레시피)"].append(f1([homo_pool[i] for i in idx]))

    fams = list(rng.choice(list(FAM), 6, replace=False))
    res["이질 6개 (계열마다 하나)"].append(
        f1([P[k][rng.integers(len(P[k]))] for k in fams]))

    a = [homo_pool[i] for i in rng.choice(len(homo_pool), 3, replace=False)]
    fams = list(rng.choice(others, 3, replace=False))
    res["반반 (제출 3 + 다른 계열 3)"].append(
        f1(a + [P[k][rng.integers(len(P[k]))] for k in fams]))

print(f"무작위 추출 {N}회씩, 전부 멤버 6개\n")
print(f"{'구성':<28}{'F1 평균':>10}{'표준편차':>10}{'+TFIDF 평균':>13}{'표준편차':>10}")
arr = {}
for k, v in res.items():
    a = np.array(v); arr[k] = a
    print(f"{k:<28}{a[:,0].mean():>10.5f}{a[:,0].std():>10.5f}"
          f"{a[:,1].mean():>13.5f}{a[:,1].std():>10.5f}")

h = arr["동질 6개 (제출 레시피)"]
for k in ("이질 6개 (계열마다 하나)", "반반 (제출 3 + 다른 계열 3)"):
    d = arr[k][:, 1] - h[:, 1]
    se = np.sqrt(arr[k][:, 1].var() / N + h[:, 1].var() / N)
    print(f"\n{k} − 동질  (TF-IDF 포함)")
    print(f"  평균 차이 {d.mean():+.5f} · 표준오차 {se:.5f} · "
          f"{'유의' if abs(d.mean()) > 2 * se else '구분 불가'}")
    print(f"  이질 쪽이 더 높은 비율 {(arr[k][:,1].mean() > h[:,1]).mean():.0%}")
