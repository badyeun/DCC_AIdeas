"""멤버를 똑같은 비중으로 평균하는 게 최선인가.

기여도가 +0.00001 에서 +0.00144 까지 100배 차이납니다. 기여 큰 멤버에
무게를 더 주면 나을 것 같지만, 1라운드에서 스태킹(학습된 결합)이 -0.033 로
크게 졌습니다. 그건 증상별 로지스틱이라 출력 분포를 재보정해 0.5 와 어긋난
탓이었습니다. 멤버 가중치는 분포를 바꾸지 않으므로(가중 평균은 여전히
확률의 볼록 결합) 다를 수 있습니다.

가중치를 홀드아웃에서 고르면 거기 과적합하므로 반분 검증으로 판정합니다.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
T = np.load(R / "tfidf_holdout_probs.npy")
L = lambda n: np.load(R / f"probs_{n}.npy")
SHIP = ["roberta-base_raw_s42_lr2e-5e3", "roberta-base_raw_s1_lr2e-5e3",
        "roberta-base_attn_raw_s42_lr2e5_LS", "roberta-base_attn_raw_s1_lr2e5_LS",
        "roberta-base_raw_s42", "roberta-base_attn_raw_s7",
        "roberta-base_synonym+filler_s42", "roberta-base_synonym+filler_s1",
        "domain_adapted_raw_s42_DAPT_LS", "domain_adapted_attn_raw_s42_DAPT_LS",
        "roberta-base_cls_raw_s42_LS"]
M = np.stack([L(n) for n in SHIP])          # (11, N, 9)


def sc(w, idx, tw=0.10):
    w = np.asarray(w, float); w = w / w.sum()
    P = (1 - tw) * np.tensordot(w, M, axes=(0, 0)) + tw * T
    return macro_f1(Y[idx], (P[idx] >= .5).astype(np.int8))


def loo_weights(idx):
    """그 절반에서 잰 기여도에 비례한 가중치."""
    eq = np.ones(len(M))
    full = sc(eq, idx)
    c = np.array([full - sc([0 if j == i else 1 for j in range(len(M))], idx)
                  for i in range(len(M))])
    c = np.clip(c, 0, None)
    return (c + c.mean()) if c.sum() > 0 else eq      # 0 기여 멤버도 완전히 빼지 않음


def greedy(idx, rounds=14):
    """기여 큰 멤버를 하나씩 더해 가는 탐욕적 선택(중복 허용 = 가중치)."""
    chosen = []
    cur = np.zeros(len(M))
    best = -1
    for _ in range(rounds):
        gains = []
        for i in range(len(M)):
            t = cur.copy(); t[i] += 1
            gains.append((sc(t, idx), i))
        g, i = max(gains)
        if g <= best:
            break
        best = g; cur[i] += 1; chosen.append(i)
    return cur if cur.sum() else np.ones(len(M))


rng = np.random.default_rng(271)
res = {"균등 (현재)": [], "기여도 비례": [], "탐욕적 선택": []}
for rep in range(40):
    o = rng.permutation(len(Y))
    for a, b in ((o[:len(Y)//2], o[len(Y)//2:]), (o[len(Y)//2:], o[:len(Y)//2])):
        eq = np.ones(len(M))
        base = sc(eq, b)
        res["균등 (현재)"].append(0.0)
        res["기여도 비례"].append(sc(loo_weights(a), b) - base)
        res["탐욕적 선택"].append(sc(greedy(a), b) - base)

print(f"반분 검증 {len(res['기여도 비례'])}회 — 한쪽에서 가중치 정하고 반대쪽에서 채점")
print(f"전체 홀드아웃 균등 가중: {sc(np.ones(len(M)), slice(None)):.5f}\n")
print(f"{'결합 방식':<18}{'ΔF1 평균':>11}{'표준오차':>10}{'승률':>7}")
for k, v in res.items():
    v = np.array(v)
    se = v.std(ddof=1) / np.sqrt(len(v)) if v.std() > 0 else 0.0
    print(f"{k:<18}{v.mean():>+11.5f}{se:>10.5f}{(v > 0).mean():>7.0%}")
