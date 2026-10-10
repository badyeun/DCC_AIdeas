"""오심과 구토가 실제로 어떻게 섞이는지.

처음 질문은 "오심은 너무 많이 답하고 구토는 너무 적게 답한다"였습니다. 그게
정말 둘 사이의 혼동인지, 아니면 각자 따로 틀리는 것인지를 가립니다.
"""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load, split, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
P = np.load(R / "_baseline_ens_probs.npy")
pred = (P >= .5).astype(np.int8)
O, G = TARGETS.index("오심"), TARGETS.index("구토")

print("=== 예측 편향 ===")
for i, t in ((O, "오심"), (G, "구토")):
    print(f"  {t}: 정답 {int(Y[:,i].sum()):>4} · 예측 {int(pred[:,i].sum()):>4} "
          f"({pred[:,i].sum()/Y[:,i].sum():.2f}배) · 재현율 "
          f"{(pred[:,i]&(Y[:,i]==1)).sum()/Y[:,i].sum():.3f} · 정밀도 "
          f"{(pred[:,i]&(Y[:,i]==1)).sum()/max(pred[:,i].sum(),1):.3f}")

print("\n=== 둘의 결합 혼동 행렬 (행=정답, 열=예측) ===")
def key(a, b):
    return {(0,0): "둘다없음", (1,0): "오심만", (0,1): "구토만", (1,1): "둘다"}[(a, b)]
cells = ["둘다없음", "오심만", "구토만", "둘다"]
print(f"  {'정답\예측':<10}" + "".join(f"{c:>10}" for c in cells) + f"{'합계':>8}")
for ta in (0, 1):
    for tb in (0, 1):
        row = [((Y[:,O]==ta)&(Y[:,G]==tb)&(pred[:,O]==pa)&(pred[:,G]==pb)).sum()
               for pa, pb in ((0,0),(1,0),(0,1),(1,1))]
        print(f"  {key(ta,tb):<10}" + "".join(f"{int(v):>10}" for v in row) + f"{int(sum(row)):>8}")

print("\n=== 오심 오답은 구토와 바꿔친 것인가 ===")
fp = (pred[:,O]==1) & (Y[:,O]==0)
fn = (pred[:,O]==0) & (Y[:,O]==1)
print(f"  오심 FP {int(fp.sum())}건 중 정답이 구토인 비율: {Y[fp,G].mean():.1%}"
      f"  (전체 구토 기저율 {Y[:,G].mean():.1%})")
print(f"  오심 FN {int(fn.sum())}건 중 모델이 구토를 찍은 비율: {pred[fn,G].mean():.1%}"
      f"  (전체 구토 예측률 {pred[:,G].mean():.1%})")
fpg = (pred[:,G]==1) & (Y[:,G]==0); fng = (pred[:,G]==0) & (Y[:,G]==1)
print(f"  구토 FN {int(fng.sum())}건 중 모델이 오심을 찍은 비율: {pred[fng,O].mean():.1%}"
      f"  (전체 오심 예측률 {pred[:,O].mean():.1%})")

print("\n=== 오심 FP가 실제로는 어떤 증상이었나 ===")
for i, t in sorted(enumerate(TARGETS), key=lambda x: -Y[fp, x[0]].mean()):
    print(f"  {t:<8} FP통화에서 {Y[fp,i].mean():>6.1%} · 전체 {Y[:,i].mean():>6.1%}"
          f" · 배율 {Y[fp,i].mean()/Y[:,i].mean():>5.2f}")

print("\n=== 전사에 구토성 표현이 있을 때의 성적 ===")
names, texts, Yall = load(TRAIN_DIR, "train")
tr, te = split(texts, Yall, seed=2026)
ht = [texts[i] for i in te]
VOM = re.compile(r"토했|토하|구토|게워|욱질")
NAU = re.compile(r"울렁|메스|미식|느글|토할")
for label, pat in (("구토성 표현", VOM), ("오심성 표현", NAU)):
    m = np.array([bool(pat.search(t)) for t in ht])
    if m.sum() < 20: continue
    print(f"\n  {label} 포함 {int(m.sum())}건:")
    for i, t in ((O, "오심"), (G, "구토")):
        tp = int((pred[m,i]&(Y[m,i]==1)).sum())
        print(f"    {t}: 정답 {int(Y[m,i].sum()):>4} · 예측 {int(pred[m,i].sum()):>4}"
              f" · F1 {2*tp/max(Y[m,i].sum()+pred[m,i].sum(),1):.3f}")
