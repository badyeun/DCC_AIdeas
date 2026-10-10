"""Is the 오심 label drifting with something outside the transcript?

If the rate at which annotators apply 오심 swings by month while the calls
themselves do not, the swing is annotator behaviour, not patient symptoms --
and no model reading only the transcript can track it. That part of the error
is permanently out of reach, and knowing its size stops us spending more GPU
hours chasing it.

Metadata is used HERE ONLY, for analysis. The model never sees anything but
utterances[].text.
"""
import sys, re, numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from common_local import TARGETS, TRAIN_DIR, load

names, texts, Y = load(TRAIN_DIR, "train")
dates = np.array([re.search(r"_(\d{8})\.json$", n).group(1) if re.search(r"_(\d{8})\.json$", n)
                  else "" for n in names])
print(f"날짜 파싱 성공 {int((dates != '').sum()):,} / {len(names):,}")
months = np.array([d[:6] for d in dates])
uniq = sorted(set(months) - {""})
print(f"월 {len(uniq)}개: {uniq[0]} ~ {uniq[-1]}\n")

print("=== 월별 증상 양성률 (전사 길이는 통제 삼아 함께 표시) ===")
hdr = f"{'월':<8}{'건수':>7}" + "".join(f"{t:>8}" for t in TARGETS) + f"{'평균길이':>9}"
print(hdr)
rates = {}
for m in uniq:
    k = months == m
    r = Y[k].mean(0)
    rates[m] = r
    L = np.mean([len(t) for t, kk in zip(texts, k) if kk])
    print(f"{m:<8}{int(k.sum()):>7}" + "".join(f"{v:>8.1%}" for v in r) + f"{L:>9.0f}")

print("\n=== 증상별 월간 변동폭 (최대-최소) ===")
M = np.array([rates[m] for m in uniq])
order = np.argsort(-(M.max(0) - M.min(0)))
print(f"{'증상':<8}{'최소':>8}{'최대':>8}{'변동폭':>9}{'평균':>8}{'상대변동':>10}")
for i in order:
    lo, hi, mu = M[:, i].min(), M[:, i].max(), M[:, i].mean()
    print(f"{TARGETS[i]:<8}{lo:>8.1%}{hi:>8.1%}{hi-lo:>9.1%}{mu:>8.1%}{(hi-lo)/mu:>10.2f}")
print("\n상대변동이 크면 '같은 증상인데 시기에 따라 다르게 적혔다'는 뜻입니다.")

print("\n=== 전사가 같은 말을 해도 월마다 라벨이 다른가 ===")
VOM = re.compile(r"울렁|메스|미식|느글")
has = np.array([bool(VOM.search(t)) for t in texts])
print(f"오심성 표현 포함 {int(has.sum()):,}건, 월별 오심 양성률:")
for m in uniq:
    k = (months == m) & has
    if k.sum() < 15: continue
    print(f"  {m}  {int(k.sum()):>4}건 → 오심 {Y[k, TARGETS.index('오심')].mean():>6.1%}")
