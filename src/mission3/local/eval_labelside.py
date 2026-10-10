"""Score the label-side runs against seed-matched baselines.

F1@0.5 is the competition metric and, after testing the alternatives, also the
best judge available. Average precision looked attractive because it is
quieter, but it correlates only 0.207 with F1@0.5 across 42 runs -- it rewards
ordering in the tail, which the frozen cut never reaches. The one useful
companion is the ranking ceiling: the F1 the same ordering would reach at the
best per-symptom threshold. It is five times quieter than F1@0.5 and still
correlates 0.372, so it is reported as a secondary signal, never as the
verdict. (A ceiling computed per symptom is a diagnostic; the submission
always decides at 0.5.)

Seed spread on F1@0.5 is 0.0047, so a single run cannot settle anything.
Effects are judged on paired seeds.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
Y = np.load(R / "holdout_Y.npy")
BASE = {42: "probs_roberta-base_raw_s42_lr2e-5e3.npy",
        1: "probs_roberta-base_raw_s1_lr2e-5e3.npy",
        7: "probs_roberta-base_raw_s7_lr2e-5e3.npy"}
NOISE_F1, NOISE_CEIL = 0.0047, 0.0010


def ceiling(p):
    out = []
    for i in range(9):
        best = 0.0
        for t in np.arange(.03, .97, .01):
            pr = p[:, i] >= t
            tp = int((pr & (Y[:, i] == 1)).sum()); d = int(Y[:, i].sum()) + int(pr.sum())
            best = max(best, 2 * tp / d if d else 0.0)
        out.append(best)
    return float(np.mean(out))


def metrics(P):
    f1, per = macro_f1(Y, (P >= .5).astype(np.int8), per_symptom=True)
    return f1, ceiling(P), per


def seed_of(stem):
    for part in stem.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return None


base = {s: metrics(np.load(R / f)) for s, f in BASE.items() if (R / f).exists()}
bf = np.mean([v[0] for v in base.values()])
print(f"기준선 (lr2e-5, 3 epoch, seed 42/1/7)   F1@0.5 {bf:.5f} · 순위상한 "
      f"{np.mean([v[1] for v in base.values()]):.5f}")
print(f"  판정 문턱: ΔF1 는 {NOISE_F1:.4f}, Δ순위상한은 {NOISE_CEIL:.4f}를 넘어야 신호\n")

runs = sorted(R.glob("probs_*_LS.npy"))
if not runs:
    raise SystemExit("아직 _LS 실행 결과가 없습니다.")
print(f"{'실행':<40}{'F1@0.5':>9}{'ΔF1':>9}{'순위상한':>10}{'Δ상한':>9}{'오심':>7}{'두통':>7}")
rows = {}
for p in runs:
    f1, ceil_, per = metrics(np.load(p))
    s = seed_of(p.stem)
    ref = base.get(s)
    d1, d2 = f1 - (ref[0] if ref else bf), ceil_ - (ref[1] if ref else 0)
    name = p.stem[6:-3]
    rows.setdefault(name.replace(f"_s{s}", ""), []).append(d1)
    print(f"{name:<40}{f1:>9.5f}{d1:>+9.5f}{ceil_:>10.5f}{d2:>+9.5f}{per[6]:>7.3f}{per[2]:>7.3f}")

print(f"\n{'설정별 요약 (seed 짝지음)':<40}{'n':>4}{'평균 ΔF1':>11}{'판정':>12}")
for k, v in sorted(rows.items(), key=lambda kv: -np.mean(kv[1])):
    m, n = float(np.mean(v)), len(v)
    verdict = ("판정 보류 (seed 1개)" if n == 1 else
               "○ 개선" if m > NOISE_F1 / np.sqrt(n) else
               "✗ 손해" if m < -NOISE_F1 / np.sqrt(n) else "− 잡음 안")
    print(f"{k:<40}{n:>4}{m:>+11.5f}{verdict:>12}")
