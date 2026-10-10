"""Collect every local run, test ensembling and post-processing, write a report.

All scores are on the Training-internal grouped holdout (5,840 calls) at the
mandated 0.5 threshold. Official Validation is untouched, so these numbers are
not directly comparable to ver4's 0.6566 -- see the caveat in the output.
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, macro_f1                                   # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
REPORT = OUT / "SUMMARY.md"
lines: list[str] = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def score(P, Y, force_one=True):
    pred = (P >= 0.5).astype(np.int8)
    blank = ~pred.any(axis=1)
    if force_one and blank.any():
        pred[blank, P[blank].argmax(axis=1)] = 1
    value, per = macro_f1(Y, pred, per_symptom=True)
    return value, per, int(blank.sum())


def best_of(P, Y):
    """Forcing a symptom onto empty rows does not always pay -- take whichever wins.

    Which one wins is itself a choice that has to be made on Training data, so
    the caller must not read this off the official Validation set.
    """
    return max(score(P, Y, force_one=False)[0], score(P, Y, force_one=True)[0])


def main():
    Y = np.load(OUT / "holdout_Y.npy")
    runs = {}
    for path in sorted(OUT.glob("probs_*.npy")):
        match = re.fullmatch(
            r"probs_(.+?)_(raw|synonym\+filler|synonym|privacy[^_]*)_s(\d+)(_\w+)?", path.stem)
        if not match:
            continue
        model, norm, seed, suffix = match.groups()
        norm = norm + (suffix or "")
        runs[(model, norm, int(seed))] = np.load(path)

    if not runs:
        say("완료된 학습 결과가 없습니다.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        return

    say("# 로컬 GPU 실험 결과 요약\n")
    say(f"자체검증 {len(Y):,}통화 · 임계값 전 증상 0.5 · 공식 Validation 미사용\n")

    say("## 1. 개별 학습 결과\n")
    say("| 모델 | 전처리 | seed | 그대로 | 최소1개 강제 | 빈 예측 |")
    say("|---|---|---:|---:|---:|---:|")
    for (model, norm, seed), P in sorted(runs.items()):
        plain, _, blank = score(P, Y, force_one=False)
        forced, _, _ = score(P, Y, force_one=True)
        say(f"| {model} | {norm} | {seed} | {plain:.5f} | **{forced:.5f}** | {blank} |")

    say("\n## 2. 정규화가 실제로 도움이 되는가\n")
    models = sorted({m for m, _, _ in runs})
    for model in models:
        seeds = sorted({s for m, n, s in runs if m == model and ("raw", s) and
                        (model, "raw", s) in runs and (model, "synonym+filler", s) in runs})
        if not seeds:
            continue
        say(f"\n### {model}\n")
        say("| seed | 원문 | 정규화 | 차이 |")
        say("|---:|---:|---:|---:|")
        deltas = []
        for seed in seeds:
            a = best_of(runs[(model, "raw", seed)], Y)
            b = best_of(runs[(model, "synonym+filler", seed)], Y)
            deltas.append(b - a)
            say(f"| {seed} | {a:.5f} | {b:.5f} | {b - a:+.5f} |")
        mean = float(np.mean(deltas))
        say(f"\n평균 차이 **{mean:+.5f}** · 이긴 seed {sum(d > 0 for d in deltas)}/{len(deltas)}")
        if len(deltas) > 1:
            spread = max(deltas) - min(deltas)
            say(f"seed별 편차 {spread:.5f} — "
                + ("차이가 편차보다 커서 실효 있음" if abs(mean) > spread
                   else "편차 안이라 이 표본으로는 단정 불가"))

    say("\n## 3. 앙상블 (확률 평균)\n")
    singles = {key: best_of(P, Y) for key, P in runs.items()}
    best_single = max(singles.values())
    mean_single = float(np.mean(list(singles.values())))

    def ensemble(keys):
        return best_of(np.mean([runs[k] for k in keys], axis=0), Y)

    say("미리 정한 구성만 봅니다. 수십 개 조합 중 최고를 고르면 그 숫자는 "
        "검증 세트에 맞춘 값이 되어 실제 성능을 과대평가합니다.\n")
    say("| 구성 | 모델 수 | Macro F1 | 단일 평균 대비 |")
    say("|---|---:|---:|---:|")
    groups = {
        "원문 전체": [k for k in runs if k[1] == "raw"],
        "정규화 전체": [k for k in runs if k[1] == "synonym+filler"],
        "base 전체": [k for k in runs if k[0] == "roberta-base"],
        "large 전체": [k for k in runs if k[0] == "roberta-large"],
        "전부": list(runs),
    }
    for label, keys in groups.items():
        if len(keys) < 2:
            continue
        value = ensemble(keys)
        say(f"| {label} | {len(keys)} | **{value:.5f}** | {value - mean_single:+.5f} |")

    say(f"\n단일 모델: 평균 {mean_single:.5f} · 최고 {best_single:.5f}")

    say("\n### 모델 수에 따른 앙상블 효과\n")
    say("각 크기마다 조합을 무작위로 뽑아 평균낸 값입니다. 최고값이 아니라 "
        "기댓값이라 실제로 기대할 수 있는 이득에 가깝습니다.\n")
    say("| 모델 수 | 표본 조합 | 평균 | 최저 | 최고 |")
    say("|---:|---:|---:|---:|---:|")
    keys = sorted(runs)
    rng = np.random.default_rng(0)
    for size in range(2, len(keys) + 1):
        # Enumerating every combination is 2^N; sample instead once that bites.
        total = 1
        for i in range(size):
            total = total * (len(keys) - i) // (i + 1)
        if total <= 60:
            picks = list(combinations(keys, size))
        else:
            picks = [tuple(rng.choice(len(keys), size, replace=False)) for _ in range(60)]
            picks = [tuple(keys[i] for i in p) for p in picks]
        values = [ensemble(p) for p in picks]
        say(f"| {size} | {len(values)} | **{np.mean(values):.5f}** | "
            f"{min(values):.5f} | {max(values):.5f} |")

    say("\n## 4. 최소 1개 증상 강제 출력의 효과\n")
    say("| 구성 | 그대로 | 강제 | 이득 |")
    say("|---|---:|---:|---:|")
    for key, P in sorted(runs.items()):
        plain = score(P, Y, force_one=False)[0]
        forced = score(P, Y, force_one=True)[0]
        say(f"| {key[0]}/{key[1]}/s{key[2]} | {plain:.5f} | {forced:.5f} | {forced - plain:+.5f} |")

    gains = [score(P, Y, True)[0] - score(P, Y, False)[0] for P in runs.values()]
    helped = sum(g > 0 for g in gains)
    say(f"\n최소1개 강제가 이득인 실행: {helped}/{len(gains)} · 평균 {np.mean(gains):+.5f}")
    if helped < len(gains):
        say("→ 항상 이득은 아닙니다. 빈 예측이 적고 모델이 잘 맞출수록, "
            "억지로 채운 한 개가 정답보다 오답일 확률이 높아집니다. "
            "쓸지 말지는 Training 내부에서 정해야 합니다.")

    best_key = max(singles, key=singles.get)
    say(f"\n## 5. 증상별 (최고 단일: {best_key[0]}/{best_key[1]}/s{best_key[2]})\n")
    _, per, _ = score(runs[best_key], Y, force_one=False)
    ver4 = {"고열": .6973, "구토": .6088, "두통": .5497, "복통": .8254, "어지러움": .6590,
            "열상": .8936, "오심": .4031, "전신쇠약": .5933, "호흡곤란": .6795}
    say("| 증상 | 로컬 | ver4 (공식 Validation) | 참고 차이 |")
    say("|---|---:|---:|---:|")
    for symptom, value in zip(TARGETS, per):
        say(f"| {symptom} | {value:.4f} | {ver4[symptom]:.4f} | {value - ver4[symptom]:+.4f} |")

    say("\n> ver4 열은 **공식 Validation 3,640통화** 기준이고 로컬 열은 "
        "**Training 내부 자체검증 5,840통화** 기준입니다. 평가 대상이 다르므로 "
        "직접 우열을 가릴 수 없고, 증상별 경향을 보는 용도로만 씁니다.")

    say("\n## 6. 학습 이력\n")
    for path in sorted(OUT.glob("history_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        arguments, best = data["args"], data["best"]
        total = sum(h["minutes"] for h in data["history"])
        peak = best.get("best_macro_f1", best["macro_f1_forced"])
        curve = " → ".join(f"{h['macro_f1']:.4f}" for h in data["history"])
        say(f"- `{path.stem[8:]}` — 최고 epoch {best['epoch']}, {peak:.5f}, "
            f"총 {total:.1f}분 · epoch별 {curve}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n보고서: {REPORT}")


if __name__ == "__main__":
    main()
