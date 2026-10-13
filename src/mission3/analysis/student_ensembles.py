"""증류 학생 앙상블을 Validation 에서 평가한다.

홀드아웃 검증은 음수였다 -- 학생 3개 앙상블이 교사 3개에 -0.00109(승률 15%),
원인은 다양성 손실(내부 불일치 0.0299 -> 0.0238)이었다. 다만 그 교사는 평균
멤버보다 +0.004 밖에 좋지 않았고, 16개 교사는 단일 모델보다 +0.012 좋다.
교사가 강하면 학생 개별 이득도 커지므로 다양성 손실을 넘어설 수 있어, 여기서
실측한다.

Validation 은 제출 결정용으로만 쓴다. 그래서 **미리 정한 구성만** 비교하고,
'가장 좋은 조합 찾기'는 하지 않는다. 무작위 k개 분포도 함께 내서, 특정
구성이 높게 나왔을 때 그게 잡음인지 판단할 수 있게 한다.

Run:  python src/mission3/analysis/student_ensembles.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results" / "mission3" / "local"
V = R / "val_probs"
VAL = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
T9 = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")

SHIP16 = ["roberta-base_raw_s42_FINAL", "roberta-base_raw_s1_FINAL",
          "roberta-base_attn_raw_s42_FINAL", "roberta-base_attn_raw_s1_FINAL",
          "roberta-base_raw_s11_D3", "roberta-base_attn_raw_s13_D4",
          "roberta-base_synonym+filler_s15_D5", "roberta-base_synonym+filler_s16_D5",
          "domain_adapted_full_raw_s17_D6", "domain_adapted_full_attn_raw_s18_D6",
          "roberta-base_cls_raw_s19_D7", "roberta-large_raw_s1_LGE2",
          "roberta-large_raw_s7_LGE2", "roberta-large_attn_raw_s42_LGE2",
          "roberta-large_synonym+filler_s42_LGF", "klue-roberta-large-tapt_raw_s42_LGF"]


def load_all():
    names = list(np.load(V / "_names.npy"))
    truth = {}
    for p in sorted(VAL.rglob("*.json")):
        ann = json.loads(p.read_text(encoding="utf-8"))
        truth[p.name] = {str(s).strip() for s in (ann.get("symptom") or [])}
    Y = np.array([[1 if s in truth[n] else 0 for s in T9] for n in names], np.int8)
    TF = np.load(R / "val_tfidf_probs.npy").astype(np.float64)
    P = {f.stem: np.load(f).astype(np.float64)
         for f in sorted(V.glob("*.npy")) if not f.stem.startswith("_")}
    return names, Y, TF, P


def main():
    names, Y, TF, P = load_all()
    def macro(M):
        pr = M >= .5
        out = []
        for i in range(9):
            tp = int(((Y[:, i] == 1) & pr[:, i]).sum())
            d = int(Y[:, i].sum()) + int(pr[:, i].sum())
            out.append(2 * tp / d if d else 0.0)
        return float(np.mean(out))
    sc = lambda ks: macro(.9 * np.mean([P[k] for k in ks], axis=0) + .1 * TF)
    cost = lambda ks: sum(3.04 if ("large" in k or "tapt" in k) else 1.0 for k in ks)

    students = sorted(k for k in P if "_S16" in k)
    split = sorted(k for k in P if k.endswith(("_TA", "_TB")))
    missing = [k for k in SHIP16 if k not in P]
    if missing:
        print(f"출하 16개 중 확률 없음: {missing}")
        return
    ref = sc(SHIP16)
    print(f"기준 출하 16개: {ref:.5f}  (모델 16 · 비용 {cost(SHIP16):.1f})\n")

    if students:
        print(f"=== 증류 학생 {len(students)}개 (교사 = 16개) ===")
        print(f"{'학생':<46}{'단독':>10}{'16개 대비':>11}")
        for k in students:
            s = sc([k])
            print(f"{k:<46}{s:>10.5f}{s-ref:>+11.5f}")
        print(f"\n{'구성':<46}{'점수':>10}{'16개 대비':>11}{'모델':>6}{'비용':>7}")
        for n in range(1, len(students) + 1):
            # 선택 없이 '앞에서 n개' 가 아니라 모든 조합 평균을 쓴다
            combos = list(itertools.combinations(students, n))
            v = np.array([sc(list(c)) for c in combos])
            print(f"{f'학생 {n}개 (조합 {len(combos)}개 평균)':<46}"
                  f"{v.mean():>10.5f}{v.mean()-ref:>+11.5f}{n:>6}"
                  f"{np.mean([cost(list(c)) for c in combos]):>7.1f}")
        allst = sc(students)
        print(f"{'학생 전부':<46}{allst:>10.5f}{allst-ref:>+11.5f}"
              f"{len(students):>6}{cost(students):>7.1f}")
        both = sc(SHIP16 + students)
        print(f"{'출하 16개 + 학생 전부':<46}{both:>10.5f}{both-ref:>+11.5f}"
              f"{len(SHIP16)+len(students):>6}{cost(SHIP16+students):>7.1f}")

    if split:
        print(f"\n=== 교사를 쪼갠 학생 {len(split)}개 ===")
        print(f"{'학생':<46}{'단독':>10}{'16개 대비':>11}")
        for k in split:
            s = sc([k])
            print(f"{k:<46}{s:>10.5f}{s-ref:>+11.5f}")
        s = sc(split)
        print(f"{'쪼갠 학생 전부':<46}{s:>10.5f}{s-ref:>+11.5f}{len(split):>6}")

    if students and split:
        s = sc(students + split)
        print(f"{'학생 전부 + 쪼갠 학생 전부':<46}{s:>10.5f}{s-ref:>+11.5f}"
              f"{len(students)+len(split):>6}")

    print("\n=== 예측 불일치 (다양성) ===")
    dis = lambda a, b: np.mean((P[a] >= .5) != (P[b] >= .5))
    for lab, grp in (("출하 16개", SHIP16), ("학생", students), ("쪼갠 학생", split)):
        if len(grp) > 1:
            print(f"  {lab:<16}내부 평균 {np.mean([dis(a,b) for a,b in itertools.combinations(grp,2)]):.4f}")

    if students:
        print("\n=== 무작위 k개 분포 (어떤 구성이 높게 나왔을 때 잡음인지 판단용) ===")
        pool = SHIP16 + students + split
        rng = np.random.default_rng(3)
        print(f"{'k':>3}{'평균':>10}{'표준편차':>9}{'최고':>10}")
        for k in (2, 4, 6, 8, 12, 16):
            v = np.array([sc(list(rng.choice(pool, k, replace=False))) for _ in range(150)])
            print(f"{k:>3}{v.mean():>10.5f}{v.std():>9.5f}{v.max():>10.5f}")


if __name__ == "__main__":
    main()
