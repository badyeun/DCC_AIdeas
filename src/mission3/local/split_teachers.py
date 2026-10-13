"""교사를 계열로 쪼개 서로 다른 소프트 타깃 파일을 만든다.

같은 교사를 본 학생들은 서로 닮아져서 앙상블 이득이 사라졌다 -- 내부 불일치가
0.0299 에서 0.0238 로 떨어지고, 개별 이득 +0.00109 가 그만큼 상쇄됐다.
학생마다 다른 교사를 주면 그 수렴이 생기지 않는다.

쪼개는 기준은 **아키텍처**다. 점수를 보고 고른 게 아니라 앙상블이 원래
가지고 있는 구조이며, base 11개와 large 5개는 서로 가장 다른 두 묶음이다.

Run:  python src/mission3/local/split_teachers.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
SRC = R / "train_probs16"

SPLITS = {
    "soft_targets_base11.npy": lambda n: not ("large" in n or "tapt" in n),
    "soft_targets_large5.npy": lambda n: ("large" in n or "tapt" in n),
}


def main():
    files = sorted(SRC.glob("*.npy"))
    if not files:
        raise SystemExit(f"멤버별 Training 확률이 없습니다: {SRC}")
    print(f"멤버 {len(files)}개\n")
    for out, keep in SPLITS.items():
        picked = [f for f in files if keep(f.stem)]
        if not picked:
            print(f"  {out}: 해당 멤버 없음 — 건너뜀")
            continue
        acc = None
        for f in picked:
            q = np.load(f).astype(np.float64)
            acc = q if acc is None else acc + q
        soft = (acc / len(picked)).astype(np.float32)
        np.save(R / out, soft)
        print(f"  {out}  멤버 {len(picked)}개 · 평균 확률 {soft.mean():.4f}")
        for f in picked:
            print(f"      {f.stem}")
    # 두 교사가 실제로 다른 것을 말하는지 — 다르지 않으면 쪼개도 의미가 없다
    a = np.load(R / "soft_targets_base11.npy")
    b = np.load(R / "soft_targets_large5.npy")
    print(f"\n두 교사의 평균 절대차 {np.abs(a-b).mean():.4f} · "
          f"0.5 판정 불일치 {np.mean((a>=.5)!=(b>=.5)):.4f}")
    print("  (불일치가 0 에 가까우면 쪼개는 의미가 없습니다)")


if __name__ == "__main__":
    main()
