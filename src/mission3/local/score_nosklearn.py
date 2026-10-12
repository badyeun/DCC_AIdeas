"""sklearn 이 차단된 PC 에서 제출 폴더를 Validation 에서 채점한다 — 측정 전용.

이 PC 의 Smart App Control 이 sklearn 의 서명 없는 .pyd 를 막는다. 끄는 것은
되돌릴 수 없으므로 측정만 우회한다. **제출물은 바꾸지 않는다.**

우회 방법은 두 가지뿐이고 둘 다 제출 코드를 건드리지 않는다:
  1. transformers 가 요구하는 sklearn 심볼(roc_curve)만 대체본으로 채운다.
  2. inference.py 가 import 하는 `tfidf_model` 을 numpy 재구현으로 바꿔 끼운다.
inference.py 자체는 수정 없이 그대로 실행되므로, 이 스크립트가 실제 제출
경로를 검증한다.

정확성 검증: 실제 sklearn 이 동작할 때 만든 11개 구성의 Validation CSV 를
그대로 재현하는지 본다(--verify). 재현하면 numpy TF-IDF 가 맞다는 뜻이다.

Run:  python src/mission3/local/score_nosklearn.py --dir <폴더> [--verify <기준 CSV>]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VAL = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")


def install_stubs():
    """transformers 가 쓰는 sklearn 심볼만 채우고, tfidf_model 을 바꿔 끼운다."""
    for name in ("sklearn", "sklearn.metrics"):
        m = types.ModuleType(name)
        m.__spec__ = ModuleSpec(name, loader=None)
        m.__version__ = "1.9.1-stub"
        sys.modules[name] = m
    sys.modules["sklearn"].metrics = sys.modules["sklearn.metrics"]
    def _blocked(*a, **k):
        raise RuntimeError("stub: 측정 경로는 sklearn 함수를 쓰지 않습니다")
    sys.modules["sklearn.metrics"].roc_curve = _blocked
    sys.path.insert(0, str(HERE))
    import tfidf_nosklearn
    sys.modules["tfidf_model"] = tfidf_nosklearn      # inference.py 가 이걸 import 한다


def macro_f1(Y, P):
    per = []
    for i in range(len(TARGETS)):
        tp = int(((Y[:, i] == 1) & (P[:, i] == 1)).sum())
        d = int(Y[:, i].sum()) + int(P[:, i].sum())
        per.append(2 * tp / d if d else 0.0)
    return float(np.mean(per)), per


def read_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for name, sym in list(csv.reader(f))[1:]:
            out[name] = set(ast.literal_eval(sym))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--verify", default="", help="재현을 확인할 기준 CSV")
    args = ap.parse_args()
    sub = Path(args.dir)
    n_ckpt = len(list(sub.glob("*.ckpt")))
    out = Path(args.out) if args.out else \
        ROOT / "results" / "mission3" / "local" / f"val_{sub.name}_ns.csv"

    install_stubs()
    import inference
    print(f"폴더 {sub.name} · 체크포인트 {n_ckpt}개 · batch {args.batch_size}", flush=True)
    sys.argv = ["inference.py", "--audio_dir", "", "--label_dir", str(VAL),
                "--ckpt_path", str(sub), "--output", str(out),
                "--batch_size", str(args.batch_size)]
    t0 = time.time()
    inference.main()
    elapsed = time.time() - t0

    truth = {}
    for p in sorted(VAL.rglob("*.json")):
        ann = json.loads(p.read_text(encoding="utf-8"))
        truth[p.name] = {str(s).strip() for s in (ann.get("symptom") or [])}
    pred = read_csv(out)
    names = sorted(truth)
    Y = np.array([[1 if s in truth[n] else 0 for s in TARGETS] for n in names], np.int8)
    P = np.array([[1 if s in pred[n] else 0 for s in TARGETS] for n in names], np.int8)
    score, per = macro_f1(Y, P)

    print(f"\n통화 {len(names):,}개 · 추론 {elapsed/60:.1f}분 "
          f"({elapsed/len(names)*1000:.0f} ms/통화)")
    print(f"\n{'증상':<10}{'F1':>8}{'실제':>7}{'예측':>7}")
    for i, s in enumerate(TARGETS):
        print(f"{s:<10}{per[i]:>8.4f}{int(Y[:,i].sum()):>7,}{int(P[:,i].sum()):>7,}")
    print(f"\n*** Macro F1 @0.5 = {score:.5f} ***")

    if args.verify:
        ref = read_csv(args.verify)
        diff = sum(1 for n in names if ref.get(n) != pred.get(n))
        R = np.array([[1 if s in ref[n] else 0 for s in TARGETS] for n in names], np.int8)
        rscore, _ = macro_f1(Y, R)
        print(f"\n=== 재현 검증 (기준: {Path(args.verify).name}) ===")
        print(f"  기준 점수 {rscore:.5f} / 이번 {score:.5f}  차이 {score-rscore:+.5f}")
        print(f"  통화 단위 불일치 {diff} / {len(names):,}")
        print(f"  라벨 단위 불일치 {int((R != P).sum())} / {R.size:,}")
        ok = diff == 0
        print("  →", "완전 재현 — numpy TF-IDF 가 sklearn 과 동일합니다"
              if ok else "불일치 — 측정에 쓸 수 없습니다")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
