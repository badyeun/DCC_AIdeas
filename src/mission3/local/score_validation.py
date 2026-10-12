"""제출 폴더를 공식 Validation 에서 한 번 채점하고, 추론 시간을 실측합니다.

왜 Validation 을 쓰는가: 이것은 **제출 결정**을 위한 측정이고 튜닝이 아닙니다.
구성은 모두 Training 내부 홀드아웃에서 확정했습니다. 규칙이 금지하는 것은
Validation 데이터를 **학습에 쓰는 것**입니다. 다만 같은 세트를 반복해서 보고
고치면 그때부터는 사실상 거기에 맞추는 것이 되므로, 후보당 한 번만 잽니다.

추론 시간도 함께 냅니다. 이전에 보고한 "14분"은 계산값이어서 철회했고,
16개 구성은 base 11개의 2.6배로 **계산만** 해둔 상태입니다. 실제로 재야
제출 규격(시간 제한)에 답할 수 있습니다.

Run:  python src/mission3/local/score_validation.py --dir <제출 폴더> [--limit N]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
PY_EXE = r"C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe"


def macro_f1(y, p):
    per = []
    for i in range(len(TARGETS)):
        tp = int(((y[:, i] == 1) & (p[:, i] == 1)).sum())
        d = int(y[:, i].sum()) + int(p[:, i].sum())
        per.append(2 * tp / d if d else 0.0)
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    sub = Path(args.dir)
    if not sub.is_dir():
        raise SystemExit(f"제출 폴더가 없습니다: {sub}")
    n_ckpt = len(list(sub.glob("*.ckpt")))
    out = Path(args.out) if args.out else ROOT / "results" / "mission3" / "local" / \
        f"val_{sub.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONPATH=r"C:\DevBox\pylibs", PYTHONIOENCODING="utf-8",
               TRANSFORMERS_VERBOSITY="error", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1")
    cmd = [PY_EXE, str(ROOT / "src" / "mission3" / "local" / "inference.py"),
           "--audio_dir", "", "--label_dir", str(VAL), "--ckpt_path", str(sub),
           "--output", str(out), "--batch_size", str(args.batch_size)]
    print(f"폴더 {sub.name} · 체크포인트 {n_ckpt}개")
    print(f"추론 시작 {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"추론 실패 (exit {r.returncode})")

    truth = {}
    for p in sorted(VAL.rglob("*.json")):
        ann = json.loads(p.read_text(encoding="utf-8"))
        truth[p.name] = {str(s).strip() for s in (ann.get("symptom") or [])}
    pred = {}
    with out.open(encoding="utf-8-sig", newline="") as f:
        for name, sym in list(csv.reader(f))[1:]:
            pred[name] = set(ast.literal_eval(sym))
    missing = set(truth) - set(pred)
    if missing:
        raise SystemExit(f"예측에 빠진 통화 {len(missing)}개")

    names = sorted(truth)
    Y = np.array([[1 if s in truth[n] else 0 for s in TARGETS] for n in names], np.int8)
    P = np.array([[1 if s in pred[n] else 0 for s in TARGETS] for n in names], np.int8)
    score, per = macro_f1(Y, P)

    print(f"\n통화 {len(names):,}개 · 추론 {elapsed/60:.1f}분 "
          f"({elapsed/len(names)*1000:.0f} ms/통화 · 모델당 {elapsed/60/n_ckpt:.2f}분)")
    print(f"\n{'증상':<10}{'F1':>8}{'실제':>7}{'예측':>7}")
    for s, f1v, i in zip(TARGETS, per, range(9)):
        print(f"{s:<10}{f1v:>8.4f}{int(Y[:,i].sum()):>7,}{int(P[:,i].sum()):>7,}")
    print(f"\n*** Macro F1 @0.5 = {score:.5f} ***")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
