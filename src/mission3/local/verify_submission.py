"""제출 전 마지막 관문.

조립된 폴더를 실제 채점과 같은 방식으로 한 번 돌려 보고, 결과 CSV가 규격에
맞는지 확인합니다. 오늘 잡은 버그 넷이 전부 "에러 없이 조용히 틀리는"
종류였기 때문에, 돌아간다는 사실만으로는 부족합니다.

검사 항목
  1. 체크포인트가 전부 네트워크 없이 열리는가
  2. 멤버가 기대하는 전처리·아키텍처가 기록돼 있는가
  3. 임계값이 전부 0.5인가
  4. 추론이 실제로 돌아가는가 (오프라인)
  5. CSV 행 수가 입력 통화 수와 정확히 같은가
  6. CSV 형식이 규격과 같은가 (헤더, 증상 문자열)
  7. 예측된 증상이 9개 목록 안에만 있는가

Run:  python src/mission3/local/verify_submission.py [--n 30]
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
# 어느 폴더를 검증할지는 인자로 받습니다. 11개 폴더(Validation 0.65658)와
# 16개 폴더가 공존하는 동안, 검증한 폴더와 제출할 폴더가 어긋나면 안 됩니다.
DEFAULT_SUB = ROOT / "results" / "mission3" / "local" / "submission_final"
TRAIN = ROOT / "data" / "대학부 데이터" / "Training" / "2.라벨링데이터" / "TL_서울_구급"
TARGETS = {"고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"}
PY_EXE = r"C:\DevBox\python\cpython-3.12-windows-x86_64-none\python.exe"

ok = True


def check(label, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"  [{'OK' if passed else '실패'}] {label}" + (f" — {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="검증에 쓸 통화 수")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="기본 cpu. 학습이 GPU를 쓰는 동안에도 검증할 수 있게.")
    ap.add_argument("--dir", default=str(DEFAULT_SUB),
                    help="검증할 제출 폴더. 기본은 11개 구성.")
    args = ap.parse_args()

    SUB = Path(args.dir)
    if not SUB.is_dir():
        raise SystemExit(f"제출 폴더가 없습니다: {SUB}")
    print(f"제출 폴더: {SUB}\n")
    ckpts = sorted(SUB.glob("*.ckpt"))
    check("체크포인트가 존재한다", bool(ckpts), f"{len(ckpts)}개")
    if not ckpts:
        sys.exit(1)

    print("\n1~3. 체크포인트 메타데이터")
    families = {}
    for p in ckpts:
        b = torch.load(p, map_location="cpu", weights_only=False)
        key = (b.get("model_name"), b.get("head", "mean"), b.get("normalize", "raw"))
        families.setdefault(key, []).append(p.name)
        if b.get("threshold") != 0.5:
            check(f"{p.name} 임계값 0.5", False, str(b.get("threshold")))
        if "encoder_config" not in b:
            check(f"{p.name} 인코더 설정", False, "없음 → 추론 시 다운로드 발생")
    check("모든 체크포인트 임계값 0.5", all(
        torch.load(p, map_location="cpu", weights_only=False).get("threshold") == 0.5
        for p in ckpts))
    check("모든 체크포인트 오프라인 로드 가능", all(
        "encoder_config" in torch.load(p, map_location="cpu", weights_only=False)
        for p in ckpts))
    print(f"\n  계열 {len(families)}개:")
    for (name, head, norm), files in sorted(families.items(), key=lambda kv: str(kv[0])):
        print(f"    {Path(name).name:<20} head={head:<16} 전처리={norm:<16} {len(files)}개")
    check("계열이 둘 이상이다 (이질 구성)", len(families) >= 2, f"{len(families)}계열")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "calls"
        inp.mkdir()
        files = sorted(TRAIN.rglob("*.json"))[:args.n]
        for f in files:
            (inp / f.name).write_bytes(f.read_bytes())
        out = td / "mission3.csv"

        print(f"\n4. 오프라인 추론 ({len(files)}통화)")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONPATH=r"C:\DevBox\pylibs", PYTHONIOENCODING="utf-8")
        r = subprocess.run(
            [PY_EXE, str(ROOT / "src/mission3/local/inference.py"),
             "--audio_dir", "unused", "--label_dir", str(inp),
             "--ckpt_path", str(SUB), "--output", str(out),
             "--device", args.device],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        check("추론이 오프라인으로 완료된다", r.returncode == 0,
              (r.stderr or "")[-200:] if r.returncode else "")
        if r.returncode:
            print(r.stdout[-1500:])
            sys.exit(1)

        print("\n5~7. 출력 CSV")
        rows = list(csv.reader(out.open(encoding="utf-8-sig")))
        check("헤더가 규격과 같다", rows[0] == ["label file name", "symptom"], str(rows[0]))
        check("행 수가 입력 통화 수와 같다", len(rows) - 1 == len(files),
              f"{len(rows)-1} vs {len(files)}")
        names_in = {f.name for f in files}
        check("파일명이 입력과 정확히 일치한다", {r[0] for r in rows[1:]} == names_in)
        bad, empty = [], 0
        for name, sym in rows[1:]:
            try:
                lst = ast.literal_eval(sym)
            except (ValueError, SyntaxError):
                bad.append((name, sym)); continue
            if not isinstance(lst, list) or not set(lst) <= TARGETS:
                bad.append((name, sym))
            if not lst:
                empty += 1
        check("증상 칸이 전부 리스트 문자열이고 9개 안에 있다", not bad,
              str(bad[:2]) if bad else "")
        print(f"       빈 예측 {empty}건 / {len(rows)-1}건")
        print("\n  예시:")
        for r_ in rows[1:4]:
            print(f"    {r_[0]},{r_[1]}")

    size = sum(f.stat().st_size for f in SUB.rglob("*")) / 1e9
    print(f"\n제출 폴더 총 {size:.2f} GB")
    print("\n" + ("전체 통과 — 제출 가능" if ok else "실패 항목이 있습니다"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
