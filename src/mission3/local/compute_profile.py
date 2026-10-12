"""제출 버전별 계산 프로파일 — 파라미터 수, 환경, batch size, 추론 시간.

조직위 안내 4개 항목을 버전별로 냅니다.

'Active 파라미터'는 보통 MoE 처럼 입력마다 일부만 켜지는 모델을 구분하려는
항목입니다. 이 제출물은 **앙상블**이므로 모든 멤버가 모든 통화를 읽습니다 --
즉 Total = Active 입니다. 작게 보이도록 쓸 여지가 없습니다.
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "results" / "mission3" / "local"
VERSIONS = {
    "11개 (기존)": "submission_final",
    "16개 fp32": "submission_final16",
    "16개 fp16": "submission_final16_fp16",
}
# 실측값 (score_validation.py, 3,640통화, batch_size 32)
MEASURED = {"submission_final": 2.8, "submission_final16": 6.2,
            "submission_final16_fp16": 6.1}
N_CALLS = 3640


def tfidf_params(path: Path):
    """TF-IDF + LogisticRegression 의 학습된 계수 수."""
    if not path.exists():
        return 0, 0
    try:
        with path.open("rb") as f:
            blob = pickle.load(f)
    except Exception as exc:        # sklearn DLL 차단 등
        print(f"    (TF-IDF 집계 실패: {type(exc).__name__}) ", end="")
        return -1, -1
    coef = feat = 0
    def walk(o):
        nonlocal coef, feat
        if hasattr(o, "coef_"):
            coef += int(o.coef_.size) + int(getattr(o, "intercept_", [0]).size
                                           if hasattr(o, "intercept_") else 0)
        if hasattr(o, "vocabulary_"):
            feat += len(o.vocabulary_)
        for a in ("estimators_", "steps", "transformer_list", "named_steps"):
            v = getattr(o, a, None)
            if isinstance(v, dict):
                v = list(v.values())
            if isinstance(v, (list, tuple)):
                for it in v:
                    walk(it[1] if isinstance(it, tuple) and len(it) == 2 else it)
    if isinstance(blob, dict):
        for v in blob.values():
            walk(v)
    else:
        walk(blob)
    return coef, feat


print("=== 환경 ===")
print(f"  OS           {platform.system()} {platform.release()} ({platform.machine()})")
print(f"  Python       {sys.version.split()[0]}")
print(f"  PyTorch      {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU          {p.name}  VRAM {p.total_memory/1024**3:.0f} GiB  "
          f"SM {p.major}.{p.minor}  CUDA {torch.version.cuda}")
    print(f"  bf16 지원    {torch.cuda.is_bf16_supported()}")
else:
    print("  GPU          (현재 세션에서 감지 안 됨)")
print(f"  CPU 코어     {torch.get_num_threads()} (torch threads)")

print("\n=== 버전별 파라미터 ===")
rows = []
for label, folder in VERSIONS.items():
    d = LOCAL / folder
    if not d.is_dir():
        print(f"  {label}: 폴더 없음"); continue
    total = 0
    per_arch = {}
    for f in sorted(d.glob("*.ckpt")):
        b = torch.load(f, map_location="meta" if False else "cpu", weights_only=False)
        nparam = sum(v.numel() for v in b["state_dict"].values()
                     if torch.is_tensor(v) and v.is_floating_point())
        total += nparam
        arch = Path(b["model_name"]).name
        per_arch.setdefault(arch, [0, 0])
        per_arch[arch][0] += 1
        per_arch[arch][1] = nparam
    coef, feat = tfidf_params(d / "tfidf.pkl")
    if coef < 0:
        coef = feat = 0
    size = sum(f.stat().st_size for f in d.rglob("*")) / 1e9
    rows.append((label, folder, total, coef, per_arch, size))
    print(f"\n  {label}  ({folder})")
    for arch, (cnt, each) in sorted(per_arch.items()):
        print(f"    {arch:<26} {cnt}개 × {each/1e6:,.1f}M = {cnt*each/1e6:,.1f}M")
    print(f"    {'TF-IDF + LogisticRegression':<26} 계수 {coef/1e6:,.2f}M "
          f"(어휘 {feat:,})")
    print(f"    {'합계 (Total = Active)':<26} {(total+coef)/1e6:,.1f}M "
          f"= {(total+coef)/1e9:.3f}B · 디스크 {size:.2f} GB")

print("\n=== 추론 (Validation 3,640통화, batch_size 32, RTX 5060 Ti, bf16) ===")
print(f"{'버전':<16}{'모델':>5}{'파라미터':>12}{'전체 시간':>11}{'통화당':>11}{'용량':>10}")
for label, folder, total, coef, per_arch, size in rows:
    mins = MEASURED.get(folder)
    n = sum(c for c, _ in per_arch.values())
    t = f"{mins:.1f}분" if mins else "-"
    per = f"{mins*60/N_CALLS*1000:.0f} ms" if mins else "-"
    print(f"{label:<16}{n:>5}{(total+coef)/1e6:>10,.0f}M{t:>11}{per:>11}{size:>9.2f}G")
