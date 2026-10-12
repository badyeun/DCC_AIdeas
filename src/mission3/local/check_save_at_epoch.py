"""--save_at_epoch 이 약속한 가중치를 저장했는지 확인한다.

3시간짜리 재학습 세 개를 걸기 전에, 새 플래그가 실제로 'epoch 2 가중치'를
저장하는지 봐야 한다. 확인하는 것:
  * 체크포인트가 saved_at_epoch = 2 를 기록했는가
  * --stop_at_save 가 epoch 3 을 돌리지 않았는가 (history 길이 2)
  * 저장된 가중치가 마지막 상태가 아니라 보관된 사본인가
마지막 항목은 history 가 2개인데 saved_at_epoch 가 2 이므로 간접 확인이다.
실패하면 0 이 아닌 코드로 끝나 큐가 본 실행으로 넘어가지 않는다.
"""
import json, sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"
# Find the file rather than predict its name. The tag is assembled from the
# model, head, preprocessing, max_length, train_frac and seed in an order this
# script has no business reproducing -- guessing it is what made the first
# run of this check report a missing checkpoint for a feature that worked.
suffix = sys.argv[1] if len(sys.argv) > 1 else "_SMOKE"
found = sorted((R / "submission_models").glob(f"*{suffix}.ckpt"),
               key=lambda f: f.stat().st_mtime)
if not found:
    print(f"검증 실패: *{suffix}.ckpt 가 submission_models 에 없습니다")
    sys.exit(1)
ck = found[-1]
tag = ck.stem
print(f"  검사 대상: {ck.name}")
hist = R / f"history_{tag}.json"
bad = []
if True:
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    got = blob.get("saved_at_epoch")
    if got != 2:
        bad.append(f"saved_at_epoch 가 2 가 아니라 {got!r} 입니다")
    else:
        print(f"  saved_at_epoch = {got}  (기대 2)")
    if "state_dict" not in blob or not blob["state_dict"]:
        bad.append("state_dict 가 비어 있습니다")
if not hist.exists():
    bad.append(f"history 가 없습니다: {hist.name}")
else:
    h = json.loads(hist.read_text(encoding="utf-8"))
    n = len(h["history"])
    if n != 2:
        bad.append(f"--stop_at_save 가 안 먹었습니다: epoch {n}개 기록")
    else:
        print(f"  epoch 기록 {n}개  (3 epoch 스케줄에서 2 에 멈춤)")
    if h["args"].get("epochs") != 3:
        bad.append(f"스케줄이 3 epoch 이 아닙니다: {h['args'].get('epochs')}")
    else:
        print(f"  스케줄 길이 = {h['args']['epochs']} epoch 유지")
if bad:
    print("\n--save_at_epoch 검증 실패:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("  --save_at_epoch 검증 통과")
