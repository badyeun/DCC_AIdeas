"""제출 체크포인트가 inference.py 의 모델 정의로 실제 적재되는지 확인한다.

학습기에 있고 추론기에 없는 헤드는 load_state_dict 에서 터진다 -- `cls` 헤드가
그렇게 잡혔다. large 급 멤버는 hidden 1024, 레이어 24, 그리고 TAPT 는 다른
체크포인트이므로 같은 확인을 다시 거친다. 오프라인으로, 네트워크 없이 된다는
것까지 본다(encoder_config 가 체크포인트 안에 있으므로).

Run:  python src/mission3/local/check_ckpt_loads.py [폴더 또는 파일...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inference import Classifier, TARGETS  # noqa: E402

R = HERE.parents[2] / "results" / "mission3" / "local"


def check(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = Classifier(blob["model_name"], len(blob["target_symptoms"]),
                       blob.get("dropout", 0.1), blob.get("head", "mean"),
                       encoder_config=blob.get("encoder_config"))
    missing, unexpected = [], []
    try:
        model.load_state_dict(blob["state_dict"])
    except RuntimeError as exc:
        return False, str(exc).splitlines()[0][:150], blob
    # 적재만으로는 부족하다 -- 실제로 순전파가 되는지 본다.
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, 100, (2, 32))
        mask = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask)
    if tuple(out.shape) != (2, len(TARGETS)):
        return False, f"출력 모양 {tuple(out.shape)} (기대 (2, {len(TARGETS)}))", blob
    return True, "", blob


def main():
    targets = [Path(a) for a in sys.argv[1:]] or [R / "submission_models"]
    files = []
    for t in targets:
        files += sorted(t.glob("*.ckpt")) if t.is_dir() else [t]
    if not files:
        raise SystemExit("확인할 체크포인트가 없습니다")
    print(f"{'체크포인트':<46}{'아키텍처':>22}{'헤드':>6}{'hidden':>8}{'판정':>7}")
    bad = 0
    for f in files:
        ok, why, blob = check(f)
        h = blob.get("encoder_config", {}).get("hidden_size", "?")
        print(f"{f.name:<46}{Path(blob['model_name']).name:>22}"
              f"{blob.get('head','mean')[:4]:>6}{h:>8}{'OK' if ok else '실패':>7}")
        if not ok:
            bad += 1
            print(f"    {why}")
    print(f"\n{len(files)}개 중 {len(files)-bad}개 적재·순전파 성공")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
