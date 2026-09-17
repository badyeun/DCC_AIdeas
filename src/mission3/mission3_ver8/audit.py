"""Training-calibration error review; never edit labels or feed reviews to training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np

if __package__:
    from .ver4_runtime.data import TARGET_SYMPTOMS, write_json
else:
    from ver4_runtime.data import TARGET_SYMPTOMS, write_json


FOCUS = ("오심", "두통", "전신쇠약")


def write_audit(directory, paths, texts, targets, probabilities, thresholds, *, seed=42, per_error=20):
    """Random, reproducible FP/FN samples, not just extreme-confidence examples.

    The caller passes only the original Training calibration rows and the frozen
    original baseline. These are descriptive errors at fitted thresholds, not an
    unbiased score. Human review columns stay blank and are not model inputs.
    """
    truth, probs = np.asarray(targets), np.asarray(probabilities)
    thresholds = np.asarray(thresholds)
    if (truth.shape != (len(paths), len(TARGET_SYMPTOMS)) or probs.shape != truth.shape
            or len(texts) != len(paths) or thresholds.shape != (len(TARGET_SYMPTOMS),)
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(probs).all()
            or np.any((probs < 0) | (probs > 1)) or not np.isfinite(thresholds).all()
            or type(per_error) is not int or per_error < 1):
        raise ValueError("Invalid audit rows, targets, probabilities or sample count")
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate audit filenames")
    predicted = probs >= thresholds
    records, counts = [], {}
    for col, label in enumerate(TARGET_SYMPTOMS):
        fp = np.flatnonzero((truth[:, col] == 0) & predicted[:, col]).tolist()
        fn = np.flatnonzero((truth[:, col] == 1) & ~predicted[:, col]).tolist()
        counts[label] = {"tp": int(np.sum((truth[:, col] == 1) & predicted[:, col])),
                         "fp": len(fp), "fn": len(fn), "support": int(truth[:, col].sum())}
        if label not in FOCUS:
            continue
        for error_index, (kind, indices) in enumerate((("FP", fp), ("FN", fn))):
            indices.sort(key=lambda i: names[i])
            rng = random.Random(seed + col * 1009 + error_index)
            selected = rng.sample(indices, min(per_error, len(indices)))
            counts[label]["review_" + kind.lower()] = len(selected)
            for index in selected:
                records.append({"symptom": label, "error_type": kind, "label_file_name": names[index],
                                "gold": int(truth[index, col]), "prediction": int(predicted[index, col]),
                                "probability": float(probs[index, col]), "threshold": float(thresholds[col]),
                                "all_target_labels": json.dumps([s for s, yes in zip(TARGET_SYMPTOMS, truth[index]) if yes], ensure_ascii=False),
                                "transcript": texts[index], "review_category": "", "review_note": ""})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve hand-written reviews across resumes. Regenerate only the template.
    template = directory / "review_samples_template.csv"
    temporary = template.with_suffix(".tmp")
    fields = ["symptom", "error_type", "label_file_name", "gold", "prediction", "probability", "threshold",
              "all_target_labels", "transcript", "review_category", "review_note"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(template)
    summary = {"scope": "Original Training calibration at frozen baseline fitted thresholds; descriptive, not cross-fit",
               "samples": len(paths), "seed": seed, "requested_per_error": per_error,
               "review_rows": len(records), "per_symptom": counts,
               "labels_modified": False, "reviews_used_for_training": False,
               "note": "Absent text evidence or an apparent label mismatch is not a confirmed annotation error."}
    write_json(directory / "summary.json", summary)
    guide = """# Training 내부 오답 검토

`review_samples_template.csv`는 오심·두통·전신쇠약별 FP 20개/FN 20개를
무작위 추출하는 기본 설정입니다. 해당 오류가 부족하면 있는 만큼만 추출합니다.
확신이 높은 오답만 뽑는 방식이 아닙니다. 이 CSV는 원본 정답을 변경하지 않습니다.

검토할 때는 `review_samples.csv`로 복사하여 review_category와 review_note를 채우세요.
재실행은 template만 갱신하며, 복사본은 읽거나 덮어쓰지 않습니다.

- `model_missed_evidence`: 전사에 근거가 있는데 모델이 놓침
- `question_negation_history`: 질문·부정·과거 증상을 잘못 해석
- `text_label_unclear`: 전사만으로 정답 근거 또는 라벨링 기준을 판단하기 어려움
- `other`: 기타; 구체적인 근거를 review_note에 기록

정답 불일치 의심과 확정된 라벨 오류를 구분하세요. 키워드 유무만으로 자동 판정하지 않습니다.
문장·원본 라벨·가중치·임계값을 이 검토 결과로 자동 변경하는 기능은 없습니다.
이 오류 집계는 원래 보정 집합에 이미 맞춘 임계값 기준이며 새 모델 성능 추정치가 아닙니다.
"""
    (directory / "README.md").write_text(guide, encoding="utf-8")
    return summary
