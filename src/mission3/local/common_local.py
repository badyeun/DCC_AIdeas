"""Shared loading, splitting and scoring for the local (own-GPU) experiments.

Rules this module enforces, from the competition brief:
  * Only Seoul Training is read for anything that influences a choice.
  * Official Validation is never touched here. A Training-internal holdout
    stands in for it, so repeated experiments cannot leak.
  * Model input is utterances[].text only.
  * The decision threshold is 0.5 for every symptom (organisers' rule).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
THRESHOLD = 0.5

ROOT = Path(r"C:\DevBox\projects\DCC_AIdeas")
TRAIN_DIR = ROOT / "data" / "대학부 데이터" / "Training" / "2.라벨링데이터" / "TL_서울_구급"
VALID_DIR = ROOT / "data" / "대학부 데이터" / "Validation" / "2.라벨링데이터" / "VL_서울_구급"
CACHE = ROOT / "src" / "mission3" / "local" / "_cache"


def load(directory: Path, cache_name: str):
    """Return (names, texts, Y). Parsed JSON is cached as .npz for speed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{cache_name}.npz"
    if cached.exists():
        z = np.load(cached, allow_pickle=True)
        return list(z["names"]), list(z["texts"]), z["Y"]
    names, texts, rows = [], [], []
    for path in sorted(directory.rglob("*.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        utterances = ann.get("utterances")
        if not isinstance(utterances, list):
            utterances = ann.get("utterences") or []
        said = [u["text"].strip() for u in utterances
                if isinstance(u, dict) and isinstance(u.get("text"), str) and u["text"].strip()]
        raw = {str(s).strip() for s in (ann.get("symptom") or [])}
        names.append(path.name)
        texts.append("\n".join(said))
        rows.append([1 if s in raw else 0 for s in TARGETS])
    Y = np.asarray(rows, dtype=np.int8)
    np.savez_compressed(cached, names=np.array(names), texts=np.array(texts, dtype=object), Y=Y)
    return names, texts, Y


def group_ids(texts):
    """Identical transcripts must never straddle a split."""
    return np.array([hashlib.sha256(re.sub(r"\s+", " ", t).strip().encode()).hexdigest()
                     for t in texts])


def split(texts, Y, seed=2026, holdout_fraction=0.2):
    """Grouped Training-internal split. This holdout replaces official Validation.

    ``texts`` MUST be the unmodified transcripts. Grouping on preprocessed text
    would make the split depend on the preprocessing, so two arms of an A/B
    would be scored on different calls and could not be compared -- which is
    exactly the bug this signature now guards against. Callers split first and
    normalise afterwards.
    """
    groups = group_ids(texts)
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    held = set(unique[: int(len(unique) * holdout_fraction)])
    is_held = np.array([g in held for g in groups])
    return np.flatnonzero(~is_held), np.flatnonzero(is_held)


def macro_f1(Y_true, Y_pred, per_symptom=False):
    """대회 평가 지표: 9개 증상 F1 의 단순 평균.

    증상별 데이터 양이 크게 다르지만(복통 838건 vs 두통 374건) Macro 평균은
    희귀한 증상도 같은 비중으로 봅니다. 그래서 흔한 증상만 잘 맞혀도 점수가
    오르지 않습니다.

    한 증상의 예측과 정답이 모두 없으면(tp=fp=fn=0) F1 을 0 으로 둡니다.
    분모가 0 이 되는 경우를 1.0 으로 보는 구현도 있는데, 그러면 "아무것도
    예측하지 않기"가 보상을 받으므로 여기서는 쓰지 않습니다.

    Args:
        Y_true: (통화 수, 9) 0/1 정답 행렬.
        Y_pred: (통화 수, 9) 0/1 예측 행렬. 확률이 아니라 **판정 결과**이며,
            판정은 전 증상 임계값 0.5 로 이루어집니다.
        per_symptom: True 면 (평균, 증상별 F1 리스트) 를 함께 돌려줍니다.

    Returns:
        Macro F1 (float), 또는 per_symptom=True 일 때 (float, list).
    """
    scores = []
    for i in range(len(TARGETS)):
        t, p = Y_true[:, i] == 1, Y_pred[:, i] == 1
        tp = int(np.count_nonzero(t & p))
        fp = int(np.count_nonzero(~t & p))
        fn = int(np.count_nonzero(t & ~p))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return (float(np.mean(scores)), scores) if per_symptom else float(np.mean(scores))


def report(tag, Y_true, P, force_one=False, out=print):
    """Score probabilities at the mandated 0.5, optionally forcing one symptom."""
    pred = (P >= THRESHOLD).astype(np.int8)
    if force_one:
        empty = ~pred.any(axis=1)
        if empty.any():
            pred[empty, P[empty].argmax(axis=1)] = 1
    score, per = macro_f1(Y_true, pred, per_symptom=True)
    out(f"{tag:<44} Macro F1 = {score:.5f}"
        + (f"   (빈 예측 {int((~(P >= THRESHOLD).any(axis=1)).sum())}건 강제 채움)" if force_one else ""))
    return score, per, pred
