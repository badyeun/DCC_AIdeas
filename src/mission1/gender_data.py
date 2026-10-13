"""Mission 1 로컬 실험용 데이터: Mission 2 발화별 캐시 + 통화별 성별 라벨.

- 발화별 특징은 디스크 배열(memmap, float16)을 **읽기만** 한다.
  `mission2_local/work/{split}_ecapa.npy`(192), `{split}_mfcc.npy`(261, 마지막 열 = log1p(발화 초)).
  이 배열은 특징 캐시 shard(`feature_cache/make_feature_cache.py`로 생성)를 `build_work_arrays`로 이어 붙인 것이다.
- 발화별 speaker(0 대원 / 1 신고자)와 통화 경계(counts)는 `mission2_embed_cache` shard에서 읽는다.
  Mission 1 규칙상 `speaker`는 추론 때도 쓸 수 있으므로 입력으로 써도 된다.
- 통화별 `gender`·파일 stem은 Colab에서 `export_labels_for_local.py`로 뽑은 `labels_{split}.npz`.
  캐시와 같은 순서(Mission 2 `collect_pairs`)이며, 발화 수가 한 통화라도 다르면 멈춘다.

예은 모델과 같은 통화로 비교하기 위해 예은의 내부 holdout(노트북 셀 80)을 그대로 재현한다:
stem 오름차순으로 놓은 Train 29,200통화에 `train_test_split(test_size=0.1, random_state=42, stratify=gender)`.
예은이 저장한 split 파일이 있으면 그것과 한 통화도 빠짐없이 같은지 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ECAPA_DIM = 192
MFCC_DIM = 261


def read_speaker_cache(embed_cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = [], []
    for index in range(10_000):
        path = embed_cache_dir / f"embed_{split}_{index:05d}.npz"
        if not path.is_file():
            break
        with np.load(path) as data:
            labels.append(data["y"].astype(np.int8))
            counts.append(data["counts"].astype(np.int64))
    if not labels:
        raise FileNotFoundError(f"shard가 없습니다: {embed_cache_dir}/embed_{split}_00000.npz")
    return np.concatenate(labels), np.concatenate(counts)


def build_work_arrays(mfcc_cache_dir: Path, embed_cache_dir: Path, work_dir: Path) -> None:
    """특징 캐시 shard들을 통화 순서 그대로 이어 붙여 float16 디스크 배열로 저장한다.

    `{work_dir}/{split}_ecapa.npy`(발화 × 192), `{split}_mfcc.npy`(발화 × 261). 학습 때 필요한 통화만
    memmap으로 읽어 RAM이 적은 PC에서도 학습할 수 있게 하려는 것이다(값은 캐시를 float16으로 반올림한 것).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for cache_dir, prefix, width, name in ((embed_cache_dir, f"embed_{split}", ECAPA_DIM, "ecapa"),
                                               (mfcc_cache_dir, split, MFCC_DIM, "mfcc")):
            paths = sorted(cache_dir.glob(f"{prefix}_[0-9][0-9][0-9][0-9][0-9].npz"))
            if not paths:
                raise FileNotFoundError(f"shard가 없습니다: {cache_dir}/{prefix}_00000.npz")
            rows = 0
            for path in paths:
                with np.load(path) as data:
                    rows += len(data["y"])
            output = np.lib.format.open_memmap(work_dir / f"{split}_{name}.npy", mode="w+", dtype=np.float16, shape=(rows, width))
            start = 0
            for path in paths:
                with np.load(path) as data:
                    block = data["x"]
                    output[start:start + len(block)] = block.astype(np.float16)
                    start += len(block)
            output.flush()
            del output
            print(f"[{split}] {name}: 발화 {rows:,} × {width}")


def load_split(m2_local: Path, export_dir: Path | None, split: str) -> dict:
    """export_dir가 None이면 라벨 없이 특징만 읽는다(풀링 특징 미리 계산용)."""
    speaker, counts = read_speaker_cache(m2_local / "mission2_embed_cache", split)
    ecapa = np.load(m2_local / "work" / f"{split}_ecapa.npy", mmap_mode="r")
    mfcc = np.load(m2_local / "work" / f"{split}_mfcc.npy", mmap_mode="r")
    if ecapa.shape != (len(speaker), ECAPA_DIM) or mfcc.shape != (len(speaker), MFCC_DIM):
        raise ValueError(f"[{split}] memmap 크기가 캐시와 다릅니다: {ecapa.shape}, {mfcc.shape}, 발화 {len(speaker)}")
    offsets = np.concatenate([[0], np.cumsum(counts)])
    data = {"ecapa": ecapa, "mfcc": mfcc, "speaker": speaker, "counts": counts, "offsets": offsets}
    if export_dir is None:
        return data
    with np.load(export_dir / f"labels_{split}.npz") as saved:
        stems = saved["stems"].astype(str)
        gender = saved["gender"].astype(np.int64)
        label_counts = saved["counts"].astype(np.int64)
    if not np.array_equal(label_counts, counts):
        bad = np.flatnonzero(label_counts != counts) if len(label_counts) == len(counts) else []
        raise ValueError(f"[{split}] JSON 발화 수와 캐시 counts가 다릅니다 (통화 {len(label_counts)} vs {len(counts)}, 불일치 {len(bad)}개)")
    if len(set(stems)) != len(stems):
        raise ValueError(f"[{split}] 파일 stem이 중복됩니다.")
    data.update(stems=stems, gender=gender)
    return data


def yeeun_holdout_mask(stems: np.ndarray, gender: np.ndarray, export_dir: Path | None = None) -> np.ndarray:
    """예은 내부 holdout(2,920통화)을 Mission 2 통화 순서의 bool 배열로 돌려준다."""
    from sklearn.model_selection import train_test_split

    order = np.argsort(stems, kind="stable")  # 예은 캐시 순서 = stem 오름차순
    sorted_gender = gender[order]
    _, held_sorted = train_test_split(np.arange(len(stems)), test_size=0.10, random_state=42, stratify=sorted_gender)
    if export_dir is not None:
        names_file = export_dir / "yeeun" / "wavlm_train_names_restored.npz"
        split_file = export_dir / "yeeun" / "internal_split_seed42.npz"
        if names_file.is_file():
            with np.load(names_file) as data:
                key = next(k for k in data.files if data[k].shape == (len(stems),))
                names = np.asarray([Path(str(n)).stem for n in data[key]])
            if not np.array_equal(names, stems[order]):
                raise ValueError("예은 캐시 순서가 stem 오름차순과 다릅니다.")
            print("확인: 예은 캐시 순서 = stem 오름차순")
        if split_file.is_file():
            with np.load(split_file) as data:
                saved = np.sort(data["val_idx"])
            if not np.array_equal(saved, np.sort(held_sorted)):
                raise ValueError("재현한 holdout이 예은의 split 파일과 다릅니다.")
            print("확인: 재현한 holdout = 예은 split 파일 (2,920통화)")
    mask = np.zeros(len(stems), dtype=bool)
    mask[order[held_sorted]] = True
    return mask


def pooled_features(data: dict) -> np.ndarray:
    """통화 하나를 고정 길이 벡터로 요약 (기준선용). 순서 정보는 쓰지 않는다.

    신고자 ECAPA(길이 1) 평균·표준편차, 신고자 MFCC 평균·표준편차, 대원 ECAPA·MFCC 평균
    (같은 회선의 기준점), 신고자 − 대원 MFCC 평균 차, 발화 수·총 길이.
    """
    rows = []
    for call in range(len(data["counts"])):
        span = slice(int(data["offsets"][call]), int(data["offsets"][call + 1]))
        ecapa = np.asarray(data["ecapa"][span], dtype=np.float32)
        ecapa /= np.maximum(np.linalg.norm(ecapa, axis=1, keepdims=True), 1e-8)
        mfcc = np.asarray(data["mfcc"][span], dtype=np.float32)
        role = data["speaker"][span] == 1
        caller_e, caller_m = ecapa[role], mfcc[role]
        agent_e = ecapa[~role] if (~role).any() else ecapa[role]
        agent_m = mfcc[~role] if (~role).any() else mfcc[role]
        seconds = np.expm1(caller_m[:, -1])
        weight = seconds / seconds.sum()
        rows.append(np.concatenate([
            (caller_e * weight[:, None]).sum(0), caller_e.std(0),
            (caller_m * weight[:, None]).sum(0), caller_m.std(0),
            agent_e.mean(0), agent_m.mean(0), caller_m.mean(0) - agent_m.mean(0),
            [np.log1p(role.sum()), np.log1p((~role).sum()), np.log1p(seconds.sum())],
        ]).astype(np.float32))
    return np.stack(rows)
