"""Reuse raw inference probabilities across experiments without reading labels.

Keys bind ordered inputs, weights, tokenizer, inference code and execution
settings. Calibration/thresholds are deliberately excluded: they are applied
after loading raw probabilities. A missing or damaged cache is recomputed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PredictionCache:
    def __init__(self, directory, paths, texts, tokenizer_dir, labels, execution):
        self.directory = Path(directory).expanduser().resolve()
        self.shape = (len(paths), len(labels))
        if len(paths) != len(texts):
            raise ValueError("Cache paths/texts length mismatch")
        tokenizer_dir = Path(tokenizer_dir)
        self.context = {
            "format": "mission3_raw_probabilities_v1",
            "inputs": hashlib.sha256(_json([
                [Path(path).name, text] for path, text in zip(paths, texts)
            ]).encode("utf-8")).hexdigest(),
            "labels": list(labels), "execution": execution,
            "tokenizer": {p.relative_to(tokenizer_dir).as_posix(): _file_hash(p)
                          for p in sorted(tokenizer_dir.rglob("*")) if p.is_file()},
        }

    def _valid(self, values):
        return (values.shape == self.shape and np.isfinite(values).all()
                and np.all((values >= 0) & (values <= 1)))

    def get(self, kind, checkpoint, config, sources, compute):
        # Do not bind absolute paths: a copied identical bundle can reuse results.
        identity = _json({
            **self.context, "kind": kind, "weights": _file_hash(checkpoint),
            "settings": {key: config[key] for key in
                         ("device", "max_length", "stride", "window_batch_size", "inference_batch_size")},
            "code": [_file_hash(path) for path in sources],
        })
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        path = self.directory / (key + ".npz")
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as stored:
                    values = np.array(stored["probabilities"], dtype=np.float64)
                    checksum = hashlib.sha256(values.tobytes()).hexdigest()
                    if (stored["identity"].item() == identity and self._valid(values)
                            and stored["checksum"].item() == checksum):
                        print(f"예측 캐시 재사용: {kind} / {Path(checkpoint).name}", flush=True)
                        return values
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                pass
            print(f"예측 캐시 손상: {kind}, 다시 계산합니다.", flush=True)
        values = np.asarray(compute(), dtype=np.float64)
        if not self._valid(values):
            raise ValueError("Invalid raw probabilities; cache was not written")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Unique temporary files keep interrupted/concurrent writes out of reads.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, identity=np.asarray(identity), probabilities=values,
                                    checksum=np.asarray(hashlib.sha256(values.tobytes()).hexdigest()))
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"예측 캐시 저장: {kind} / {Path(checkpoint).name}", flush=True)
        return values
