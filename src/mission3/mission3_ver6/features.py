"""One frozen encoder pass, reused by every head trial; no labels in the cache.

Completed shards are atomic and content-verified. A new session rebuilds only
missing shards; repeated trials mmap local float16 features without loading the
whole corpus into RAM. CLS/SEP are retained for the exact legacy pooling path.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


def native_bf16_supported():
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def amp_context(device):
    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16 if native_bf16_supported() else torch.float16)


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value), encoding="utf-8")
    temporary.replace(path)


def _encoder(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = dict(checkpoint["encoder_config"])
    model_type = config.pop("model_type")
    # Lazy extraction can run after restoring a head's RNG state. Constructing
    # a temporary encoder must not perturb that state's subsequent dropout.
    with torch.random.fork_rng(devices=[]):
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **config))
    state = {key.removeprefix("encoder."): value for key, value in checkpoint["model_state"].items()
             if key.startswith("encoder.")}
    if not state:
        raise ValueError("Base checkpoint contains no encoder")
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


class FeatureStore:
    """Sequence of documents, each a list of hidden/content_mask window dicts."""

    def __init__(self, directory, documents, special_ids, shards, identity):
        self.directory = Path(directory)
        self.documents = documents
        self.special_ids = set(special_ids)
        self.shards = shards
        self.identity = identity
        self.arrays = {}
        self.legacy_summaries = {}
        self.lookup = []
        for number, shard in enumerate(shards):
            cursor = 0
            for index in range(shard["start"], shard["end"]):
                windows = []
                for window in documents[index]:
                    length = len(window["input_ids"])
                    windows.append((cursor, cursor + length))
                    cursor += length
                self.lookup.append((number, windows))
            if cursor != shard["shape"][0]:
                raise ValueError("Feature shard token offsets disagree with input")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("FeatureStore indices must be integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        number, offsets = self.lookup[index]
        if number not in self.arrays:
            path = self.directory / self.shards[number]["file"]
            self.arrays[number] = np.load(path, mmap_mode="r", allow_pickle=False)
        values = self.arrays[number]
        result = []
        for window, (start, end) in zip(self.documents[index], offsets):
            ids = window["input_ids"]
            attention = window.get("attention_mask", [1] * len(ids))
            content = [bool(active) and token not in self.special_ids for token, active in zip(ids, attention)]
            # Copies only one window, avoiding writable aliases into the cache.
            result.append({"hidden": torch.from_numpy(np.array(values[start:end], copy=True)),
                           "content_mask": torch.tensor(content, dtype=torch.bool)})
        return result

    def get_legacy(self, index):
        """Keep only CLS+mean in RAM for A-C, avoiding repeated full-token I/O."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("FeatureStore indices must be integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.legacy_summaries:
            summaries = []
            for window in self[index]:
                hidden = window["hidden"].float()
                summaries.append(torch.cat([hidden[0], hidden.mean(0)]))
            self.legacy_summaries[index] = summaries
        return [{"legacy_summary": value.clone()} for value in self.legacy_summaries[index]]

    def close(self):
        self.arrays.clear()
        self.legacy_summaries.clear()


def build_feature_store(documents, tokenizer, checkpoint_path, cache_dir, config,
                        checkpoint_sha256=None, shard_documents=256):
    if not documents or any(not windows for windows in documents):
        raise ValueError("Features require nonempty documents/windows")
    if type(shard_documents) is not int or shard_documents < 1:
        raise ValueError("shard_documents must be positive")
    device = torch.device(config.get("device", "cpu"))
    batch_size = int(config.get("window_batch_size", 2))
    if batch_size < 1:
        raise ValueError("window_batch_size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction requires a GPU runtime")
    special_ids = sorted({int(value) for name in ("pad_token_id", "cls_token_id", "sep_token_id",
                                                  "bos_token_id", "eos_token_id")
                          if (value := getattr(tokenizer, name, None)) is not None})
    execution = {"device": str(device), "torch": torch.__version__,
                 "transformers": importlib.metadata.version("transformers"),
                 "window_batch_size": batch_size, "stored_dtype": "float16"}
    if device.type == "cuda":
        execution.update(gpu=torch.cuda.get_device_name(device),
                         capability=list(torch.cuda.get_device_capability(device)),
                         native_bf16=native_bf16_supported(), cuda_version=torch.version.cuda)
    identity = {"format": "mission3_frozen_features_v1", "execution": execution,
                "weights": checkpoint_sha256 or _hash(checkpoint_path),
                "inputs": hashlib.sha256(_json(documents).encode()).hexdigest(),
                "special_ids": special_ids, "shard_documents": shard_documents,
                "code": _hash(__file__)}
    signature = hashlib.sha256(_json(identity).encode()).hexdigest()
    directory = Path(cache_dir).expanduser().resolve() / signature
    directory.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    hidden_size = int(saved["encoder_config"]["hidden_size"])
    del saved
    shards, pending = [], []
    for number, start in enumerate(range(0, len(documents), shard_documents)):
        end = min(start + shard_documents, len(documents))
        tokens = sum(len(window["input_ids"]) for doc in documents[start:end] for window in doc)
        record = {"start": start, "end": end, "shape": [tokens, hidden_size], "file": f"{number:05d}.npy"}
        path = directory / record["file"]
        meta = path.with_suffix(".json")
        valid = False
        try:
            previous = json.loads(meta.read_text())
            if all(previous.get(key) == value for key, value in record.items()):
                stat = path.stat()
                valid = stat.st_size == previous["size"]
                if valid and stat.st_mtime_ns != previous["mtime_ns"]:
                    valid = _hash(path) == previous["sha256"]
                if valid:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    valid = list(array.shape) == record["shape"] and array.dtype == np.float16
                    del array
        except (OSError, ValueError, KeyError, EOFError):
            pass
        shards.append(record)
        if not valid:
            pending.append(number)
    required = sum(np.prod(shards[n]["shape"], dtype=np.int64) * 2 + 256 for n in pending)
    if required > shutil.disk_usage(directory).free:
        raise OSError(f"Frozen feature cache needs {required / 2**30:.2f} GiB of free LOCAL disk space")
    total_bytes = sum(np.prod(s["shape"], dtype=np.int64) * 2 for s in shards)
    print(f"동결 특징: {len(documents):,}개 통화 / 약 {total_bytes / 2**30:.2f} GiB / "
          f"재사용 {len(shards)-len(pending)}/{len(shards)} 조각", flush=True)
    encoder = None
    started = time.monotonic()
    try:
        if pending:
            encoder = _encoder(checkpoint_path).to(device).eval()
        for number in pending:
            record = shards[number]
            path = directory / record["file"]
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as f:
                    temporary = Path(f.name)
                array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                                 shape=tuple(record["shape"]))
                windows = [window for doc in documents[record["start"]:record["end"]] for window in doc]
                cursor = 0
                for start in range(0, len(windows), batch_size):
                    chunk = windows[start:start + batch_size]
                    inputs = tokenizer.pad(chunk, padding=True, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.inference_mode(), amp_context(device):
                        hidden = encoder(**inputs).last_hidden_state
                    for i, window in enumerate(chunk):
                        length = len(window["input_ids"])
                        values = hidden[i, :length].float().cpu().numpy().astype(np.float16)
                        if not np.isfinite(values).all():
                            raise FloatingPointError("Non-finite frozen token features")
                        array[cursor:cursor + length] = values
                        cursor += length
                if cursor != record["shape"][0]:
                    raise RuntimeError("Feature extraction lost token rows")
                array.flush()
                del array
                digest = _hash(temporary)
                temporary.replace(path)
                stat = path.stat()
                _write_json(path.with_suffix(".json"), {**record, "sha256": digest,
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                print(f"동결 특징 저장 {number+1}/{len(shards)}: "
                      f"통화 {record['end']:,}/{len(documents):,}, {(time.monotonic()-started)/60:.1f}분", flush=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    finally:
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(directory / "index.json", {"identity": identity, "signature": signature,
                "documents": len(documents), "size_bytes": int(total_bytes), "shards": shards})
    return FeatureStore(directory, documents, special_ids, shards, identity)
