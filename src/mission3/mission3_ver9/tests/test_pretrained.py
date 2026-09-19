"""Tiny real encoder/tokenizer serialization; no public weights or network."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from transformers import AutoModel, AutoTokenizer, DebertaV2Config, DebertaV2Model, BertTokenizerFast

from mission3_ver9 import pretrained


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n환자\n두통\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"))
    tokenizer.save_pretrained(source)
    model = DebertaV2Model(DebertaV2Config(vocab_size=7, embedding_size=8, hidden_size=8,
                                      num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
    model.save_pretrained(source)
    model_loader = AutoModel.from_pretrained
    tokenizer_loader = AutoTokenizer.from_pretrained
    downloads = []

    def load_model(location, **kwargs):
        if str(location) == pretrained.MODEL_ID:
            downloads.append(("model", kwargs))
            assert kwargs["revision"] == pretrained.MODEL_REVISION
            assert kwargs["local_files_only"] is False
            assert kwargs["trust_remote_code"] is False
            return model_loader(source, local_files_only=True)
        assert kwargs["local_files_only"] is True
        return model_loader(location, **kwargs)

    def load_tokenizer(location, **kwargs):
        if str(location) == pretrained.MODEL_ID:
            downloads.append(("tokenizer", kwargs))
            assert kwargs["revision"] == pretrained.MODEL_REVISION
            assert kwargs["local_files_only"] is False
            return tokenizer_loader(source, local_files_only=True)
        assert kwargs["local_files_only"] is True
        return tokenizer_loader(location, **kwargs)

    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    return downloads


def test_atomic_local_bundle_roundtrip_and_offline_reuse(tiny_source, tmp_path, monkeypatch):
    destination = tmp_path / "prepared"
    metadata = pretrained.prepare_pretrained(destination)
    assert metadata == pretrained.validate_pretrained(destination)
    assert metadata["revision"] == "47a3b00828c5dc2b8b080c6738ef7a58ebd419d5"
    assert "model.safetensors" in metadata["files"]
    assert "tokenizer.json" in metadata["files"]
    assert not (destination / "pytorch_model.bin").exists()
    assert sorted(kind for kind, _ in tiny_source) == ["model", "tokenizer"]
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in destination.iterdir()}
    monkeypatch.setattr(AutoModel, "from_pretrained", Mock(side_effect=AssertionError("no repeat model load/download")))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(side_effect=AssertionError("no repeat tokenizer download")))
    assert pretrained.prepare_pretrained(destination) == metadata
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in destination.iterdir()} == before


def test_corruption_detected_and_repaired_atomically(tiny_source, tmp_path):
    destination = tmp_path / "prepared"
    pretrained.prepare_pretrained(destination)
    path = destination / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="무결성"):
        pretrained.validate_pretrained(destination)
    pretrained.prepare_pretrained(destination)
    pretrained.validate_pretrained(destination)
    assert len(tiny_source) == 4
    assert not list(tmp_path.glob(".prepared.*"))


def test_failed_download_leaves_no_complete_destination(tiny_source, tmp_path, monkeypatch):
    destination = tmp_path / "prepared"
    monkeypatch.setattr(AutoModel, "from_pretrained", Mock(side_effect=RuntimeError("download interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        pretrained.prepare_pretrained(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".prepared.*"))


def test_failed_repair_preserves_previous_bundle(tiny_source, tmp_path, monkeypatch):
    destination = tmp_path / "prepared"
    pretrained.prepare_pretrained(destination)
    (destination / "config.json").write_text("corrupt", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(side_effect=RuntimeError("offline failure")))
    with pytest.raises(RuntimeError, match="offline failure"):
        pretrained.prepare_pretrained(destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before


def test_identity_mismatch_and_unmanaged_destination_are_not_overwritten(tiny_source, tmp_path):
    destination = tmp_path / "prepared"
    pretrained.prepare_pretrained(destination)
    with pytest.raises(ValueError, match="다릅니다"):
        pretrained.prepare_pretrained(destination, revision="a" * 40)
    with pytest.raises(ValueError, match="40자리"):
        pretrained.prepare_pretrained(tmp_path / "unpinned", revision="main")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="새 경로"):
        pretrained.prepare_pretrained(unrelated)
    assert (unrelated / "important.txt").read_text() == "preserve me"


def test_manifest_requires_full_files_and_rejects_links(tiny_source, tmp_path):
    destination = tmp_path / "prepared"
    pretrained.prepare_pretrained(destination)
    marker = destination / pretrained.MANIFEST_NAME
    metadata = json.loads(marker.read_text())
    changed = dict(metadata, complete=False)
    marker.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest"):
        pretrained.validate_pretrained(destination)
    marker.write_text(json.dumps(metadata))
    target = destination / "tokenizer.json"
    original = tmp_path / "tokenizer-outside.json"
    target.replace(original)
    target.symlink_to(original)
    with pytest.raises(ValueError, match="로컬"):
        pretrained.validate_pretrained(destination)
