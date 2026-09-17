"""Real offline encoder features, cache reuse, and recovery without a GPU."""
import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mission3_ver6 import features
from mission3_ver6.ver4_runtime.neural import tokenize_dialogues


@pytest.fixture
def feature_assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]
    vocabulary_path = tmp_path / "vocab.txt"
    vocabulary_path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary_path), do_lower_case=False)
    configuration = BertConfig(vocab_size=len(vocabulary), hidden_size=12, num_hidden_layers=1,
                               num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
                               hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
    torch.manual_seed(53)
    encoder = BertModel(configuration).eval()
    checkpoint_path = tmp_path / "fold_1.ckpt"
    torch.save({"encoder_config": configuration.to_dict(),
                "model_state": {"encoder." + name: tensor for name, tensor in encoder.state_dict().items()}},
               checkpoint_path)
    documents = tokenize_dialogues(["alpha omega " * 8, "beta", "", "alpha beta omega", "omega beta"],
                                    tokenizer, max_length=8, stride=2)
    return SimpleNamespace(tokenizer=tokenizer, encoder=encoder, checkpoint=checkpoint_path,
                           documents=documents, cache=tmp_path / "feature_cache",
                           config={"device": "cpu", "window_batch_size": 2})


def build(assets, **overrides):
    arguments = dict(documents=assets.documents, tokenizer=assets.tokenizer,
                     checkpoint_path=assets.checkpoint, cache_dir=assets.cache,
                     config=assets.config, shard_documents=2)
    arguments.update(overrides)
    return features.build_feature_store(**arguments)


def shard_snapshot(store):
    return {record["file"]: ((store.directory / record["file"]).read_bytes(),
                              (store.directory / record["file"]).stat().st_mtime_ns)
            for record in store.shards}


def test_cached_token_features_match_direct_encoder_and_preserve_masks(feature_assets):
    assets = feature_assets
    original_rng = torch.get_rng_state().clone()
    with patch.object(features.AutoModel, "from_pretrained", side_effect=AssertionError("No network downloads")):
        store = build(assets)
    assert torch.equal(torch.get_rng_state(), original_rng), "Feature cache misses must not change training dropout RNG"
    assert len(store) == len(assets.documents)
    assert len(assets.documents[0]) > 1  # Exercise overlapping long-dialogue windows.
    assert len(store.shards) == 3
    for row, document in enumerate(assets.documents):
        cached_windows = store[row]
        assert len(cached_windows) == len(document)
        for cached, original in zip(cached_windows, document):
            inputs = assets.tokenizer.pad([original], padding=True, return_tensors="pt")
            with torch.inference_mode():
                expected = assets.encoder(**inputs).last_hidden_state[0]
            assert cached["hidden"].dtype == torch.float16
            torch.testing.assert_close(cached["hidden"].float(), expected, atol=1e-3, rtol=1e-3)
            ids = original["input_ids"]
            expected_mask = torch.tensor([token not in (assets.tokenizer.cls_token_id,
                                                       assets.tokenizer.sep_token_id,
                                                       assets.tokenizer.pad_token_id) for token in ids])
            torch.testing.assert_close(cached["content_mask"], expected_mask)
    assert not store[2][0]["content_mask"].any()  # Empty text retains only special-token states.

    original = store[0][0]["hidden"].clone()
    returned_window = store[0][0]
    returned_window["hidden"].fill_(0)
    torch.testing.assert_close(store[0][0]["hidden"], original, atol=0, rtol=0)
    store.close()


def test_valid_shards_are_reused_without_constructing_encoder(feature_assets):
    store = build(feature_assets)
    directory, snapshot = store.directory, shard_snapshot(store)
    expected = store[-1][0]["hidden"].clone()
    store.close()
    with patch.object(features, "_encoder", side_effect=AssertionError("Valid cache must skip encoder loading")):
        reused = build(feature_assets)
    assert reused.directory == directory
    assert shard_snapshot(reused) == snapshot
    torch.testing.assert_close(reused[-1][0]["hidden"], expected, atol=0, rtol=0)
    reused.close()


def test_legacy_summaries_match_token_pooling_and_reuse_without_token_reads(feature_assets):
    store = build(feature_assets)
    token_windows = store[0]
    expected = [torch.cat([window["hidden"][0].float(), window["hidden"].float().mean(0)])
                for window in token_windows]
    summaries = store.get_legacy(0)
    assert len(summaries) == len(expected)
    for actual, reference in zip(summaries, expected):
        assert set(actual) == {"legacy_summary"}
        assert actual["legacy_summary"].dtype == torch.float32
        torch.testing.assert_close(actual["legacy_summary"], reference, atol=0, rtol=0)
    summaries[0]["legacy_summary"].fill_(0)
    with patch.object(features.FeatureStore, "__getitem__",
                      side_effect=AssertionError("A cached summary must not read all token states again")):
        repeated = store.get_legacy(0)
    for actual, reference in zip(repeated, expected):
        torch.testing.assert_close(actual["legacy_summary"], reference, atol=0, rtol=0)
    assert store.arrays and store.legacy_summaries
    store.close()
    assert store.arrays == {}
    assert store.legacy_summaries == {}


def test_one_corrupted_shard_is_rebuilt_and_other_shards_stay_untouched(feature_assets):
    store = build(feature_assets)
    directory, snapshot = store.directory, shard_snapshot(store)
    first_name = store.shards[0]["file"]
    store.close()
    damaged_path = directory / first_name
    damaged = bytearray(damaged_path.read_bytes())
    damaged[-1] ^= 0x40  # Change feature data while preserving the NumPy header and file size.
    damaged_path.write_bytes(damaged)
    modified = snapshot[first_name][1] + 10_000_000
    os.utime(damaged_path, ns=(modified, modified))
    with patch.object(features, "_encoder", wraps=features._encoder) as loader:
        repaired = build(feature_assets)
    assert loader.call_count == 1
    assert repaired.directory == directory
    assert damaged_path.read_bytes() == snapshot[first_name][0]
    after = shard_snapshot(repaired)
    for name, before in snapshot.items():
        if name != first_name:
            assert after[name] == before
    repaired.close()


def test_changed_input_tokens_use_a_new_cache_namespace(feature_assets):
    original = build(feature_assets)
    first_directory, before = original.directory, shard_snapshot(original)
    original_hidden = original[0][0]["hidden"].clone()
    changed = copy.deepcopy(feature_assets.documents)
    assert changed[0][0]["input_ids"][1] == feature_assets.tokenizer.convert_tokens_to_ids("alpha")
    changed[0][0]["input_ids"][1] = feature_assets.tokenizer.convert_tokens_to_ids("beta")
    with patch.object(features, "_encoder", wraps=features._encoder) as loader:
        alternate = build(feature_assets, documents=changed)
    assert loader.call_count == 1
    assert alternate.directory != first_directory
    assert alternate.identity["inputs"] != original.identity["inputs"]
    assert shard_snapshot(original) == before
    assert not torch.equal(alternate[0][0]["hidden"], original_hidden)
    original.close()
    alternate.close()


def test_cache_artifacts_contain_features_and_provenance_without_targets(feature_assets):
    store = build(feature_assets)
    index = json.loads((store.directory / "index.json").read_text(encoding="utf-8"))
    assert index["documents"] == len(feature_assets.documents)
    assert isinstance(index["identity"]["inputs"], str)
    assert len(index["identity"]["inputs"]) == 64  # Token input identity; no document contents in metadata.
    for path in store.directory.iterdir():
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            assert array.ndim == 2
            assert array.shape[1] == feature_assets.encoder.config.hidden_size
            assert array.dtype == np.float16
        else:
            assert path.suffix == ".json"
            metadata = json.loads(path.read_text(encoding="utf-8"))

            def check_keys(value):
                if isinstance(value, dict):
                    assert not {"labels", "targets", "symptom", "fit_targets", "stop_targets"} & value.keys()
                    for child in value.values():
                        check_keys(child)
                elif isinstance(value, list):
                    for child in value:
                        check_keys(child)

            check_keys(metadata)
    assert set(store[0][0]) == {"hidden", "content_mask"}
    store.close()


def test_insufficient_local_disk_fails_before_encoder_loading(feature_assets):
    with patch.object(features.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
         patch.object(features, "_encoder", side_effect=AssertionError("Disk preflight must run first")):
        with pytest.raises(OSError, match="free LOCAL disk space"):
            build(feature_assets)
    assert not list(feature_assets.cache.rglob("*.npy"))
