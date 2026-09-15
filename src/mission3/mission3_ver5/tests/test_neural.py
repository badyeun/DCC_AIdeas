"""Actual tiny local-transformer tests; no hub downloads or GPU are needed."""
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mission3_ver5 import neural
from mission3_ver5.ver4_runtime.neural import DialogueClassifier as LegacyClassifier


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    config = BertConfig(vocab_size=len(vocabulary), hidden_size=12, num_hidden_layers=1,
                        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
                        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    torch.manual_seed(7)
    encoder = BertModel(config)
    legacy = LegacyClassifier(encoder, dropout=0.1, window_batch_size=2)
    checkpoint = tmp_path / "fold_1.ckpt"
    torch.save({"format_version": 1, "encoder_config": config.to_dict(),
                "architecture": {"num_labels": 9, "dropout": 0.1, "window_batch_size": 2},
                "model_state": legacy.state_dict(), "model_name": "local-tiny-bert",
                "epoch": 3, "seed": 1051, "stop_ap": 0.5}, checkpoint)
    documents = neural.tokenize_dialogues(
        ["alpha " * 25 + "omega", "beta", "alpha beta " * 6,
         "omega alpha beta", "alpha beta " * 8], tokenizer, max_length=12, stride=3,
    )
    targets = ((np.arange(5)[:, None] + np.arange(9)[None, :]) % 3 == 0).astype(np.float32)
    stop_targets = np.array([[1] * 9, [0] * 9], dtype=np.float32)
    settings = {"device": "cpu", "warmup_epochs": 1, "finetune_epochs": 1,
                "patience": 2, "attention_dim": 8, "batch_size": 2,
                "window_batch_size": 2, "inference_batch_size": 2,
                "gradient_accumulation_steps": 2, "gradient_checkpointing": True,
                "dropout": 0.1, "learning_rate": 1e-3, "head_learning_rate": 1e-3,
                "warmup_head_learning_rate": 2e-3, "warmup_ratio": 0,
                "weight_decay": 0, "layerwise_lr_decay": 1, "log_every": 0}
    return tokenizer, legacy, checkpoint, documents, targets, stop_targets, settings


def train(assets, output, **kwargs):
    tokenizer, _, checkpoint, documents, targets, stop_targets, settings = assets
    return neural.train_model(documents, targets, documents[:2], stop_targets,
                              tokenizer, settings, output, seed=42,
                              init_checkpoint_path=checkpoint, run_signature=kwargs.pop("run_signature", "same-run"),
                              **kwargs)


def test_zero_residual_exact_warmstart_and_offline_architecture(assets):
    tokenizer, legacy, checkpoint, documents, _, _, settings = assets
    with patch.object(neural.AutoModel, "from_pretrained", side_effect=AssertionError("network forbidden")):
        model, metadata = neural.warm_start(checkpoint, tokenizer, settings)
    legacy.eval()
    model.eval()
    batch = neural.DialogueCollator(tokenizer)([{"windows": doc} for doc in documents[:2]])
    with torch.no_grad():
        torch.testing.assert_close(model(**batch), legacy(**batch), rtol=0, atol=0)
    assert metadata["init_epoch"] == 3
    assert not model.token_evidence_weights.detach().any()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_labelwise_token_mask_excludes_padding_cls_sep_and_empty_is_finite(assets):
    tokenizer, _, checkpoint, _, _, _, settings = assets
    model, _ = neural.warm_start(checkpoint, tokenizer, settings)
    model.eval()
    hidden = torch.randn(2, 5, 12, requires_grad=True)
    ids = torch.tensor([[tokenizer.cls_token_id, 5, 6, tokenizer.sep_token_id, tokenizer.pad_token_id],
                        [tokenizer.cls_token_id, tokenizer.sep_token_id, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    pooled = model._pool_tokens(hidden, ids, mask)
    altered = hidden.detach().clone()
    altered[0, [0, 3, 4]] += 1000
    altered[1] -= 1000
    torch.testing.assert_close(pooled, model._pool_tokens(altered, ids, mask), rtol=0, atol=0)
    assert pooled.shape == (2, 9, 8)
    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(pooled[1], torch.zeros_like(pooled[1]), rtol=0, atol=0)
    pooled.sum().backward()
    assert hidden.grad[0, 1:3].abs().sum() > 0
    assert not hidden.grad[0, [0, 3, 4]].any()
    assert not hidden.grad[1].any()
    # CPU BF16 autocast exercises the same normalization precision guarantee
    # as the GPU path without requiring accelerator access in these tests.
    with patch("torch.softmax", wraps=torch.softmax) as softmax:
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            mixed_precision = model._pool_tokens(hidden.detach(), ids, mask)
    assert softmax.call_args.args[0].dtype == torch.float32
    assert torch.isfinite(mixed_precision).all()
    assert not mixed_precision[1].any()


def test_long_dialogue_tail_reaches_residual_and_window_microbatch_equivalence(assets):
    tokenizer, _, checkpoint, _, _, _, settings = assets
    texts = ["alpha " * 67 + "omega", ""]
    documents = neural.tokenize_dialogues(texts, tokenizer, max_length=12, stride=3)
    reconstructed = documents[0][0]["input_ids"][1:-1]
    for window in documents[0][1:]:
        reconstructed += window["input_ids"][1:-1][3:]
    assert reconstructed == tokenizer(texts[0], add_special_tokens=False)["input_ids"]
    model, _ = neural.warm_start(checkpoint, tokenizer, settings)
    with torch.no_grad():
        model.token_evidence_weights.fill_(0.1)
    model.eval()
    model.window_batch_size = 1
    together = copy.deepcopy(model)
    together.window_batch_size = 100
    batch = neural.DialogueCollator(tokenizer)([{"windows": doc} for doc in documents])
    logits = model(**batch)
    torch.testing.assert_close(logits, together(**batch), rtol=1e-5, atol=1e-6)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    tail_id = tokenizer.convert_tokens_to_ids("omega")
    assert model.encoder.embeddings.word_embeddings.weight.grad[tail_id].abs().sum() > 0
    assert model.token_projection[0].weight.grad.abs().sum() > 0


def test_actual_warmup_finetune_resume_rng_and_best_fingerprint(assets, tmp_path):
    tokenizer, legacy, _, documents, _, _, settings = assets
    full_path = tmp_path / "full.ckpt"
    full_result = train(assets, full_path)
    assert [row["phase"] for row in full_result["history"]] == ["baseline", "warmup", "finetune"]
    assert full_result["completed_epochs"] == 2
    assert full_result["best_epoch"] in (0, 1, 2)
    assert full_result["history"][0]["train_loss"] is None
    full_last = torch.load(tmp_path / "full_last.ckpt", map_location="cpu", weights_only=True)
    assert full_last["completed"]
    assert {"optimizer", "scheduler", "scaler", "rng"} <= full_last.keys()

    resumed_path = tmp_path / "resumed.ckpt"
    original_save = neural._atomic_save

    def interrupt_after_committed_warmup(value, path):
        original_save(value, path)
        if value.get("format") == neural.LAST_FORMAT and value["completed_epochs"] == 1:
            raise KeyboardInterrupt("Simulated stop after committing the epoch, before writing best")

    with patch.object(neural, "_atomic_save", side_effect=interrupt_after_committed_warmup):
        with pytest.raises(KeyboardInterrupt):
            train(assets, resumed_path)
    warmup_last = torch.load(tmp_path / "resumed_last.ckpt", map_location="cpu", weights_only=True)
    for name, value in legacy.state_dict().items():
        if name.startswith("encoder."):
            torch.testing.assert_close(warmup_last["model_state"][name], value, rtol=0, atol=0)
    assert warmup_last["model_state"]["token_evidence_weights"].abs().sum() > 0
    resumed_result = train(assets, resumed_path)
    assert resumed_result["resumed"]
    resumed_last = torch.load(tmp_path / "resumed_last.ckpt", map_location="cpu", weights_only=True)
    for name, value in full_last["model_state"].items():
        torch.testing.assert_close(resumed_last["model_state"][name], value, rtol=0, atol=0)
    assert resumed_result["best_epoch"] == full_result["best_epoch"]
    for one, two in zip(full_result["history"], resumed_result["history"]):
        for key in ("epoch", "phase", "phase_epoch", "train_loss", "stop_macro_ap"):
            assert one[key] == two[key]
    for key in ("torch", "loader"):
        torch.testing.assert_close(full_last["rng"][key], resumed_last["rng"][key], rtol=0, atol=0)
    assert full_last["scheduler"] == resumed_last["scheduler"]
    assert not list(tmp_path.glob("*.tmp"))

    with patch.object(neural.AutoModel, "from_pretrained", side_effect=AssertionError("network forbidden")):
        first = neural.predict_checkpoint(full_path, documents, tokenizer, settings)
        second = neural.predict_checkpoint(resumed_path, documents, tokenizer, settings)
        empty = neural.predict_checkpoint(full_path, [], tokenizer, settings)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 9) and first.dtype == np.float32
    assert empty.shape == (0, 9) and np.isfinite(first).all()
    # An already completed run returns without fitting or even rebuilding the encoder.
    with patch.object(neural, "_from_saved", side_effect=AssertionError("must reuse complete run")):
        complete_again = train(assets, resumed_path)
    assert complete_again["completed_epochs"] == 2
    with pytest.raises(ValueError, match="signature mismatch"):
        train(assets, resumed_path, run_signature="different-data-or-code")
    modified_assets = (*assets[:-1], {**settings, "batch_size": 3})
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        train(modified_assets, resumed_path)


def test_document_accumulation_handles_short_final_group(assets, tmp_path):
    tokenizer, legacy, checkpoint, documents, targets, stop_targets, settings = assets
    # Deterministic dropout-free encoder isolates unequal-size gradient weighting.
    original = torch.load(checkpoint, weights_only=True)
    original["encoder_config"]["hidden_dropout_prob"] = 0
    original["encoder_config"]["attention_probs_dropout_prob"] = 0
    deterministic = tmp_path / "deterministic_init.ckpt"
    torch.save(original, deterministic)
    deterministic_settings = {**settings, "dropout": 0, "warmup_epochs": 0,
                              "batch_size": 3, "gradient_accumulation_steps": 2,
                              "gradient_checkpointing": False}
    first_assets = (tokenizer, legacy, deterministic, documents, targets, stop_targets, deterministic_settings)
    second_assets = (*first_assets[:-1], {**deterministic_settings, "batch_size": 5,
                                        "gradient_accumulation_steps": 1})
    train(first_assets, tmp_path / "micro.ckpt")
    train(second_assets, tmp_path / "whole.ckpt")
    first = torch.load(tmp_path / "micro_last.ckpt", weights_only=True)["model_state"]
    second = torch.load(tmp_path / "whole_last.ckpt", weights_only=True)["model_state"]
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=1e-4, atol=2e-6)


def test_resume_inside_finetune_restores_adam_and_scheduler_states(assets, tmp_path):
    settings = {**assets[-1], "finetune_epochs": 2}
    configured = (*assets[:-1], settings)
    uninterrupted = tmp_path / "uninterrupted.ckpt"
    train(configured, uninterrupted)
    interrupted = tmp_path / "interrupted.ckpt"
    original_save = neural._atomic_save

    def stop_after_first_finetune(value, path):
        original_save(value, path)
        if value.get("format") == neural.LAST_FORMAT and value["completed_epochs"] == 2:
            raise KeyboardInterrupt("Stop after one fine-tuning epoch")

    with patch.object(neural, "_atomic_save", side_effect=stop_after_first_finetune):
        with pytest.raises(KeyboardInterrupt):
            train(configured, interrupted)
    restored_result = train(configured, interrupted)
    assert restored_result["completed_epochs"] == 3
    first = torch.load(tmp_path / "uninterrupted_last.ckpt", weights_only=True)
    second = torch.load(tmp_path / "interrupted_last.ckpt", weights_only=True)
    for name in first["model_state"]:
        torch.testing.assert_close(first["model_state"][name], second["model_state"][name], rtol=0, atol=0)
    assert first["scheduler"] == second["scheduler"]
    assert first["optimizer"]["param_groups"] == second["optimizer"]["param_groups"]
    for parameter, state in first["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, second["optimizer"]["state"][parameter][name], rtol=0, atol=0)
    assert first["best_epoch"] == second["best_epoch"]
    assert first["best_stop_ap"] == second["best_stop_ap"]
