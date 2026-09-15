"""Small actual transformer checks; all assets are created locally, without downloads."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mission3_ver4.neural import (
    DialogueClassifier, DialogueCollator, predict_checkpoint, tokenize_dialogues,
    train_model,
)


@pytest.fixture
def tiny_assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    configuration = BertConfig(
        vocab_size=len(vocabulary), hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    )
    torch.manual_seed(7)
    encoder = BertModel(configuration)
    model_path = tmp_path / "encoder"
    encoder.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return tokenizer, encoder, model_path


def test_overflow_preserves_every_token_and_tail(tiny_assets):
    tokenizer, _, _ = tiny_assets
    text = "alpha " * 67 + "omega"
    documents = tokenize_dialogues([text, ""], tokenizer, max_length=12, stride=3)
    assert len(documents[0]) > 5
    reconstructed = documents[0][0]["input_ids"][1:-1]
    for window in documents[0][1:]:
        reconstructed += window["input_ids"][1:-1][3:]
    assert reconstructed == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert reconstructed[-1] == tokenizer.convert_tokens_to_ids("omega")
    assert len(documents[1]) == 1
    with pytest.raises(ValueError, match="stride"):
        tokenize_dialogues([text], tokenizer, max_length=12, stride=10)


def test_document_loss_and_microbatch_gradients_reach_last_window(tiny_assets):
    tokenizer, encoder, _ = tiny_assets
    documents = tokenize_dialogues(["alpha " * 35 + "omega", "beta"], tokenizer, 12, 3)
    batch = DialogueCollator(tokenizer)([
        {"windows": documents[0], "labels": [1] * 9},
        {"windows": documents[1], "labels": [0] * 9},
    ])
    labels = batch.pop("labels")
    assert labels.shape == (2, 9)
    assert len(batch["doc_indices"]) > 2
    model = DialogueClassifier(encoder, dropout=0, window_batch_size=1)
    combined = copy.deepcopy(model)
    combined.window_batch_size = 64
    model.eval()
    combined.eval()
    logits = model(**batch)
    torch.testing.assert_close(logits, combined(**batch), atol=1e-6, rtol=1e-5)
    assert logits.shape == labels.shape
    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
    tail_id = tokenizer.convert_tokens_to_ids("omega")
    assert model.encoder.embeddings.word_embeddings.weight.grad[tail_id].abs().sum() > 0


def test_actual_training_offline_reload_and_unequal_batch_accumulation(tiny_assets, tmp_path):
    tokenizer, _, model_path = tiny_assets
    texts = ["alpha " * 25 + "omega", "beta", "alpha beta " * 6,
             "omega alpha beta", "alpha beta " * 8]
    documents = tokenize_dialogues(texts, tokenizer, max_length=12, stride=3)
    targets = ((np.arange(5)[:, None] + np.arange(9)[None, :]) % 3 == 0).astype(np.float32)
    stopping_targets = np.array([[1] * 9, [0] * 9], dtype=np.float32)
    config = {
        "device": "cpu", "model_name": str(model_path), "batch_size": 3,
        "inference_batch_size": 2, "gradient_accumulation_steps": 2,
        "max_epochs": 1, "patience": 1, "dropout": 0,
        "gradient_checkpointing": True, "window_batch_size": 2,
        "learning_rate": 1e-3, "head_learning_rate": 1e-3,
        "warmup_ratio": 0, "weight_decay": 0, "layerwise_lr_decay": 1,
        "log_every": 1,
    }
    accumulated = tmp_path / "accumulated.pt"
    result = train_model(documents, targets, documents[:2], stopping_targets,
                         tokenizer, config, accumulated, seed=42)
    assert result["best_epoch"] == 1
    assert np.isfinite(result["best_stop_ap"])
    assert result["history"][0]["minutes"] > 0
    assert not list(tmp_path.glob("*.tmp"))
    with patch("mission3_ver4.neural.AutoModel.from_pretrained", side_effect=AssertionError("Network forbidden")):
        predictions = predict_checkpoint(accumulated, documents, tokenizer, config)
        smaller = predict_checkpoint(accumulated, documents, tokenizer,
                                     {**config, "inference_batch_size": 1, "window_batch_size": 1})
        empty = predict_checkpoint(accumulated, [], tokenizer, config)
    assert predictions.shape == (5, 9)
    assert ((predictions >= 0) & (predictions <= 1)).all()
    np.testing.assert_allclose(predictions, smaller, rtol=1e-5, atol=1e-6)
    assert empty.shape == (0, 9)
    whole_batch = tmp_path / "whole_batch.pt"
    train_model(documents, targets, documents[:2], stopping_targets, tokenizer,
                {**config, "batch_size": 5, "gradient_accumulation_steps": 1},
                whole_batch, seed=42)
    first = torch.load(accumulated, weights_only=True)["model_state"]
    second = torch.load(whole_batch, weights_only=True)["model_state"]
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=1e-4, atol=2e-6)
