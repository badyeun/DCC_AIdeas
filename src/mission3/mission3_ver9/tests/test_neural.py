"""Real tiny local DeBERTa: full updates, tail gradients and exact epoch resume."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

from mission3_ver9 import neural
from mission3_ver9.data import candidate_texts


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    configuration = DebertaV2Config(vocab_size=len(vocabulary), embedding_size=12, hidden_size=12,
                                  num_hidden_layers=3, num_attention_heads=3, intermediate_size=24,
                                  max_position_embeddings=32, relative_attention=True, pos_att_type=["p2c", "c2p"],
                                  position_buckets=16, max_relative_positions=32, hidden_dropout_prob=.1,
                                  attention_probs_dropout_prob=.1)
    torch.manual_seed(7)
    public_encoder = DebertaV2Model(configuration)
    initial = tmp_path / "public_pretrained"
    public_encoder.save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    texts = ["alpha " * (1 + i % 4) + "\n" + "beta " * (i % 3) + "omega" for i in range(15)]
    texts[0] = "alpha " * 23 + "\nomega"
    texts[-1] = ""
    documents = neural.tokenize_dialogues(candidate_texts(texts, tokenizer), tokenizer, max_length=12, stride=3)
    targets = ((np.arange(15)[:, None] + np.arange(9)[None, :]) % 3 == 0).astype(np.float32)
    targets[3] = 0  # All-negative fit call must remain a training row.
    targets[-1] = 0  # Empty all-negative stop call stays in prediction order.
    settings = {"device": "cpu", "max_epochs": 3, "patience": 3,
                "batch_size": 2, "window_batch_size": 2, "inference_batch_size": 3,
                "gradient_accumulation_steps": 2, "gradient_checkpointing": True,
                "dropout": .1, "learning_rate": 1e-3, "head_learning_rate": 2e-3,
                "warmup_ratio": 0, "weight_decay": 0, "layerwise_lr_decay": .95,
                "max_length": 12, "stride": 3, "log_every": 0}
    return tokenizer, public_encoder, initial, documents, targets, settings


def train(assets, path, **overrides):
    tokenizer, _, initial, documents, targets, settings = assets
    return neural.train_model(documents[:9], targets[:9], documents[9:], targets[9:], tokenizer,
                              {**settings, **overrides.pop("config", {})}, path, seed=42,
                              initial_model_dir=overrides.pop("initial_model_dir", initial),
                              run_signature=overrides.pop("run_signature", "same-run"),
                              stop_groups=np.arange(6), **overrides)


def model_for(assets):
    tokenizer, encoder, _, _, _, _ = assets
    return neural.DialogueClassifier(copy.deepcopy(encoder), dropout=.1, window_batch_size=2)


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def without_timing(history):
    return [{key: value for key, value in row.items() if key != "minutes"} for row in history]


def test_tail_changes_predictions_and_receives_encoder_gradients(assets):
    tokenizer, _, _, documents, _, _ = assets
    model = model_for(assets).eval()
    assert len(documents[0]) > 1
    omega = tokenizer.convert_tokens_to_ids("omega")
    assert omega not in documents[0][0]["input_ids"]
    assert omega in documents[0][-1]["input_ids"]
    batch = neural.DialogueCollator(tokenizer)([{"windows": documents[0]}, {"windows": documents[-1]}])
    output = model(**batch)
    assert output.shape == (2, 9)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad[omega].abs().sum() > 0
    for parameter in (model.attention_queries, model.projection[0].weight, model.label_weights):
        assert parameter.grad.abs().sum() > 0
    changed = {key: value.clone() for key, value in batch.items()}
    changed["input_ids"][changed["input_ids"] == omega] = tokenizer.convert_tokens_to_ids("beta")
    assert not torch.equal(output[0], model(**changed)[0])


def test_all_encoder_blocks_train_from_first_step_with_checkpointing(assets):
    tokenizer, _, _, documents, targets, settings = assets
    model = model_for(assets)
    model.encoder.requires_grad_(False)  # Configuration must undo any inherited freeze.
    before = {name: value.clone() for name, value in model.named_parameters()}
    neural._configure_training(model, neural._settings(settings))
    model.train()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.encoder.is_gradient_checkpointing
    assert model.encoder.embeddings.training
    assert all(block.training for block in model.encoder.encoder.layer)
    optimizer = neural._optimizer(model, settings)
    included = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert included == {id(parameter) for parameter in model.parameters()}
    batch = neural.DialogueCollator(tokenizer)([{"windows": document} for document in documents[:2]])
    torch.nn.functional.binary_cross_entropy_with_logits(model(**batch), torch.tensor(targets[:2])).backward()
    optimizer.step()
    for prefix in ["encoder.embeddings"] + [f"encoder.encoder.layer.{i}." for i in range(3)]:
        assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters() if name.startswith(prefix))


def test_fresh_epoch_zero_is_exact_public_encoder_and_new_window_head(assets, tmp_path):
    tokenizer, public, initial, documents, _, settings = assets
    checkpoint = tmp_path / "zero.ckpt"
    original = neural.AutoModel.from_pretrained
    with patch.object(neural.AutoModel, "from_pretrained", wraps=original) as load:
        result = train(assets, checkpoint, config={"max_epochs": 0})
    load.assert_called_once_with(str(initial), local_files_only=True, trust_remote_code=False, use_safetensors=True)
    assert result["completed_epochs"] == 0 and result["best_epoch"] == 0
    assert result["best_is_initial_baseline"]
    assert result["history"][0]["phase"] == "initial_public_encoder"
    saved = neural._read_checkpoint(checkpoint)
    assert saved["encoder_config"]["model_type"] == "deberta-v2"
    assert saved["model_name"] == "team-lucid/deberta-v3-base-korean"
    for name, parameter in public.state_dict().items():
        torch.testing.assert_close(saved["model_state"]["encoder." + name], parameter, atol=0, rtol=0)
    assert "attention_queries" in saved["model_state"]
    assert "projection.0.weight" in saved["model_state"]
    assert saved["loss"] == "unweighted_bce"
    assert "optimizer" not in saved and "rng" not in saved
    # Completed state and inference must survive relocation/removal of initial weights.
    moved = tmp_path / "relocated_initial"
    initial.rename(moved)
    with patch.object(neural.AutoModel, "from_pretrained", side_effect=AssertionError("offline reconstruction required")):
        assert train(assets, checkpoint, config={"max_epochs": 0})["resumed"]
        probabilities = neural.predict_checkpoint(checkpoint, documents, tokenizer, settings)
    assert probabilities.shape == (15, 9) and np.isfinite(probabilities).all()
    assert neural.predict_checkpoint(checkpoint, [], tokenizer, settings).shape == (0, 9)


@pytest.mark.parametrize("interrupt_epoch", [0, 1])
def test_exact_interrupted_resume_repairs_best_and_retains_optimizer_rng(assets, tmp_path, interrupt_epoch):
    full = tmp_path / "full" / "best.ckpt"
    interrupted = tmp_path / "interrupted" / "best.ckpt"
    expected = train(assets, full)
    original_save = neural._atomic_save

    def interrupt(state, path):
        original_save(state, path)
        if state.get("format") == neural.LAST_FORMAT and state.get("completed_epochs") == interrupt_epoch:
            raise KeyboardInterrupt("Committed epoch interrupted before best write")

    with patch.object(neural, "_atomic_save", side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        train(assets, interrupted)
    with patch.object(neural.AutoModel, "from_pretrained", side_effect=AssertionError("resume must use local state")):
        actual = train(assets, interrupted, initial_model_dir=tmp_path / "irrelevant_relocated_directory")
    assert actual["resumed"]
    assert without_timing(expected["history"]) == without_timing(actual["history"])
    assert actual["best_epoch"] == expected["best_epoch"]
    for path_a, path_b in [(full, interrupted), (full.with_name("best_last.ckpt"), interrupted.with_name("best_last.ckpt"))]:
        first, second = neural._read_checkpoint(path_a), neural._read_checkpoint(path_b)
        for key in ("model_state", "best_score") + (("optimizer", "scheduler", "scaler", "rng") if "optimizer" in first else ()):
            assert_nested_equal(first[key], second[key])
    with patch.object(neural, "_from_saved", side_effect=AssertionError("already complete")), \
         patch.object(neural.AutoModel, "from_pretrained", side_effect=AssertionError("already complete")):
        assert train(assets, interrupted)["resumed"]
    with pytest.raises(ValueError, match="run signature"):
        train(assets, interrupted, run_signature="changed")
    with pytest.raises(ValueError, match="config fingerprint"):
        train(assets, interrupted, config={"head_learning_rate": .01})
    assert not list(tmp_path.rglob("*.tmp"))


def test_partial_and_completed_resume_reject_changed_gpu_precision(assets, tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    train(assets, checkpoint, config={"max_epochs": 0})
    changed = {**neural.execution_identity("cpu"), "gpu": "another GPU family", "amp_dtype": "bfloat16"}
    with patch.object(neural, "execution_identity", return_value=changed), pytest.raises(ValueError, match="same execution environment"):
        train(assets, checkpoint, config={"max_epochs": 0})


def test_training_accumulation_matches_large_batch_on_short_remainder(assets, tmp_path):
    # Actual train_model: 9 fit calls => microbatches 2+2+2 then 2+1.
    # With dropout disabled those must have the same weights as batches 6+3.
    _, encoder, _, _, _, _ = assets
    encoder.config.hidden_dropout_prob = 0
    encoder.config.attention_probs_dropout_prob = 0
    initial = tmp_path / "zero_dropout_public"
    encoder.save_pretrained(initial)
    micro, large = tmp_path / "micro.ckpt", tmp_path / "large.ckpt"
    shared = {"max_epochs": 1, "dropout": 0}
    train(assets, micro, initial_model_dir=initial,
          config={**shared, "batch_size": 2, "gradient_accumulation_steps": 3})
    train(assets, large, initial_model_dir=initial,
          config={**shared, "batch_size": 6, "gradient_accumulation_steps": 1})
    first = neural._read_checkpoint(micro.with_name("micro_last.ckpt"))
    second = neural._read_checkpoint(large.with_name("large_last.ckpt"))
    assert first["scheduler"]["last_epoch"] == second["scheduler"]["last_epoch"] == 2
    assert first["history"][1]["train_loss"] == pytest.approx(second["history"][1]["train_loss"], abs=1e-7)
    for key in first["model_state"]:
        torch.testing.assert_close(first["model_state"][key], second["model_state"][key], atol=2e-6, rtol=3e-5)


def test_live_stop_ties_preserve_epoch_zero_and_stop_at_patience(assets, tmp_path):
    checkpoint = tmp_path / "ties.ckpt"
    score = {"stop_macro_f1": .25, "stop_macro_ap": .4}
    with patch.object(neural.StopSelector, "score", return_value=score) as scorer:
        report = train(assets, checkpoint, config={"max_epochs": 3, "patience": 2})
    assert scorer.call_count == 3  # The live initial baseline plus two live epochs.
    assert report["completed_epochs"] == 2 and report["best_epoch"] == 0
    assert report["best_is_initial_baseline"]
    assert neural._read_checkpoint(checkpoint)["epoch"] == 0
    assert neural._read_checkpoint(checkpoint.with_name("ties_last.ckpt"))["completed"]


def test_nonfinite_live_prediction_is_rejected(assets):
    tokenizer, _, _, documents, _, _ = assets
    model = model_for(assets).eval()
    with torch.no_grad():
        model.label_bias[0] = float("nan")
    loader = neural._loader(documents[-1:], None, tokenizer, 1)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        neural._predict(model, loader, torch.device("cpu"))


def test_native_bf16_uses_hardware_support_not_emulation():
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "is_bf16_supported", return_value=False) as check:
        assert not neural.native_bf16_supported()
        check.assert_called_once_with(including_emulation=False)
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "is_bf16_supported", side_effect=TypeError), \
         patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
        assert not neural.native_bf16_supported()


@pytest.mark.parametrize("config", [{"max_epochs": 4}, {"max_epochs": -1}, {"max_epochs": True},
                                   {"patience": 0}, {"batch_size": 0}, {"dropout": 1},
                                   {"learning_rate": float("nan")}, {"layerwise_lr_decay": 2},
                                   {"stop_crossfit_splits": 1}, {"max_length": 513}, {"stride": 510}])
def test_invalid_settings_rejected(config):
    with pytest.raises(ValueError):
        neural._settings(config)


def test_deadline_before_initial_evaluation_is_explicit_and_resumable(assets, tmp_path):
    checkpoint = tmp_path / "budget_initial.ckpt"
    report = train(assets, checkpoint, deadline_epoch=1)
    assert report["stopped_for_budget"] and not report["initial_evaluated"]
    assert report["completed_epochs"] == report["best_epoch"] == 0
    assert report["best_stop_macro_f1"] is None
    assert report["history"] == []
    saved = neural._read_checkpoint(checkpoint)
    assert saved["epoch"] == 0 and not saved["initial_evaluated"]
    assert not saved["completed"]
    resumed = train(assets, checkpoint)
    uninterrupted = train(assets, tmp_path / "full_initial.ckpt")
    assert resumed["resumed"] and not resumed["stopped_for_budget"]
    assert without_timing(resumed["history"]) == without_timing(uninterrupted["history"])


def test_mid_epoch_deadline_preserves_exact_last_complete_epoch(assets, tmp_path):
    checkpoint = tmp_path / "budget_partial.ckpt"
    control = {"after_one": False, "checks": 0, "last_bytes": None, "best_bytes": None}
    real_save = neural._atomic_save

    def capture(state, path):
        real_save(state, path)
        if state.get("format") == neural.LAST_FORMAT and state["completed_epochs"] == 1:
            control["after_one"] = True
            control["last_bytes"] = path.read_bytes()
            control["best_bytes"] = checkpoint.read_bytes()
        elif state.get("format") == neural.MODEL_FORMAT and state.get("epoch") == 1:
            control["best_bytes"] = checkpoint.read_bytes()

    def cutoff(*args, **kwargs):
        if control["after_one"]:
            control["checks"] += 1
            # Next-epoch entry, first-batch entry, then first-batch exit.
            if control["checks"] == 3:
                raise neural.BudgetDeadlineExceeded("test cutoff after partial update")

    with patch.object(neural, "_atomic_save", side_effect=capture), \
         patch.object(neural, "_check_deadline", side_effect=cutoff):
        report = train(assets, checkpoint)
    assert report["stopped_for_budget"] and report["budget_reason"] == "partial_epoch_abandoned"
    assert report["completed_epochs"] == 1
    assert [row["epoch"] for row in report["history"]] == [0, 1]
    last_path = checkpoint.with_name("budget_partial_last.ckpt")
    assert last_path.read_bytes() == control["last_bytes"]
    assert checkpoint.read_bytes() == control["best_bytes"]
    committed = neural._read_checkpoint(last_path)
    assert not committed["completed"] and committed["completed_epochs"] == 1
    resumed = train(assets, checkpoint)
    full_path = tmp_path / "budget_full.ckpt"
    full = train(assets, full_path)
    assert without_timing(resumed["history"]) == without_timing(full["history"])
    actual, expected = neural._read_checkpoint(last_path), neural._read_checkpoint(full_path.with_name("budget_full_last.ckpt"))
    for key in ("model_state", "optimizer", "scheduler", "scaler", "rng"):
        assert_nested_equal(actual[key], expected[key])


def test_next_epoch_requires_measured_duration_and_save_reserve(assets, tmp_path):
    checkpoint = tmp_path / "budget_next.ckpt"
    clock = {"now": 1000.0}
    real_save = neural._atomic_save

    def capture(state, path):
        real_save(state, path)
        if state.get("format") == neural.LAST_FORMAT and state["completed_epochs"] == 1:
            clock["now"] = 1880.0  # 120 seconds remain; insufficient even for a tiny epoch.

    with patch.object(neural, "_atomic_save", side_effect=capture), \
         patch.object(neural.time, "time", side_effect=lambda: clock["now"]):
        report = train(assets, checkpoint, deadline_epoch=2000.0)
    assert report["stopped_for_budget"]
    assert report["budget_reason"] == "insufficient_time_for_next_epoch"
    assert report["completed_epochs"] == 1


def test_inference_checks_wall_clock_cutoff(assets, tmp_path):
    checkpoint = tmp_path / "prediction_cutoff.ckpt"
    train(assets, checkpoint, config={"max_epochs": 0})
    tokenizer, _, _, documents, _, settings = assets
    with pytest.raises(neural.BudgetDeadlineExceeded):
        neural.predict_checkpoint(checkpoint, documents, tokenizer, {**settings, "deadline_epoch": 1})

