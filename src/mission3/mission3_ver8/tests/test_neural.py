"""Real tiny local Electra: full updates, tail gradients and exact epoch resume."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast, ElectraConfig, ElectraModel

from mission3_ver8 import neural
from mission3_ver8.data import candidate_texts


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    configuration = ElectraConfig(vocab_size=len(vocabulary), embedding_size=12, hidden_size=12,
                                  num_hidden_layers=3, num_attention_heads=3, intermediate_size=24,
                                  max_position_embeddings=32, hidden_dropout_prob=.1,
                                  attention_probs_dropout_prob=.1)
    torch.manual_seed(7)
    public_encoder = ElectraModel(configuration)
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
    settings = {"device": "cpu", "max_epochs": 3, "patience": 3, "attention_dim": 8,
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
    return neural.DialogueClassifier(copy.deepcopy(encoder), dropout=.1, window_batch_size=2,
                                     attention_dim=8, special_token_ids=neural._special_ids(tokenizer))


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
    torch.testing.assert_close(output[-1], model.head.label_bias, atol=0, rtol=0)
    output.sum().backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad[omega].abs().sum() > 0
    for parameter in (model.head.token_queries, model.head.token_projection.weight, model.head.label_weights):
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


def test_fresh_epoch_zero_is_exact_public_encoder_and_new_direct_head(assets, tmp_path):
    tokenizer, public, initial, documents, _, settings = assets
    checkpoint = tmp_path / "zero.ckpt"
    original = neural.AutoModel.from_pretrained
    with patch.object(neural.AutoModel, "from_pretrained", wraps=original) as load:
        result = train(assets, checkpoint, config={"max_epochs": 0})
    load.assert_called_once_with(str(initial), local_files_only=True)
    assert result["completed_epochs"] == 0 and result["best_epoch"] == 0
    assert result["best_is_initial_baseline"]
    assert result["history"][0]["phase"] == "initial_public_encoder"
    saved = neural._read_checkpoint(checkpoint)
    assert saved["encoder_config"]["model_type"] == "electra"
    assert saved["model_name"] == "klue/roberta-large"
    for name, parameter in public.state_dict().items():
        torch.testing.assert_close(saved["model_state"]["encoder." + name], parameter, atol=0, rtol=0)
    assert "head.token_queries" in saved["model_state"]
    assert not any("projection.0" in key for key in saved["model_state"])
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
        model.head.label_bias[0] = float("nan")
    loader = neural._loader(documents[-1:], None, tokenizer, 1)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        neural._predict(model, loader, torch.device("cpu"))


def test_positive_weight_sqrt_clamp_and_empty_negative_rows():
    targets = np.zeros((100, 9), dtype=np.float32)
    targets[:20, 0] = 1
    targets[:1, 1] = 1
    targets[:, 2] = 1
    weights = neural._positive_weights(targets)
    np.testing.assert_array_equal(weights[:3], [2., 5., 1.])
    assert np.isfinite(weights).all()
    assert np.all((weights >= 1) & (weights <= 5))


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


@pytest.mark.parametrize('head_type', ['window', 'token'])
def test_both_ablations_train_all_layers_and_export_without_optimizer(assets, tmp_path, head_type):
    checkpoint = tmp_path / (head_type + '.ckpt')
    result = train(assets, checkpoint, config={'max_epochs': 1, 'head_type': head_type})
    saved = neural._read_checkpoint(checkpoint)
    last = neural._read_checkpoint(checkpoint.with_name(head_type + '_last.ckpt'))
    assert saved['architecture']['head_type'] == head_type
    assert 'optimizer' not in saved and 'rng' not in saved
    assert 'optimizer' in last and 'rng' in last
    public = assets[1]
    for i in range(public.config.num_hidden_layers):
        assert any(not torch.equal(parameter, last['model_state']['encoder.' + name])
                   for name, parameter in public.state_dict().items() if name.startswith(f'encoder.layer.{i}.'))
    assert result['completed_epochs'] == 1
    probabilities = neural.predict_checkpoint(checkpoint, assets[3], assets[0], assets[5])
    assert probabilities.shape == (15, 9) and np.isfinite(probabilities).all()


def test_token_head_initialization_is_identical_after_encoder_adaptation(assets, tmp_path):
    first = tmp_path / 'public_init.ckpt'
    train(assets, first, config={'max_epochs': 0})
    changed_encoder = copy.deepcopy(assets[1])
    with torch.no_grad():
        changed_encoder.embeddings.word_embeddings.weight.add_(.03)
    adapted = tmp_path / 'adapted'
    changed_encoder.save_pretrained(adapted)
    second = tmp_path / 'adapted_init.ckpt'
    report = train(assets, second, initial_model_dir=adapted,
                   config={'max_epochs': 0, 'initialization_type': 'fit_only_tapt'})
    a, b = neural._read_checkpoint(first), neural._read_checkpoint(second)
    for key in a['model_state']:
        if key.startswith('head.'):
            torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    assert not torch.equal(a['model_state']['encoder.embeddings.word_embeddings.weight'],
                           b['model_state']['encoder.embeddings.word_embeddings.weight'])
    assert report['history'][0]['phase'] == 'initial_tapt_encoder'
    assert 'fit_only_tapt' in report['initialization']
