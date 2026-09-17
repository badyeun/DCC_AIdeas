import json
from unittest.mock import patch

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

from mission3_ver8 import tapt, neural


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    initial = tmp_path / 'public'
    initial.mkdir()
    (initial / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nbeta\nomega\n')
    tokenizer = BertTokenizerFast(vocab_file=str(initial / 'vocab.txt'), do_lower_case=False)
    tokenizer.save_pretrained(initial)
    torch.manual_seed(31)
    BertForMaskedLM(BertConfig(vocab_size=8, hidden_size=12, num_hidden_layers=2,
        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)).save_pretrained(initial)
    texts = ['alpha beta omega ' * (1 + i % 3) for i in range(7)] + ['']
    names = [f'fit_{i}.json' for i in range(len(texts))]
    config = {'device': 'cpu', 'gradient_checkpointing': True, 'tapt_steps': 5,
              'tapt_save_every': 1, 'tapt_max_length': 12, 'tapt_batch_size': 2,
              'tapt_accumulation_steps': 2, 'tapt_learning_rate': 1e-3,
              'keep_completed_resume_states': True}
    return texts, names, tokenizer, initial, config


def run(assets, output):
    texts, names, tokenizer, initial, config = assets
    return tapt.adapt(texts, names, tokenizer, initial, output, config, run_signature='fit-only-run', seed=123)


def test_exact_step_resume_and_completed_reuse(assets, tmp_path):
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    _, expected = run(assets, full)
    original = neural._atomic_save
    def stop_after_two(value, path):
        original(value, path)
        if value.get('format') == 'mission3_ver8_tapt_last' and value['completed_steps'] == 2:
            raise KeyboardInterrupt('step boundary')
    with patch.object(neural, '_atomic_save', side_effect=stop_after_two), pytest.raises(KeyboardInterrupt):
        run(assets, interrupted)
    assert not (interrupted / 'tapt_done.json').exists()
    _, actual = run(assets, interrupted)
    assert actual['resumed'] and actual['completed_steps'] == 5
    assert actual['masked_token_loss'] == expected['masked_token_loss']
    a = torch.load(full / 'tapt_last.ckpt', weights_only=True)
    b = torch.load(interrupted / 'tapt_last.ckpt', weights_only=True)
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    assert a['scheduler'] == b['scheduler']
    public = assets[3]; public.rename(tmp_path / 'hidden_public')
    with patch.object(tapt.AutoModelForMaskedLM, 'from_pretrained', side_effect=AssertionError('already complete')):
        assert run(assets, interrupted)[1] == actual
    # Different fit membership or content must never reuse an adapted encoder.
    texts, names, tokenizer, initial, config = assets
    with pytest.raises(ValueError, match='mismatch'):
        tapt.adapt(texts, names[:-1] + ['different.json'], tokenizer, initial, interrupted,
                   config, run_signature='fit-only-run', seed=123)


def test_only_semantic_tokens_are_mlm_targets_and_original_tokens_survive(assets):
    texts, _, tokenizer, _, _ = assets
    docs = neural.tokenize_dialogues(texts[:2], tokenizer, 12, 0)
    windows = [w for doc in docs for w in doc]
    before = json.dumps(windows)
    batch = tapt.masked_batch(windows, tokenizer, .000001)
    values = batch['labels'][batch['labels'] != -100]
    assert len(values) >= 1
    assert not any(int(x) in tokenizer.all_special_ids for x in values)
    assert json.dumps(windows) == before


def test_empty_fit_cannot_create_nan_mlm_loss(assets, tmp_path):
    _, _, tokenizer, initial, config = assets
    with pytest.raises(ValueError, match='semantic tokens'):
        tapt.adapt([''], ['empty.json'], tokenizer, initial, tmp_path / 'empty', config,
                   run_signature='x', seed=1)
