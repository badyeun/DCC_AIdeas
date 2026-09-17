"""Actual tiny CPU training, cache isolation, and portable offline CLI inference."""
import ast
from contextlib import contextmanager, ExitStack
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, ElectraConfig, ElectraModel

from mission3_ver8 import pipeline, pretrained as public_model
from mission3_ver8.ver4_runtime import pipeline as legacy
from mission3_ver8.ver4_runtime.data import TARGET_SYMPTOMS, load_dataset


def _annotation(path, text, symptoms):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"utterances": [{"text": text, "speaker": "caller"}],
                                "symptom": symptoms}, ensure_ascii=False), encoding="utf-8")


@contextmanager
def _restore(path):
    original = path.read_bytes()
    try:
        yield
    finally:
        path.write_bytes(original)


def _build_base(root):
    torch.set_num_threads(1)
    pretrained = root / "pretrained"
    pretrained.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "call", "context"]
    vocab += [f"cue{i}" for i in range(9)] + [f"id{i}" for i in range(144)]
    vocab_path = pretrained / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(pretrained)
    torch.manual_seed(3)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=12, num_hidden_layers=1,
                         num_attention_heads=3, intermediate_size=24, max_position_embeddings=32,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(pretrained)
    train_dir = root / "Training" / "TL_서울_구급"
    truth = np.random.default_rng(24).integers(0, 2, size=(144, 9), dtype=np.int8)
    texts = {}
    for i in reversed(range(144)):
        prefix = "context " * (25 if i % 11 == 0 else 1)
        text = f"call id{i} {prefix}" + " ".join(f"cue{j}" for j in np.flatnonzero(truth[i]))
        name = f"call_{i:03d}.json"
        texts[name] = text
        _annotation(train_dir / name, text, [s for s, flag in zip(TARGET_SYMPTOMS, truth[i]) if flag])
    config = {
        "model_name": str(pretrained), "device": "cpu", "require_cuda": False,
        "n_folds": 2, "calibration_size": 0.2, "calibration_folds": 2,
        "max_length": 16, "stride": 4, "max_epochs": 1, "patience": 1,
        "batch_size": 8, "inference_batch_size": 8, "window_batch_size": 4,
        "gradient_accumulation_steps": 2, "dropout": 0, "gradient_checkpointing": True,
        "char_features": 100, "word_features": 100, "min_df": 1,
    }
    base_dir = root / "base_original"
    manifest_path, report = legacy.train_pipeline(train_dir, base_dir, config)
    return pretrained, train_dir, texts, base_dir, legacy.load_manifest(manifest_path), report


def _assert_bad_base_rejected(train_dir, base_dir, manifest):
    # A changed dataset must fail before trusting saved probability arrays.
    first = next(train_dir.glob("*.json"))
    with _restore(first):
        annotation = json.loads(first.read_text())
        annotation["utterances"][0]["text"] += " changed"
        first.write_text(json.dumps(annotation), encoding="utf-8")
        with pytest.raises(ValueError, match="Training"):
            pipeline._read_base(train_dir, base_dir, 1)

    cache_path = base_dir / "calibration_predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        original = {key: cache[key].copy() for key in cache.files}
    for corruption, message in [("targets", "targets mismatch"), ("names", "filenames mismatch"),
                                ("nonfinite", "Invalid ver4 raw"), ("shape", "Invalid ver4 raw")]:
        with _restore(cache_path):
            values = {key: value.copy() for key, value in original.items()}
            if corruption == "targets":
                values["targets"][0, 0] = 1 - values["targets"][0, 0]
            elif corruption == "names":
                values["filenames"][0] = values["filenames"][1]
            elif corruption == "nonfinite":
                values["neural"][0, 0] = np.nan
            else:
                values["sparse"] = values["sparse"][:, :8]
            np.savez_compressed(cache_path, **values)
            with pytest.raises(ValueError, match=message):
                pipeline._read_base(train_dir, base_dir, 1)

    split_path = base_dir / "splits.json"
    splits = json.loads(split_path.read_text())
    with _restore(split_path):
        invalid = copy.deepcopy(splits)
        invalid["folds"][0]["fit"][0] = splits["calibration"][0]
        split_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="leakage"):
            pipeline._read_base(train_dir, base_dir, 1)

    # Keep the synthetic fingerprint internally consistent to reach the separate
    # normalized-text group guard rather than merely failing dataset identity.
    fit_file = train_dir / splits["folds"][0]["fit"][0]
    cal_file = train_dir / splits["calibration"][0]
    checkpoint_path = base_dir / "mission3_ver4_model.ckpt"
    with _restore(fit_file), _restore(checkpoint_path):
        annotation = json.loads(fit_file.read_text())
        annotation["utterances"] = json.loads(cal_file.read_text())["utterances"]
        fit_file.write_text(json.dumps(annotation), encoding="utf-8")
        paths, texts, targets = load_dataset(train_dir, include_targets=True)
        changed = copy.deepcopy(manifest)
        changed["signature"] = legacy._fingerprint(paths, texts, targets, changed["config"])
        torch.save(changed, checkpoint_path)
        with pytest.raises(ValueError, match="text-group leakage"):
            pipeline._read_base(train_dir, base_dir, 1)


def _csv_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["label file name", "symptom"]
        rows = list(reader)
    result = {}
    for row in rows:
        labels = ast.literal_eval(row["symptom"])
        assert isinstance(labels, list) and len(set(labels)) == len(labels)
        assert all(label in TARGET_SYMPTOMS for label in labels)
        assert row["label file name"] not in result
        result[row["label file name"]] = [int(s in labels) for s in TARGET_SYMPTOMS]
    return rows, result


def _public_mlm(root, base_pretrained):
    from transformers import RobertaConfig, RobertaForMaskedLM
    directory = root / 'public_roberta'
    directory.mkdir()
    vocabulary = (base_pretrained / 'vocab.txt').read_text().splitlines()
    (directory / 'vocab.txt').write_text('\n'.join(vocabulary))
    tokenizer = BertTokenizerFast(vocab_file=str(directory / 'vocab.txt'), do_lower_case=False)
    tokenizer.save_pretrained(directory)
    torch.manual_seed(37)
    model = RobertaForMaskedLM(RobertaConfig(vocab_size=len(vocabulary), hidden_size=12,
        num_hidden_layers=2, num_attention_heads=3, intermediate_size=24, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=2, eos_token_id=3, type_vocab_size=2,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1))
    model.save_pretrained(directory)
    metadata = {'format': public_model.FORMAT, 'complete': True, 'model_id': 'test/tiny-roberta',
                'revision': 'a' * 40, 'files': {p.name: {'size': p.stat().st_size, 'sha256': pipeline._sha256(p)}
                    for p in directory.iterdir() if p.is_file()}}
    (directory / public_model.MANIFEST_NAME).write_text(json.dumps(metadata))
    return directory


@pytest.mark.parametrize('portable', [True, False])
def test_abc_fit_only_tapt_resume_export_and_inference(tmp_path, portable):
    initial, train_dir, texts, base_dir, base_manifest, _ = _build_base(tmp_path)
    _assert_bad_base_rejected(train_dir, base_dir, base_manifest)
    public_dir = _public_mlm(tmp_path, initial)
    out = tmp_path / 'ver8'
    config = {'model_name': 'test/tiny-roberta', 'device': 'cpu', 'require_cuda': False,
              'profiles': ['A', 'B', 'C'], 'max_epochs': 1, 'patience': 1, 'stop_crossfit_splits': 2,
              'bundle_base': portable, 'attention_dim': 8, 'batch_size': 8, 'window_batch_size': 4,
              'inference_batch_size': 8, 'gradient_accumulation_steps': 2, 'dropout': 0,
              'gradient_checkpointing': True, 'log_every': 0, 'max_length': 20, 'stride': 4,
              'tapt_steps': 3, 'tapt_save_every': 1, 'tapt_max_length': 16,
              'tapt_batch_size': 4, 'tapt_accumulation_steps': 2, 'audit_per_error': 2,
              'min_calibration_gain': 1.0}
    original = pipeline.neural.train_model
    def interrupt_after_a(*args, **kwargs):
        if args[5]['head_type'] == 'token':
            raise KeyboardInterrupt('A committed; interrupt before B')
        return original(*args, **kwargs)
    with patch.object(pipeline.neural, 'train_model', side_effect=interrupt_after_a), pytest.raises(KeyboardInterrupt):
        pipeline.train_pipeline(train_dir, base_dir, out, config, pretrained_dir=public_dir)
    assert (out / 'trials/A/done.json').is_file()
    a_hash = pipeline._sha256(out / 'trials/A/best.ckpt')
    assert not (out / 'trials/A/best_last.ckpt').exists()
    review_copy = out / 'error_analysis/review_samples.csv'
    review_copy.write_text('human notes remain')
    with patch.object(pipeline.tapt, 'adapt', wraps=pipeline.tapt.adapt) as adapt, \
         patch.object(pipeline.neural, 'train_model', wraps=original) as train, \
         patch.object(legacy, 'train_model', side_effect=AssertionError('do not retrain old model')), \
         patch.object(legacy, 'fit_sparse', side_effect=AssertionError('do not refit sparse')):
        checkpoint, report = pipeline.train_pipeline(train_dir, base_dir, out, config, pretrained_dir=public_dir)
    assert train.call_count == 2  # B/C only; A was committed before interruption.
    assert adapt.call_count == 1
    split = json.loads((out / 'splits.json').read_text())
    passed_texts, passed_names = adapt.call_args.args[:2]
    assert passed_names == split['fit']
    assert set(passed_names).isdisjoint(split['stop'] + split['calibration'])
    assert len(passed_texts) == len(passed_names)
    assert review_copy.read_text() == 'human notes remain'
    assert pipeline._sha256(out / 'trials/A/best.ckpt') == a_hash
    assert report['new_models_trained'] == 3
    assert report['tapt']['completed_steps'] == 3
    assert not report['tapt']['stop_or_calibration_or_validation_used']
    assert report['selected_profile'] == pipeline.choose_profile(report['trials'])
    assert report['selected_name'] == 'ver4_baseline'
    assert report['calibrator']['candidate_weights'] == [0.] * 9
    assert report['calibrator']['thresholds'] == base_manifest['calibrator']['thresholds']
    assert not report['validation_used_for_training_or_selection']
    for profile in 'ABC':
        saved = torch.load(out / f'trials/{profile}/best.ckpt', weights_only=True)
        assert saved['architecture']['head_type'] == ('window' if profile == 'A' else 'token')
        assert 'optimizer' not in saved and 'scheduler' not in saved
        assert not (out / f'trials/{profile}/best_last.ckpt').exists()
    assert not (out / 'tapt/tapt_last.ckpt').exists()
    with patch.object(pipeline.neural, 'train_model', side_effect=AssertionError('completed retrained')), \
         patch.object(pipeline.tapt, 'adapt', side_effect=AssertionError('completed TAPT repeated')):
        assert pipeline.train_pipeline(train_dir, base_dir, out, config, pretrained_dir=public_dir)[1] == report
    # Final inference uses only saved weights, including a candidate whose original MLM directory is gone.
    public_dir.rename(tmp_path / 'public_removed')
    relocated = tmp_path / 'relocated'
    shutil.copytree(out, relocated)
    validation = tmp_path / 'unlabelled'
    for i in range(4):
        _annotation(validation / f'new_{i}.json', f'call cue{i}', [])
        path = validation / f'new_{i}.json'
        annotation = json.loads(path.read_text()); annotation.pop('symptom')
        path.write_text(json.dumps(annotation))
    final_csv, candidate_csv = tmp_path / 'prediction.csv', tmp_path / 'candidate.csv'
    command = [sys.executable, str(relocated / 'submission/inference.py'), '--label_dir', str(validation),
               '--ckpt_path', str(relocated / checkpoint.name), '--output', str(final_csv),
               '--candidate_output', str(candidate_csv), '--device', 'cpu', '--batch_size', '2']
    if not portable:
        command += ['--base_dir', str(base_dir)]
    env = dict(os.environ, PYTHONPATH='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=90)
    with final_csv.open(encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
    with candidate_csv.open(encoding='utf-8-sig') as f: candidate_rows = list(csv.DictReader(f))
    assert len(rows) == len(candidate_rows) == 4
    assert [r['label file name'] for r in rows] == [f'new_{i}.json' for i in range(4)]
    assert all(set(ast.literal_eval(r['symptom'])) <= set(TARGET_SYMPTOMS) for r in rows + candidate_rows)
    # Changed labels on identical text cannot affect inference, including cache reuse.
    for path in validation.glob('*.json'):
        d = json.loads(path.read_text()); d['symptom'] = list(TARGET_SYMPTOMS); path.write_text(json.dumps(d))
    before = final_csv.read_bytes(), candidate_csv.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=90)
    assert before == (final_csv.read_bytes(), candidate_csv.read_bytes())


def test_profile_selection_ignores_diagnostic_or_validation_scores():
    trials = {name: {'best_stop_macro_f1': .6, 'best_stop_ap': .7, 'validation_f1': v}
              for name, v in [('C', 1.), ('B', .9), ('A', .1)]}
    assert pipeline.choose_profile(trials) == 'A'
    trials['B']['best_stop_macro_f1'] = .61
    assert pipeline.choose_profile(trials) == 'B'
