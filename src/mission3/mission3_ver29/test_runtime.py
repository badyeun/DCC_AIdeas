"""Orchestration tests with deterministic model fixtures, never claimed as GPU F1."""
from __future__ import annotations

from contextlib import ExitStack
import json
import hashlib
import csv
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import data, inference, reference, runtime
from .budget import Budget, BudgetExceeded
from .settings import default_settings, validate_settings
from .test_reference import create_fixture


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".ver29_runtime_", dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.texts = ["fixture dialogue " + str(i) for i in range(36)]
        self.targets = np.random.default_rng(19).integers(0, 2, (36, 9), dtype=np.int8)
        self.valid_texts = ["validation fixture " + str(i) for i in range(4)]
        self.valid_targets = np.random.default_rng(20).integers(0, 2, (4, 9), dtype=np.int8)
        self.truth = {text: target for text, target in zip(self.texts + self.valid_texts,
                      np.concatenate([self.targets, self.valid_targets]))}
        self.train_dir, self.valid_dir = self.root / "train", self.root / "valid"
        self.calls = []
        self.dataset_reads = []
        self.settings = default_settings()
        self.settings.update(allow_cpu=True, output_dir=str(self.root / "results"), work_dir=str(self.root / "work"))
        self.stack = ExitStack()
        self.reference_dir = self.root / 'reference'
        _, _, digest = create_fixture(self.reference_dir, self.texts, self.targets)
        self.settings['reference_run'] = str(self.reference_dir)
        self.stack.enter_context(mock.patch.object(reference, 'EXPECTED_MANIFEST_SHA256', digest))
        self.stack.enter_context(mock.patch.object(runtime, "EXPECTED_TRAINING_SAMPLES", len(self.texts)))
        self.stack.enter_context(mock.patch.object(runtime, "EXPECTED_VALIDATION_SAMPLES", len(self.valid_texts)))
        self.stack.enter_context(mock.patch.object(runtime, "EXPECTED_DATA_DIGEST", data.dataset_digest(self.texts, self.targets)))
        split = data.three_way_split(self.texts, self.targets)
        self.stack.enter_context(mock.patch.object(runtime, "EXPECTED_SPLIT_DIGEST", split[3]["split_digest"]))
        self.real_resolve_inputs = runtime.resolve_inputs
        self.stack.enter_context(mock.patch.object(runtime, "resolve_inputs", return_value=(self.train_dir, self.valid_dir)))
        self.stack.enter_context(mock.patch.object(data, "load_dataset", side_effect=self.load_dataset))
        self.stack.enter_context(mock.patch.object(runtime, "resolve_spec", side_effect=self.resolve_spec))
        self.stack.enter_context(mock.patch.object(runtime.trainer, "train_model", side_effect=self.train_model))
        self.stack.enter_context(mock.patch.object(runtime.trainer, "predict_checkpoint", side_effect=self.predict))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def load_dataset(self, directory, include_targets=False):
        is_valid = Path(directory) == self.valid_dir
        self.dataset_reads.append((is_valid, include_targets, len(self.calls)))
        texts = self.valid_texts if is_valid else self.texts
        targets = self.valid_targets if is_valid else self.targets
        return [Path(directory) / f"{i}.json" for i in range(len(texts))], texts, targets if include_targets else None

    def resolve_spec(self, kind, spec, settings, budget, output):
        value = {**spec, "revision": "a" * 40, "kind": kind, "snapshot_path": "fixture"}
        path = output / "resolved_models.json"
        values = runtime.read_json(path) if path.exists() else {}
        values[kind] = value
        data.write_json(path, values)
        return value

    def predict(self, checkpoint_dir, texts, indices=None, **kwargs):
        ids = np.arange(len(texts)) if indices is None else indices
        kind = Path(checkpoint_dir).parent.name
        truth = np.asarray([self.truth[texts[int(i)]] for i in ids])
        return np.full(truth.shape, -2., dtype=np.float32) if kind == "control" else (truth * 4. - 2.).astype(np.float32)

    def train_model(self, kind, spec, settings, texts, targets, train_ids, dev_ids, output, budget, epochs, resume=False):
        self.calls.append({"kind": kind, "train": list(train_ids), "dev": None if dev_ids is None else list(dev_ids),
                           "epochs": epochs, "resume": resume, "output": str(output)})
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "best_model").mkdir(exist_ok=True)
        data.write_json(output / "training_identity.json", {"fixture": True})
        data.write_json(output / "batch_benchmark.json", {"selected_batch_size": 2})
        reference = train_ids[:2]
        runtime.save_npz(output / "reload_reference.npz", indices=reference,
                         logits=self.predict(output / "best_model", texts, reference))
        if dev_ids is not None:
            np.save(output / "epoch_2_development_logits.npy", self.predict(output / "best_model", texts, dev_ids))
        return {"status": "complete", "best_epoch": 2, "best_score": 0., "checkpoint_dir": str(output / "best_model")}

    def test_screening_then_frozen_full_and_no_validation_selection(self):
        screened = runtime.run(self.settings)
        self.assertEqual(screened["status"], "screening_complete")
        self.assertFalse(any(valid for valid, _, _ in self.dataset_reads))
        source = Path(screened["run_dir"])
        selection = runtime.read_json(source / "selection.json")
        self.assertEqual(selection["members"], ["klue_asl"])
        self.assertEqual([call['kind'] for call in self.calls], ['klue_asl', 'qwen_embedding'])
        self.assertEqual(set(selection['scores']), set(runtime.RECIPES))
        self.assertFalse((source / 'models/control').exists())
        self.assertFalse((source / 'models/kure').exists())
        for call in self.calls:
            self.assertFalse(set(call["train"]) & set(call["dev"]))
        self.dataset_reads.clear()
        self.calls.clear()
        result = runtime.run({**self.settings, "run_mode": "full", "run_validation": True,
                              "source_run": str(source)})
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["submission_ready"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["train"], list(range(len(self.texts))))
        self.assertIsNone(self.calls[0]["dev"])
        self.assertEqual(self.calls[0]["epochs"], 2)
        self.assertEqual([count for valid, _, count in self.dataset_reads if valid], [1])
        runtime.verify_run(Path(result["run_dir"]))

    def test_autotune_pause_with_identity_but_no_optimizer_can_resume_in_new_folder(self):
        def pause(kind, spec, settings, texts, targets, train_ids, dev_ids, output, budget, epochs, resume=False):
            Path(output).mkdir(parents=True)
            data.write_json(Path(output) / "training_identity.json", {"fixture": True})
            return {"status": "paused"}
        with mock.patch.object(runtime.trainer, "train_model", side_effect=pause):
            paused = runtime.run(self.settings)
        self.assertEqual(paused["status"], "paused")
        result = runtime.run({**self.settings, "resume_run": paused["run_dir"]})
        self.assertEqual(result["status"], "screening_complete")
        self.assertTrue(self.calls[0]["resume"])
        self.assertNotEqual(result["run_dir"], paused["run_dir"])
        self.assertEqual(runtime.read_json(Path(paused["run_dir"]) / "run_state.json")["status"], "paused")
        self.assertNotIn('pause_stage', runtime.read_json(Path(result['run_dir']) / 'run_state.json'))

    def test_setup_pause_without_split_is_restartable(self):
        class PauseSetup(Budget):
            def check(self, stage, training=False):
                raise BudgetExceeded(stage)
        with mock.patch.object(runtime, "Budget", PauseSetup):
            paused = runtime.run(self.settings)
        self.assertFalse((Path(paused["run_dir"]) / "split_manifest.json").exists())
        result = runtime.run({**self.settings, "resume_run": paused["run_dir"]})
        self.assertEqual(result["status"], "screening_complete")

    def test_full_resume_keeps_frozen_revisions_and_completed_members(self):
        screened = runtime.run(self.settings)
        source = Path(screened["run_dir"])
        selection = runtime.read_json(source / "selection.json")
        selection.update(selected_recipe="all_four", members=runtime.RECIPES['all_four'])
        data.write_json(source / "selection.json", selection)
        runtime.write_manifest(source)
        self.calls.clear()

        def pause_second(kind, spec, settings, texts, targets, train_ids, dev_ids, output, budget, epochs, resume=False):
            if kind in ("control", "kure", "klue_asl"):
                return self.train_model(kind, spec, settings, texts, targets, train_ids, dev_ids, output, budget, epochs, resume)
            Path(output).mkdir(parents=True)
            data.write_json(Path(output) / "training_identity.json", {"fixture": True})
            return {"status": "paused"}

        with mock.patch.object(runtime.trainer, "train_model", side_effect=pause_second):
            paused = runtime.run({**self.settings, "run_mode": "full", "run_validation": True,
                                  "source_run": str(source)})
        self.assertEqual(paused["status"], "paused")
        self.calls.clear()
        full = runtime.run({**self.settings, "run_mode": "full", "run_validation": True,
                            "resume_run": paused["run_dir"]})
        self.assertEqual(full["status"], "complete")
        self.assertEqual([call["kind"] for call in self.calls], ["qwen_embedding"])
        self.assertTrue(self.calls[0]["resume"])
        self.assertEqual(runtime.read_json(Path(full["run_dir"]) / "resolved_models.json")["qwen_embedding"]["revision"], "a" * 40)
        self.assertEqual(list(runtime.read_json(Path(full['run_dir']) / 'submission_manifest.json')['members']), runtime.RECIPES['all_four'])
        self.assertEqual(runtime.find_run(Path(full['run_dir'])), Path(full['run_dir']))

    def test_confirmation_cannot_reselect_and_pause_keeps_selection(self):
        class PauseConfirmation(Budget):
            def check(self, stage, training=False):
                if stage == 'confirmation report':
                    raise BudgetExceeded(stage)
                return super().check(stage, training)
        with mock.patch.object(runtime, 'Budget', PauseConfirmation):
            paused = runtime.run(self.settings)
        source = Path(paused['run_dir'])
        self.assertTrue((source / 'selection.json').is_file())
        frozen = (source / 'selection.json').read_bytes()
        self.calls.clear()
        with mock.patch.object(runtime, 'choose_candidate', side_effect=AssertionError('Selection must remain frozen')):
            completed = runtime.run({**self.settings, 'resume_run': str(source)})
        self.assertEqual(completed['status'], 'screening_complete')
        self.assertEqual(self.calls, [])
        self.assertEqual((Path(completed['run_dir']) / 'selection.json').read_bytes(), frozen)
        self.assertFalse(runtime.read_json(Path(completed['run_dir']) / 'confirmation_report.json')['selection_changed'])

    def test_reference_failure_stops_before_any_training(self):
        (self.reference_dir / 'development_predictions.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            runtime.run(self.settings)
        self.assertEqual(self.calls, [])

    def test_kaggle_auto_extracted_dataset_detection_and_hash_check(self):
        base = self.root / "mounted" / "private-dataset"
        files = []
        for prefix, name in ((runtime.bootstrap.TRAIN_PREFIX, "train.json"),
                             (runtime.bootstrap.VALID_PREFIX, "valid.json")):
            path = base / prefix / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{"utterances":[],"symptom":[]}')
            files.append({"path": path.relative_to(base).as_posix(), "size": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        data.write_json(base / "bundle_manifest.json", {"format": "mission3_ver5_inputs_v1", "files": files})
        # Call the real resolver, not the orchestration fixture used by other tests.
        real_resolver = self.real_resolve_inputs
        found = real_resolver({**self.settings, "input_root": str(base.parent)})
        self.assertEqual(found, (base / runtime.bootstrap.TRAIN_PREFIX, base / runtime.bootstrap.VALID_PREFIX))
        (base / files[0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            real_resolver({**self.settings, "input_root": str(base.parent)})

    def test_mutated_or_unlisted_source_cannot_be_used_for_full(self):
        screened = runtime.run(self.settings)
        source = Path(screened["run_dir"])
        (source / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            runtime.verify_run(source)
        (source / "extra.json").unlink()
        (source / "selection.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            runtime.verify_run(source)

    def test_settings_guard_validation_and_full_source(self):
        with self.assertRaises(ValueError):
            validate_settings({**self.settings, "run_validation": True})
        with self.assertRaises(ValueError):
            validate_settings({**self.settings, "run_mode": "full", "run_validation": True})
        with self.assertRaises(ValueError):
            validate_settings({**self.settings, "time_budget_minutes": 481})
        with self.assertRaises(ValueError):
            validate_settings({**self.settings, "candidate": "friend_full_training"})

    def test_inference_does_not_read_targets(self):
        screened = runtime.run(self.settings)
        full = runtime.run({**self.settings, "run_mode": "full", "run_validation": True,
                            "source_run": screened["run_dir"]})
        self.dataset_reads.clear()
        target = self.root / "prediction.csv"
        inference.predict(self.valid_dir, full["run_dir"], target)
        self.assertTrue(target.is_file())
        self.assertEqual([read[1] for read in self.dataset_reads], [False])


class InferenceCliTests(unittest.TestCase):
    def test_direct_competition_cli_reloads_real_tiny_checkpoint_and_ignores_audio_targets(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from . import trainer
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='.ver29_cli_', dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            vocab = root / 'vocab.txt'
            vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npatient\nfever\n', encoding='utf-8')
            tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
            model = trainer.EncoderClassifier(BertModel(BertConfig(vocab_size=7, hidden_size=8,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                max_position_embeddings=32), add_pooling_layer=False), 'klue_asl', .1)
            checkpoint = root / 'models/klue_asl/best_model'
            trainer._save_assets(checkpoint, model, tokenizer, {'kind': 'klue_asl', 'dropout': .1,
                'max_length': 24, 'pooling': trainer.model_protocol('klue_asl')['pooling']})
            data.write_json(root / 'run_state.json', {'format': runtime.FORMAT, 'status': 'complete', 'run_mode': 'full'})
            data.write_json(root / 'submission_manifest.json', {'format': runtime.FORMAT,
                'label_order': list(data.TARGET_SYMPTOMS), 'decision_threshold': .5,
                'members': {'klue_asl': 'models/klue_asl/best_model'}})
            labels = root / 'labels'
            labels.mkdir()
            data.write_json(labels / 'synthetic.json', {'utterances': [{'text': 'patient fever'}],
                'symptom': {'invalid_training_target': True}})
            output = root / 'prediction.csv'
            args = ['inference.py', '--audio_dir', str(root / 'audio_does_not_exist'), '--label_dir', str(labels),
                    '--ckpt_path', str(root / 'submission_manifest.json'), '--output', str(output), '--device', 'cpu']
            with mock.patch.object(sys, 'argv', args), mock.patch.object(sys, 'path', sys.path.copy()):
                runpy.run_path(str(Path(inference.__file__).resolve()), run_name='__main__')
            with output.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ['label file name', 'symptom'])
            self.assertEqual(rows[1][0], 'synthetic.json')
            self.assertEqual(len(rows), 2)
            # Exercise the exact wrapper exported by completed full runs too.
            package = root / 'source_snapshot'
            package.mkdir()
            for path in Path(__file__).parent.glob('*.py'):
                if not path.name.startswith('test_'):
                    shutil.copyfile(path, package / path.name)
            (root / 'inference.py').write_text(runtime.INFERENCE_ENTRYPOINT, encoding='utf-8')
            args[args.index('--ckpt_path') + 1] = str(root)
            with mock.patch.object(sys, 'argv', args), mock.patch.object(sys, 'path', [str(root), *sys.path]):
                runpy.run_path(str(root / 'inference.py'), run_name='__main__')
            self.assertEqual(output.read_text(encoding='utf-8-sig').splitlines()[0], 'label file name,symptom')
            # Avoid sharing this temporary source_snapshot package with later tests.
            for name in list(sys.modules):
                if name == 'source_snapshot' or name.startswith('source_snapshot.'):
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
