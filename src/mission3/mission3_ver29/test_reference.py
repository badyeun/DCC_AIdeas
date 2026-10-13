"""Small synthetic reference import and fixed ensemble contract tests."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from . import analysis, data, reference
from .settings import default_settings


def create_fixture(root, texts, targets):
    root.mkdir(parents=True, exist_ok=True)
    fit, dev, confirmation, split = data.three_way_split(texts, targets)
    indices = {'fit': fit, 'development': dev, 'confirmation': confirmation}
    settings = default_settings()
    settings.update(version='ver28', candidate='kure')
    specs = {kind: {'model_id': 'fixture/' + kind, 'revision': 'a' * 40, 'kind': kind}
             for kind in ('control', 'kure')}
    def baseline(ids):
        values = (targets[ids] * 4. - 2.).astype(np.float32)
        values[:, :2] = -2.
        return values
    logits = baseline(dev)
    score = analysis.metrics(targets[dev], logits)
    selection = {'format': reference.REFERENCE_FORMAT, 'selected_recipe': 'mean',
        'members': ['control', 'kure'], 'selected_epochs': {'control': 3, 'kure': 2},
        'split_digest': split['split_digest'], 'training_rows': len(fit),
        'decision_rule': reference.POLICY, 'scores': {kind: score for kind in ('control', 'kure', 'mean')}}
    content = {}
    def add(name, value):
        content[name] = json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    def array(name, **values):
        stream = io.BytesIO()
        np.savez_compressed(stream, **values)
        content[name] = stream.getvalue()
    add('run_state.json', {'format': reference.REFERENCE_FORMAT, 'status': 'screening_complete',
                          'run_mode': 'screening', 'submission_ready': False})
    add('settings.json', settings)
    add('screening_complete.json', {'format': reference.REFERENCE_FORMAT})
    add('selection.json', selection)
    add('split_manifest.json', split)
    add('resolved_models.json', specs)
    array('split_indices.npz', **indices)
    array('development_predictions.npz', indices=dev, targets=targets[dev],
          control=logits, kure=logits, mean=analysis.ensemble_logits(logits, logits))
    confirm = baseline(confirmation)
    array('confirmation_predictions.npz', indices=confirmation, targets=targets[confirmation],
          control=confirm, selected=analysis.ensemble_logits(confirm, confirm))
    add('confirmation_report.json', {'selected': analysis.metrics(targets[confirmation],
                                                               analysis.ensemble_logits(confirm, confirm))})
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'little'))
        digest.update(raw)
    digest.update(np.asarray(targets, dtype=np.float32).tobytes())
    for kind, epoch in selection['selected_epochs'].items():
        history = [{'epoch': e, 'macro_f1': score['macro_f1'] if e == epoch else score['macro_f1'] / 2}
                   for e in (1, 2, 3)]
        add(f'models/{kind}/history.json', history)
        add(f'models/{kind}/training_result.json', {'status': 'complete', 'best_epoch': epoch})
        keys = ['seed', 'effective_batch_size', 'head_lr', 'weight_decay', 'warmup_ratio',
                'dropout', kind + '_lr', kind + '_max_length']
        add(f'models/{kind}/training_identity.json', {'format': 'mission3_ver28_full_encoder_v1',
            **specs[kind], 'target_symptoms': list(data.TARGET_SYMPTOMS), 'decision': reference.IDENTITY_POLICY,
            'data_sha256': digest.hexdigest(), 'train_indices': fit.tolist(), 'dev_indices': dev.tolist(),
            'epochs': 3, 'settings': {key: settings[key] for key in keys}})
    for name in ('trainer.py', 'runtime.py', 'data.py', 'analysis.py', 'settings.py'):
        content['source_snapshot/' + name] = b'# Synthetic provenance fixture; never executed.\n'
    manifest = {'format': reference.REFERENCE_FORMAT, 'files': {
        name: {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in content.items()}}
    add('artifact_manifest.json', manifest)
    for name, raw in content.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return indices, split, hashlib.sha256(content['artifact_manifest.json']).hexdigest()


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.ver29_reference_', dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.texts = ['synthetic call ' + str(i) for i in range(48)]
        self.targets = np.random.default_rng(14).integers(0, 2, size=(48, 9), dtype=np.int8)
        self.source = self.root / 'source'
        self.indices, self.split, self.digest = create_fixture(self.source, self.texts, self.targets)
        self.settings = {**default_settings(), 'reference_run': str(self.source)}
        self.patch = mock.patch.object(reference, 'EXPECTED_MANIFEST_SHA256', self.digest)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def load(self, settings=None, output=None):
        return reference.import_reference(settings or self.settings, output or self.root / 'output',
            self.texts, self.targets, self.indices, self.split)

    def zip_source(self):
        target = self.root / reference.REFERENCE_ARCHIVE
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in self.source.rglob('*'):
                if path.is_file():
                    archive.write(path, 'run/' + path.relative_to(self.source).as_posix())
        return target

    def test_directory_import_preserves_small_self_contained_snapshot(self):
        value = self.load()
        self.assertEqual(value['selected_epochs'], {'control': 3, 'kure': 2})
        self.assertEqual(value['development'].shape, (len(self.indices['development']), 9))
        self.assertFalse(value['provenance']['weights_required'])
        self.assertTrue((self.root / 'output/reference_snapshot/source_snapshot/trainer.py').exists())
        again = self.load({**self.settings, 'reference_run': str(self.root / 'missing')})
        np.testing.assert_array_equal(value['development'], again['development'])

    def test_original_target_and_fit_indices_must_match(self):
        old = self.targets[0, 0]
        self.targets[0, 0] = 1 - old
        with self.assertRaisesRegex(ValueError, 'data hash'):
            self.load()
        self.targets[0, 0] = old
        self.indices['fit'] = self.indices['fit'][::-1]
        with self.assertRaisesRegex(ValueError, 'indices'):
            self.load()

    def test_corrupt_source_bytes_and_unpinned_manifest_rejected(self):
        path = self.source / 'source_snapshot/trainer.py'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.load()
        (self.source / 'artifact_manifest.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'exact audited'):
            self.load()

    def test_raw_portable_zip_and_autoexpanded_reference_are_supported(self):
        zipped = self.zip_source()
        outer = self.root / 'mission3_ver29_inputs.zip'
        raw = zipped.read_bytes()
        member = 'ver28_reference/' + zipped.name
        with zipfile.ZipFile(outer, 'w') as archive:
            archive.writestr(member, raw)
            archive.writestr('bundle_manifest.json', json.dumps({'reference_analysis': {
                'path': member, 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}))
        value = self.load({**self.settings, 'reference_run': '', 'input_zip': str(outer)})
        self.assertEqual(value['development'].shape[1], 9)
        found = reference.locate({**self.settings, 'reference_run': '', 'input_root': str(self.root)})
        self.assertEqual(found, zipped)

    def test_zip_path_traversal_is_rejected(self):
        path = self.zip_source()
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('../bad', b'no')
        with self.assertRaisesRegex(ValueError, 'Unsafe bundle path'):
            self.load({**self.settings, 'reference_run': str(path)})

    def test_registered_four_model_average_matches_probability_mean(self):
        values = [np.random.default_rng(i).normal(size=(100, 9)) for i in range(4)]
        expected = np.mean([1 / (1 + np.exp(-v)) for v in values], axis=0)
        result = analysis.equal_mean_logits(values)
        np.testing.assert_allclose(1 / (1 + np.exp(-result)), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result >= 0, expected >= .5)
        baseline = analysis.ensemble_logits(*values[:2])
        np.testing.assert_array_equal(result, analysis.ensemble_logits(baseline, analysis.ensemble_logits(*values[2:])))
        with self.assertRaises(ValueError):
            analysis.equal_mean_logits(values[:3])


if __name__ == '__main__':
    unittest.main()
