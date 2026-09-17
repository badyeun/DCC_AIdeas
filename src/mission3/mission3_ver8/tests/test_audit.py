import csv
import json

import numpy as np

from mission3_ver8 import audit


def test_random_stratified_review_is_stable_and_preserves_human_notes(tmp_path):
    paths = [f'call_{i:03d}.json' for i in range(100)]
    texts = [f'transcript {i}' for i in range(100)]
    targets = np.zeros((100, 9), dtype=np.int8)
    targets[:40] = 1
    probabilities = 1. - targets.astype(float)
    original = targets.copy()
    root = tmp_path / 'audit'
    summary = audit.write_audit(root, paths, texts, targets, probabilities, [.5] * 9)
    assert summary['review_rows'] == 120
    assert not summary['labels_modified'] and not summary['reviews_used_for_training']
    template = root / 'review_samples_template.csv'
    first = template.read_bytes()
    with template.open(encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
    assert all(row['review_category'] == row['review_note'] == '' for row in rows)
    assert len({(row['symptom'], row['error_type'], row['label_file_name']) for row in rows}) == 120
    for symptom in audit.FOCUS:
        for error in ['FP', 'FN']:
            assert sum(r['symptom'] == symptom and r['error_type'] == error for r in rows) == 20
    (root / 'review_samples.csv').write_text('my review')
    audit.write_audit(root, paths[::-1], texts[::-1], targets[::-1], probabilities[::-1], [.5] * 9)
    assert first == template.read_bytes()
    assert (root / 'review_samples.csv').read_text() == 'my review'
    np.testing.assert_array_equal(targets, original)


def test_insufficient_errors_are_reported_without_duplicates(tmp_path):
    summary = audit.write_audit(tmp_path, ['a.json'], ['text'], np.zeros((1, 9)), np.ones((1, 9)), [.5] * 9)
    assert summary['review_rows'] == 3
    assert all(summary['per_symptom'][s]['review_fn'] == 0 for s in audit.FOCUS)
