"""Import the pinned completed ver28 prediction reference, never old weights/code."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import stat
import zipfile

import numpy as np

from . import analysis, bootstrap, data

REFERENCE_FORMAT = "mission3_ver28_v1"
REFERENCE_ARCHIVE = "ver28_analysis_20261013T005822Z_1de5c4ec.zip"
EXPECTED_MANIFEST_SHA256 = "97d5ea739790354e4cc73c2a0643945dc7ff2595522c7829ef9e012024e3ceaf"
MAX_FILE_BYTES = 32 * 1024 * 1024
BASE_FILES = {"run_state.json", "settings.json", "screening_complete.json", "selection.json",
              "resolved_models.json", "split_manifest.json", "split_indices.npz",
              "development_predictions.npz", "confirmation_predictions.npz", "confirmation_report.json"}
POLICY = "independent sigmoid >= 0.5; ensemble is fixed equal probability mean"
IDENTITY_POLICY = "independent_logits_greater_than_or_equal_to_zero"


def _json(raw):
    return json.loads(raw.decode("utf-8-sig"))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _required(manifest):
    names = set(BASE_FILES)
    for kind in ("control", "kure"):
        names.update(f"models/{kind}/{name}" for name in
                     ("training_identity.json", "history.json", "training_result.json"))
    source = {name for name in manifest["files"] if name.startswith("source_snapshot/")}
    if not {"source_snapshot/trainer.py", "source_snapshot/runtime.py", "source_snapshot/data.py",
            "source_snapshot/analysis.py", "source_snapshot/settings.py"}.issubset(source):
        raise ValueError("Reference manifest omits its source provenance")
    return names | source


def _verified_contents(read):
    raw = read("artifact_manifest.json")
    if _sha(raw) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Reference must be the exact audited completed ver28 run")
    manifest = _json(raw)
    if manifest.get("format") != REFERENCE_FORMAT or not isinstance(manifest.get("files"), dict):
        raise ValueError("Reference manifest format is invalid")
    result = {"artifact_manifest.json": raw}
    for name in sorted(_required(manifest)):
        bootstrap._relative(name)
        metadata = manifest["files"].get(name, {})
        if type(metadata.get("size")) is not int or not 0 <= metadata["size"] <= MAX_FILE_BYTES:
            raise ValueError("Reference required artifact missing or unexpectedly large")
        payload = read(name)
        if len(payload) != metadata["size"] or _sha(payload) != metadata.get("sha256"):
            raise ValueError("Reference artifact integrity failed: " + name)
        result[name] = payload
    return result


def _read_directory(root):
    def read(name):
        path = bootstrap._target(root.resolve(), name)
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Reference required artifact missing or too large")
        return path.read_bytes()
    return _verified_contents(read)


def _read_zip(payload):
    with zipfile.ZipFile(payload) as archive:
        names = set()
        for info in archive.infolist():
            bootstrap._relative(info.orig_filename)
            key = info.filename.rstrip('/').casefold()
            if key in names or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("Unsafe reference ZIP entry")
            names.add(key)
        matches = [n for n in archive.namelist() if n.endswith('/artifact_manifest.json') or n == 'artifact_manifest.json']
        if len(matches) != 1:
            raise ValueError("Reference ZIP must contain exactly one run manifest")
        prefix = matches[0][:-len('artifact_manifest.json')]
        def read(name):
            info = archive.getinfo(prefix + name)
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError("Reference ZIP artifact is unexpectedly large")
            return archive.read(info)
        return _verified_contents(read)


def _read_source(path):
    if path.is_dir():
        return _read_directory(path)
    if path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            if 'bundle_manifest.json' in archive.namelist():
                manifest = _json(archive.read('bundle_manifest.json'))
                item = manifest.get('reference_analysis', {})
                name = item.get('path', '')
                bootstrap._relative(name)
                info = archive.getinfo(name)
                if info.file_size > MAX_FILE_BYTES or info.file_size != item.get('size'):
                    raise ValueError("Bundled reference size mismatch")
                raw = archive.read(info)
                if _sha(raw) != item.get('sha256'):
                    raise ValueError("Bundled reference hash mismatch")
                return _read_zip(io.BytesIO(raw))
        return _read_zip(path)
    raise ValueError("Reference path is not a run folder or ZIP")


def locate(settings):
    explicit = settings.get('reference_run', '')
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.is_file() or (root / 'artifact_manifest.json').is_file():
            return root
        roots = [root]
    else:
        roots = [Path(settings['input_root']).expanduser()]
        if settings.get('input_zip'):
            item = Path(settings['input_zip']).expanduser()
            if item.is_file():
                return item.resolve()
            if item.is_dir():
                roots.append(item)
    candidates = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(REFERENCE_ARCHIVE):
            candidates.add(path.resolve())
        for path in root.rglob('mission3_ver29_inputs.zip'):
            candidates.add(path.resolve())
        for marker in root.rglob('artifact_manifest.json'):
            if marker.stat().st_size <= MAX_FILE_BYTES and _sha(marker.read_bytes()) == EXPECTED_MANIFEST_SHA256:
                candidates.add(marker.parent.resolve())
    # Multiple representations of the pinned reference are equivalent, but
    # every chosen source still undergoes exact manifest/content verification.
    if not candidates:
        raise FileNotFoundError("Attach the ver29 input bundle with the audited ver28 reference")
    return sorted(candidates, key=lambda p: (p.name != REFERENCE_ARCHIVE, str(p)))[0]


def _arrays(contents, name):
    with np.load(io.BytesIO(contents[name]), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def import_reference(settings, output, texts, targets, indices, split_manifest):
    """Verify against current original data, then preserve only small evidence."""
    snapshot = Path(output) / 'reference_snapshot'
    if (snapshot / 'artifact_manifest.json').is_file():
        contents, source = _read_directory(snapshot), 'self-contained reference_snapshot'
    else:
        path = locate(settings)
        contents, source = _read_source(path), str(path)
    state = _json(contents['run_state.json'])
    config = _json(contents['settings.json'])
    split = _json(contents['split_manifest.json'])
    selection = _json(contents['selection.json'])
    if (state.get('format') != REFERENCE_FORMAT or state.get('status') != 'screening_complete'
            or state.get('run_mode') != 'screening' or state.get('submission_ready') is not False
            or config.get('run_mode') != 'screening' or config.get('run_validation') is not False):
        raise ValueError("Reference must be completed screening without official Validation selection")
    if (split.get('data_digest') != split_manifest['data_digest']
            or split.get('split_digest') != split_manifest['split_digest']
            or split.get('label_order') != list(data.TARGET_SYMPTOMS)
            or selection.get('split_digest') != split_manifest['split_digest']
            or selection.get('training_rows') != len(indices['fit'])
            or selection.get('selected_recipe') != 'mean' or selection.get('members') != ['control', 'kure']
            or selection.get('decision_rule') != POLICY
            or selection.get('selected_epochs') != {'control': 3, 'kure': 2}):
        raise ValueError("Reference data, split, labels, epochs, or policy differs")
    if data.dataset_digest(texts, targets) != split['data_digest']:
        raise ValueError("Reference original data hash mismatch")
    old_indices = _arrays(contents, 'split_indices.npz')
    if set(old_indices) != set(indices) or any(not np.array_equal(old_indices[k], v) for k,v in indices.items()):
        raise ValueError("Reference row indices differ")
    specs = _json(contents['resolved_models.json'])
    identity_digest = hashlib.sha256()
    for text in texts:
        raw = text.encode('utf-8')
        identity_digest.update(len(raw).to_bytes(8, 'little'))
        identity_digest.update(raw)
    # ver28 trainer validates/casts labels to float32 before its identity hash.
    identity_digest.update(np.asarray(targets, dtype=np.float32).tobytes())
    for kind in ('control', 'kure'):
        identity = _json(contents[f'models/{kind}/training_identity.json'])
        history = _json(contents[f'models/{kind}/history.json'])
        result = _json(contents[f'models/{kind}/training_result.json'])
        if (identity.get('format') != 'mission3_ver28_full_encoder_v1'
                or identity.get('kind') != kind or identity.get('target_symptoms') != list(data.TARGET_SYMPTOMS)
                or identity.get('decision') != IDENTITY_POLICY or identity.get('data_sha256') != identity_digest.hexdigest()
                or identity.get('train_indices') != indices['fit'].tolist()
                or identity.get('dev_indices') != indices['development'].tolist()
                or identity.get('epochs') != 3 or result.get('status') != 'complete'
                or result.get('best_epoch') != selection['selected_epochs'][kind]
                or max(history, key=lambda r: r['macro_f1'])['epoch'] != result['best_epoch']
                or identity.get('model_id') != specs[kind]['model_id']
                or identity.get('revision') != specs[kind]['revision']
                or not re.fullmatch('[0-9a-f]{40}', specs[kind]['revision'])):
            raise ValueError("Reference model provenance or original fit rows differ")
        for name, value in identity['settings'].items():
            if settings.get(name) != value:
                raise ValueError("Reference shared training recipe differs: " + name)
    development = _arrays(contents, 'development_predictions.npz')
    confirmation = _arrays(contents, 'confirmation_predictions.npz')
    for name, arrays in (('development', development), ('confirmation', confirmation)):
        if not np.array_equal(arrays['indices'], indices[name]) or not np.array_equal(arrays['targets'], targets[indices[name]]):
            raise ValueError("Reference predictions do not align with original targets")
    observed = analysis.ensemble_logits(development['control'], development['kure'])
    if not np.allclose(observed, development['mean'], rtol=1e-12, atol=1e-12) or not np.array_equal(observed >= 0, development['mean'] >= 0):
        raise ValueError("Reference is not the fixed equal baseline")
    for kind in ('control', 'kure', 'mean'):
        if analysis.metrics(development['targets'], development[kind]) != selection['scores'][kind]:
            raise ValueError("Reference development score failed reproduction")
    confirm_report = _json(contents['confirmation_report.json'])
    if analysis.metrics(confirmation['targets'], confirmation['selected']) != confirm_report['selected']:
        raise ValueError("Reference confirmation score failed reproduction")
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        dest = bootstrap._target(snapshot.resolve(), name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    old_provenance = Path(output) / 'reference_provenance.json'
    original_source = _json(old_provenance.read_bytes()).get('source', source) if old_provenance.is_file() else source
    provenance = {'format': 'mission3_ver29_reference_v1', 'source': original_source,
        'reference_manifest_sha256': EXPECTED_MANIFEST_SHA256, 'verified_artifacts': len(contents)-1,
        'partial_artifact_copy': True, 'weights_required': False, 'original_labels_and_indices_verified': True,
        'baseline_epochs': selection['selected_epochs'], 'baseline_training_rows': len(indices['fit']),
        'baseline_precision': 'BF16 on T4; ver29 fresh training uses native-supported precision',
        'comparison_limitation': 'Historical baseline reuses the same split; precision changes mean this is not a pure loss ablation.',
        'source_not_executed': True}
    data.write_json(Path(output) / 'reference_provenance.json', provenance)
    return {'development': analysis._logits(development['mean'], len(indices['development'])),
            'confirmation': analysis._logits(confirmation['selected'], len(indices['confirmation'])),
            'selected_epochs': selection['selected_epochs'], 'specs': specs, 'provenance': provenance}
