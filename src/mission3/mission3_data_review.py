"""CPU-only review of frozen ver9 baseline errors on Training calibration.

This module reads the original input ZIP and small JSON/NPZ artifacts. It does
not load model weights, train, tune thresholds, or read official Validation.
The companion Colab notebook embeds this source; uploading it alone is enough.
"""
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ['고열', '구토', '두통', '복통', '어지러움', '열상', '오심', '전신쇠약', '호흡곤란']
CATEGORIES = {
    'model_missed_evidence': '명확한 근거를 모델이 놓침',
    'question_negation_history': '질문·부정·과거 문맥을 혼동',
    'symptom_confusion': '다른 증상과 혼동한 것으로 의심',
    'text_label_unclear': '전사만으로 정답 근거를 판단하기 어려움',
    'correct_clear': '맞힌 사례이며 근거도 명확함',
    'other': '기타: 메모에 구체적으로 기록',
}
SCOPE = 'Training calibration; frozen fitted thresholds; descriptive, not independent evaluation'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_analysis(input_zip, ver9_dir):
    ver9_dir = Path(ver9_dir)
    manifest = read_json(ver9_dir / 'mission3_ver9_model.ckpt')
    identity = read_json(ver9_dir / 'run_identity.json')
    require(manifest.get('format') == 'mission3_ver9' and manifest.get('labels') == LABELS,
            'ver9 manifest 또는 증상 순서가 다릅니다.')
    require(manifest.get('signature') == identity.get('signature'), '서로 다른 실행의 ver9 파일입니다.')
    selection = manifest['selection']
    require(manifest.get('selected_name') == 'ver4_baseline'
            and manifest.get('status') == 'selection_frozen'
            and selection.get('blend_space') == 'raw'
            and selection.get('candidate_weights') == [0.0] * 9,
            '이 노트북은 현재 ver9의 확정된 ver4 baseline 결과 전용입니다.')
    alpha = np.asarray(selection['neural_weights'], dtype=np.float64)
    thresholds = np.asarray(selection['thresholds'], dtype=np.float64)
    require(alpha.shape == thresholds.shape == (9,) and np.isfinite(alpha).all()
            and np.isfinite(thresholds).all() and ((alpha >= 0) & (alpha <= 1)).all(),
            '잘못된 혼합 비율 또는 임계값입니다.')

    with zipfile.ZipFile(input_zip) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), 'ZIP 안에 중복 경로가 있습니다.')

        def unique(suffix):
            found = [name for name in members if name.endswith(suffix)]
            require(len(found) == 1, f'ZIP 안에 {suffix} 파일이 정확히 하나 있어야 합니다.')
            return found[0]

        cache_bytes = archive.read(unique('/ver4/calibration_predictions.npz'))
        split_bytes = archive.read(unique('/ver4/splits.json'))
        require(sha(cache_bytes) == identity.get('calibration_sha256'), 'ver9와 원본 예측 캐시의 해시가 다릅니다.')
        require(sha(split_bytes) == identity.get('splits_sha256'), 'ver9와 원본 데이터 분할의 해시가 다릅니다.')
        splits = json.loads(split_bytes)
        with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as cache:
            names = cache['filenames'].tolist()
            truth = cache['targets'].copy()
            neural = cache['neural'].astype(np.float64)
            sparse = cache['sparse'].astype(np.float64)
        require(len(names) > 0 and len(names) == len(set(names)), '예측 파일명이 비어 있거나 중복됩니다.')
        require(len(splits['calibration']) == len(set(splits['calibration']))
                and set(names) == set(splits['calibration']), '보정 집합의 파일 목록이 다릅니다.')
        for fold in splits['folds']:
            require(not set(names).intersection(fold['fit'] + fold['stop']), '학습/stop과 보정 집합이 겹칩니다.')
        for arr in (truth, neural, sparse):
            require(arr.shape == (len(names), 9) and np.isfinite(arr).all()
                    and ((arr >= 0) & (arr <= 1)).all(), '잘못된 예측 배열입니다.')
        require(np.isin(truth, [0, 1]).all(), '정답은 0 또는 1이어야 합니다.')

        # Read only the original Training calibration JSONs, without extracting
        # the multi-GB model weights or opening any Validation member.
        training = {}
        for name in members:
            if '/TL_서울_구급/' in name and name.endswith('.json'):
                basename = name.rsplit('/', 1)[-1]
                require(basename not in training, 'Training에 중복 파일명이 있습니다.')
                training[basename] = name
        require(set(names) <= set(training), 'ZIP에 필요한 서울 Training JSON이 없습니다.')
        texts = []
        for i, name in enumerate(names):
            raw = json.loads(archive.read(training[name]).decode('utf-8-sig'))
            symptoms = raw['symptom']
            if symptoms is None:
                symptoms = []
            if isinstance(symptoms, str):
                symptoms = [symptoms]
            require(isinstance(symptoms, list) and all(isinstance(s, str) for s in symptoms),
                    f'정답 형식 오류: {name}')
            symptoms = {s.strip() for s in symptoms}
            require(np.array_equal([int(s in symptoms) for s in LABELS], truth[i]),
                    f'원본과 캐시의 정답 불일치: {name}')
            utterances = raw.get('utterances', raw.get('utterences'))
            require(isinstance(utterances, list), f'전사 형식 오류: {name}')
            turns = []
            for turn in utterances:
                if isinstance(turn, dict) and isinstance(turn.get('text'), str):
                    text = ' '.join(unicodedata.normalize('NFC', turn['text']).split())
                    if text:
                        turns.append(text)
            texts.append('\n'.join(turns))
            if (i + 1) % 500 == 0 or i + 1 == len(names):
                print(f'Training 전사 확인: {i + 1:,}/{len(names):,}', flush=True)
    probabilities = alpha * neural + (1 - alpha) * sparse
    text_digest = sha(json.dumps(list(zip(names, texts)), ensure_ascii=False).encode())
    return dict(names=names, truth=truth, texts=texts, probabilities=probabilities,
                predictions=probabilities >= thresholds, thresholds=thresholds,
                cache_sha256=sha(cache_bytes), text_sha256=text_digest,
                signature=manifest['signature'], scope=SCOPE)


def confusion_rows(data):
    rows = []
    for j, symptom in enumerate(LABELS):
        y, p = data['truth'][:, j].astype(bool), data['predictions'][:, j]
        tp, fp = int((y & p).sum()), int((~y & p).sum())
        fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
        rows.append(dict(symptom=symptom, TP=tp, FP=fp, FN=fn, TN=tn,
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                         f1=2 * tp / max(2 * tp + fp + fn, 1)))
    return rows


def make_samples(data, focus=('오심',), per_group=20, seed=42):
    require(type(per_group) is int and per_group > 0, '표본 수는 양의 정수여야 합니다.')
    require(len(focus) > 0 and len(focus) == len(set(focus)) and set(focus) <= set(LABELS), '증상 설정을 확인하세요.')
    records = []
    for symptom in focus:
        j = LABELS.index(symptom)
        y, p = data['truth'][:, j].astype(bool), data['predictions'][:, j]
        masks = [('FP', ~y & p), ('FN', y & ~p), ('TP', y & p), ('TN', ~y & ~p)]
        for group_index, (kind, mask) in enumerate(masks):
            indices = sorted(np.flatnonzero(mask).tolist(), key=lambda i: data['names'][i])
            rng = random.Random(seed + j * 1009 + group_index)
            for i in rng.sample(indices, min(per_group, len(indices))):
                records.append(dict(
                    review_id=symptom + ':' + data['names'][i], symptom=symptom, error_type=kind,
                    label_file_name=data['names'][i], gold=int(y[i]), prediction=int(p[i]),
                    probability=float(data['probabilities'][i, j]), threshold=float(data['thresholds'][j]),
                    all_target_labels=json.dumps([s for s, v in zip(LABELS, data['truth'][i]) if v], ensure_ascii=False),
                    transcript=data['texts'][i], review_category='', evidence='', review_note='',
                ))
    # Mix error/control order so blind inspection does not reveal the group.
    random.Random(seed).shuffle(records)
    return records


def write_csv(path, rows):
    path = Path(path)
    require(bool(rows), '저장할 행이 없습니다.')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_review(data, output_root, reviewer='reviewer1', focus=('오심',), per_group=20, seed=42):
    require(re.fullmatch(r'[A-Za-z0-9_-]+', reviewer) is not None, '검토자 이름은 영문·숫자·밑줄·하이픈만 사용하세요.')
    samples = make_samples(data, focus, per_group, seed)
    settings = dict(signature=data['signature'], cache_sha256=data['cache_sha256'],
                    text_sha256=data['text_sha256'], focus=list(focus), per_group=per_group, seed=seed)
    key = sha(json.dumps(settings, sort_keys=True).encode())[:12]
    directory = Path(output_root) / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'review_{reviewer}.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as f:
            existing = list(csv.DictReader(f))
        require([r['review_id'] for r in existing] == [r['review_id'] for r in samples],
                '기존 검토 파일의 사례 목록이 달라 덮어쓰지 않습니다.')
        for new, old in zip(samples, existing):
            require(old.get('review_category', '') in ('', *CATEGORIES),
                    '기존 검토 CSV의 분류 코드가 올바르지 않습니다. 파일을 유지하고 중단합니다.')
            for field in ('review_category', 'evidence', 'review_note'):
                new[field] = old[field]
    else:
        write_csv(path, samples)
    write_csv(directory / 'symptom_counts.csv', confusion_rows(data))
    cooccurrence = []
    nausea, vomiting = data['truth'][:, 6], data['truth'][:, 1]
    for n, v, label in [(1, 0, '오심만'), (0, 1, '구토만'), (1, 1, '둘 다'), (0, 0, '둘 다 없음')]:
        mask = (nausea == n) & (vomiting == v)
        cooccurrence.append(dict(group=label, samples=int(mask.sum()),
                                 predicted_nausea=int(data['predictions'][mask, 6].sum())))
    write_csv(directory / 'nausea_vomiting_counts.csv', cooccurrence)
    (directory / 'analysis_info.json').write_text(json.dumps(
        {**settings, 'scope': SCOPE, 'samples': len(data['names']), 'review_samples': len(samples),
         'labels_modified': False, 'automatic_cause_assignment': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    return path, samples


def save_review(path, rows, index, category, evidence='', note=''):
    require(category in CATEGORIES, '분류를 선택하세요.')
    require(bool(evidence.strip() or note.strip()), '근거 문장 또는 판단 메모를 적어주세요.')
    require(type(index) is int and 0 <= index < len(rows), '사례 번호를 확인하세요.')
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        saved = list(csv.DictReader(f))
    fields = ('review_id', 'review_category', 'evidence', 'review_note')
    require(len(saved) == len(rows) and all(
        all(a.get(field) == b.get(field) for field in fields)
        for a, b in zip(saved, rows)),
        '다른 탭 또는 파일에서 검토 내용이 바뀌었습니다. 분석 셀과 검토 화면을 다시 실행해 불러오세요.')
    updated = [dict(row) for row in rows]
    updated[index].update(review_category=category, evidence=evidence.strip(), review_note=note.strip())
    write_csv(path, updated)
    rows[index].update(updated[index])


def review_summary(rows):
    result = []
    for symptom in dict.fromkeys(row['symptom'] for row in rows):
        for kind in ('FP', 'FN', 'TP', 'TN'):
            group = [r for r in rows if r['symptom'] == symptom and r['error_type'] == kind]
            counts = Counter(r['review_category'] for r in group if r['review_category'])
            result.append(dict(symptom=symptom, group=kind, sampled=len(group), reviewed=sum(counts.values()),
                               **{code: counts[code] for code in CATEGORIES}))
    return result
