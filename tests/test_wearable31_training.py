"""v3-specific loader guards, without retraining or touching real raw data."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.train import wearable31 as runner


def pair():
    deep = {'values': np.ones((2, 5, 30), np.float32), 'masks': np.ones((2, 2, 30), np.float32),
            'y': np.array(['W', 'N2']), 'subject_id': np.array(['s', 's']),
            'start_seconds': np.array([0., 30.]), 'end_seconds': np.array([30., 60.]),
            'supervised_mask': np.ones(2, dtype=bool)}
    classic = {key: deep[key].copy() for key in ('y', 'subject_id', 'start_seconds', 'end_seconds')}
    return deep, classic


def test_missing_preserved_and_identity_aligned():
    deep, classic = pair()
    deep['values'][0, 0, 0] = np.nan
    deep['masks'][0, 0, 0] = 0
    runner.validate_arrays(deep, classic, 's')
    assert np.isnan(deep['values'][0, 0, 0])


@pytest.mark.parametrize('fault', ['mask', 'label', 'identity', 'time', 'supervised', 'shape', 'inf', 'fields'])
def test_semantic_tampering_rejected(fault):
    deep, classic = pair()
    if fault == 'mask': deep['masks'][0, 0, 0] = 0
    elif fault == 'label': deep['y'][0] = 'X'
    elif fault == 'identity': deep['subject_id'][0] = 'x'
    elif fault == 'time': deep['start_seconds'][0] = 1
    elif fault == 'supervised': deep['supervised_mask'][0] = False
    elif fault == 'shape': deep['values'] = deep['values'][:, :, :29]
    elif fault == 'inf': deep['values'][0, 1, 0] = np.inf
    elif fault == 'fields': deep['unexpected'] = np.zeros(2)
    with pytest.raises(ValueError):
        runner.validate_arrays(deep, classic, 's')


@pytest.mark.parametrize('fault', ['incomplete', 'duplicate', 'split', 'source', 'errors', 'version'])
def test_index_tampering_rejected_before_raw_access(tmp_path, monkeypatch, fault):
    grouped = {str(i): {} for i in range(31)}
    manifest = {'id': 'sleepaccel-cohort-31', 'subjects': {'subjects': {'train': list(grouped)[:18],
                'validation': list(grouped)[18:24], 'test': list(grouped)[24:]}}}
    monkeypatch.setattr(runner.audit, 'metadata', lambda root: (manifest, grouped))
    monkeypatch.setattr(runner.prepare, 'source_hashes', lambda root: {'x': 'fixed'})
    item = {'dataset': manifest['id'], 'split_policy': manifest['subjects'], 'source_sha256': {'x': 'fixed'},
            'complete': True, 'errors': [], 'records': [{'record_id': s} for s in grouped],
            'feature_version': runner.prepare.FEATURE_VERSION, 'version': runner.prepare.DEEP_VERSION, 'task': 'wearable'}
    classic, deep = copy.deepcopy(item), copy.deepcopy(item)
    if fault == 'incomplete': deep['complete'] = False
    elif fault == 'duplicate': deep['records'][-1] = deep['records'][0]
    elif fault == 'split': deep['split_policy'] = {}
    elif fault == 'source': deep['source_sha256'] = {'x': 'changed'}
    elif fault == 'errors': deep['errors'] = ['failed record']
    elif fault == 'version': deep['version'] = 'deep-inputs-v1'
    for relative, data in ((runner.prepare.BASE, classic), (runner.prepare.DEEP, deep)):
        folder = tmp_path / relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'corpus.json').write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        runner.load_inputs(tmp_path)


def test_public_report_removes_private_fitted_and_identity_details():
    result = runner.public_report({'normalizer': {'mean': [1]}, 'provenance': ['private'],
        'split_subjects': {'train': ['private']}, 'split_policy': {'subjects': ['private']}, 'models': {}}, {'exact': True})
    assert not {'normalizer', 'provenance', 'split_subjects', 'split_policy'} & set(result)
    assert result['subject_counts'] == {'train': 18, 'validation': 6, 'test': 7}
