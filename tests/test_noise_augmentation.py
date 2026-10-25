"""Small artificial signals only; augmentation never reads validation/test values."""
import json

import numpy as np
import pytest

from scripts.datasets import noise_augmentation as cache
from scripts.datasets import noise_features as base
from test_noise_features import _setup


def setup(tmp_path, monkeypatch):
    result = _setup(tmp_path, monkeypatch)
    (tmp_path / 'docs/NOISE_AUGMENTATION_PROTOCOL.md').write_text('fixed augmentation fixture')
    monkeypatch.setattr(cache, 'EXPECTED_TOTAL_ROWS', 18)
    return result


def index_path(root):
    return root / f'data/processed/{base.DATASET}/{cache.VERSION}/corpus.json'


def test_selector_is_stable_native_grid_and_label_independent():
    import hashlib
    record = '00000995-100507'
    expected = int(hashlib.sha256(json.dumps([record, 240000, 42]).encode()).hexdigest(), 16) % 8
    assert cache.condition_index(record, 240000) == expected
    assert [cache.condition_index(record, i) for i in range(30)] == [cache.condition_index(record, i) for i in range(30)]


def test_dry_train_scope_zero_writes(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    before = set(tmp_path.rglob('*'))
    monkeypatch.setattr(base, '_verified_context', lambda *_: pytest.fail('dry source read'))
    result = cache.prepare_augmentation(tmp_path)
    assert result['complete'] is result['writes'] is False
    assert len(result['records']) == 6
    assert {r['record_id'].split('-')[0] for r in result['records']} == set(base.FROZEN_SUBJECTS['train'])
    assert before == set(tmp_path.rglob('*'))


def test_real_joint_pipeline_once_each_supervised_and_reuse(tmp_path, monkeypatch):
    _, _, _, reads, _ = setup(tmp_path, monkeypatch)
    original = base._verified_context
    visited = []
    def checked(root, manifest, record):
        assert record['record_id'].split('-')[0] in base.FROZEN_SUBJECTS['train']
        visited.append(record['record_id'])
        return original(root, manifest, record)
    monkeypatch.setattr(base, '_verified_context', checked)
    result = cache.prepare_augmentation(tmp_path, True)
    assert result['complete'] and len(visited) == 6
    for record in result['records']:
        folder = tmp_path / record['path']
        data = base._load(folder / 'inputs.npz')
        np.testing.assert_array_equal(data['source_window_index'], [0, 1, 2])
        assert data['X'].shape == (3, 67) and data['values'].shape == (3, 32, 100)
        assert data['X'].dtype == data['values'].dtype == np.float32
        assert data['kind'].dtype.kind == data['y'].dtype.kind == 'U'
        receipt = json.loads((folder / 'metadata.json').read_text())
        assert sum(receipt['condition_counts'].values()) == 3
        assert len(receipt['condition_counts']) == 8
        assert all(abs(c['actual_rounded_analysis_snr_db'] - c['snr_db']) < 1e-4 for c in receipt['conditions'])
    assert len(reads) == 6
    assert cache.prepare_augmentation(tmp_path, True)['complete']
    assert len(visited) == 12  # raw/source context is always reverified


def test_mask_selection_and_only_selected_fft(monkeypatch):
    expected = {'y': np.array(['normal', '', 'apnea']), 'subject_id': np.array(['s'] * 3),
                'start_seconds': np.array([0., 5., 10.]), 'end_seconds': np.array([10., 15., 20.]),
                'supervised_mask': np.array([True, False, True])}
    arrays, choices = cache._allocate('r', 16000, expected)
    assert arrays['source_window_index'].tolist() == [0, 2]
    calls = []
    monkeypatch.setattr(base.representations, 'joint_features', lambda wave,*_: (calls.append(wave.copy()) or np.zeros(67, np.float32), np.zeros((32, 100), np.float32)))
    rate = 16000
    original = (.1*np.sin(2*np.pi*400*np.arange(rate*20)/rate)).astype(np.float32)
    observed = base.noise.NoiseRecording(original,base.noise.synthetic_noise(len(original),rate,'white',42),rate)
    filled = np.zeros(2, bool)
    for condition in range(8):
        calibration = observed.calibrate(cache.CONDITIONS[condition][1])
        streamed = list(base.noise_windows.iter_noise_windows(observed,calibration))
        first = len(calls)
        cache._fill_condition(observed,calibration,expected,choices,condition,arrays,filled)
        chosen_indices = [int(index) for row,index in enumerate(arrays['source_window_index']) if choices[row] == condition]
        for actual,index in zip(calls[first:],chosen_indices,strict=True):
            np.testing.assert_array_equal(actual,streamed[index]['noisy'])
    assert filled.all() and len(calls) == 2
    with pytest.raises(ValueError, match='duplicate'):
        cache._fill_condition(observed,observed.calibrate(cache.CONDITIONS[int(choices[0])][1]),expected,choices,int(choices[0]),arrays,filled)


@pytest.mark.parametrize('tamper', ['artifact', 'metadata', 'source', 'protocol', 'implementation'])
def test_tamper_refused_never_complete(tmp_path, monkeypatch, tamper):
    _, _, provenance, _, _ = setup(tmp_path, monkeypatch)
    result = cache.prepare_augmentation(tmp_path, True)
    folder = tmp_path / result['records'][0]['path']
    if tamper in ('artifact', 'metadata'):
        path = folder / ('inputs.npz' if tamper == 'artifact' else 'metadata.json')
        with path.open('ab') as stream:
            stream.write(b' ' if tamper == 'metadata' else b'tamper')
    elif tamper == 'source':
        provenance['fixture_source'] = 'changed'
    elif tamper == 'protocol':
        (tmp_path / 'docs/NOISE_AUGMENTATION_PROTOCOL.md').write_text('changed')
    else:
        original = cache._sources
        monkeypatch.setattr(cache, '_sources', lambda root: {**original(root), 'noise_augmentation': 'changed'})
    with pytest.raises(ValueError):
        cache.prepare_augmentation(tmp_path, True)
    assert not json.loads(index_path(tmp_path).read_text())['complete']


@pytest.mark.parametrize('error', [ValueError('failure'), KeyboardInterrupt()])
def test_failure_or_interrupt_never_complete(tmp_path, monkeypatch, error):
    setup(tmp_path, monkeypatch)
    def fail(*_):
        raise error
    monkeypatch.setattr(cache, '_generate', fail)
    with pytest.raises(type(error)):
        cache.prepare_augmentation(tmp_path, True)
    assert not json.loads(index_path(tmp_path).read_text())['complete']


def test_raw_verification_failure_blocks_reuse(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    cache.prepare_augmentation(tmp_path, True)
    def bad_raw(*_):
        raise ValueError('raw input differs from official receipt')
    monkeypatch.setattr(base, '_verified_context', bad_raw)
    with pytest.raises(ValueError, match='raw input'):
        cache.prepare_augmentation(tmp_path, True)
    assert not json.loads(index_path(tmp_path).read_text())['complete']
