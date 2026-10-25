import json
import hashlib
from importlib.util import find_spec

import numpy as np
import pytest

from scripts.datasets import deep_cache as cache
from nightlab.audio.features import AudioFeatureConfig
from nightlab.audio.reference import window_target
from scripts.datasets.prepare import prepare


def _setup(tmp_path, monkeypatch):
    dataset = cache.DATASETS['wearable']
    folder = tmp_path / 'data/processed' / dataset
    folder.mkdir(parents=True)
    split = {'unit': 'subject', 'subjects': {'train': ['1'], 'validation': [], 'test': []}}
    manifest = {'id': dataset, 'version': 'test', 'subjects': split,
                'license': {'verified': True}, 'files': [{'subject_id': '1'}]}
    manifest_folder = tmp_path / 'data/manifests'
    manifest_folder.mkdir()
    (manifest_folder / f'{dataset}.json').write_text(json.dumps(manifest))
    corpus = {'dataset': dataset, 'version': 'test', 'complete': True, 'split_policy': split,
              'records': [{'record_id': '1', 'ok': True, 'rows': 1,
                           'path': f'data/processed/{dataset}/1/features-v2'}]}
    (folder / 'corpus.json').write_text(json.dumps(corpus), encoding='utf-8')
    monkeypatch.setattr(cache, 'load_manifest', lambda *_: manifest)
    expected = {'y': np.array(['W']), 'subject_id': np.array(['1']),
                'start_seconds': np.array([0.]), 'end_seconds': np.array([30.]),
                'supervised_mask': np.array([True])}
    arrays = {**expected, 'values': np.zeros((1, 5, 30), dtype=np.float32),
              'masks': np.ones((1, 2, 30), dtype=np.float32)}
    provenance = {'input_sha256': {'raw': 'original'}}
    calls = []

    def context(*_):
        calls.append('verify')
        return provenance, expected, lambda: arrays

    monkeypatch.setattr(cache, '_record_inputs', context)
    return folder, arrays, provenance, calls, corpus


def test_dry_run_performs_no_writes_or_input_preparation(tmp_path, monkeypatch):
    folder, _, _, calls, _ = _setup(tmp_path, monkeypatch)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob('*'))
    plan = cache.prepare_deep(tmp_path, 'wearable')
    assert plan['writes'] is plan['network'] is plan['complete'] is False
    assert plan['workers'] == 1 and plan['numeric_threads'] <= 2
    assert calls == []
    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob('*'))
    assert not (folder / cache.VERSION).exists()


def test_atomic_cache_reuse_integrity_and_source_change(tmp_path, monkeypatch):
    folder, arrays, provenance, calls, _ = _setup(tmp_path, monkeypatch)
    result = cache.prepare_deep(tmp_path, 'wearable', execute=True)
    assert result['complete'] and result['records'][0]['rows'] == 1
    destination = folder / cache.VERSION / '1'
    receipt = json.loads((destination / 'metadata.json').read_text())
    assert receipt['artifact_sha256'] == cache.sha256(destination / 'inputs.npz')
    assert receipt['shape'] == [1, 5, 30]
    with np.load(destination / 'inputs.npz', allow_pickle=False) as stored:
        for key in stored.files:
            np.testing.assert_array_equal(stored[key], arrays[key])
            assert not stored[key].dtype.hasobject
    assert cache.prepare_deep(tmp_path, 'wearable', True)['complete']
    assert calls == ['verify', 'verify']
    provenance['input_sha256']['raw'] = 'changed'
    with pytest.raises(ValueError, match='provenance/checksum'):
        cache.prepare_deep(tmp_path, 'wearable', True)
    assert not json.loads((folder / cache.VERSION / 'corpus.json').read_text())['complete']
    assert cache.sha256(destination / 'inputs.npz') == receipt['artifact_sha256']


@pytest.mark.skipif(find_spec('torch') is None,reason='optional torch required for training entrypoint')
def test_training_entry_rechecks_current_source_not_only_saved_npz(tmp_path,monkeypatch):
    from scripts.train import cnn
    from scripts.datasets.prepare import feature_source_hashes
    _,_,provenance,calls,_ = _setup(tmp_path,monkeypatch)
    provenance.update(record_id='1',feature_source_sha256=feature_source_hashes('sleepaccel'))
    monkeypatch.setattr(cnn,'load_manifest',cache.load_manifest)
    cache.prepare_deep(tmp_path,'wearable',True)
    assert cnn.load_inputs(tmp_path,'wearable')[0]['dataset_version'] == 'test'
    assert calls == ['verify','verify']
    # Same valid NPZ, changed current raw/old-source context: training must refuse it.
    provenance['input_sha256']['raw'] = 'changed'
    with pytest.raises(ValueError,match='integrity/source'):
        cnn.load_inputs(tmp_path,'wearable')


def test_tampered_npz_refused_and_no_false_complete(tmp_path, monkeypatch):
    folder, *_ = _setup(tmp_path, monkeypatch)
    cache.prepare_deep(tmp_path, 'wearable', True)
    with (folder / cache.VERSION / '1/inputs.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='provenance/checksum'):
        cache.prepare_deep(tmp_path, 'wearable', True)
    assert not json.loads((folder / cache.VERSION / 'corpus.json').read_text())['complete']


@pytest.mark.parametrize('error', [ValueError('failed generation'), KeyboardInterrupt()])
def test_failed_generation_is_not_published(tmp_path, monkeypatch, error):
    folder, arrays, _, _, _ = _setup(tmp_path, monkeypatch)
    expected = {key: value for key, value in arrays.items() if key not in {'values', 'masks'}}

    def fail():
        raise error

    monkeypatch.setattr(cache, '_record_inputs', lambda *_: ({}, expected, fail))
    with pytest.raises(type(error)):
        cache.prepare_deep(tmp_path, 'wearable', True)
    assert not (folder / cache.VERSION / '1').exists()
    assert not list((folder / cache.VERSION).glob('.deep-*'))
    assert not json.loads((folder / cache.VERSION / 'corpus.json').read_text())['complete']


def test_invalid_classic_corpus_invalidates_former_complete(tmp_path, monkeypatch):
    folder, _, _, _, corpus = _setup(tmp_path, monkeypatch)
    cache.prepare_deep(tmp_path, 'wearable', True)
    corpus['complete'] = False
    (folder / 'corpus.json').write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match='complete classic corpus'):
        cache.prepare_deep(tmp_path, 'wearable', True)
    assert not json.loads((folder / cache.VERSION / 'corpus.json').read_text())['complete']


def test_wearable_masks_and_object_arrays_rejected():
    expected = {'y': np.array(['W']), 'subject_id': np.array(['1']),
                'start_seconds': np.array([0.]), 'end_seconds': np.array([30.]),
                'supervised_mask': np.array([True])}
    arrays = {**expected, 'values': np.zeros((1, 5, 30), dtype=np.float32),
              'masks': np.ones((1, 2, 30), dtype=np.float32)}
    arrays['masks'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='observation masks'):
        cache._validate_arrays(arrays, expected, 'wearable')
    arrays['values'][0, 0, 0] = np.nan
    cache._validate_arrays(arrays, expected, 'wearable')
    arrays['extra'] = np.array([None], dtype=object)
    with pytest.raises(ValueError, match='shape/dtype'):
        cache._validate_arrays(arrays, expected, 'wearable')


def test_audio_all_window_geometry_and_classic_target_validation(tmp_path, monkeypatch):
    config = AudioFeatureConfig()
    duration = config.window_seconds + 2 * config.hop_seconds
    starts = np.arange(3) * config.hop_seconds
    ends = starts + config.window_seconds
    target, overlaps = zip(*(window_target(a, b, [], config.positive_overlap_fraction)
                             for a, b in zip(starts, ends)))
    labels = np.array([value or '' for value in target])
    mask = labels != ''
    timeline = {'start_seconds': starts, 'end_seconds': ends, 'target': labels,
                'target_valid': mask, 'overlap_fraction': np.array(overlaps)}
    np.savez(tmp_path / 'timeline.npz', **timeline)
    np.savez(tmp_path / 'features.npz', start_seconds=starts[mask], end_seconds=ends[mask],
             y=labels[mask], subject_id=np.full(mask.sum(), '00000001'),
             record_id=np.full(mask.sum(), '00000001-000001'), overlap_fraction=np.array(overlaps)[mask])

    class Recording:
        duration_seconds = duration
        calls = []

        def sample_rate(self, channel):
            return 8000

        def read_window(self, channel, start, length):
            self.calls.append(start)
            return np.zeros(round(length * 8000))

    recording = Recording()
    metadata = {'channel': 'Mic', 'config': config.metadata(), 'segment_duration_seconds': duration,
                'segment_offset_seconds': 0, 'source_rate_hz': 8000, 'candidate_windows': 3,
                'rows': int(mask.sum())}
    monkeypatch.setattr(cache, 'record_context', lambda *_:
                        (tmp_path, metadata, recording, [], 0, {'subject_id': '00000001'}))
    record = {'record_id': '00000001-000001', 'rows': int(mask.sum())}
    _, expected, generate = cache._audio_inputs(tmp_path, {}, record)
    arrays = generate()
    cache._validate_arrays(arrays, expected, 'audio')
    assert arrays['values'].shape == (3, 32, 100)
    assert recording.calls == list(starts)
    assert arrays['y'].dtype.kind == 'U'
    timeline['target'] = np.array(['corrupt'] * 3)
    np.savez(tmp_path / 'timeline.npz', **timeline)
    with pytest.raises(ValueError, match='target differs'):
        cache._audio_inputs(tmp_path, {}, record)


def test_real_small_wearable_raw_receipt_and_deep_cache(tmp_path):
    dataset = cache.DATASETS['wearable']
    split = {'unit': 'subject', 'subjects': {'train': ['1'], 'validation': [], 'test': []}}
    manifest = {'schema_version': 1, 'id': dataset, 'version': 'fixture', 'license': {'verified': True},
                'subjects': split, 'sources': [], 'reported_size': {}, 'inventory_complete': False,
                'preparation_policy': {'repeated_signal_blocks': 'collapse_identical'}, 'files': []}
    text = {'heart_rate': '0,70\n5,80\n60,60\n65,70\n' * 2,
            'motion': '0 0 0 1\n5 0 0 3\n60 1 0 0\n65 3 0 0\n' * 2,
            'labels': '0 0\n30 -1\n60 5\n'}
    for kind, body in text.items():
        path = tmp_path / f'data/raw/sleepaccel/{kind}/1_data.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding='utf-8', newline='')
        manifest['files'].append({'path': path.relative_to(tmp_path).as_posix(), 'subject_id': '1',
                                  'exact_bytes': path.stat().st_size,
                                  'official_checksums': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    output = f'data/processed/{dataset}/1/features-v2'
    prepare(tmp_path, {**manifest, 'id': 'sleepaccel'}, output, execute=True)
    manifest_folder = tmp_path / 'data/manifests'
    manifest_folder.mkdir()
    (manifest_folder / f'{dataset}.json').write_text(json.dumps(manifest))
    corpus = {'dataset': dataset, 'version': 'fixture', 'complete': True, 'split_policy': split,
              'records': [{'record_id': '1', 'ok': True, 'rows': 2, 'path': output}]}
    corpus_path = tmp_path / f'data/processed/{dataset}/corpus.json'
    corpus_path.write_text(json.dumps(corpus))
    hashes = {entry['path']: cache.sha256(tmp_path / entry['path']) for entry in manifest['files']}
    result = cache.prepare_deep(tmp_path, 'wearable', True)
    assert result['complete']
    assert result['source_corpus_sha256'] == cache.sha256(corpus_path)
    destination = tmp_path / result['records'][0]['path']
    with np.load(destination / 'inputs.npz', allow_pickle=False) as stored:
        assert stored['values'].shape == (2, 5, 30)
        assert stored['masks'].shape == (2, 2, 30)
        np.testing.assert_array_equal(stored['start_seconds'], [0, 60])
        np.testing.assert_array_equal(stored['y'], ['W', 'REM'])
        assert stored['values'][0, 0, 0] == 70
        assert np.isnan(stored['values'][0, :, 1]).all()
    assert hashes == {entry['path']: cache.sha256(tmp_path / entry['path']) for entry in manifest['files']}
    assert cache.prepare_deep(tmp_path, 'wearable', True)['complete']
    raw = tmp_path / manifest['files'][0]['path']
    raw.write_text(text['heart_rate'].replace('70', '71'), newline='')
    with pytest.raises(ValueError, match='raw wearable input differs'):
        cache.prepare_deep(tmp_path, 'wearable', True)
    assert not json.loads((destination.parent / 'corpus.json').read_text())['complete']
