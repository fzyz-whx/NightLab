"""Artificial small sources and receipts; never open the user's real recordings."""
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from nightlab.audio.features import AudioFeatureConfig, AudioFeatureExtractor
from nightlab.audio.reference import window_target
from scripts.datasets import noise_features as cache


def _setup(tmp_path, monkeypatch):
    config = AudioFeatureConfig()
    monkeypatch.setattr(cache, 'EXPECTED_ROWS', 3)
    monkeypatch.setattr(cache.noise_audit, 'available_memory', lambda: 32 * 1024**3)
    manifest = {'id': cache.DATASET, 'version': 'V3', 'license': {'verified': True},
                'subjects': {'unit': 'subject', 'subjects': cache.FROZEN_SUBJECTS}, 'files': []}
    records = []
    for split, subjects in cache.FROZEN_SUBJECTS.items():
        for subject in subjects:
            identity = subject + '-100507'
            records.append({'record_id': identity, 'ok': True, 'rows': 3, 'path': f'data/processed/fixture/{identity}'})
            for extension in ('edf', 'rml'):
                manifest['files'].append({'record_id': identity, 'subject_id': subject, 'split': split,
                                         'path': f'data/raw/fixture/{identity}.{extension}'})
    manifest_path = tmp_path / f'data/manifests/{cache.DATASET}.json'
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    corpus = {'dataset': cache.DATASET, 'version': 'V3', 'complete': True,
              'split_policy': manifest['subjects'], 'records': records}
    corpus_path = tmp_path / f'data/processed/{cache.DATASET}/corpus.json'
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps(corpus))
    protocol = tmp_path / 'docs/NOISE_PROTOCOL.md'
    protocol.parent.mkdir()
    protocol.write_text('fixed synthetic fixture protocol')
    monkeypatch.setattr(cache, 'load_manifest', lambda *_: manifest)
    rate, duration = 16000, 20
    samples = (0.01 * np.sin(2 * np.pi * 270 * np.arange(rate * duration) / rate)).astype(np.float32)
    calls = []

    class Recording:
        duration_seconds = duration

        def sample_rate(self, channel):
            assert channel == 'Mic'
            return rate

        def read_window(self, channel, start, length):
            calls.append((channel, start, length))
            return samples[round(start * rate):round((start + length) * rate)]

    recording = Recording()
    provenance = {'subject_id': None, 'fixture_source': 'original'}

    def context(root, selected_manifest, record):
        identity = record['record_id']
        subject = identity.split('-')[0]
        folder = tmp_path / record['path']
        folder.mkdir(parents=True, exist_ok=True)
        starts = np.arange(3, dtype=np.float64) * 5
        ends = starts + 10
        targets, overlaps = zip(*(window_target(start, end, [], 0.5) for start, end in zip(starts, ends, strict=True)))
        labels = np.asarray([label or '' for label in targets])
        mask = labels != ''
        with threadpool_limits(limits=1):
            x = np.stack([AudioFeatureExtractor().transform(samples[round(start * rate):round(end * rate)], rate)
                          for start, end in zip(starts[mask], ends[mask], strict=True)])
        timeline_path = folder / 'timeline.npz'
        # Leave these files in place on repeated verification so corruption is detected.
        if not timeline_path.exists():
            np.savez(timeline_path, start_seconds=starts, end_seconds=ends, target=labels,
                     target_valid=mask, overlap_fraction=np.asarray(overlaps))
            np.savez(folder / 'features.npz', start_seconds=starts[mask], end_seconds=ends[mask], y=labels[mask],
                     subject_id=np.full(mask.sum(), subject), record_id=np.full(mask.sum(), identity),
                     overlap_fraction=np.asarray(overlaps)[mask], X=x)
        metadata = {'channel': 'Mic', 'config': config.metadata(), 'feature_names': list(AudioFeatureExtractor().feature_names),
                    'source_rate_hz': rate, 'segment_duration_seconds': duration, 'segment_offset_seconds': 0,
                    'candidate_windows': 3, 'rows': int(mask.sum())}
        return folder, metadata, recording, [], 0, {**provenance, 'subject_id': subject}

    monkeypatch.setattr(cache, 'record_context', context)
    return manifest, corpus, provenance, calls, samples


def _index(tmp_path, scope='pilot'):
    version = 'noise-pilot-v1' if scope == 'pilot' else 'noise-eval-v1'
    return tmp_path / f'data/processed/{cache.DATASET}/{version}/corpus.json'


def test_dry_run_has_no_writes_source_reads_or_large_allocations(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
    monkeypatch.setattr(cache, 'record_context', lambda *_: pytest.fail('dry-run opened source'))
    monkeypatch.setattr(cache.noise, 'synthetic_noise', lambda *_: pytest.fail('dry-run allocated noise'))
    result = cache.prepare_noise(tmp_path)
    assert result['writes'] is result['network'] is result['complete'] is False
    assert result['scoring'] is False
    assert [(item['record_id'], item['kind'], item['snr_db']) for item in result['records']] == [(cache.PILOT_RECORD, 'white', -5)]
    assert before == {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
    plan = cache.prepare_noise(tmp_path, 'test')
    assert len(plan['records']) == 16
    assert {item['record_id'] for item in plan['records']} == set(cache.TEST_RECORDS)
    assert {item['split'] for item in plan['records']} == {'test'}
    for identity in cache.TEST_RECORDS:
        assert {(item['kind'], item['snr_db']) for item in plan['records'] if item['record_id'] == identity} == {
            (kind, snr) for kind in cache.KINDS for snr in cache.SNRS}


def test_real_short_signal_complete_pickle_free_geometry_and_reuse(tmp_path, monkeypatch):
    _, _, _, reads, original = _setup(tmp_path, monkeypatch)
    source = original.copy()
    result = cache.prepare_noise(tmp_path, execute=True)
    assert result['complete'] and len(result['records']) == 1
    record = result['records'][0]
    assert record['rows'] == 3 and record['split'] == 'train'
    folder = tmp_path / record['path']
    assert {path.name for path in folder.iterdir()} == {'metadata.json', 'inputs.npz'}
    assert record['artifact_sha256'] == cache.sha256(folder / 'inputs.npz')
    assert record['metadata_sha256'] == cache.sha256(folder / 'metadata.json')
    with np.load(folder / 'inputs.npz', allow_pickle=False) as data:
        assert data['X_noisy'].shape == (3, 67)
        assert data['images_filtered_noisy'].shape == (3, 32, 100)
        np.testing.assert_array_equal(data['start_seconds'], [0, 5, 10])
        np.testing.assert_array_equal(data['end_seconds'], [10, 15, 20])
        assert np.all(data['subject_id'] == '00000995')
        assert not any(data[key].dtype.hasobject for key in data.files)
        assert not any('waveform' in key or 'samples' in key for key in data.files)
    receipt = json.loads((folder / 'metadata.json').read_text())
    assert abs(receipt['provenance']['actual_rounded_analysis_snr_db'] + 5) < 1e-4
    assert receipt['provenance']['source_hashes']['protocol'] == result['protocol_sha256']
    assert receipt['provenance']['environment'] == result['environment']
    assert set(result['environment']) == {'python', 'numpy', 'scipy'}
    assert set(receipt['statistics']['clipped_window_samples'].values()) == {0}
    assert reads == [('Mic', 0.0, 20.0)]
    assert cache.prepare_noise(tmp_path, execute=True)['complete']
    np.testing.assert_array_equal(original, source)


def test_test_scope_identity_and_exact_gain_clean_feature_reuse(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    original_joint = cache.representations.joint_features
    calls = []

    def joint(*args):
        calls.append(1)
        return original_joint(*args)

    monkeypatch.setattr(cache.representations, 'joint_features', joint)
    result = cache.prepare_noise(tmp_path, 'test', True)
    assert result['complete'] and len(result['records']) == 16
    assert len(calls) == 2 * (1 + 3 * (4 + 7 * 2))
    for item in result['records']:
        with np.load(tmp_path / item['path'] / 'inputs.npz', allow_pickle=False) as data:
            assert np.all(data['subject_id'] == item['record_id'].split('-')[0])
            assert item['split'] == 'test'
    gains = [json.loads((tmp_path / item['path'] / 'metadata.json').read_text())['statistics'] for item in result['records']]
    assert all(item['common_gain_hex'] == float(1).hex() for item in gains)
    assert sum(item['clean_branches_reused'] for item in gains) == 14


@pytest.mark.parametrize('tamper', ['artifact', 'source', 'protocol'])
def test_tamper_rejected_and_old_complete_invalidated(tmp_path, monkeypatch, tamper):
    _, _, provenance, _, _ = _setup(tmp_path, monkeypatch)
    result = cache.prepare_noise(tmp_path, execute=True)
    destination = tmp_path / result['records'][0]['path']
    if tamper == 'artifact':
        with (destination / 'inputs.npz').open('ab') as stream:
            stream.write(b'tamper')
    elif tamper == 'source':
        provenance['fixture_source'] = 'modified'
    else:
        (tmp_path / 'docs/NOISE_PROTOCOL.md').write_text('modified protocol')
    with pytest.raises(ValueError, match='provenance/checksum'):
        cache.prepare_noise(tmp_path, execute=True)
    assert not json.loads(_index(tmp_path).read_text())['complete']


def test_timeline_and_failure_never_publish_complete(tmp_path, monkeypatch):
    _, corpus, _, _, _ = _setup(tmp_path, monkeypatch)
    original_generate = cache._generate_condition
    monkeypatch.setattr(cache, '_generate_condition', lambda *_: (_ for _ in ()).throw(ValueError('artificial failure')))
    with pytest.raises(ValueError, match='artificial failure'):
        cache.prepare_noise(tmp_path, execute=True)
    assert not json.loads(_index(tmp_path).read_text())['complete']
    assert not list(_index(tmp_path).parent.rglob('inputs.npz'))
    monkeypatch.setattr(cache, '_generate_condition', original_generate)
    folder = tmp_path / next(record['path'] for record in corpus['records'] if record['record_id'] == cache.PILOT_RECORD)
    timeline = cache._load(folder / 'timeline.npz')
    timeline['target'][0] = 'wrong'
    np.savez(folder / 'timeline.npz', **timeline)
    with pytest.raises(ValueError, match='classic timeline'):
        cache.prepare_noise(tmp_path, execute=True)
    assert not json.loads(_index(tmp_path).read_text())['complete']


def test_unlabelled_windows_stay_unsupervised_and_gain_cache_is_exact(tmp_path, monkeypatch):
    expected = {'y': np.array(['normal', '', 'apnea']), 'subject_id': np.array(['s', 's', 's']),
                'start_seconds': np.array([0., 5., 10.]), 'end_seconds': np.array([10., 15., 20.]),
                'supervised_mask': np.array([True, False, True])}
    class Observed:
        source_rate = 16000
    def windows(*_):
        for index in range(3):
            yield {'start_seconds': expected['start_seconds'][index], 'end_seconds': expected['end_seconds'][index],
                   **{name: np.zeros(160000, dtype=np.float32) for name in cache.noise.BRANCHES}}
    monkeypatch.setattr(cache.noise_windows, 'iter_noise_windows', windows)
    calls = []
    def joint(*_):
        calls.append(1)
        return np.zeros(67, np.float32), np.zeros((32, 100), np.float32)
    monkeypatch.setattr(cache.representations, 'joint_features', joint)
    clean_cache = {}
    for gain in (1., 1., np.nextafter(1., 0.)):
        arrays, _ = cache._generate_condition(Observed(), {'common_gain': gain}, expected, clean_cache)
        cache._validate_arrays(arrays, expected)
        assert arrays['y'][1] == '' and not arrays['supervised_mask'][1]
    assert len(calls) == 12 + 6 + 12


def test_memory_failure_and_frozen_split_failure_do_not_publish_success(tmp_path, monkeypatch):
    manifest, _, _, _, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cache.noise_audit, 'available_memory', lambda: 1)
    with pytest.raises(ValueError, match='insufficient free RAM'):
        cache.prepare_noise(tmp_path, execute=True)
    assert not json.loads(_index(tmp_path).read_text())['complete']
    manifest['files'][0]['split'] = 'test'
    with pytest.raises(ValueError, match='identity/split'):
        cache.prepare_noise(tmp_path)
