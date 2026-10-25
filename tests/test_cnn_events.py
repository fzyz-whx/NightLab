import json

import numpy as np
import pytest

pytest.importorskip('torch')

from nightlab.audio.reference import ReferenceEvent
from scripts.evaluate import cnn_events as events
from scripts.evaluate.event_failures import analyze


def _prediction_arrays():
    return {'validation_rows': np.array([3, 4, 5]), 'test_rows': np.array([6, 7, 8]),
            'validation_probabilities': np.tile([.1, .9], (3, 1)),
            'test_probabilities': np.tile([.9, .1], (3, 1))}


def test_prediction_coverage_and_class_order():
    subjects = np.repeat(['a', 'b', 'c'], 3)
    assignments = {'train': ['a'], 'validation': ['b'], 'test': ['c']}
    result = events.validate_predictions(_prediction_arrays(), subjects, assignments, events.LABEL_ORDER)
    np.testing.assert_array_equal(result['validation'][1], [.9, .9, .9])
    with pytest.raises(ValueError, match='class order'):
        events.validate_predictions(_prediction_arrays(), subjects, assignments, events.LABEL_ORDER[::-1])


@pytest.mark.parametrize('rows', [np.array([3, 4]), np.array([3, 4, 4]), np.array([5, 4, 3]),
                                 np.array([3., 4., 5.]), np.array([0, 4, 5]), np.array([3, 4, 9])])
def test_missing_duplicate_shuffled_training_and_invalid_rows_rejected(rows):
    predictions = _prediction_arrays()
    predictions['validation_rows'] = rows
    with pytest.raises(ValueError, match='cover all held-out windows'):
        events.validate_predictions(predictions, np.repeat(['a', 'b', 'c'], 3),
                                    {'train': ['a'], 'validation': ['b'], 'test': ['c']}, events.LABEL_ORDER)


@pytest.mark.parametrize('probabilities', [np.ones((3, 2)), np.ones((2, 2)) * .5,
                                         np.array([[np.nan, .5]] * 3),
                                         np.array([[-.1, 1.1]] * 3),
                                         np.array([[np.inf, .5]] * 3)])
def test_invalid_probabilities_rejected(probabilities):
    predictions = _prediction_arrays()
    predictions['test_probabilities'] = probabilities
    with pytest.raises(ValueError, match='finite normalized'):
        events.validate_predictions(predictions, np.repeat(['a', 'b', 'c'], 3),
                                    {'train': ['a'], 'validation': ['b'], 'test': ['c']}, events.LABEL_ORDER)


def _fixture(root, monkeypatch):
    name = 'audio-cnn-pilot-v1'
    source = root / 'runs' / name
    source.mkdir(parents=True)
    model = root / 'models' / name
    model.mkdir(parents=True)
    (model / 'weights.pt').write_bytes(b'hashed-only-never-deserialize')
    np.savez(source / 'predictions.npz', **_prediction_arrays())
    ids = ['00000001-000001', '00000002-000001', '00000003-000001']
    subjects = [record.split('-')[0] for record in ids]
    assignments = dict(zip(('train', 'validation', 'test'), ([subject] for subject in subjects)))
    split = {'unit': 'subject', 'subjects': assignments}
    records, provenance, files = [], [], []
    for index, record in enumerate(ids):
        relative = f'data/processed/psg-audio-pilot-10/deep-inputs-v1/{record}'
        folder = root / relative
        folder.mkdir(parents=True)
        receipt = {'provenance': {'record_id': record, 'offset_seconds': 100., 'duration_seconds': 20.}}
        (folder / 'metadata.json').write_text(json.dumps(receipt))
        records.append({'record_id': record, 'path': relative, 'rows': 3, 'ok': True})
        provenance.append({'record_id': record, 'row_offset': index * 3, 'rows': 3,
                           'metadata_sha256': events.sha256(folder / 'metadata.json')})
        files.append({'record_id': record, 'path': f'data/raw/psg/{record}/source.rml'})
    corpus = {'dataset': 'psg-audio-pilot-10', 'split_policy': split, 'records': records}
    arrays = {'subject_id': np.repeat(subjects, 3),
              'start_seconds': np.tile([0., 5., 10.], 3), 'end_seconds': np.tile([10., 15., 20.], 3)}
    report = {'task': 'audio', 'feature_set': 'all', 'smoke_only': False,
              'saved_loaded_predictions_exact': True, 'label_order': events.LABEL_ORDER,
              'checkpoint_sha256': events.sha256(model / 'weights.pt'),
              'predictions_sha256': events.sha256(source / 'predictions.npz'),
              'provenance': provenance, 'split_policy': split, 'dataset': corpus['dataset']}
    report_path = source / 'metrics.json'
    report_path.write_text(json.dumps(report))
    for relative in ('docs/EVENT_PROTOCOL.md', 'backend/src/nightlab/audio/events.py',
                     'data/manifests/psg-audio-pilot-10.json'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source')
    monkeypatch.setattr(events, 'load_inputs', lambda *_: (corpus, arrays, provenance))
    monkeypatch.setattr(events, 'load_manifest', lambda *_: {'files': files})
    monkeypatch.setattr(events, 'read_reference', lambda *_:
                        (None, [], [ReferenceEvent(100, 120, 'Hypopnea')]))
    return name, report, report_path, source, model


def test_manual_events_are_record_local_and_public_summary_anonymous(tmp_path, monkeypatch):
    name, *_ = _fixture(tmp_path, monkeypatch)
    public = events.run(tmp_path, name, 'audio-cnn-events-v1')
    assert public['models']['cnn']['validation']['tp'] == 1
    assert public['models']['cnn']['test']['fn'] == 1
    assert public['models']['cnn']['test']['tp'] == 0  # no cross-record matching
    encoded = json.dumps(public)
    for forbidden in ('00000002', 'reference_intervals', 'window_probabilities', 'matches', 'bin_probabilities'):
        assert forbidden not in encoded
    assert [item['participant'] for item in public['records']] == ['validation-01', 'test-01']
    output = tmp_path / 'runs/audio-cnn-events-v1'
    local = json.loads((output / 'events.json').read_text())
    assert local['records'][0]['models']['cnn']['decoded']['events'] == [[0., 20.]]
    assert local['records'][0]['reference_intervals'] == [[0., 20.]]
    failure_report = analyze(output / 'events.json', tmp_path / 'failures.json')
    assert failure_report['models']['cnn']['test']['false_negatives']['no_reference_overlap'] == 1
    with pytest.raises(ValueError, match='output exists'):
        events.run(tmp_path, name, 'audio-cnn-events-v1')


@pytest.mark.parametrize('field,value', [('smoke_only', True), ('task', 'wearable'),
                                       ('feature_set', 'heart_rate'), ('saved_loaded_predictions_exact', False)])
def test_smoke_or_incompatible_run_rejected(tmp_path, monkeypatch, field, value):
    name, report, path, *_ = _fixture(tmp_path, monkeypatch)
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='completed full audio CNN'):
        events.run(tmp_path, name, 'rejected')
    assert not (tmp_path / 'runs/rejected').exists()


@pytest.mark.parametrize('artifact', ['predictions.npz', 'weights.pt'])
def test_tampered_saved_artifacts_rejected(tmp_path, monkeypatch, artifact):
    name, _, _, source, model = _fixture(tmp_path, monkeypatch)
    path = (source if artifact.endswith('.npz') else model) / artifact
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum differs'):
        events.run(tmp_path, name, 'rejected')
    assert not (tmp_path / 'runs/rejected').exists()


def test_changed_run_provenance_rejected(tmp_path, monkeypatch):
    name, report, path, *_ = _fixture(tmp_path, monkeypatch)
    report['provenance'] = []
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='provenance/split'):
        events.run(tmp_path, name, 'rejected')


@pytest.mark.parametrize('name', ['../outside', 'a/b', 'C:/escape'])
def test_unsafe_names_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='portable'):
        events.run(tmp_path, name, 'out')
