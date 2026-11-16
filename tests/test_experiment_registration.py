import copy
import json
from pathlib import Path

import pytest

from nightlab.experiments.baseline import classification_metrics
from nightlab.services.experiments import ExperimentService
from nightlab.services.local_experiments import MAX_BYTES, register


def metrics():
    labels = ['no_target_reference_event', 'reference_respiratory_event']
    value = classification_metrics(labels, labels, labels, [0.1, 0.9], labels[1])
    return {'models': {name: {split: copy.deepcopy(value) for split in ('validation', 'test')}
                       for name in ('majority', 'random_forest')},
            'dataset': 'psg-audio-pilot-10', 'feature_subset': 'all', 'feature_count': 67,
            'subject_count': 3, 'split_subjects': {'train': ['private-name'], 'validation': ['v'], 'test': ['t']},
            'split_rows': {'train': 2, 'validation': 2, 'test': 2},
            'config': {'seed': 42, 'trees': 200, 'max_depth': 12, 'min_samples_leaf': 3},
            'provenance': [{'features_path': 'C:/secret/personal'}], 'environment': {'secret': 'secret'}}


def write(root, report=None):
    path = root / 'runs' / 'my-run' / 'metrics.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report if report is not None else metrics()), encoding='utf-8')
    return path


def test_synthetic_metrics_dry_run_register_browse(tmp_path):
    write(tmp_path)
    assert register(tmp_path, 'my-run')['mode'] == 'dry-run'
    assert not (tmp_path / 'private').exists()
    result = register(tmp_path, 'my-run', True)
    assert result['mode'] == 'registered'
    service = ExperimentService(tmp_path)
    assert [row['id'] for row in service.list()] == ['local_my-run']
    detail = service.detail('local_my-run')
    assert detail['origin'] == 'local' and detail['figures'] == []
    assert detail['report']['models']['random_forest']['test']['macro_f1'] == 1
    payload = json.dumps(detail)
    assert all(secret not in payload for secret in ('private-name', 'secret', 'features_path'))
    with pytest.raises(ValueError, match='already registered'):
        register(tmp_path, 'my-run', True)


def test_fresh_checkout_still_has_builtin_eight():
    root = Path(__file__).resolve().parents[1]
    assert len([r for r in ExperimentService(root).list() if r['origin'] == 'builtin']) == 8


@pytest.mark.parametrize('name', ['../x', 'a/b', 'a\\b', 'C:x', '.', 'CON', 'a' * 81])
def test_names_rejected(tmp_path, name):
    with pytest.raises(ValueError): register(tmp_path, name)
    assert not (tmp_path / 'private').exists()


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(models={'cnn': {}}),
    lambda r: r['models']['majority']['test'].update(predictions=[1, 2]),
    lambda r: r['models']['majority']['test'].update(macro_f1=float('nan')),
    lambda r: r['models']['majority']['test'].update(macro_f1=float('inf')),
    lambda r: r['models']['majority']['test'].update(macro_f1=1.1),
    lambda r: r['models']['majority']['test'].update(confusion_matrix=[[1]]),
    lambda r: r['split_subjects'].update(test=['private-name']),
    lambda r: r.update(feature_count=True),
    lambda r: r.update(dataset='C:/secret/data'),
])
def test_bad_aggregate_rejected(tmp_path, mutation):
    report = metrics(); mutation(report); write(tmp_path, report)
    with pytest.raises((ValueError, KeyError, TypeError)):
        register(tmp_path, 'my-run', True)
    assert not (tmp_path / 'private').exists()


def test_strict_json_and_size(tmp_path):
    path = write(tmp_path)
    for raw in ('{"dataset":"x","dataset":"y"}', '{"number":1e999}', 'x' * (MAX_BYTES + 1)):
        path.write_text(raw)
        with pytest.raises(ValueError): register(tmp_path, 'my-run')


@pytest.mark.parametrize('component', ['runs', 'private'])
def test_linked_storage_rejected(tmp_path, component):
    write(tmp_path)
    outside = tmp_path / 'outside'; outside.mkdir()
    target = tmp_path / component
    if component == 'runs':
        target.rename(outside / 'runs')
        destination = outside / 'runs'
    else:
        destination = outside
    try:
        target.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip('Windows symbolic link permission unavailable')
    with pytest.raises(ValueError, match='linked'):
        register(tmp_path, 'my-run', True)


def test_snapshot_tampering_cannot_publish_arbitrary_fields(tmp_path):
    write(tmp_path); register(tmp_path, 'my-run', True)
    path = tmp_path / 'private/local-experiments/my-run.json'
    data = json.loads(path.read_text()); data['report']['secret'] = 'private'
    path.write_text(json.dumps(data))
    detail = ExperimentService(tmp_path).detail('local_my-run')
    assert detail['available'] is False
    assert 'private' not in json.dumps(detail['report'])


def test_bad_local_entries_preserve_all_builtins_and_valid_local(tmp_path):
    import shutil
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'docs/results', tmp_path / 'docs/results')
    write(tmp_path); register(tmp_path, 'my-run', True)
    folder = tmp_path / 'private/local-experiments'
    (folder / 'broken.json').write_text('{C:/private/patient}')
    (folder / 'unsafe sensitive name.json').write_text('{}')
    (folder / 'obsolete.json').write_text('{"schema":-1}')
    service = ExperimentService(tmp_path)
    rows = service.list()
    assert len([r for r in rows if r['origin'] == 'builtin']) == 8
    assert any(r['id'] == 'local_my-run' and r['available'] for r in rows)
    failures = [r for r in rows if r.get('available') is False]
    assert len(failures) == 3
    assert all('sensitive' not in json.dumps(r) and 'patient' not in json.dumps(r) for r in failures)
    assert all(service.detail(r['id'])['available'] is False for r in failures)
    assert all(service.detail(r['id'])['id'] == r['id'] for r in failures)
    assert service.detail('audio_baseline')['report']['models']


def test_unreadable_local_directory_preserves_builtin_reports(tmp_path, monkeypatch):
    import shutil
    from nightlab.services import local_experiments
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / 'docs/results', tmp_path / 'docs/results')
    def reject(*args):
        raise ValueError('C:/secret')
    monkeypatch.setattr(local_experiments, 'owned', reject)
    rows = ExperimentService(tmp_path).list()
    assert len([r for r in rows if r['origin'] == 'builtin']) == 8
    assert len(rows) == 9 and rows[-1]['available'] is False
    assert 'C:/secret' not in json.dumps(rows[-1])


def test_missing_classes_and_ap_explanation_normalized(tmp_path):
    report = metrics()
    labels = ['no_target_reference_event', 'reference_respiratory_event']
    value = classification_metrics([labels[0], labels[0]], [labels[0], labels[0]], labels, [0.1, 0.1], labels[1])
    value['macro_f1_scope'] = 'C:/secret'
    value['pr_auc_undefined_reason'] = 'private patient details'
    report['models'] = {name: {split: copy.deepcopy(value) for split in ('validation', 'test')} for name in ('majority', 'random_forest')}
    write(tmp_path, report); register(tmp_path, 'my-run', True)
    detail = ExperimentService(tmp_path).detail('local_my-run')
    clean = detail['report']['models']['random_forest']['test']
    assert clean['absent_reference_classes'] == [labels[1]]
    assert clean['macro_f1_scope'] == 'fixed declared class list; absent classes contribute zero'
    assert clean['balanced_accuracy_scope'] == 'mean recall over classes present in this reference split'
    assert clean['pr_auc_average_precision'] is None
    assert clean['pr_auc_undefined_reason'] == 'reference split must contain both positive and negative classes'
    assert 'secret' not in json.dumps(detail) and 'patient' not in json.dumps(detail)
