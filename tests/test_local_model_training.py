"""Synthetic fixture integration only: these arrays are not PSG research results."""
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest

from nightlab.audio.features import AudioFeatureExtractor
from nightlab.services.model_registry import LABELS, ModelRegistry, controlled_path, digest
from scripts.datasets.prepare import feature_source_hashes
from scripts.train import app_audio_rf


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_corpus(tmp_path):
    """Mirror the fixed cohort schema with small, explicitly synthetic matrices."""
    manifest_path = tmp_path/'data/manifests/psg-audio-pilot-10.json'
    manifest_path.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT/'data/manifests/psg-audio-pilot-10.json',manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = list(AudioFeatureExtractor().feature_names)
    rng = np.random.default_rng(123)
    records = []
    for record_id in sorted({item['record_id'] for item in manifest['files']}):
        relative = f'data/processed/synthetic-fixture/{record_id}'
        folder = tmp_path/relative
        folder.mkdir(parents=True)
        target = np.array(LABELS*4)
        features = rng.normal(size=(len(target),len(names)))
        features[target==LABELS[1],0] += 5
        np.savez(folder/'features.npz',X=features,y=target,
                 subject_id=np.repeat(record_id.split('-')[0],len(target)))
        np.savez(folder/'timeline.npz',start_seconds=np.arange(len(target)))
        entries = [item for item in manifest['files'] if item['record_id']==record_id]
        metadata = {'fixture':'synthetic; no raw PSG input or clinical result',
                    'dataset':'psg-audio','dataset_version':manifest['version'],
                    'feature_version':'psg-rms-logmel-v1','feature_names':names,'rows':len(target),
                    'feature_source_sha256':feature_source_hashes('psg-audio'),
                    'input_integrity':{'ok':True,'files':[
                        {'path':item['path'],'local_checksums':item['official_checksums']} for item in entries]},
                    'artifact_sha256':{name:digest(folder/name) for name in ('features.npz','timeline.npz')}}
        (folder/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
        records.append({'record_id':record_id,'path':relative,'ok':True})
    corpus = {'complete':True,'dataset':manifest['id'],'version':manifest['version'],
              'split_policy':manifest['subjects'],'records':records}
    path = tmp_path/app_audio_rf.CORPUS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(corpus),encoding='utf-8')
    return tmp_path


@pytest.fixture
def trained(synthetic_corpus):
    result = app_audio_rf.train('synthetic-rf',execute=True,root=synthetic_corpus)
    return synthetic_corpus,result


def test_real_classic_training_receipt_and_desktop_prediction(trained,monkeypatch):
    root,result = trained
    assert result['status']=='trained'
    assert result['scope'].startswith('fresh local classic training')
    assert not (root/'private/app-models/synthetic-rf.lock').exists()
    assert not list((root/'private/app-models').glob('*.tmp'))
    receipt = json.loads((root/'private/app-models/synthetic-rf.json').read_text(encoding='utf-8'))
    assert receipt['model_sha256']==digest(root/'models/synthetic-rf/random_forest.joblib')
    assert receipt['metrics_sha256']==digest(root/'runs/synthetic-rf/metrics.json')
    registry = ModelRegistry(root)
    entries = registry.list_models()
    assert [item['id'] for item in entries[:2]]==['audio_rf_pilot_v1','audio_cnn_augmented_v1']
    assert next(item for item in entries if item['id']==result['model_id'])['available']
    original = joblib.load
    loads = []
    def observed_load(path):
        loads.append(path)
        return original(path)
    monkeypatch.setattr(joblib,'load',observed_load)
    rate = 16000
    waves = [.05*np.sin(2*np.pi*300*np.arange(rate)/rate)]
    a = registry.predict_windows(result['model_id'],waves,rate)
    b = registry.predict_windows(result['model_id'],waves,rate)
    assert a.shape==(1,) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    np.testing.assert_array_equal(a,b)
    assert len(loads)==1
    with pytest.raises(ValueError,match='exists'):
        app_audio_rf.train('synthetic-rf',execute=True,root=root)


@pytest.mark.parametrize('target', ['model','metrics','schema','features','labels','dependency','source','run','extra','malformed'])
def test_invalid_proof_never_deserializes(trained,monkeypatch,target):
    root,result = trained
    receipt_path = root/'private/app-models/synthetic-rf.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if target=='model':
        (root/'models/synthetic-rf/random_forest.joblib').write_bytes(b'untrusted pickle')
    elif target=='metrics':
        (root/'runs/synthetic-rf/metrics.json').write_text('{}')
    elif target=='malformed':
        receipt_path.write_text('[]')
    else:
        if target=='schema': receipt['schema_version']=True
        if target=='features': receipt['feature_names'].reverse()
        if target=='labels': receipt['labels'].reverse()
        if target=='dependency': receipt['environment']['scikit-learn']='unknown'
        if target=='source': receipt['source_sha256']={}
        if target=='run': receipt['run_name']='other'
        if target=='extra': receipt['pickle_path']='external.joblib'
        receipt_path.write_text(json.dumps(receipt),encoding='utf-8')
    def forbidden(*args,**kwargs):
        pytest.fail('joblib.load must not run before receipt/hash verification')
    monkeypatch.setattr(joblib,'load',forbidden)
    with pytest.raises((ValueError,TypeError)):
        ModelRegistry(root).predict_windows(result['model_id'],[np.zeros(16000)],16000)


@pytest.mark.parametrize('name',['../escape','a/b','a\\b','A','con','nul','a'*65,'','x:stream'])
def test_run_names_and_default_dry_run(tmp_path,name):
    with pytest.raises(ValueError):
        app_audio_rf.train(name,root=tmp_path)
    plan = app_audio_rf.train('safe-name',root=tmp_path)
    assert plan['mode']=='dry-run' and not plan['writes']
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('existing',['runs/new','models/new','private/app-models/new.json'])
def test_never_registers_existing_output(tmp_path,existing):
    path = tmp_path/existing
    path.parent.mkdir(parents=True)
    path.write_bytes(b'unknown saved artifact')
    with pytest.raises(ValueError,match='exists'):
        app_audio_rf.train('new',execute=True,root=tmp_path)


def test_wrong_task_and_corrupt_artifact_fail_before_training(synthetic_corpus,monkeypatch):
    root = synthetic_corpus
    path = root/app_audio_rf.CORPUS
    corpus = json.loads(path.read_text())
    corpus['dataset']='sleepaccel-pilot-31'
    path.write_text(json.dumps(corpus))
    monkeypatch.setattr(app_audio_rf.baselines,'run',lambda *a,**k:pytest.fail('must not train'))
    with pytest.raises(ValueError,match='PSG'):
        app_audio_rf.train('wrong',execute=True,root=root)
    corpus['dataset']='psg-audio-pilot-10'
    path.write_text(json.dumps(corpus))
    (root/corpus['records'][0]['path']/'features.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):
        app_audio_rf.train('corrupt',execute=True,root=root)
    assert not (root/'private').exists()


def test_paths_reject_escape_and_links(tmp_path):
    with pytest.raises(ValueError):
        controlled_path(tmp_path,'models','../outside')
    real = tmp_path/'real'
    real.mkdir()
    try:
        (tmp_path/'models').symlink_to(real,target_is_directory=True)
    except OSError:
        pytest.skip('Windows account does not permit symlink creation')
    with pytest.raises(ValueError,match='链接'):
        app_audio_rf.train('safe',root=tmp_path)


def test_cli_has_no_existing_model_import_option():
    with pytest.raises(SystemExit):
        app_audio_rf.main(['--run-name','safe','--pickle','untrusted.joblib'])
