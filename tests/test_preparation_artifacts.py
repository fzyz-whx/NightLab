import hashlib
import json

import numpy as np
import pytest

from scripts.datasets.prepare import output_path, prepare


def fixture_manifest(root):
    inputs = [('heart_rate','5383425_heartrate.txt','-1,200\n0,70\n5,80\n60,60\n65,70\n'),
              ('motion','5383425_acceleration.txt','-1 100 100 100\n0 0 0 1\n5 0 0 3\n60 1 0 0\n65 3 0 0\n'),
              ('labels','5383425_labeled_sleep.txt','0 0\n30 -1\n60 5\n')]
    manifest = {'id':'sleepaccel','version':'fixture','license':{'verified':True},
                'inventory_complete':False,'files':[]}
    for kind, name, text in inputs:
        path = root/'data/raw/sleepaccel'/kind/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8',newline='')
        body = path.read_bytes()
        manifest['files'].append({'path':path.relative_to(root).as_posix(),'exact_bytes':len(body),
                                  'official_checksums':{'sha256':hashlib.sha256(body).hexdigest()}})
    return manifest


def test_preparation_default_does_not_create_output(tmp_path):
    manifest = fixture_manifest(tmp_path)
    result = prepare(tmp_path,manifest)
    assert result['writes'] is False
    assert not (tmp_path/'data/processed').exists()


def test_artifacts_preserve_masks_and_load_without_pickle(tmp_path):
    manifest = fixture_manifest(tmp_path)
    result = prepare(tmp_path,manifest,execute=True)
    assert result['rows'] == 2
    assert result['target_counts'] == {'W':1,'REM':1}
    folder = tmp_path/result['output']
    with np.load(folder/'features.npz',allow_pickle=False) as artifact:
        assert artifact['X'].shape == (2,13)
        np.testing.assert_array_equal(artifact['start_seconds'], [0,60])
        assert artifact['X'][0,1] == 75  # outside-PSG 200 bpm excluded
        for value in artifact.values():
            assert not value.dtype.hasobject
    with np.load(folder/'timeline.npz',allow_pickle=False) as timeline:
        assert timeline['stage'].tolist() == ['W','','REM']
        assert timeline['target_valid'].tolist() == [True,False,True]
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert metadata['formal_evaluation_eligible'] is False
    with pytest.raises(ValueError,match='refusing overwrite'):
        prepare(tmp_path,manifest,execute=True)


def test_corrupt_or_unlicensed_inputs_do_not_create_output(tmp_path):
    manifest = fixture_manifest(tmp_path)
    manifest['license']['verified'] = False
    with pytest.raises(ValueError,match='license'):
        prepare(tmp_path,manifest,execute=True)
    manifest['license']['verified'] = True
    (tmp_path/manifest['files'][0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):
        prepare(tmp_path,manifest,execute=True)
    assert not (tmp_path/'data/processed').exists()


@pytest.mark.parametrize('value',['../raw','data/processed/../raw','C:/data/processed/out','data/raw/out','/data/processed/out'])
def test_preparation_paths_reject_escape(tmp_path,value):
    with pytest.raises(ValueError):
        output_path(tmp_path,value)
