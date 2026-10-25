import numpy as np
import pytest

torch = pytest.importorskip('torch')
from nightlab.deep.training import (TrainingConfig,fit_pilot,prepare_tensors,select_device)


def test_modality_masks_and_train_only_moments():
    values = np.ones((3,5,30),dtype=np.float32)
    values[1] = 3
    values[2] = 101
    values[2,0,0] = np.nan
    masks = np.ones((3,2,30),dtype=np.float32)
    masks[2,0,0] = 0
    for subset,channels in (('heart_rate',2),('acceleration',5),('all',7)):
        tensor,normalizer = prepare_tensors('wearable',values,masks,np.array([0,1]),subset)
        assert tensor.shape == (3,channels,30)
        assert set(normalizer['mean']) == {2} and set(normalizer['scale']) == {1}
        assert tensor[2,0,1] == 99
        if subset != 'acceleration':
            assert tensor[2,0,0] == 0
            assert tensor[2,-(2 if subset == 'all' else 1),0] == 0
    assert np.isnan(values[2,0,0])  # preprocessing does not overwrite input gaps


def test_smoke_seed_selection_and_ambiguous_train_rows_excluded():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(9,32,100)).astype(np.float32)
    values[4] = 1000  # train identity but ambiguous; must not fit moments or loss
    bg,pos = 'no_target_reference_event','reference_respiratory_event'
    y = np.array([bg,pos,bg,pos,'',bg,pos,bg,pos])
    subjects = np.array(['a']*5+['b']*2+['c']*2)
    mask = y != ''
    config = TrainingConfig(epochs=2,batch_size=2,patience=1)
    args = ('audio',values,None,y,subjects,mask,{'train':['a'],'validation':['b'],'test':['c']})
    first,report,outputs,_ = fit_pilot(*args,smoke=True,config=config)
    second,repeated,repeated_outputs,_ = fit_pilot(*args,smoke=True,config=config)
    assert report['normalizer']['fit_rows'] == 4
    np.testing.assert_allclose(report['normalizer']['mean'],values[:4].mean(axis=(0,2)),atol=1e-6)
    assert report['class_weights_training_only'] == [1,1]
    assert report['history'][0]['training_rows_seen'] == 4
    assert set(report['models']['cnn']) == set(outputs) == {'validation'}
    assert report['smoke_only'] and report['device'] == 'cpu'
    assert report['best_epoch'] == 1
    for key in first:
        assert torch.equal(first[key],second[key])
    np.testing.assert_array_equal(outputs['validation']['probabilities'],repeated_outputs['validation']['probabilities'])
    assert report['models'] == repeated['models']


def test_cpu_fallback_is_explicit_and_config_thread_limit(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert select_device('auto').type == select_device('cpu').type == 'cpu'
    with pytest.raises(ValueError,match='unavailable'):
        select_device('cuda')
    for kwargs in ({'cpu_threads':3},{'epochs':0},{'batch_size':0},{'learning_rate':np.nan}):
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)
