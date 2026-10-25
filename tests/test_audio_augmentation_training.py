import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.train.audio_augmentation import combine_training_arrays


def fixture():
    base = {'values':np.arange(6*32*100,dtype=np.float32).reshape(6,32,100),
            'y':np.array(['a','b','','a','b','a']),'subject_id':np.array(['train']*3+['val']+['test']*2),
            'start_seconds':np.arange(6)*5.,'end_seconds':np.arange(6)*5.+10,
            'supervised_mask':np.array([True,True,False,True,True,True])}
    origins = np.array([0,1],dtype=np.int64)
    extra = {key:base[key][origins].copy() for key in ('y','subject_id','start_seconds','end_seconds')}
    extra['values'] = np.full((2,32,100),99,dtype=np.float32)
    groups = {'train':['train'],'validation':['val'],'test':['test']}
    return base,extra,origins,groups


def test_duplicate_control_and_augmented_share_rows_labels_and_order():
    base,extra,origins,groups = fixture()
    clean = combine_training_arrays(base,extra,origins,groups,'clean_duplicate')
    noisy = combine_training_arrays(base,extra,origins,groups,'noise_augmented')
    for key in ('y','subject_id','start_seconds','end_seconds','supervised_mask'):
        np.testing.assert_array_equal(clean[key],noisy[key])
    np.testing.assert_array_equal(clean['values'][-2:],base['values'][origins])
    np.testing.assert_array_equal(noisy['values'][-2:],extra['values'])
    np.testing.assert_array_equal(noisy['values'][:6],base['values'])
    assert not noisy['supervised_mask'][2] and noisy['supervised_mask'][-2:].all()


@pytest.mark.parametrize('origins',[[0,0],[0,2],[0,3],[0,4],[0],[-1,0],[0,99]])
def test_no_unscored_validation_test_duplicates_or_missing_origins(origins):
    base,extra,_,groups = fixture()
    with pytest.raises(ValueError,match='training row'):
        combine_training_arrays(base,extra,np.array(origins,dtype=np.int64),groups,'noise_augmented')


def test_copied_label_or_identity_must_match_original():
    base,extra,origins,groups = fixture()
    extra['y'][0] = 'z'
    with pytest.raises(ValueError,match='differs'):
        combine_training_arrays(base,extra,origins,groups,'noise_augmented')
