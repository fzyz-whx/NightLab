import numpy as np
import pytest

from nightlab.experiments.splits import subject_split


def test_all_windows_of_each_subject_stay_together_and_cover_rows():
    ids = np.repeat([f's{index}' for index in range(10)], [1,2,3,4,5,6,7,8,9,10])
    split = subject_split(ids)
    assert {name:len(group) for name,group in split.subjects.items()} == {'train':6,'validation':2,'test':2}
    indices = list(split.row_indices.values())
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(ids)))
    for name,rows in split.row_indices.items():
        assert set(ids[rows]) == set(split.subjects[name])
        for other,other_rows in split.row_indices.items():
            if other != name:
                assert not set(ids[rows]) & set(ids[other_rows])
    assert subject_split(ids[::-1]).subjects == split.subjects


def test_single_subject_cannot_become_fake_train_test_by_window_count():
    with pytest.raises(ValueError,match='three independent subjects'):
        subject_split(['same']*1000)
    split = subject_split(['a','b','c'])
    assert all(len(group)==1 for group in split.subjects.values())


@pytest.mark.parametrize('values',[[],[['a','b']],['a',None],['a',' ', 'b']])
def test_invalid_identity_arrays_rejected(values):
    with pytest.raises(ValueError):
        subject_split(values)
