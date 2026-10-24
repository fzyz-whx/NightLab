import joblib
import numpy as np
import pytest

from nightlab.experiments.baseline import BaselineConfig, classification_metrics, fit_baselines


def arrays():
    return np.array([[0.],[2.],[0.],[np.nan],[100.],[200.]]), np.array(['a','b','a','b','a','b']), np.array(['train','train','train','train','val','test'])


def test_manual_metric_example():
    result = classification_metrics(['a','a','b','b'],['a','b','b','b'],['a','b'],[.1,.8,.8,.9],'b')
    assert result['confusion_matrix'] == [[1,1],[0,2]]
    assert result['accuracy'] == result['balanced_accuracy'] == .75
    assert abs(result['macro_f1']-(2/3+.8)/2) < 1e-12
    assert abs(result['pr_auc_average_precision']-5/6) < 1e-12


def test_fit_is_train_only_and_saved_pipeline_round_trips(tmp_path):
    x,y,subjects = arrays()
    assignments = {'train':['train'],'validation':['val'],'test':['test']}
    models,result = fit_baselines(x,y,subjects,assignments,['a','b'],BaselineConfig(trees=5,min_samples_leaf=1))
    assert models['random_forest'].named_steps['impute'].statistics_[0] == 0
    assert result['split_rows'] == {'train':4,'validation':1,'test':1}
    path = tmp_path/'owned-model.joblib'
    joblib.dump(models['random_forest'],path)
    loaded = joblib.load(path)  # only the model created by this test, never an external artifact
    np.testing.assert_array_equal(loaded.predict(x),models['random_forest'].predict(x))


def test_leaky_or_incomplete_assignments_are_rejected():
    x,y,subjects = arrays()
    for assignments in [{'train':['train'],'validation':['train','val'],'test':['test']},
                        {'train':['train'],'validation':['val'],'test':['absent']},
                        {'train':['train'],'validation':[],'test':['val','test']}]:
        with pytest.raises(ValueError):
            fit_baselines(x,y,subjects,assignments,['a','b'])
