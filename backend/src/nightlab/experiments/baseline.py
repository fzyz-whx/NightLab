"""Train-only fitted classic baselines with disjoint, fixed subject sets."""
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 42
    trees: int = 200
    max_depth: int = 12
    min_samples_leaf: int = 3


def classification_metrics(truth, predicted, labels, positive_probability=None, positive_label=None):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or not len(truth) or truth.shape != predicted.shape:
        raise ValueError('nonempty equal-length metric arrays required')
    if not set(truth).issubset(labels) or not set(predicted).issubset(labels):
        raise ValueError('unknown metric class')
    precision, recall, scores, counts = precision_recall_fscore_support(truth,predicted,labels=labels,zero_division=0)
    result = {'macro_f1':float(f1_score(truth,predicted,labels=labels,average='macro',zero_division=0)),
              'balanced_accuracy':float(recall[counts>0].mean()),
              'accuracy':float(accuracy_score(truth,predicted)),
              'label_order':list(labels),'confusion_matrix':confusion_matrix(truth,predicted,labels=labels).tolist(),
              'per_class':{str(label):{'precision':float(p),'recall':float(r),'f1':float(f),'support':int(n)}
                           for label,p,r,f,n in zip(labels,precision,recall,scores,counts)},
              'absent_reference_classes':[str(label) for label,n in zip(labels,counts) if n==0],
              'balanced_accuracy_scope':'mean recall over classes present in this reference split',
              'macro_f1_scope':'fixed declared class list; absent classes contribute zero'}
    if positive_label is not None:
        binary = truth == positive_label
        probability = np.asarray(positive_probability,dtype=float)
        if probability.shape != truth.shape or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError('finite positive-class probabilities in [0,1] required')
        result['pr_auc_average_precision'] = float(average_precision_score(binary,probability)) if binary.any() and (~binary).any() else None
        if result['pr_auc_average_precision'] is None:
            result['pr_auc_undefined_reason'] = 'reference split must contain both positive and negative classes'
    return result


def fit_baselines(features, target, subjects, assignments, labels, config=None, positive_label=None):
    config = config or BaselineConfig()
    features = np.asarray(features,dtype=np.float64)
    target, subjects = np.asarray(target), np.asarray(subjects)
    if features.ndim != 2 or not len(features) or target.shape != (len(features),) or subjects.shape != target.shape:
        raise ValueError('inconsistent feature/target/subject shapes')
    if np.isinf(features).any() or not set(target).issubset(labels):
        raise ValueError('infinite features or unknown target classes')
    if set(assignments) != {'train','validation','test'}:
        raise ValueError('three named subject sets required')
    groups = {name:set(group) for name,group in assignments.items()}
    if any(not group for group in groups.values()):
        raise ValueError('three nonempty subject sets required')
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('subject leakage across splits')
    if set.union(*groups.values()) != set(subjects):
        raise ValueError('subject assignment must match all prepared subjects exactly')
    indices = {name:np.flatnonzero(np.isin(subjects,list(group))) for name,group in groups.items()}
    if any(not len(rows) for rows in indices.values()):
        raise ValueError('split has no prepared rows')
    train = indices['train']
    if set(target[train]) != set(labels):
        raise ValueError('training set does not contain every declared target class')
    models, results = {}, {}
    estimators = {'majority':DummyClassifier(strategy='most_frequent'),
                  'random_forest':RandomForestClassifier(n_estimators=config.trees,max_depth=config.max_depth,
                       min_samples_leaf=config.min_samples_leaf,class_weight='balanced_subsample',
                       random_state=config.seed,n_jobs=1)}
    for name, estimator in estimators.items():
        pipeline = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('classifier',estimator)])
        pipeline.fit(features[train],target[train])
        models[name] = pipeline
        results[name] = {}
        for partition in ('validation','test'):
            rows = indices[partition]
            predicted = pipeline.predict(features[rows])
            probabilities = None
            if positive_label is not None:
                slot = pipeline.classes_.tolist().index(positive_label)
                probabilities = pipeline.predict_proba(features[rows])[:,slot]
            results[name][partition] = classification_metrics(target[rows],predicted,labels,probabilities,positive_label)
    return models, {'models':results,'config':asdict(config),'hyperparameter_policy':'fixed before test evaluation; no search or test-based threshold selection',
                    'split_subjects':{name:sorted(group) for name,group in groups.items()},
                    'split_rows':{name:len(rows) for name,rows in indices.items()},
                    'imputer_fit_scope':'training rows only', 'evaluation_unit':'windows/epochs in held-out subjects; not population or clinical validation'}
