"""Split subject identities before windows; never randomly split correlated rows."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    subjects: dict
    row_indices: dict
    seed: int


def subject_split(subject_ids, seed=42, train_fraction=.6, validation_fraction=.2):
    values = np.asarray(subject_ids,dtype=object)
    if values.ndim != 1 or not len(values) or any(not isinstance(value,str) or not value.strip() for value in values):
        raise ValueError('subject identities must be a nonempty one-dimensional string array')
    if not np.isfinite([train_fraction,validation_fraction]).all() or min(train_fraction,validation_fraction) <= 0 or train_fraction+validation_fraction >= 1:
        raise ValueError('positive train/validation/test fractions required')
    unique = sorted(set(values))
    if len(unique) < 3:
        raise ValueError('at least three independent subjects required; single-subject windows are not an evaluation split')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    validation_count = max(1,int(len(unique)*validation_fraction))
    train_count = min(max(1,int(len(unique)*train_fraction)),len(unique)-validation_count-1)
    assignments = {'train':shuffled[:train_count],
                   'validation':shuffled[train_count:train_count+validation_count],
                   'test':shuffled[train_count+validation_count:]}
    if any(not subjects for subjects in assignments.values()):
        raise ValueError('unable to allocate three nonempty subject sets')
    indices = {name:np.flatnonzero(np.isin(values,subjects)) for name,subjects in assignments.items()}
    return SubjectSplit(assignments,indices,seed)
