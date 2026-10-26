"""SleepAccel31-only canonicalization, following fixed author remove_repeats.

Does not mutate inputs or labels, infer clock offsets, average or interpolate.
The label interval is used without inspecting stage values for signal selection.
"""
import numpy as np

from .features import _validate_rows

VERSION = 'sleepaccel-canonical-v3'
AUTHOR_SOURCE = 'https://github.com/ojwalch/sleep_classifiers/blob/ec9704f0290563016adbe48184365ab102daf9ea/source/utils.py#L112-L115'


def _signal(rows, kind, start, end):
    raw = np.asarray(rows, dtype=np.float64)
    width = 4 if kind == 'motion' else 2
    if raw.ndim != 2 or raw.shape[1] != width or not len(raw) or not np.isfinite(raw).all():
        raise ValueError(f'{kind}: expected finite nonempty {width}-column rows')
    differences = np.diff(raw[:, 0])
    stats = {'raw_samples': len(raw), 'raw_time_reversals': int((differences < 0).sum()),
             'raw_adjacent_equal_times': int((differences == 0).sum())}
    del differences
    # Author's exact full-row uniqueness, followed by original timestamp ordering.
    unique = np.unique(raw, axis=0)
    unique = unique[np.argsort(unique[:, 0], kind='stable')]
    differences = np.diff(unique[:, 0])
    equal = np.flatnonzero(differences == 0)
    conflict_times = np.unique(unique[equal, 0])
    conflict_inside = (conflict_times >= start) & (conflict_times < end)
    stats.update(unique_samples_before_crop=len(unique), exact_duplicate_rows_removed=len(raw)-len(unique),
                 conflicting_timestamp_groups_before_crop=len(conflict_times),
                 conflicting_timestamp_groups_inside_label_interval=int(conflict_inside.sum()),
                 conflicting_timestamp_groups_outside_label_interval=int((~conflict_inside).sum()),
                 unique_samples_before_label_interval=int((unique[:, 0] < start).sum()),
                 unique_samples_after_label_interval=int((unique[:, 0] >= end).sum()),
                 raw_samples_before_label_interval=int((raw[:, 0] < start).sum()),
                 raw_samples_after_label_interval=int((raw[:, 0] >= end).sum()))
    inside = (unique[:, 0] >= start) & (unique[:, 0] < end)
    canonical = unique[inside].copy()
    stats.update(canonical_samples_after_crop=len(canonical),
                 unique_samples_cropped=int((~inside).sum()),
                 sample_count_semantics='raw counts precede all cleaning; downstream feature input counts describe canonical cropped arrays')
    if conflict_inside.any():
        raise ValueError(f'{kind}: conflicting same-timestamp values inside label interval')
    _validate_rows(canonical, kind, 'reject')  # Also rejects empty cropped signal.
    stats['strictly_increasing_after_crop'] = True
    return canonical, stats


def canonicalize(hr, motion, labels):
    """Return independent canonical arrays and raw-cleaning statistics.

    Labels remain in original order; strict 30-second/encoding validation happens
    before any signal cleaning. All label epochs, including -1, bound the crop.
    """
    label_rows = _validate_rows(np.array(labels, dtype=np.float64, copy=True), 'labels', 'reject')
    start, end = float(label_rows[0, 0]), float(label_rows[-1, 0]+30)
    heart_rate, hr_stats = _signal(hr, 'heart_rate', start, end)
    accel, motion_stats = _signal(motion, 'motion', start, end)
    return {'heart_rate': heart_rate, 'motion': accel, 'labels': label_rows,
            'metadata': {'version': VERSION, 'author_source': AUTHOR_SOURCE,
                         'policy': 'exact full-row deduplication; timestamp ordering; full label interval crop; reject remaining repeated timestamps',
                         'label_epochs': len(label_rows), 'label_interval_duration_seconds': end-start,
                         'crop_includes_unscored_epochs': True,
                         'stage_values_used_for_signal_selection': False,
                         'signals': {'heart_rate': hr_stats, 'motion': motion_stats}}}
