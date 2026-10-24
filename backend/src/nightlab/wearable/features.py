"""Deterministic SleepAccel epoch features; times remain relative to PSG in seconds.

Epochs are [start, start + 30). Statistics use the available samples without
resampling. Standard deviations are population (ddof=0). Movement is the median
absolute deviation of acceleration magnitude, in g (no normal-consistency scale).
Coverage gaps describe sample spacing, including interval edges; they are not an
estimate of missing duration or a statement that a signal is usable for training.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STAGES = {-1: None, 0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}
FEATURE_NAMES = (
    "hr_sample_count", "hr_mean_bpm", "hr_std_bpm", "hr_min_bpm", "hr_max_bpm",
    "hr_linear_slope_bpm_per_second", "accel_sample_count", "accel_x_std_g",
    "accel_y_std_g", "accel_z_std_g", "accel_norm_mean_g", "accel_norm_std_g",
    "accel_norm_median_absolute_deviation_g",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Optional count flags only: thresholds never remove epochs or fill values."""

    min_hr_samples: int = 0
    min_accel_samples: int = 0
    repeated_block_policy: str = 'reject'

    def __post_init__(self):
        for value in (self.min_hr_samples, self.min_accel_samples):
            if type(value) is not int or value < 0:
                raise ValueError("minimum sample counts must be nonnegative integers")
        if self.repeated_block_policy not in {'reject','collapse_identical'}:
            raise ValueError('unknown repeated-block policy')


@dataclass
class WearableFeatures:
    subject_id: np.ndarray
    epoch_start_seconds: np.ndarray
    epoch_end_seconds: np.ndarray
    raw_stage: np.ndarray
    stage: np.ndarray
    target_valid: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    hr_sample_count: np.ndarray
    accel_sample_count: np.ndarray
    coverage: dict
    label_table: dict
    metadata: dict


def _validate_rows(rows, kind, repeated_block_policy='reject'):
    if kind not in {"heart_rate", "motion", "labels"}:
        raise ValueError(f"unsupported SleepAccel kind: {kind}")
    rows = np.asarray(rows, dtype=np.float64)
    width = 4 if kind == "motion" else 2
    if rows.ndim != 2 or rows.shape[1] != width or len(rows) == 0:
        raise ValueError(f"{kind}: expected nonempty {width}-column rows")
    if not np.isfinite(rows).all():
        raise ValueError(f"{kind}: nonfinite value")
    differences = np.diff(rows[:,0])
    if np.any(differences <= 0):
        if repeated_block_policy!='collapse_identical' or kind=='labels' or np.any(differences==0):
            raise ValueError(f"{kind}: timestamps must strictly increase")
        boundaries = np.r_[0,np.flatnonzero(differences<0)+1,len(rows)]
        lengths = np.diff(boundaries)
        if not np.all(lengths==lengths[0]) or not all(np.array_equal(rows[:lengths[0]],rows[left:right])
                    for left,right in zip(boundaries[1:-1],boundaries[2:])):
            raise ValueError(f'{kind}: non-identical time-reversal blocks; refusing repair')
        rows = rows[:lengths[0]]
    if kind == "labels":
        if np.any(rows[:, 1] != np.floor(rows[:, 1])):
            raise ValueError("labels: noninteger stage")
        if not np.isin(rows[:, 1], tuple(STAGES)).all():
            raise ValueError("labels: undocumented SleepAccel stage")
        if np.any(np.diff(rows[:, 0]) != 30):
            raise ValueError("labels: epochs must be contiguous and exactly 30 seconds")
    return rows


def _parse_fields(text, kind):
    if kind not in {"heart_rate", "motion", "labels"}:
        raise ValueError(f"unsupported SleepAccel kind: {kind}")
    width = 4 if kind == "motion" else 2
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split(",") if kind == "heart_rate" else line.split()
        if len(fields) != width:
            raise ValueError(f"{kind} line {number}: expected {width} columns")
        try:
            rows.append([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError(f"{kind} line {number}: invalid number") from exc
    return np.asarray(rows,dtype=np.float64)


def parse_sleepaccel_rows(text: str, kind: str) -> np.ndarray:
    """Strict public parser; repeated blocks need explicit preparation policy."""
    return _validate_rows(_parse_fields(text,kind),kind)


def read_sleepaccel_rows(path: str | Path, kind: str) -> np.ndarray:
    return parse_sleepaccel_rows(Path(path).read_text(encoding="utf-8"), kind)


def _coverage(times, starts, ends):
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, ends, side="left")
    count = right - left
    leading = np.full(len(starts), np.nan)
    trailing = np.full(len(starts), np.nan)
    internal = np.full(len(starts), np.nan)
    maximum = np.full(len(starts), 30.0)
    for i, (lo, hi) in enumerate(zip(left, right)):
        if hi > lo:
            chunk = times[lo:hi]
            leading[i] = chunk[0] - starts[i]
            trailing[i] = ends[i] - chunk[-1]
            if len(chunk) > 1:
                internal[i] = np.max(np.diff(chunk))
            maximum[i] = max(leading[i], trailing[i], internal[i] if hi-lo > 1 else 0)
    return left, right, {
        "sample_count": count,
        "leading_gap_seconds": leading,
        "trailing_gap_seconds": trailing,
        "maximum_internal_gap_seconds": internal,
        "maximum_gap_including_edges_seconds": maximum,
        "no_samples": count == 0,
    }


def extract_sleepaccel_features(subject_id, heart_rate, motion, labels, config=None):
    """Build scored epochs, retaining the complete label timeline separately.

    Input arrays are timestamp/value rows (HR 2, motion 4, labels 2 columns).
    A single sample has std=0 but cannot define a slope (NaN). Missing signals
    have count=0 and all their value features NaN. Unscored epochs are excluded
    from feature rows and preserved with target_valid=False in label_table.
    """
    config = config if config is not None else FeatureConfig()
    if not isinstance(config, FeatureConfig):
        raise TypeError("config must be FeatureConfig")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a nonempty string")
    hr = _validate_rows(heart_rate, "heart_rate",config.repeated_block_policy)
    accel = _validate_rows(motion, "motion",config.repeated_block_policy)
    labels = _validate_rows(labels, "labels")
    raw = labels[:, 1].astype(np.int64)
    valid = raw != -1
    all_stages = np.array([STAGES[int(code)] for code in raw], dtype=object)
    starts = labels[valid, 0].copy()
    ends = starts + 30
    values = np.full((len(starts), len(FEATURE_NAMES)), np.nan)
    hr_lo, hr_hi, hr_coverage = _coverage(hr[:, 0], starts, ends)
    acc_lo, acc_hi, acc_coverage = _coverage(accel[:, 0], starts, ends)
    values[:, 0] = hr_coverage["sample_count"]
    values[:, 6] = acc_coverage["sample_count"]
    for i, (lo, hi) in enumerate(zip(hr_lo, hr_hi)):
        if hi > lo:
            times, bpm = hr[lo:hi, 0], hr[lo:hi, 1]
            values[i, 1:5] = (bpm.mean(), bpm.std(), bpm.min(), bpm.max())
            if hi-lo > 1:
                centered = times - times.mean()
                values[i, 5] = np.dot(centered, bpm-bpm.mean()) / np.dot(centered, centered)
    for i, (lo, hi) in enumerate(zip(acc_lo, acc_hi)):
        if hi > lo:
            xyz = accel[lo:hi, 1:4]
            magnitude = np.linalg.norm(xyz, axis=1)
            values[i, 7:10] = xyz.std(axis=0)
            values[i, 10:13] = (magnitude.mean(), magnitude.std(),
                                  np.median(np.abs(magnitude - np.median(magnitude))))
    hr_coverage["below_configured_min_samples"] = hr_coverage["sample_count"] < config.min_hr_samples
    acc_coverage["below_configured_min_samples"] = acc_coverage["sample_count"] < config.min_accel_samples
    range_start, range_end = float(labels[0, 0]), float(labels[-1, 0]+30)
    signal_metadata = {}
    for name, rows, original in (("heart_rate", hr, heart_rate), ("motion", accel, motion)):
        times = rows[:, 0]
        signal_metadata[name] = {
            "input_samples": len(rows),
            'original_input_samples':len(original),
            'identical_repeated_blocks':len(original)//len(rows),
            'discarded_exact_repeat_samples':len(original)-len(rows),
            "samples_before_label_range": int(np.searchsorted(times, range_start)),
            "samples_after_label_range": int(len(times)-np.searchsorted(times, range_end)),
        }
    return WearableFeatures(
        subject_id=np.full(len(starts), subject_id), epoch_start_seconds=starts,
        epoch_end_seconds=ends, raw_stage=raw[valid], stage=all_stages[valid],
        target_valid=np.ones(len(starts), dtype=bool), features=values,
        feature_names=FEATURE_NAMES, hr_sample_count=hr_coverage["sample_count"],
        accel_sample_count=acc_coverage["sample_count"],
        coverage={"heart_rate": hr_coverage, "motion": acc_coverage},
        label_table={"epoch_start_seconds": labels[:, 0].copy(),
                     "epoch_end_seconds": labels[:, 0]+30, "raw_stage": raw,
                     "stage": all_stages, "target_valid": valid},
        metadata={"epoch_seconds": 30, "mapping_version": "sleepaccel-five-stage-v1",
                  "label_epochs": len(labels), "scored_epochs": int(valid.sum()),
                  "unscored_epochs": int((~valid).sum()),
                  "label_interval_seconds": [range_start, range_end],
                  "minimum_sample_thresholds_report_only": True,
                  "min_hr_samples": config.min_hr_samples,
                  "min_accel_samples": config.min_accel_samples,
                  'repeated_block_policy':config.repeated_block_policy,
                  "signals": signal_metadata,
                  "standard_deviation_ddof": 0,
                  "movement_definition": "median(abs(norm_g - median(norm_g)))",
                  "coverage_definition": "sample spacing inside each epoch plus edges; empty epoch maximum gap is 30 seconds; unavailable individual gaps are NaN"},
    )


def prepare_sleepaccel_features(subject_id, heart_rate_path, motion_path, labels_path, config=None):
    """Read local files only. Caller owns provenance/checksum verification."""
    return extract_sleepaccel_features(
        subject_id, _parse_fields(Path(heart_rate_path).read_text(encoding='utf-8'), "heart_rate"),
        _parse_fields(Path(motion_path).read_text(encoding='utf-8'), "motion"),
        _parse_fields(Path(labels_path).read_text(encoding='utf-8'), "labels"), config,
    )
