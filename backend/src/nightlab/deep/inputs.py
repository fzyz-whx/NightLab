"""Observed-sample tensor inputs; no imputation or fitted dataset statistics."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from nightlab.audio.features import AudioFeatureExtractor
from nightlab.wearable.features import STAGES, _validate_rows


def log_mel_image(signal, source_rate, time_bins=100):
    """Average actual frame log powers into contiguous, nonempty time groups."""
    if type(time_bins) is not int or time_bins <= 0:
        raise ValueError('time_bins must be a positive integer')
    if type(source_rate) is not int or source_rate <= 0:
        raise ValueError('source rate must be a positive integer')
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError('audio must be nonempty finite mono samples')
    extractor = AudioFeatureExtractor()
    c = extractor.config
    if source_rate != c.sample_rate:
        factor = gcd(source_rate, c.sample_rate)
        signal = resample_poly(signal, c.sample_rate // factor, source_rate // factor)
    if len(signal) < c.frame_samples:
        raise ValueError('audio shorter than one analysis frame')
    frames = np.lib.stride_tricks.sliding_window_view(signal, c.frame_samples)[::c.frame_hop_samples]
    if len(frames) < time_bins:
        raise ValueError('too few analysis frames for nonempty time groups')
    power = np.abs(np.fft.rfft(frames * extractor.window, n=c.fft_samples, axis=1)) ** 2
    power /= np.square(extractor.window).sum()
    log_power = np.log(np.maximum(power @ extractor.filters, c.power_floor))
    edges = np.linspace(0, len(frames), time_bins + 1, dtype=np.int64)
    return np.stack([log_power[left:right].mean(axis=0)
                     for left, right in zip(edges[:-1], edges[1:])], axis=1).astype(np.float32)


def wearable_epoch_tensors(subject_id, hr_rows, motion_rows, label_rows,
                           repeat_policy='collapse_identical'):
    """Build 1 Hz HR/XYZ/norm-std tensors from scored 30 second epochs.

    Masks indicate actual observations, never interpolation. Missing value bins
    stay NaN until a caller applies a fitted training-set standardizer.
    """
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError('subject_id must be a nonempty string')
    if repeat_policy not in {'reject', 'collapse_identical'}:
        raise ValueError('unknown repeated-block policy')
    hr = _validate_rows(hr_rows, 'heart_rate', repeat_policy)
    accel = _validate_rows(motion_rows, 'motion', repeat_policy)
    labels = _validate_rows(label_rows, 'labels', 'reject')
    raw = labels[:, 1].astype(np.int64)
    valid = raw != -1
    starts = labels[valid, 0].copy()
    values = np.full((len(starts), 5, 30), np.nan, dtype=np.float32)
    counts = np.zeros((len(starts), 2, 30), dtype=np.int64)
    for i, start in enumerate(starts):
        for channel, rows in enumerate((hr, accel)):
            edges = np.searchsorted(rows[:, 0], start + np.arange(31), side='left')
            for second, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                counts[i, channel, second] = right - left
                if right == left:
                    continue
                chunk = rows[left:right, 1:]
                if channel == 0:
                    values[i, 0, second] = chunk[:, 0].mean()
                else:
                    values[i, 1:4, second] = chunk.mean(axis=0)
                    values[i, 4, second] = np.linalg.norm(chunk, axis=1).std(ddof=0)
    range_start, range_end = float(labels[0, 0]), float(labels[-1, 0] + 30)
    signals = {}
    for name, rows, original, index in (('heart_rate', hr, hr_rows, 0),
                                        ('motion', accel, motion_rows, 1)):
        signals[name] = {
            'original_input_samples': len(original), 'input_samples': len(rows),
            'identical_repeated_blocks': len(original) // len(rows),
            'discarded_exact_repeat_samples': len(original) - len(rows),
            'samples_before_label_range': int(np.searchsorted(rows[:, 0], range_start)),
            'samples_after_label_range': int(len(rows) - np.searchsorted(rows[:, 0], range_end)),
            'samples_in_scored_epochs': int(counts[:, index].sum()),
            'samples_in_unscored_epochs': int(sum(
                np.searchsorted(rows[:, 0], start + 30) - np.searchsorted(rows[:, 0], start)
                for start in labels[~valid, 0])),
        }
    return {
        'values': values, 'masks': (counts > 0).astype(np.float32),
        'y': np.asarray([STAGES[int(code)] for code in raw[valid]], dtype=str),
        'subject_id': np.full(len(starts), subject_id, dtype=f'<U{len(subject_id)}'),
        'epoch_start_seconds': starts, 'raw_stage': raw[valid],
        'hr_sample_count': counts[:, 0], 'accel_sample_count': counts[:, 1],
        'metadata': {'epoch_seconds': 30, 'bin_seconds': 1,
                     'channel_names': ['hr_mean_bpm', 'accel_x_mean_g', 'accel_y_mean_g',
                                       'accel_z_mean_g', 'accel_norm_std_g'],
                     'mask_names': ['heart_rate_observed', 'accel_observed'],
                     'label_epochs': len(labels), 'scored_epochs': int(valid.sum()),
                     'unscored_epochs': int((~valid).sum()),
                     'repeated_block_policy': repeat_policy, 'signals': signals,
                     'mapping_version': 'sleepaccel-five-stage-v1',
                     'standard_deviation_ddof': 0, 'missing_values': 'NaN; no interpolation'},
    }


class MaskedChannelStandardizer:
    """Fit per-channel population moments on observed training values only."""

    @staticmethod
    def _array(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] == 0:
            raise ValueError('expected N x C x T values with at least one channel')
        if np.isinf(values).any():
            raise ValueError('infinite values are invalid')
        return values

    def fit(self, values):
        values = self._array(values)
        means, scales = [], []
        for channel in range(values.shape[1]):
            observed = values[:, channel, :]
            observed = observed[np.isfinite(observed)]
            if not observed.size:
                raise ValueError(f'channel {channel} has no observed training values')
            means.append(observed.mean())
            scale = observed.std(ddof=0)
            scales.append(scale if scale > 0 else 1.0)
        self.mean_ = np.asarray(means)
        self.scale_ = np.asarray(scales)
        return self

    def transform(self, values):
        values = self._array(values)
        if not hasattr(self, 'mean_') or not hasattr(self, 'scale_'):
            raise ValueError('standardizer must be fitted before transform')
        if values.shape[1] != len(self.mean_):
            raise ValueError('channel count differs from fitted training values')
        standardized = (values - self.mean_[None, :, None]) / self.scale_[None, :, None]
        return np.where(np.isnan(values), 0.0, standardized).astype(np.float32)

    def fit_transform(self, values):
        return self.fit(values).transform(values)
