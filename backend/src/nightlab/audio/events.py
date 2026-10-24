"""Fixed event decoding and interval evaluation for observed research recordings.

Protocol: 5-second bins (including a shorter final bin), mean probabilities of
windows covering each bin center, threshold >= 0.5, and merge touching positive
bins. Matching uses one-to-one IoU >= 0.5, maximizing cardinality before IoU.
These are research measurements, not clinical diagnoses.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


BIN_SECONDS = 5.0
PROBABILITY_THRESHOLD = 0.5
MATCH_IOU_THRESHOLD = 0.5


def _duration(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('observed duration must be finite and positive')
    return value


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return result


def decode_events(window_starts, window_ends, window_probabilities, duration):
    """Return JSON-ready bin arrays and merged ``events`` as [start, end] pairs.

    Windows are half-open [start, end). Every bin center must have at least one
    prediction; omitted/ambiguous training windows must not become negatives.
    Inputs must describe windows within the selected recording in seconds.
    """
    duration = _duration(duration)
    starts = _vector(window_starts, 'window_starts')
    ends = _vector(window_ends, 'window_ends')
    probabilities = _vector(window_probabilities, 'window_probabilities')
    if not (len(starts) == len(ends) == len(probabilities)):
        raise ValueError('window arrays must have equal lengths')
    if (np.any(starts < 0) or np.any(ends > duration)
            or np.any(ends <= starts) or np.any(np.diff(starts) <= 0)):
        raise ValueError('windows must be valid, in recording bounds, with strictly increasing starts')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('window probabilities must be in [0, 1]')
    bin_starts = np.arange(0.0, duration, BIN_SECONDS)
    bin_ends = np.minimum(bin_starts + BIN_SECONDS, duration)
    bin_probabilities = []
    events = []
    for left, right in zip(bin_starts, bin_ends, strict=True):
        center = (left + right) / 2
        covered = (starts <= center) & (center < ends)
        if not covered.any():
            raise ValueError(f'uncovered bin [{left}, {right})')
        probability = float(probabilities[covered].mean())
        bin_probabilities.append(probability)
        if probability >= PROBABILITY_THRESHOLD:
            if events and events[-1][1] == left:
                events[-1][1] = float(right)
            else:
                events.append([float(left), float(right)])
    return {
        'bin_starts': bin_starts.tolist(),
        'bin_ends': bin_ends.tolist(),
        'bin_probabilities': bin_probabilities,
        'events': events,
    }


def _intervals(values, duration, name):
    result = np.asarray(values, dtype=float)
    if result.shape == (0,):
        result = result.reshape(0, 2)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite [start, end] pairs')
    if (np.any(result[:, 0] < 0) or np.any(result[:, 1] > duration)
            or np.any(result[:, 1] <= result[:, 0])):
        raise ValueError(f'{name} intervals must have positive lengths and lie in recording bounds')
    return result


def evaluate_events(predicted_intervals, reference_intervals, observed_seconds, censored_flags=None):
    """Evaluate clipped observed intervals with fixed IoU matching.

    Reference ``censored_flags`` indicate events clipped at recording boundaries.
    Such references count for detection, but their matches are excluded from
    onset/end localization errors. Signed errors are prediction minus reference.
    Undefined rates and localization means are returned as None.
    """
    observed_seconds = _duration(observed_seconds)
    predictions = _intervals(predicted_intervals, observed_seconds, 'predicted')
    references = _intervals(reference_intervals, observed_seconds, 'reference')
    if censored_flags is None:
        flags = np.zeros(len(references), dtype=bool)
    else:
        flags = np.asarray(censored_flags)
        if flags.ndim != 1 or len(flags) != len(references) or (flags.size and flags.dtype.kind != 'b'):
            raise ValueError('censored_flags must contain one boolean per reference')
        flags = flags.astype(bool)
    matches = []
    onset_errors, end_errors = [], []
    if len(predictions) and len(references):
        intersection = np.maximum(0, np.minimum(predictions[:, None, 1], references[None, :, 1])
                                  - np.maximum(predictions[:, None, 0], references[None, :, 0]))
        union = (predictions[:, 1] - predictions[:, 0])[:, None] + (references[:, 1] - references[:, 0])[None, :] - intersection
        iou = intersection / union
        valid = iou >= MATCH_IOU_THRESHOLD
        # One additional valid pair outweighs every possible IoU improvement.
        reward = valid * (min(len(predictions), len(references)) + 1 + iou)
        rows, columns = linear_sum_assignment(reward, maximize=True)
        for pred_index, ref_index in zip(rows, columns, strict=True):
            if not valid[pred_index, ref_index]:
                continue
            censored = bool(flags[ref_index])
            onset = float(predictions[pred_index, 0] - references[ref_index, 0])
            end = float(predictions[pred_index, 1] - references[ref_index, 1])
            matches.append({
                'prediction_index': int(pred_index), 'reference_index': int(ref_index),
                'iou': float(iou[pred_index, ref_index]), 'reference_censored': censored,
                'onset_error_seconds': None if censored else onset,
                'end_error_seconds': None if censored else end,
                'onset_absolute_error_seconds': None if censored else abs(onset),
                'end_absolute_error_seconds': None if censored else abs(end),
            })
            if not censored:
                onset_errors.append(onset)
                end_errors.append(end)
    tp = len(matches)
    fp, fn = len(predictions) - tp, len(references) - tp
    mean = lambda values: float(np.mean(values)) if values else None
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': tp / len(predictions) if len(predictions) else None,
        'recall': tp / len(references) if len(references) else None,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
        'false_positives_per_hour': fp * 3600 / observed_seconds,
        'observed_seconds': observed_seconds,
        'matches': matches,
        'localization': {
            'matched_uncensored_count': len(onset_errors),
            'mean_onset_error_seconds': mean(onset_errors),
            'mean_end_error_seconds': mean(end_errors),
            'mean_absolute_onset_error_seconds': mean([abs(value) for value in onset_errors]),
            'mean_absolute_end_error_seconds': mean([abs(value) for value in end_errors]),
        },
    }
