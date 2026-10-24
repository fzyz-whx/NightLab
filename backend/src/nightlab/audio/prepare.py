"""Prepare single-segment features without fitting or splitting any model."""
from dataclasses import dataclass
import math

import numpy as np
from threadpoolctl import threadpool_limits

from .edf import EDFRecording
from .features import AudioFeatureConfig, AudioFeatureExtractor
from .reference import read_reference, window_target


@dataclass
class AudioPrepared:
    features: np.ndarray
    target: np.ndarray
    subject_id: np.ndarray
    record_id: np.ndarray
    start_seconds: np.ndarray
    end_seconds: np.ndarray
    overlap_fraction: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict
    all_windows: dict


def prepare_psg_segment(edf_path, rml_path, subject_id, record_id, config=None, channel='Mic', progress=None):
    if not all(isinstance(value, str) and value.strip() for value in (subject_id, record_id)):
        raise ValueError('subject and recording identities must be explicit nonempty strings')
    c = config or AudioFeatureConfig()
    recording = EDFRecording(edf_path)
    origin, segments, events = read_reference(rml_path)
    offset = (recording.start-origin).total_seconds()
    if (offset, recording.duration_seconds) not in segments:
        raise ValueError('EDF clock/duration does not match an RML segment')
    rate = recording.sample_rate(channel)
    if rate != round(rate) or rate < 8000:
        raise ValueError('selected channel must have an integral audio sampling rate >= 8000 Hz')
    if c.window_seconds > recording.duration_seconds:
        raise ValueError('segment shorter than configured window')
    count = 1 + math.floor((recording.duration_seconds-c.window_seconds)/c.hop_seconds+1e-9)
    starts = np.arange(count, dtype=np.float64)*c.hop_seconds
    targets, fractions, features, keep = [], [], [], []
    extractor = AudioFeatureExtractor(c)
    with threadpool_limits(limits=1, user_api='blas'):
        for index, start in enumerate(starts):
            target, fraction = window_target(offset+start, offset+start+c.window_seconds, events, c.positive_overlap_fraction)
            targets.append(target or '')
            fractions.append(fraction)
            if target is not None:
                signal = recording.read_window(channel, float(start), c.window_seconds)
                features.append(extractor.transform(signal, int(rate)))
                keep.append(index)
            if progress is not None and (index % 100 == 0 or index == count-1):
                progress(index+1, count)
    if not features:
        raise ValueError('no unambiguous windows available')
    keep = np.asarray(keep, dtype=np.int64)
    targets = np.asarray(targets, dtype=str)
    fractions = np.asarray(fractions, dtype=np.float64)
    return AudioPrepared(
        features=np.stack(features), target=targets[keep],
        subject_id=np.full(len(keep), subject_id), record_id=np.full(len(keep), record_id),
        start_seconds=starts[keep], end_seconds=starts[keep]+c.window_seconds,
        overlap_fraction=fractions[keep], feature_names=extractor.feature_names,
        all_windows={'start_seconds': starts, 'end_seconds': starts+c.window_seconds,
                     'target': targets, 'target_valid': targets != '', 'overlap_fraction': fractions},
        metadata={'channel': channel, 'source_rate_hz': int(rate), 'segment_offset_seconds': offset,
                  'segment_duration_seconds': recording.duration_seconds, 'candidate_windows': count,
                  'scored_windows': len(keep), 'ambiguous_windows_excluded': count-len(keep),
                  'target_definition': 'union of four PSG reference respiratory event types covers >=50% by default; zero respiratory overlap is background; ambiguous/other-respiratory windows excluded',
                  'target_interpretation': 'reference physiological event windows; not audible snoring or a clinical diagnosis',
                  'config': c.metadata(), 'feature_version': 'psg-rms-logmel-v1', 'feature_blas_threads':1,
                  'subject_count': 1, 'formal_evaluation_eligible': False,
                  'scope': 'feature preparation only; no normalization fit, model fit, split or performance metric'},
    )
