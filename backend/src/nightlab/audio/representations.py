"""Joint frozen classic features and CNN image from a single spectral analysis."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .features import AudioFeatureExtractor


def joint_features(signal, source_rate):
    """Match both historical representations exactly; no fitted statistics."""
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
    if len(frames) < 100:
        raise ValueError('too few analysis frames for nonempty time groups')
    power = np.abs(np.fft.rfft(frames * extractor.window, n=c.fft_samples, axis=1)) ** 2
    power /= np.square(extractor.window).sum()
    total_power = power.sum()
    centroid = float((power * extractor.frequencies).sum() / total_power) if total_power > 0 else 0.0
    log_mel = np.log(np.maximum(power @ extractor.filters, c.power_floor))
    classic = np.r_[np.sqrt(np.square(signal).mean()),
                    np.mean(np.signbit(signal[1:]) != np.signbit(signal[:-1])),
                    centroid, log_mel.mean(axis=0), log_mel.std(axis=0)].astype(np.float32)
    edges = np.linspace(0, len(frames), 101, dtype=np.int64)
    image = np.stack([log_mel[left:right].mean(axis=0)
                      for left, right in zip(edges[:-1], edges[1:], strict=True)], axis=1).astype(np.float32)
    if classic.shape != (67,) or image.shape != (32, 100) or not np.isfinite(classic).all() or not np.isfinite(image).all():
        raise ValueError('nonfinite or inconsistent representations')
    return classic, image
