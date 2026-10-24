"""Deterministic RMS, zero crossing, spectral and log-mel statistics."""
from dataclasses import asdict, dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    window_seconds: float = 10.0
    hop_seconds: float = 5.0
    positive_overlap_fraction: float = 0.5
    frame_samples: int = 400
    frame_hop_samples: int = 160
    fft_samples: int = 512
    mel_bands: int = 32
    power_floor: float = 1e-12

    def __post_init__(self):
        ints = (self.sample_rate, self.frame_samples, self.frame_hop_samples, self.fft_samples, self.mel_bands)
        if any(type(value) is not int or value <= 0 for value in ints):
            raise ValueError('audio dimensions/rates must be positive integers')
        if not np.isfinite([self.window_seconds, self.hop_seconds, self.power_floor, self.positive_overlap_fraction]).all():
            raise ValueError('audio configuration must be finite')
        if min(self.window_seconds, self.hop_seconds, self.power_floor) <= 0 or not 0 < self.positive_overlap_fraction <= 1:
            raise ValueError('invalid audio window or power parameters')
        if self.fft_samples < self.frame_samples:
            raise ValueError('FFT must cover the full frame')
        if self.window_seconds*self.sample_rate < self.frame_samples:
            raise ValueError('window shorter than one analysis frame')

    def metadata(self):
        return {**asdict(self), 'mel_formula': 'HTK 2595*log10(1+Hz/700)',
                'spectrum': 'Hann-window rFFT power divided by sum(window**2)',
                'log': 'natural log; mel power floored before log',
                'input_amplitude': 'fixed EDF digital ADC full scale; not physical dB/SPL',
                'resampling': 'scipy.signal.resample_poly, integer rate ratio, default Kaiser window'}


class AudioFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or AudioFeatureConfig()
        c = self.config
        self.window = np.hanning(c.frame_samples)
        self.frequencies = np.fft.rfftfreq(c.fft_samples, 1/c.sample_rate)
        mel_limit = 2595*np.log10(1+(c.sample_rate/2)/700)
        boundaries = 700*(10**(np.linspace(0, mel_limit, c.mel_bands+2)/2595)-1)
        filters = []
        for left, center, right in zip(boundaries, boundaries[1:], boundaries[2:]):
            filters.append(np.maximum(0, np.minimum((self.frequencies-left)/(center-left),
                                                   (right-self.frequencies)/(right-center))))
        self.filters = np.asarray(filters).T
        if np.any(self.filters.sum(axis=0) == 0):
            raise ValueError('mel band has no FFT support; adjust bands or FFT size')
        self.feature_names = ('rms_adc_full_scale', 'zero_crossing_fraction', 'power_centroid_hz') + tuple(
            f'log_mel_{stat}_{band:02d}' for stat in ('mean', 'std') for band in range(c.mel_bands))

    def transform(self, signal, source_rate):
        c = self.config
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
            raise ValueError('audio must be nonempty finite mono samples')
        if type(source_rate) is not int or source_rate <= 0:
            raise ValueError('source rate must be a positive integer')
        if source_rate != c.sample_rate:
            factor = gcd(source_rate, c.sample_rate)
            signal = resample_poly(signal, c.sample_rate//factor, source_rate//factor)
        if len(signal) < c.frame_samples:
            raise ValueError('audio shorter than one analysis frame')
        frames = np.lib.stride_tricks.sliding_window_view(signal, c.frame_samples)[::c.frame_hop_samples]
        power = np.abs(np.fft.rfft(frames*self.window, n=c.fft_samples, axis=1))**2 / np.square(self.window).sum()
        total_power = power.sum()
        centroid = float((power*self.frequencies).sum()/total_power) if total_power > 0 else 0.0
        log_mel = np.log(np.maximum(power @ self.filters, c.power_floor))
        values = np.r_[np.sqrt(np.square(signal).mean()), np.mean(np.signbit(signal[1:]) != np.signbit(signal[:-1])),
                       centroid, log_mel.mean(axis=0), log_mel.std(axis=0)]
        if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError('nonfinite or inconsistent audio feature vector')
        return values.astype(np.float32)
