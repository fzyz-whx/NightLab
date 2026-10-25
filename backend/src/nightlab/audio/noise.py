"""Synthetic noise controls with recording-wide SNR and causal filtering.

SNR is measured in the 16 kHz analysis band over a whole recording.  Neither
synthetic control is a recording of real environmental noise.  Working arrays
are bounded by the block size; callers retain ownership of the input arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator

import numpy as np
from scipy.signal import butter, firwin, resample_poly, sosfilt

BRANCHES = ("matched_clean", "matched_filtered_clean", "noisy", "filtered_noisy")


def _rate(source_rate: int) -> int:
    if isinstance(source_rate, (bool, np.bool_)) or not isinstance(source_rate, (int, np.integer)):
        raise ValueError("source_rate must be an integer multiple of 16000")
    if source_rate < 16000 or source_rate % 16000:
        raise ValueError("source_rate must be an integer multiple of 16000")
    return int(source_rate)


def _block_size(block_samples: int | None, source_rate: int) -> int:
    if block_samples is None:
        return source_rate * 10
    if isinstance(block_samples, (bool, np.bool_)) or not isinstance(block_samples, (int, np.integer)) or block_samples <= 0:
        raise ValueError("block_samples must be a positive integer")
    return int(block_samples)


def _signal(value: np.ndarray, block: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "fiu":
        raise ValueError("signal must be a nonempty one-dimensional real numeric array")
    for start in range(0, array.size, block):
        if not np.isfinite(array[start : start + block]).all():
            raise ValueError("signal must contain only finite samples")
    return array


def noise_seed(record_id: str, split: str, kind: str, seed: int = 42) -> int:
    """Stable independent random stream identifier, independent of Python hash."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("invalid split")
    if kind not in {"white", "mains_hum"}:
        raise ValueError("invalid synthetic noise kind")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be nonempty text")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    encoded = json.dumps([record_id, split, kind, int(seed)], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def synthetic_noise(count: int, source_rate: int, kind: str, seed: int) -> np.ndarray:
    """Return float32 white noise or a periodic synthetic mains-hum control."""
    rate = _rate(source_rate)
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(seed)
    if kind == "white":
        return rng.standard_normal(int(count), dtype=np.float32)
    if kind != "mains_hum":
        raise ValueError("invalid synthetic noise kind")
    period = rate // math.gcd(rate, 50)
    times = np.arange(period, dtype=np.float64) / rate
    cycle = np.zeros(period, dtype=np.float64)
    for frequency, amplitude, phase in zip((50, 100, 150, 200), (1, 0.5, 0.25, 0.125), rng.uniform(0, 2 * np.pi, 4), strict=True):
        cycle += amplitude * np.sin(2 * np.pi * frequency * times + phase)
    return np.tile(cycle.astype(np.float32), (int(count) + period - 1) // period)[:count]


def analysis_rms(signal: np.ndarray, source_rate: int, block_samples: int | None = None) -> float:
    """RMS of explicit FIR decimation, equivalent to whole-record resample_poly.

    Aligned core chunks have 11*q source-sample guards on either side.  Only
    physical recording edges see zero padding; interior guards are discarded.
    """
    rate = _rate(source_rate)
    block = _block_size(block_samples, rate)
    array = _signal(signal, block)
    return _analysis_rms_reader(array.size, rate, block, lambda left, right: array[left:right])


def _analysis_rms_reader(count_samples: int, rate: int, block: int, read: Callable[[int, int], np.ndarray]) -> float:
    """Analyze a slice reader while retaining only a core block plus FIR guards."""
    q = rate // 16000
    block = max(q, block // q * q)
    fir = firwin(2 * 10 * q + 1, 1 / q, window=("kaiser", 5.0)) if q > 1 else None
    total = np.float64(0)
    compensation = np.float64(0)
    count = 0
    for start in range(0, count_samples, block):
        end = min(start + block, count_samples)
        if q == 1:
            values = np.asarray(read(start, end), dtype=np.float64)
        else:
            left, right = max(0, start - 11 * q), min(count_samples, end + 11 * q)
            guarded = np.asarray(read(left, right), dtype=np.float64)
            decimated = resample_poly(guarded, 1, q, window=fir)
            offset = (start - left) // q
            length = (end + q - 1) // q - start // q
            values = decimated[offset : offset + length]
        increment = np.dot(values, values) - compensation
        updated = total + increment
        compensation = (updated - total) - increment
        total = updated
        count += values.size
    return float(np.sqrt(total / count))


def _peak(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


class NoiseRecording:
    """Calibrate and stream four matched branches without window resets.

    Input arrays must remain unchanged while this object is in use. Calibration
    dictionaries are JSON-ready and checked against this object's calibration
    results before streaming, preventing accidental reuse for another recording.
    """

    def __init__(self, clean: np.ndarray, noise: np.ndarray, source_rate: int, block_samples: int | None = None):
        self.source_rate = _rate(source_rate)
        self.block_samples = _block_size(block_samples, self.source_rate)
        self.clean = _signal(clean, self.block_samples)
        self.noise = _signal(noise, self.block_samples)
        if self.clean.shape != self.noise.shape:
            raise ValueError("clean and noise must have identical shapes")
        self.clean_analysis_rms = analysis_rms(self.clean, self.source_rate, self.block_samples)
        self.noise_analysis_rms = analysis_rms(self.noise, self.source_rate, self.block_samples)
        self.sos = butter(4, [80, 4000], btype="bandpass", fs=self.source_rate, output="sos")
        self._calibrations: dict[float, dict] = {}
        digest = hashlib.sha256(str(self.source_rate).encode("ascii"))
        peaks = [0.0, 0.0]
        zi = np.zeros((self.sos.shape[0], 2))
        for start, end in self._ranges():
            clean_block = np.asarray(self.clean[start:end], dtype=np.float64)
            noise_block = np.asarray(self.noise[start:end], dtype=np.float64)
            digest.update(clean_block.tobytes())
            digest.update(noise_block.tobytes())
            filtered, zi = sosfilt(self.sos, clean_block, zi=zi)
            peaks[0] = max(peaks[0], _peak(clean_block))
            peaks[1] = max(peaks[1], _peak(filtered))
        self._fingerprint = digest.hexdigest()
        self._clean_peaks = peaks

    def _ranges(self):
        for start in range(0, self.clean.size, self.block_samples):
            yield start, min(start + self.block_samples, self.clean.size)

    def calibrate(self, snr_db: float) -> dict:
        """Measure all recording-wide peaks and assign one gain to all branches."""
        if isinstance(snr_db, (bool, np.bool_)) or not np.isscalar(snr_db):
            raise ValueError("snr_db must be finite")
        snr_db = float(snr_db)
        if not math.isfinite(snr_db):
            raise ValueError("snr_db must be finite")
        if self.clean_analysis_rms <= 0 or self.noise_analysis_rms <= 0:
            raise ValueError("SNR is undefined for zero-energy clean or noise")
        try:
            scale = self.clean_analysis_rms / (self.noise_analysis_rms * 10 ** (snr_db / 20))
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("snr_db produces an unrepresentable noise scale") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("snr_db produces an unrepresentable noise scale")
        peaks = dict(zip(BRANCHES, [*self._clean_peaks, 0.0, 0.0], strict=True))
        zi = np.zeros((self.sos.shape[0], 2))
        for start, end in self._ranges():
            mixture = np.asarray(self.clean[start:end], dtype=np.float64) + scale * np.asarray(self.noise[start:end], dtype=np.float64)
            filtered, zi = sosfilt(self.sos, mixture, zi=zi)
            peaks["noisy"] = max(peaks["noisy"], _peak(mixture))
            peaks["filtered_noisy"] = max(peaks["filtered_noisy"], _peak(filtered))
        maximum = max(peaks.values())
        if not math.isfinite(maximum):
            raise ValueError("mixture must remain finite")
        gain = min(1.0, 0.98 / maximum)
        result = {
            "target_snr_db": snr_db,
            "noise_scale": scale,
            "common_gain": gain,
            "clean_analysis_rms": self.clean_analysis_rms,
            "noise_analysis_rms": self.noise_analysis_rms,
            "achieved_analysis_snr_db": 20 * math.log10(self.clean_analysis_rms / (scale * self.noise_analysis_rms)),
            "peaks_before_gain": peaks,
            "peak_bound_after_gain": maximum * gain,
            "source_rate": self.source_rate,
            "sample_count": int(self.clean.size),
            "recording_fingerprint": self._fingerprint,
        }
        self._calibrations[snr_db] = json.loads(json.dumps(result, allow_nan=False))
        return result

    def _checked_parameters(self, calibration: dict) -> tuple[float, float]:
        if not isinstance(calibration, dict):
            raise ValueError("invalid calibration")
        try:
            encoded = json.dumps(calibration, sort_keys=True, allow_nan=False)
            expected = self._calibrations[calibration["target_snr_db"]]
            if encoded != json.dumps(expected, sort_keys=True, allow_nan=False):
                raise ValueError("calibration does not match this recording")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration does not match this recording") from exc
        return calibration["noise_scale"], calibration["common_gain"]

    def measure_actual_snr(self, calibration: dict) -> float:
        """Measure actual float32 model-input SNR with bounded working memory.

        Both components use the same guarded 16 kHz analysis as calibration.
        The error component is float64 subtraction of the two rounded float32
        branches, so this measures rounding as well as the added noise. Each
        component is read in a separate streaming pass; no full output exists.
        """
        scale, gain = self._checked_parameters(calibration)

        def read_clean(left: int, right: int) -> np.ndarray:
            return (gain * np.asarray(self.clean[left:right], dtype=np.float64)).astype(np.float32)

        def read_error(left: int, right: int) -> np.ndarray:
            clean = np.asarray(self.clean[left:right], dtype=np.float64)
            noise = np.asarray(self.noise[left:right], dtype=np.float64)
            matched_clean = (gain * clean).astype(np.float32)
            noisy = (gain * (clean + scale * noise)).astype(np.float32)
            return noisy.astype(np.float64) - matched_clean.astype(np.float64)

        clean_rms = _analysis_rms_reader(self.clean.size, self.source_rate, self.block_samples, read_clean)
        error_rms = _analysis_rms_reader(self.clean.size, self.source_rate, self.block_samples, read_error)
        if not math.isfinite(clean_rms) or not math.isfinite(error_rms) or clean_rms <= 0 or error_rms <= 0:
            raise ValueError("actual SNR is undefined for zero-energy or nonfinite rounded components")
        return 20 * math.log10(clean_rms / error_rms)

    def iter_blocks(self, calibration: dict) -> Iterator[dict]:
        """Stream independently restartable causal branches as float32 blocks."""
        scale, gain = self._checked_parameters(calibration)
        clean_zi = np.zeros((self.sos.shape[0], 2))
        noisy_zi = np.zeros_like(clean_zi)
        for start, end in self._ranges():
            clean = np.asarray(self.clean[start:end], dtype=np.float64)
            noisy = clean + scale * np.asarray(self.noise[start:end], dtype=np.float64)
            filtered_clean, clean_zi = sosfilt(self.sos, clean, zi=clean_zi)
            filtered_noisy, noisy_zi = sosfilt(self.sos, noisy, zi=noisy_zi)
            result = {"start_sample": start, "end_sample": end}
            for name, values in zip(BRANCHES, (clean, filtered_clean, noisy, filtered_noisy), strict=True):
                output = (values * gain).astype(np.float32)
                if not np.isfinite(output).all() or _peak(output) > 1:
                    raise ValueError("branch is nonfinite or exceeds the amplitude bound")
                result[name] = output
            yield result
