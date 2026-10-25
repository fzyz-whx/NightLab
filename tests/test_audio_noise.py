"""Small synthetic controls; no real recordings, model training, or GPU work."""

import json

import numpy as np
import pytest
from scipy.signal import butter, firwin, resample_poly, sosfilt

from nightlab.audio.noise import BRANCHES, NoiseRecording, analysis_rms, noise_seed, synthetic_noise


def test_stable_seed_and_independent_streams():
    seed = noise_seed("record-01", "train", "white")
    assert seed == 8373482942810209319
    assert 0 <= seed < 2**64
    seeds = {noise_seed("record-01", split, kind) for split in ("train", "validation", "test") for kind in ("white", "mains_hum")}
    assert len(seeds) == 6
    first = synthetic_noise(24001, 48000, "white", seed)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, synthetic_noise(24001, 48000, "white", seed))


def test_hum_period_frequencies_and_amplitudes():
    values = synthetic_noise(48000, 48000, "mains_hum", 42)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values[:-960], values[960:])
    amplitudes = np.abs(np.fft.rfft(values.astype(np.float64))) * 2 / values.size
    np.testing.assert_allclose(amplitudes[[50, 100, 150, 200]], [1, 0.5, 0.25, 0.125], atol=1e-8)
    amplitudes[[50, 100, 150, 200]] = 0
    assert amplitudes.max() < 1e-8


@pytest.mark.parametrize("rate", [16000, 32000, 48000])
@pytest.mark.parametrize("length", [1, 2, 3, 31, 101, 48013])
@pytest.mark.parametrize("block", [1, 16, 1000, 48000])
def test_analysis_rms_matches_complete_float64_decimation(rate, length, block):
    signal = np.random.default_rng(123).standard_normal(length).astype(np.float32)
    q = rate // 16000
    if q > 1:
        fir = firwin(2 * 10 * q + 1, 1 / q, window=("kaiser", 5.0))
        whole = resample_poly(signal.astype(np.float64), 1, q, window=fir)
    else:
        whole = signal.astype(np.float64)
    expected = np.sqrt(np.mean(whole**2))
    assert analysis_rms(signal, rate, block) == pytest.approx(expected, rel=2e-14, abs=1e-15)


def _recording(block=1001, count=96017):
    rate = 48000
    t = np.arange(count) / rate
    clean = (1.7 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.cos(2 * np.pi * 3700 * t)).astype(np.float32)
    noise = synthetic_noise(count, rate, "white", 18)
    return NoiseRecording(clean, noise, rate, block), clean, noise


def _join(blocks):
    return {name: np.concatenate([block[name] for block in blocks]) for name in BRANCHES}


@pytest.mark.parametrize("snr", [-5, 0, 5, 10])
def test_whole_record_snr_common_gain_filter_continuity_and_immutability(snr):
    recording, clean, noise = _recording()
    original_clean, original_noise = clean.copy(), noise.copy()
    calibration = recording.calibrate(snr)
    calibration = json.loads(json.dumps(calibration, allow_nan=False))
    blocks = list(recording.iter_blocks(calibration))
    result = _join(blocks)
    assert blocks[0]["start_sample"] == 0
    assert blocks[-1]["end_sample"] == clean.size
    assert all(left["end_sample"] == right["start_sample"] for left, right in zip(blocks[:-1], blocks[1:], strict=True))
    gain, scale = calibration["common_gain"], calibration["noise_scale"]
    sos = butter(4, [80, 4000], btype="bandpass", fs=48000, output="sos")
    mixture = clean.astype(np.float64) + scale * noise.astype(np.float64)
    expected = {
        "matched_clean": clean.astype(np.float64),
        "matched_filtered_clean": sosfilt(sos, clean.astype(np.float64)),
        "noisy": mixture,
        "filtered_noisy": sosfilt(sos, mixture),
    }
    for name in BRANCHES:
        assert result[name].dtype == np.float32
        assert result[name].shape == clean.shape
        assert np.max(np.abs(result[name])) <= 0.9800001
        np.testing.assert_array_equal(result[name], (expected[name] * gain).astype(np.float32))
        assert calibration["peaks_before_gain"][name] == pytest.approx(np.max(np.abs(expected[name])), rel=1e-13)
    assert gain == pytest.approx(min(1, 0.98 / max(calibration["peaks_before_gain"].values())))
    residual = result["noisy"].astype(np.float64) - result["matched_clean"].astype(np.float64)
    actual_snr = 20 * np.log10(analysis_rms(result["matched_clean"], 48000) / analysis_rms(residual, 48000))
    assert actual_snr == pytest.approx(snr, abs=1e-6)
    assert recording.measure_actual_snr(calibration) == pytest.approx(actual_snr, abs=1e-12)
    assert calibration["achieved_analysis_snr_db"] == pytest.approx(snr, abs=1e-12)
    np.testing.assert_array_equal(clean, original_clean)
    np.testing.assert_array_equal(noise, original_noise)
    # Each iteration restarts from the recording's beginning, with fresh states.
    repeated = _join(list(recording.iter_blocks(calibration)))
    for name in BRANCHES:
        np.testing.assert_array_equal(repeated[name], result[name])
        # Two overlapping downstream windows retain identical shared samples.
        window_a, window_b = result[name][900:3900], repeated[name][1900:4900]
        np.testing.assert_array_equal(window_a[1000:], window_b[:-1000])


def test_block_size_independent_and_quiet_signal_does_not_increase_gain():
    recording, clean, noise = _recording(block=101, count=5003)
    other = NoiseRecording(clean, noise, 48000, 48000)
    left_cal, right_cal = recording.calibrate(5), other.calibrate(5)
    assert left_cal["noise_scale"] == pytest.approx(right_cal["noise_scale"], rel=2e-14)
    assert left_cal["common_gain"] == pytest.approx(right_cal["common_gain"], rel=2e-14)
    left, right = _join(list(recording.iter_blocks(left_cal))), _join(list(other.iter_blocks(right_cal)))
    for name in BRANCHES:
        np.testing.assert_array_equal(left[name], right[name])
    quiet = NoiseRecording(clean * 0.001, noise, 48000)
    assert quiet.calibrate(0)["common_gain"] == 1.0


@pytest.mark.parametrize("value", [np.array([]), np.zeros((2, 2)), np.array([np.nan]), np.array([np.inf]), np.array([1j]), np.array(["bad"])])
def test_invalid_signals(value):
    with pytest.raises(ValueError):
        analysis_rms(value, 48000)
    with pytest.raises(ValueError):
        NoiseRecording(value, value, 48000)


@pytest.mark.parametrize("rate", [0, 8000, 44100, 48000.0, True])
def test_invalid_rates(rate):
    with pytest.raises(ValueError):
        analysis_rms(np.ones(10), rate)
    with pytest.raises(ValueError):
        synthetic_noise(10, rate, "white", 42)


def test_zero_energy_and_bad_parameters():
    zeros, ones = np.zeros(31), np.ones(31)
    assert analysis_rms(zeros, 48000, 2) == 0
    for clean, noise in ((zeros, ones), (ones, zeros)):
        recording = NoiseRecording(clean, noise, 48000)
        with pytest.raises(ValueError, match="zero-energy"):
            recording.calibrate(0)
    with pytest.raises(ValueError):
        NoiseRecording(ones, np.ones(2), 48000)
    for block in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            analysis_rms(ones, 48000, block)
    for split, kind in (("oops", "white"), ("train", "rain")):
        with pytest.raises(ValueError):
            noise_seed("record", split, kind)
    for count, kind in ((0, "white"), (-1, "white"), (1, "rain")):
        with pytest.raises(ValueError):
            synthetic_noise(count, 48000, kind, 42)
    recording, _, _ = _recording(count=51)
    for snr in (np.nan, np.inf, -np.inf, 100000, -100000, True):
        with pytest.raises(ValueError):
            recording.calibrate(snr)


def test_calibration_cannot_be_modified_or_mixed_between_recordings():
    recording, clean, noise = _recording(count=111)
    calibration = recording.calibrate(0)
    for field, bad in (("noise_scale", 1), ("common_gain", 2), ("target_snr_db", np.nan), ("sample_count", 99)):
        changed = dict(calibration, **{field: bad})
        with pytest.raises(ValueError, match="calibration"):
            list(recording.iter_blocks(changed))
        with pytest.raises(ValueError, match="calibration"):
            recording.measure_actual_snr(changed)
    other = NoiseRecording(clean * 2, noise, 48000)
    other.calibrate(0)
    with pytest.raises(ValueError, match="calibration"):
        list(other.iter_blocks(calibration))
    with pytest.raises(ValueError, match="calibration"):
        other.measure_actual_snr(calibration)


@pytest.mark.parametrize("rate", [16000, 48000])
@pytest.mark.parametrize("block", [1, 1001])
@pytest.mark.parametrize("kind", ["white", "mains_hum"])
def test_actual_snr_reader_matches_whole_float32_branches(rate, block, kind, monkeypatch):
    count = 2117
    times = np.arange(count) / rate
    clean = (1.7 * np.sin(2 * np.pi * 270 * times)).astype(np.float32)
    noise = synthetic_noise(count, rate, kind, 77)
    recording = NoiseRecording(clean, noise, rate, block)
    calibration = recording.calibrate(-5)
    gain, scale = calibration["common_gain"], calibration["noise_scale"]
    clean32 = (gain * clean.astype(np.float64)).astype(np.float32)
    noisy32 = (gain * (clean.astype(np.float64) + scale * noise.astype(np.float64))).astype(np.float32)
    residual = noisy32.astype(np.float64) - clean32.astype(np.float64)
    q = rate // 16000
    if q > 1:
        fir = firwin(2 * 10 * q + 1, 1 / q, window=("kaiser", 5.0))
        clean16 = resample_poly(clean32.astype(np.float64), 1, q, window=fir)
        error16 = resample_poly(residual, 1, q, window=fir)
    else:
        clean16, error16 = clean32.astype(np.float64), residual
    expected = 20 * np.log10(np.sqrt(np.mean(clean16**2)) / np.sqrt(np.mean(error16**2)))
    # This measurement must never materialize the four output streams.
    monkeypatch.setattr(recording, "iter_blocks", lambda _: pytest.fail("unexpected output iteration"))
    assert recording.measure_actual_snr(calibration) == pytest.approx(expected, abs=1e-12)
