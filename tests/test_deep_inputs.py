import numpy as np
import pytest

from nightlab.audio.features import AudioFeatureExtractor
from nightlab.deep.inputs import (
    MaskedChannelStandardizer, log_mel_image, wearable_epoch_tensors,
)


def test_wearable_manual_bins_boundaries_masks_and_labels():
    hr = [[-1, 999], [0, 60], [.5, 80], [1, 90], [29.9, 100],
          [30, 777], [60, 50], [89.9, 70], [90, 999]]
    motion = [[-1, 9, 9, 9], [0, 0, 0, 0], [.5, 0, 0, 2],
              [1, 2, 3, 4], [30, 8, 8, 8], [60, 1, 0, 0], [90, 9, 9, 9]]
    result = wearable_epoch_tensors('s01', hr, motion, [[0, 3], [30, -1], [60, 5]])
    assert result['values'].shape == (2, 5, 30)
    assert result['masks'].shape == (2, 2, 30)
    assert result['values'].dtype == result['masks'].dtype == np.float32
    np.testing.assert_allclose(result['values'][0, :, 0], [70, 0, 0, 1, 1])
    np.testing.assert_allclose(result['values'][0, :, 1], [90, 2, 3, 4, 0])
    assert result['values'][0, 0, 29] == 100
    assert np.isnan(result['values'][0, :, 2]).all()
    np.testing.assert_array_equal(result['masks'][0, :, 2], [0, 0])
    np.testing.assert_array_equal(result['hr_sample_count'][0, :3], [2, 1, 0])
    np.testing.assert_array_equal(result['epoch_start_seconds'], [0, 60])
    np.testing.assert_array_equal(result['raw_stage'], [3, 5])
    np.testing.assert_array_equal(result['y'], ['N3', 'REM'])
    assert result['y'].dtype.kind == result['subject_id'].dtype.kind == 'U'
    assert result['metadata']['unscored_epochs'] == 1
    for stats in result['metadata']['signals'].values():
        assert stats['samples_before_label_range'] == stats['samples_after_label_range'] == 1
        assert stats['samples_in_unscored_epochs'] == 1


def test_repeat_blocks_explicit_policy_and_strict_labels():
    hr = np.array([[0, 60], [1, 70]])
    motion = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    result = wearable_epoch_tensors('s', np.tile(hr, (3, 1)), np.tile(motion, (2, 1)), [[0, 4]])
    stats = result['metadata']['signals']['heart_rate']
    assert stats['original_input_samples'] == 6
    assert stats['input_samples'] == 2
    assert stats['identical_repeated_blocks'] == 3
    assert stats['discarded_exact_repeat_samples'] == 4
    assert result['y'].tolist() == ['N3']
    with pytest.raises(ValueError, match='strictly increase'):
        wearable_epoch_tensors('s', np.tile(hr, (2, 1)), motion, [[0, 0]], 'reject')
    for labels in ([[0, 0], [0, 0]], [[0, 0], [31, 1]], [[0, 1.5]], [[0, 6]]):
        with pytest.raises(ValueError):
            wearable_epoch_tensors('s', hr, motion, labels)
    damaged = np.tile(hr, (2, 1))
    damaged[-1, 1] += 1
    with pytest.raises(ValueError, match='non-identical'):
        wearable_epoch_tensors('s', damaged, motion, [[0, 0]])


def test_all_unscored_retains_empty_shapes():
    result = wearable_epoch_tensors('s', [[0, 60]], [[0, 1, 2, 3]], [[0, -1]])
    assert result['values'].shape == (0, 5, 30)
    assert result['masks'].shape == (0, 2, 30)
    assert result['y'].dtype.kind == 'U'
    assert result['metadata']['signals']['heart_rate']['samples_in_unscored_epochs'] == 1


def test_log_mel_sine_peak_shape_and_exact_silence_floor():
    rate = 16000
    signal = np.sin(2 * np.pi * 1000 * np.arange(rate * 2) / rate)
    image = log_mel_image(signal, rate)
    assert image.shape == (32, 100)
    assert image.dtype == np.float32
    extractor = AudioFeatureExtractor()
    expected_band = extractor.filters[np.argmin(abs(extractor.frequencies - 1000))].argmax()
    assert image.mean(axis=1).argmax() == expected_band
    np.testing.assert_allclose(log_mel_image(np.zeros(rate * 2), rate), np.log(1e-12), atol=2e-6)
    # Different source rate takes the same physical tone through real resampling.
    low_rate = 8000
    resampled = log_mel_image(np.sin(2*np.pi*1000*np.arange(low_rate*2)/low_rate), low_rate)
    assert resampled.mean(axis=1).argmax() == expected_band


def test_log_mel_uses_contiguous_frame_groups_before_averaging():
    signal = np.r_[np.zeros(16000), np.ones(16000)]
    extractor = AudioFeatureExtractor()
    frames = np.lib.stride_tricks.sliding_window_view(signal, 400)[::160]
    power = abs(np.fft.rfft(frames * extractor.window, n=512))**2 / (extractor.window**2).sum()
    log_power = np.log(np.maximum(power @ extractor.filters, 1e-12))
    edges = np.linspace(0, len(frames), 101, dtype=int)
    expected = np.stack([log_power[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    np.testing.assert_allclose(log_mel_image(signal, 16000), expected, atol=2e-6)
    assert not np.allclose(expected[:, 0], expected[:, -1])


@pytest.mark.parametrize('signal,rate,bins', [([], 16000, 100), ([np.nan], 16000, 1),
    (np.zeros(400), 16000, 100), (np.zeros(16000), 0, 10), (np.zeros(16000), 16000, 0)])
def test_log_mel_invalid_inputs(signal, rate, bins):
    with pytest.raises(ValueError):
        log_mel_image(signal, rate, bins)


def test_standardizer_training_only_no_mutation_missing_and_constant_channel():
    training = np.array([[[1, 3, np.nan], [7, 7, np.nan]],
                         [[5, np.nan, 7], [7, np.nan, 7]]])
    original = training.copy()
    standardizer = MaskedChannelStandardizer().fit(training)
    np.testing.assert_allclose(standardizer.mean_, [4, 7])
    np.testing.assert_allclose(standardizer.scale_, [np.sqrt(5), 1])
    validation = np.array([[[104, np.nan], [9, np.nan]]])
    transformed = standardizer.transform(validation)
    np.testing.assert_allclose(transformed[0, :, 0], [100 / np.sqrt(5), 2], rtol=1e-6)
    np.testing.assert_array_equal(transformed[0, :, 1], 0)
    np.testing.assert_array_equal(training, original)
    np.testing.assert_allclose(standardizer.mean_, [4, 7])
    assert np.isnan(validation[0, 0, 1])
    assert transformed.dtype == np.float32


def test_standardizer_rejects_empty_channels_inf_shape_and_unfitted():
    with pytest.raises(ValueError, match='no observed'):
        MaskedChannelStandardizer().fit(np.full((1, 1, 2), np.nan))
    with pytest.raises(ValueError, match='infinite'):
        MaskedChannelStandardizer().fit(np.array([[[np.inf]]]))
    with pytest.raises(ValueError, match='N x C x T'):
        MaskedChannelStandardizer().fit(np.ones((2, 3)))
    with pytest.raises(ValueError, match='fitted'):
        MaskedChannelStandardizer().transform(np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match='channel count'):
        MaskedChannelStandardizer().fit(np.ones((1, 1, 1))).transform(np.ones((1, 2, 1)))
