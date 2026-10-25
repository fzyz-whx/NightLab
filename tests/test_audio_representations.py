import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from nightlab.audio.features import AudioFeatureExtractor
from nightlab.audio.representations import joint_features
from nightlab.deep.inputs import log_mel_image


@pytest.mark.parametrize('rate', [16000, 48000])
@pytest.mark.parametrize('kind', ['random', 'silence', 'tone'])
def test_joint_matches_both_frozen_representations_exactly(rate, kind):
    count = rate * 2 + 17
    signal = {'random': lambda: np.random.default_rng(42).standard_normal(count).astype(np.float32),
              'silence': lambda: np.zeros(count, dtype=np.float32),
              'tone': lambda: np.sin(np.arange(count) * 2 * np.pi * 253 / rate).astype(np.float32)}[kind]()
    original = signal.copy()
    with threadpool_limits(limits=1):
        classic, image = joint_features(signal, rate)
        np.testing.assert_array_equal(classic, AudioFeatureExtractor().transform(signal, rate))
        np.testing.assert_array_equal(image, log_mel_image(signal, rate))
    assert classic.dtype == image.dtype == np.float32
    assert classic.shape == (67,) and image.shape == (32, 100)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize('signal,rate', [(np.array([]), 16000), (np.array([np.nan]), 16000),
                                      (np.ones((2, 3)), 16000), (np.ones(10), 16000),
                                      (np.ones(32000), 0), (np.ones(32000), 16000.0)])
def test_joint_rejects_invalid_input(signal, rate):
    with pytest.raises(ValueError):
        joint_features(signal, rate)
