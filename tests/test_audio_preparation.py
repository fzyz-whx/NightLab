from pathlib import Path

import numpy as np
import pytest

from nightlab.audio.edf import EDFRecording
from nightlab.audio.features import AudioFeatureConfig, AudioFeatureExtractor
from nightlab.audio.reference import ReferenceEvent, union_overlap, window_target


def tiny_edf(path, labels=('Mic', 'Snore')):
    fixed = bytearray(b' '*256)
    for start, value in [(0,b'0       '),(168,b'07.05.10'),(176,b'22.00.00'),
                         (184,b'768     '),(236,b'2       '),(244,b'1       '),(252,b'2   ')]:
        fixed[start:start+len(value)] = value
    header = bytearray(b' '*512)
    for index, label in enumerate(labels):
        header[index*16:(index+1)*16] = label.ljust(16).encode()
    for width_offset, values in [(16+80+8+8+8, (-32768,-32768)),
                                  (16+80+8+8+8+8, (32767,32767)),
                                  (16+80+8+8+8+8+8+80, (4,2))]:
        offset = 2*width_offset
        for index, value in enumerate(values):
            header[offset+index*8:offset+(index+1)*8] = str(value).ljust(8).encode()
    payload = np.asarray([0,1,2,3,90,91,4,5,6,7,92,93], dtype='<i2').tobytes()
    path.write_bytes(fixed+header+payload)


def test_edf_cross_record_window_uses_only_selected_channel(tmp_path):
    path = tmp_path/'signal.edf'
    tiny_edf(path)
    recording = EDFRecording(path)
    np.testing.assert_allclose(recording.read_window('Mic', .5, 1), np.array([2,3,4,5])/32768)
    np.testing.assert_allclose(recording.read_window('Snore', 0, 2), np.array([90,91,92,93])/32768)
    for interval in [(-1,1),(0,3),(0,0),(.1,1)]:
        with pytest.raises(ValueError):
            recording.read_window('Mic', *interval)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='size inconsistent'):
        EDFRecording(path)


def test_edf_duplicate_channel_and_bad_header_rejected(tmp_path):
    path = tmp_path/'signal.edf'
    tiny_edf(path, ('Mic','Mic'))
    with pytest.raises(ValueError, match='unambiguous'):
        EDFRecording(path).read_window('Mic', 0, 1)
    path.write_bytes(b'')
    with pytest.raises(ValueError):
        EDFRecording(path)


def test_sine_features_and_resampling_have_known_scale_and_centroid():
    rate = 8000
    times = np.arange(rate*2)/rate
    extractor = AudioFeatureExtractor()
    features = extractor.transform(.5*np.sin(2*np.pi*1000*times), rate)
    assert features.shape == (67,)
    assert abs(features[0]-.5/np.sqrt(2)) < .005
    assert abs(features[2]-1000) < 10
    assert np.isfinite(features).all()


def test_silence_remains_finite_and_bad_audio_rejected():
    extractor = AudioFeatureExtractor()
    features = extractor.transform(np.zeros(16000), 16000)
    assert features[0] == features[1] == features[2] == 0
    np.testing.assert_allclose(features[3:35], np.log(1e-12), atol=1e-5)
    for signal in [[], [np.nan]*500, np.zeros((2,500)), np.zeros(20)]:
        with pytest.raises(ValueError):
            extractor.transform(signal, 16000)
    with pytest.raises(ValueError):
        AudioFeatureConfig(fft_samples=200)


def test_reference_window_uses_union_and_keeps_ambiguous_windows_out():
    events = [ReferenceEvent(0,6,'Hypopnea'),ReferenceEvent(4,8,'ObstructiveApnea')]
    assert union_overlap(0,10,events) == 8
    assert window_target(0,10,events) == ('reference_respiratory_event', .8)
    assert window_target(7,17,events) == (None, .1)
    assert window_target(8,18,events) == ('no_target_reference_event', 0)
    assert window_target(0,10,[ReferenceEvent(0,10,'RERA')]) == (None,0)
