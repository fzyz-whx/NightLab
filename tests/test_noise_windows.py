import numpy as np
import pytest

from nightlab.audio.noise import BRANCHES,NoiseRecording,synthetic_noise
from nightlab.audio.noise_windows import iter_noise_windows


def test_cross_block_window_samples_and_overlap_exactly_match_recording():
    rate = 16000
    t = np.arange(rate*2+3)/rate
    original = .2*np.sin(2*np.pi*500*t)
    recording = NoiseRecording(original,synthetic_noise(len(t),rate,'white',42),rate,block_samples=rate//3)
    calibration = recording.calibrate(0)
    complete = {name:np.concatenate([block[name] for block in recording.iter_blocks(calibration)]) for name in BRANCHES}
    windows = list(iter_noise_windows(recording,calibration,1,.5))
    assert len(windows) == 3
    for index,window in enumerate(windows):
        assert (window['start_seconds'],window['end_seconds']) == (index*.5,index*.5+1)
        for name in BRANCHES:
            np.testing.assert_array_equal(window[name],complete[name][index*8000:index*8000+16000])
            assert not window[name].flags.writeable
            if index:
                np.testing.assert_array_equal(windows[index-1][name][8000:],window[name][:8000])


@pytest.mark.parametrize('window,hop',[(0,1),(1,0),(1,2),(np.nan,1),(1.00001,.5)])
def test_invalid_noise_window_grid_rejected(window,hop):
    recording = NoiseRecording(np.ones(100),np.ones(100),16000)
    calibration = recording.calibrate(0)
    with pytest.raises(ValueError):
        list(iter_noise_windows(recording,calibration,window,hop))
