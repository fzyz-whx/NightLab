"""Stream common sample-aligned windows without resetting recording filters."""
import math

import numpy as np

from .noise import BRANCHES


def iter_noise_windows(recording,calibration,window_seconds=10.0,hop_seconds=5.0):
    """Yield read-only views; consume immediately rather than storing waveforms.

    Branch buffers retain at most one incoming block plus a window. Only full
    windows are emitted, matching classic preparation. No labels are changed.
    """
    rate = recording.source_rate
    if (not all(math.isfinite(value) for value in (window_seconds,hop_seconds))
            or not 0 < hop_seconds <= window_seconds):
        raise ValueError('positive hop <= window required')
    length,hop = window_seconds*rate,hop_seconds*rate
    if abs(length-round(length)) > 1e-6 or abs(hop-round(hop)) > 1e-6:
        raise ValueError('window/hop must align to the native sample grid')
    length,hop = round(length),round(hop)
    if min(length,hop) < 1:
        raise ValueError('window/hop must contain samples')
    buffers = {name:np.empty(0,dtype=np.float32) for name in BRANCHES}
    next_sample,received = 0,0
    for block in recording.iter_blocks(calibration):
        begin,end = block['start_sample'],block['end_sample']
        if begin != received or end <= begin:
            raise ValueError('condition stream is not contiguous')
        for name in BRANCHES:
            values = block[name]
            if values.shape != (end-begin,) or values.dtype != np.float32:
                raise ValueError('branch grid/dtype mismatch')
            buffers[name] = np.concatenate((buffers[name],values))
        received = end
        while len(buffers[BRANCHES[0]]) >= length:
            result = {'start_seconds':next_sample/rate,'end_seconds':(next_sample+length)/rate}
            for name in BRANCHES:
                view = buffers[name][:length].view()
                view.setflags(write=False)
                result[name] = view
            yield result
            next_sample += hop
            buffers = {name:values[hop:] for name,values in buffers.items()}
    if received != calibration['sample_count']:
        raise ValueError('condition stream ended before recording')
