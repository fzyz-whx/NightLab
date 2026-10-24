"""Bounded EDF channel reads; raw patient fields are never exposed."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class Channel:
    label: str
    samples_per_record: int
    offset_bytes: int
    digital_min: int
    digital_max: int


class EDFRecording:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            fixed = stream.read(256)
            if len(fixed) != 256 or fixed[:8] != b'0       ':
                raise ValueError('expected 16-bit EDF header')
            count = int(fixed[252:256])
            self.header_bytes = int(fixed[184:192])
            self.records = int(fixed[236:244])
            self.record_seconds = float(fixed[244:252])
            if not 1 <= count <= 4096 or self.header_bytes != (count+1)*256:
                raise ValueError('unsupported EDF header layout')
            if self.records <= 0 or not math.isfinite(self.record_seconds) or self.record_seconds <= 0:
                raise ValueError('known positive EDF record count/duration required')
            self.start = datetime.strptime((fixed[168:176]+fixed[176:184]).decode('ascii'), '%d.%m.%y%H.%M.%S')
            header = stream.read(self.header_bytes - 256)
        if len(header) != self.header_bytes - 256:
            raise ValueError('truncated EDF signal header')
        def field(offset, width, cast):
            return [cast(header[offset+i*width:offset+(i+1)*width].decode('ascii').strip()) for i in range(count)]
        labels = field(0, 16, str)
        digital_min = field(count*(16+80+8+8+8), 8, int)
        digital_max = field(count*(16+80+8+8+8+8), 8, int)
        samples = field(count*(16+80+8+8+8+8+8+80), 8, int)
        if any(value <= 0 for value in samples):
            raise ValueError('nonpositive EDF sample count')
        self.channels = []
        offset = 0
        for label, size, lo, hi in zip(labels, samples, digital_min, digital_max):
            if lo >= hi or lo < -32768 or hi > 32767:
                raise ValueError('invalid EDF digital range')
            self.channels.append(Channel(label, size, offset, lo, hi))
            offset += size*2
        self.record_bytes = offset
        self.duration_seconds = self.records*self.record_seconds
        if self.path.stat().st_size != self.header_bytes + self.records*self.record_bytes:
            raise ValueError('EDF size inconsistent with declared records')

    def channel(self, name):
        matches = [channel for channel in self.channels if channel.label == name]
        if len(matches) != 1:
            raise ValueError(f'expected one unambiguous {name!r} channel')
        return matches[0]

    def sample_rate(self, name):
        return self.channel(name).samples_per_record/self.record_seconds

    def read_window(self, name, start_seconds, duration_seconds):
        """Read sample-grid aligned windows, scaled by fixed ADC full scale.

        This hardware conversion uses no fitted population statistics. It is not
        physical sound pressure or loudness calibration. No whole-hour load.
        """
        if not all(math.isfinite(v) for v in (start_seconds, duration_seconds)) or start_seconds < 0 or duration_seconds <= 0:
            raise ValueError('invalid audio interval')
        if start_seconds + duration_seconds > self.duration_seconds:
            raise ValueError('audio interval exceeds EDF segment')
        channel = self.channel(name)
        rate = self.sample_rate(name)
        first, length = start_seconds*rate, duration_seconds*rate
        if abs(first-round(first)) > 1e-6 or abs(length-round(length)) > 1e-6:
            raise ValueError('window must align with source sample grid')
        first, length = round(first), round(length)
        if length <= 0:
            raise ValueError('empty source audio interval')
        output = np.empty(length, dtype=np.float32)
        written = 0
        with self.path.open('rb') as stream:
            while written < length:
                position = first + written
                record, within = divmod(position, channel.samples_per_record)
                take = min(length-written, channel.samples_per_record-within)
                stream.seek(self.header_bytes + record*self.record_bytes + channel.offset_bytes + within*2)
                chunk = stream.read(take*2)
                if len(chunk) != take*2:
                    raise ValueError('truncated EDF audio samples')
                values = np.frombuffer(chunk, dtype='<i2')
                if np.any(values < channel.digital_min) or np.any(values > channel.digital_max):
                    raise ValueError('EDF samples exceed declared digital range')
                output[written:written+take] = values
                written += take
        output /= max(abs(channel.digital_min), abs(channel.digital_max))
        return output
