"""Inspect verified PSG-Audio metadata without exporting patient fields or signals."""
import array
from collections import Counter
from datetime import datetime
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET

from .cli import ROOT, load_manifest, raw_path, verify


def local_tag(element):
    return element.tag.rsplit('}', 1)[-1]


def read_header(stream):
    fixed = stream.read(256)
    if len(fixed) != 256 or fixed[:8] != b'0       ':
        raise ValueError('expected EDF (16-bit) fixed header')
    signals = int(fixed[252:256])
    header_bytes = int(fixed[184:192])
    records = int(fixed[236:244])
    duration = float(fixed[244:252])
    if not 1 <= signals <= 4096 or header_bytes != 256 * (signals + 1):
        raise ValueError('inconsistent or unsupported EDF channel header')
    if records <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('known positive record count and duration required')
    channel_header = stream.read(header_bytes - 256)
    if len(channel_header) != header_bytes - 256:
        raise ValueError('truncated channel header')
    labels = [channel_header[i*16:(i+1)*16].decode('ascii').strip() for i in range(signals)]
    offset = signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples = [int(channel_header[offset+i*8:offset+(i+1)*8]) for i in range(signals)]
    if any(count <= 0 for count in samples):
        raise ValueError('nonpositive channel sample count')
    return {'header_bytes': header_bytes, 'records': records, 'record_duration': duration,
            'labels': labels, 'samples': samples, 'record_bytes': sum(samples)*2,
            # Kept internal for local alignment; do not include in public report.
            '_start': datetime.strptime((fixed[168:176]+fixed[176:184]).decode(), '%d.%m.%y%H.%M.%S')}


def read_rml(root):
    def text(name):
        found = [element.text for element in root.iter() if local_tag(element) == name]
        if len(found) != 1:
            raise ValueError(f'expected one {name}')
        return found[0]
    origin = datetime.fromisoformat(text('RecordingStart'))
    segments = []
    for segment in root.iter():
        if local_tag(segment) == 'Segment':
            values = {local_tag(child): child.text for child in segment}
            offset = (datetime.fromisoformat(values['StartTime']) - origin).total_seconds()
            duration = float(values['Duration'])
            if not math.isfinite(offset) or offset < 0 or not math.isfinite(duration) or duration <= 0:
                raise ValueError('invalid RML segment interval')
            segments.append((offset, duration))
    events = []
    for event in root.iter():
        if local_tag(event) == 'Event' and event.get('Family') == 'Respiratory':
            start, duration = float(event.get('Start')), float(event.get('Duration'))
            if not all(math.isfinite(value) for value in (start, duration)) or start < 0 or duration <= 0:
                raise ValueError('invalid respiratory event interval')
            events.append({'start': start, 'end': start + duration, 'type': event.get('Type')})
    staging = {}
    for name in ('UserStaging', 'MachineStaging'):
        groups = [element for element in root.iter() if local_tag(element) == name]
        if len(groups) != 1:
            raise ValueError(f'expected one {name}')
        staging[name] = [stage for stage in groups[0].iter() if local_tag(stage) == 'Stage']
    return origin, segments, events, staging


def inspect(edf_path, rml_path):
    origin, segments, events, staging = read_rml(ET.parse(rml_path).getroot())
    with edf_path.open('rb') as stream:
        header = read_header(stream)
        if edf_path.stat().st_size != header['header_bytes'] + header['records'] * header['record_bytes']:
            raise ValueError('EDF payload size inconsistent with channel records')
        offset = (header['_start'] - origin).total_seconds()
        duration = header['records'] * header['record_duration']
        if (offset, duration) not in segments:
            raise ValueError('EDF clock/duration does not match an RML segment')
        contained = [event for event in events if offset <= event['start'] and event['end'] <= offset + duration]
        if not contained:
            raise ValueError('no fully contained respiratory events for signal inspection')
        record = int((contained[0]['start'] - offset) // header['record_duration'])
        audio = []
        for channel in ('Mic', 'Tracheal'):
            if header['labels'].count(channel) != 1:
                raise ValueError(f'expected exactly one {channel} signal')
            index = header['labels'].index(channel)
            rate = header['samples'][index] / header['record_duration']
            if rate < 8000:
                raise ValueError('audio channel rate too low for this inspection')
            stream.seek(header['header_bytes'] + record*header['record_bytes'] + sum(header['samples'][:index])*2)
            signal = array.array('h')
            signal.frombytes(stream.read(header['samples'][index]*2))
            if sys.byteorder != 'little':
                signal.byteswap()
            if len(signal) != header['samples'][index]:
                raise ValueError('truncated audio record')
            audio.append({'channel': channel, 'sample_rate_hz': rate, 'inspected_samples': len(signal),
                          'record_start_seconds_within_segment': record * header['record_duration'],
                          'rms_digital': math.sqrt(sum(value*value for value in signal)/len(signal)),
                          'peak_digital': max(abs(value) for value in signal), 'all_zero': not any(signal)})
    return {'signals': len(header['labels']), 'duration_seconds': duration,
            'offset_from_rml_start_seconds': offset, 'clock_and_duration_match_rml_segment': True,
            'respiratory_events_in_complete_rml': len(events),
            'fully_contained_respiratory_events': len(contained),
            'contained_event_types': dict(Counter(event['type'] for event in contained)),
            'staging_change_counts': {name: len(stages) for name, stages in staging.items()},
            'reference_staging_source': 'UserStaging only; MachineStaging kept separate', 'audio_record_checks': audio,
            'scope': 'one verified segment; structural clock alignment and short signal check, not medical/event accuracy validation'}


def main():
    try:
        manifest = load_manifest(ROOT / 'data/manifests/psg-audio.json', ROOT)
        integrity = verify(ROOT, manifest)
        if not integrity['ok']:
            raise ValueError('all selected PSG files must pass integrity checks first')
        edf = raw_path(ROOT, next(entry['path'] for entry in manifest['files'] if entry['path'].endswith('.edf')))
        rml = raw_path(ROOT, next(entry['path'] for entry in manifest['files'] if entry['path'].endswith('.rml')))
        print(json.dumps(inspect(edf, rml), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, StopIteration, ET.ParseError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
