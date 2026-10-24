"""Audit original annotations and range-fetched WAV headers; never infer offsets."""
import argparse
from collections import Counter
import hashlib
import json
import math
import struct

from .cli import ROOT, raw_path


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite numeric timestamp or duration')
    return float(value)


def annotation_summary(document, duration=None):
    start = number(document['record_start'])
    events, intervals = document['events'], document['awake_intervals']
    normalized = []
    for event in events:
        # This misspelling is present in the actual V5 source file.
        if 'evnet_start' in event and 'event_start' in event and event['evnet_start'] != event['event_start']:
            raise ValueError('conflicting start fields')
        event_start = number(event['evnet_start'] if 'evnet_start' in event else event['event_start'])
        event_duration = number(event['event_duration'])
        if event_duration <= 0:
            raise ValueError('event duration must be positive')
        if not isinstance(event['event_type'], str) or not isinstance(event['sleep_stage'], str):
            raise ValueError('event type and stage must be source strings')
        normalized.append((event_start - start, event_start - start + event_duration))
    awake = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError('awake interval must have two endpoints')
        left, right = map(number, interval)
        if right <= left:
            raise ValueError('awake interval end must follow start')
        awake.append((left, right))
    report = {'record_start_seconds': start, 'event_count': len(events),
              'source_event_keys': sorted({key for event in events for key in event}),
              'raw_event_type_counts': dict(Counter(event['event_type'] for event in events)),
              'raw_stage_counts': dict(Counter(event['sleep_stage'] for event in events)),
              'awake_interval_count': len(awake), 'duplicate_awake_intervals': len(awake) - len(set(awake)),
              'relative_first_event': min((left for left, _ in normalized), default=None),
              'relative_last_event_end': max((right for _, right in normalized), default=None),
              'alignment_assumption': 'event absolute seconds minus record_start; not an independently verified audio offset'}
    if duration is not None:
        duration = number(duration)
        if duration <= 0:
            raise ValueError('audio duration must be positive')
        outside = sum(left < 0 or right > duration for left, right in normalized)
        report.update(audio_header_duration_seconds=duration,
                      events_outside_audio_under_stated_assumption=outside,
                      alignment_status='conflict' if outside else 'bounds consistent; synchronization still unverified')
        if normalized:
            span = max(right for _, right in normalized) - min(left for left, _ in normalized)
            report.update(annotated_event_span_seconds=span,
                          all_events_fit_under_any_constant_offset=span <= duration)
    return report


def wav_header_summary(header):
    if len(header) < 12 or header[:4] != b'RIFF' or header[8:12] != b'WAVE':
        raise ValueError('expected RIFF/WAVE header')
    offset, fmt = 12, None
    while offset + 8 <= len(header):
        tag, size = struct.unpack_from('<4sI', header, offset)
        offset += 8
        if tag == b'fmt ':
            if size < 16 or offset + 16 > len(header):
                raise ValueError('truncated fmt chunk')
            fmt = struct.unpack_from('<HHIIHH', header, offset)
        elif tag == b'data':
            if fmt is None:
                raise ValueError('data appears before fmt')
            encoding, channels, rate, byte_rate, block, bits = fmt
            if encoding != 1 or min(channels, rate, byte_rate, block, bits) <= 0:
                raise ValueError('only positive PCM format supported in header audit')
            if bits % 8 or block != channels * bits // 8 or byte_rate != rate * block or size % block:
                raise ValueError('inconsistent PCM header')
            return {'sample_rate_hz': rate, 'channels': channels, 'bits_per_sample': bits,
                    'duration_seconds': size / byte_rate, 'data_bytes': size,
                    'declared_file_bytes': struct.unpack_from('<I', header, 4)[0] + 8,
                    'data_offset': offset, 'full_audio_downloaded_or_verified': False}
        offset += size + size % 2
    raise ValueError('data chunk absent from available header bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=('01', '02'), default='01')
    args = parser.parse_args()
    try:
        catalog = json.loads((ROOT / 'data/catalogs/sciencedb-19070-v5.json').read_text())
        name = f'{args.subject}_annotation.json'
        entry = next(file for file in catalog['files'] if file['fileName'] == name)
        prefix = f'data/raw/sciencedb-19070/v5/Data/{args.subject}/{args.subject}'
        content = raw_path(ROOT, prefix + '_annotation.json').read_bytes()
        if len(content) != entry['size'] or hashlib.md5(content, usedforsecurity=False).hexdigest() != entry['md5']:
            raise ValueError('annotation size or official MD5 mismatch')
        audio = wav_header_summary(raw_path(ROOT, prefix + '_phone.header').read_bytes())
        result = {'subject': args.subject, 'annotation_official_md5_verified': True,
                  'annotation_local_sha256': hashlib.sha256(content).hexdigest(),
                  'audio_header_only': audio,
                  'annotation': annotation_summary(json.loads(content), audio['duration_seconds'])}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result['annotation'].get('alignment_status') == 'conflict' else 0
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
