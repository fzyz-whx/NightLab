"""Inspect verified SleepAccel files without guessing label semantics."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from .cli import ROOT, load_manifest, raw_path
from .sleepaccel_labels import normalize_stage


def inspect_rows(text, kind):
    if kind not in {'labels', 'heart_rate', 'motion'}:
        raise ValueError('unsupported signal kind')
    rows = []
    expected = 4 if kind == 'motion' else 2
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split(',') if kind == 'heart_rate' else line.split()
        if len(fields) != expected:
            raise ValueError(f'line {number}: expected {expected} columns')
        row = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f'line {number}: nonfinite value')
        if rows and row[0] <= rows[-1][0]:
            raise ValueError(f'line {number}: timestamp must increase')
        if kind == 'labels' and not row[1].is_integer():
            raise ValueError(f'line {number}: noninteger label')
        rows.append(row)
    if len(rows) < 2:
        raise ValueError('at least two rows required')
    gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    report = {'rows': len(rows), 'columns': expected,
              'time_start_seconds': rows[0][0], 'time_end_seconds': rows[-1][0],
              'gap_min_seconds': min(gaps), 'gap_median_seconds': statistics.median(gaps),
              'gap_max_seconds': max(gaps), 'before_psg_rows': sum(r[0] < 0 for r in rows)}
    if kind == 'labels':
        report['raw_label_counts'] = dict(sorted(Counter(int(r[1]) for r in rows).items()))
        normalized = [normalize_stage(int(r[1])) for r in rows]
        report['semantic_mapping_applied'] = True
        report['mapping_version'] = 'sleepaccel-five-stage-v1'
        report['valid_target_rows'] = sum(r['target_valid'] for r in normalized)
        report['normalized_stage_counts'] = dict(Counter(r['stage'] for r in normalized if r['target_valid']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('labels', 'heart_rate', 'motion'), required=True)
    args = parser.parse_args()
    manifest = load_manifest(ROOT / 'data/manifests/sleepaccel.json', ROOT)
    results = []
    try:
        for entry in manifest['files']:
            if Path(entry['path']).parent.name != args.kind:
                continue
            path = raw_path(ROOT, entry['path'])
            content = path.read_bytes()
            expected_hash = entry.get('official_checksums', {}).get('sha256')
            if not expected_hash or hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError('official SHA-256 required and must match before inspection')
            results.append({'file': entry['path'], 'official_sha256_verified': True,
                            **inspect_rows(content.decode('utf-8'), args.kind)})
        if not results:
            raise ValueError('no matching files in manifest')
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
