"""Independent, one-worker quality audit. Default is metadata-only and read-only.

No feature caches, training, sorting, interpolation, or nonidentical repair.
Report contains counts/durations only, never raw samples or absolute timestamps.
"""
import argparse
import ctypes
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from threadpoolctl import threadpool_limits
from nightlab.wearable.features import _parse_fields, _validate_rows, _coverage, STAGES
from .cli import ROOT, load_manifest, raw_path
from .sleepaccel_labels import normalize_stage

MANIFEST = 'data/manifests/sleepaccel-cohort-31.json'
REPORT = 'data/processed/sleepaccel-cohort-31/quality-audit.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def available_memory():
    if os.name != 'nt':
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in ('total_phys', 'available_phys',
             'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('cannot inspect available RAM')
    return int(status.available_phys)


def read_rows(path, kind):
    """Bound Python list/text overhead, preserving original row order exactly."""
    size = path.stat().st_size
    # Arrays plus concatenation and parser overhead, with 512 MiB left for peers.
    if available_memory() < size * 4 + 512 * 1024**2:
        raise MemoryError('insufficient available RAM for conservative file parsing budget')
    chunks = []
    with path.open(encoding='utf-8') as stream:
        while lines := list(itertools.islice(stream, 20000)):
            if available_memory() < 256 * 1024**2:
                raise MemoryError('available RAM fell below reserve while parsing')
            chunks.append(_parse_fields(''.join(lines), kind))
    if not chunks:
        return np.empty((0, 4 if kind == 'motion' else 2))
    return np.concatenate(chunks)


def validate_signal(rows, kind):
    strict_error = None
    try:
        _validate_rows(rows, kind)
    except ValueError as exc:
        strict_error = str(exc)
    cleaned = _validate_rows(rows, kind, repeated_block_policy='collapse_identical')
    summary = {'raw_rows': len(rows), 'validated_rows': len(cleaned),
               'strict_original_valid': strict_error is None, 'strict_original_error': strict_error,
               'identical_blocks': len(rows) // len(cleaned),
               'collapsed_identical_rows': len(rows) - len(cleaned)}
    # A collapsed view would otherwise retain the entire original backing array.
    return (cleaned.copy() if len(cleaned) != len(rows) else cleaned), summary


def raw_diagnostics(rows, labels):
    """Describe rejected order without sorting, repairing, or suggesting usability."""
    result = {'validated': False, 'raw_rows': len(rows)}
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        result['finite_nonempty_rows'] = False
        return result
    times = rows[:, 0]
    differences = np.diff(times)
    result.update(finite_nonempty_rows=True,
                  duration_range_seconds=float(times.max() - times.min()),
                  negative_time_steps=int((differences < 0).sum()),
                  equal_time_steps=int((differences == 0).sum()),
                  raw_interval_seconds={'minimum': float(differences.min()) if len(differences) else None,
                                        'median': float(np.median(differences)) if len(differences) else None,
                                        'maximum': float(differences.max()) if len(differences) else None})
    if np.any(differences <= 0):
        boundaries = np.r_[0, np.flatnonzero(differences <= 0) + 1, len(rows)]
        origin = labels[0, 0] if labels is not None else times[0]
        result['segment_time_origin'] = 'seconds relative to first label; deidentified, not wall-clock' if labels is not None else 'seconds relative to first signal sample'
        result['strictly_increasing_segments'] = [
            {'rows': int(right-left), 'start_seconds': float(times[left]-origin),
             'end_seconds': float(times[right-1]-origin)}
            for left, right in zip(boundaries[:-1], boundaries[1:])]
        reverse_indices = np.flatnonzero(differences < 0)
        result['reversal_magnitude_seconds'] = {
            'minimum': float(-differences[reverse_indices].max()) if len(reverse_indices) else None,
            'maximum': float(-differences[reverse_indices].min()) if len(reverse_indices) else None}
        if labels is not None:
            overlaps = (times[reverse_indices] >= labels[0, 0]) & (times[reverse_indices+1] < labels[-1, 0]+30)
            result['reversal_spans_intersecting_label_range'] = int(overlaps.sum())
        # Diagnostic indexing only: never reorder rows or feed these groups to features.
        unique_times, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
        duplicated = counts > 1
        conflicting = np.zeros(len(unique_times), dtype=bool)
        for column in range(1, rows.shape[1]):
            low = np.full(len(unique_times), np.inf)
            high = np.full(len(unique_times), -np.inf)
            np.minimum.at(low, inverse, rows[:, column])
            np.maximum.at(high, inverse, rows[:, column])
            conflicting |= low != high
            del low, high
        in_labels = ((unique_times >= labels[0, 0]) & (unique_times < labels[-1, 0]+30)) if labels is not None else np.zeros(len(unique_times), dtype=bool)
        result['repeated_timestamp_groups'] = {
            'groups': int(duplicated.sum()), 'extra_rows': int((counts[duplicated]-1).sum()),
            'identical_values': int((duplicated & ~conflicting).sum()),
            'conflicting_values': int((duplicated & conflicting).sum()),
            'inside_label_range': int((duplicated & in_labels).sum()),
            'conflicting_inside_label_range': int((duplicated & conflicting & in_labels).sum()),
            'identical_inside_label_range': int((duplicated & ~conflicting & in_labels).sum())}
        reset_boundaries = np.r_[0, reverse_indices+1, len(rows)]
        lengths = np.diff(reset_boundaries)
        full_equal = bool(len(lengths)>1 and np.all(lengths==lengths[0]) and
            all(np.array_equal(rows[:lengths[0]],rows[left:right])
                for left,right in zip(reset_boundaries[1:-1],reset_boundaries[2:])))
        result['all_reset_blocks_exactly_identical'] = full_equal
        result['clock_interpretation'] = 'No clock offset inference: repeated timestamp conflict is reported, not resolved.'
        del unique_times, inverse, counts, conflicting, duplicated, in_labels
    del differences
    if labels is not None:
        # Assign samples to existing contiguous label bins in their ORIGINAL order.
        # Counts on invalid signals are diagnostic only; repeated rows are retained.
        inside = (times >= labels[0, 0]) & (times < labels[-1, 0] + 30)
        indices = ((times[inside] - labels[0, 0]) // 30).astype(np.int64)
        counts = np.bincount(indices, minlength=len(labels))
        counts = counts[labels[:, 1] != -1]
        result['raw_scored_epoch_presence'] = {
            'epochs': len(counts), 'no_samples': int((counts == 0).sum()),
            'sample_count_min': int(counts.min()) if len(counts) else None,
            'sample_count_max': int(counts.max()) if len(counts) else None,
            'scope': 'Diagnostic presence only; original order invalid and repeated rows retained; not usable feature coverage'}
    return result


def audit_subject(root, subject, entries):
    result = {'subject_id': subject, 'valid': False, 'files': {}, 'signals': {}, 'errors': []}
    good = {}
    for kind, entry in entries.items():
        state = {'size_verified': False, 'official_sha256_verified': False}
        result['files'][kind] = state
        try:
            path = raw_path(root, entry['path'])
            state['size_verified'] = path.stat().st_size == entry['exact_bytes']
            state['official_sha256_verified'] = sha256(path) == entry['official_checksums']['sha256']
            if not state['size_verified'] or not state['official_sha256_verified']:
                raise ValueError('raw size or official SHA256 mismatch')
            good[kind] = path
        except (OSError, ValueError) as exc:
            result['errors'].append(f'{kind}: {exc}')
    labels = None
    try:
        labels = _validate_rows(read_rows(good['labels'], 'labels'), 'labels')
        raw_codes, counts = np.unique(labels[:, 1].astype(int), return_counts=True)
        normalized = {int(code): normalize_stage(int(code)) for code in raw_codes}
        assert all(STAGES[code] == item['stage'] for code, item in normalized.items())
        valid = labels[:, 1] != -1
        starts = labels[valid, 0].copy()
        result['labels'] = {'epochs': len(labels), 'scored_epochs': int(valid.sum()),
                            'unscored_epochs': int((~valid).sum()),
                            'raw_stage_counts': {str(k): int(v) for k, v in zip(raw_codes, counts)},
                            'unknown_codes': [], 'contiguous_30_second_epochs': True,
                            'mapping_version': 'sleepaccel-five-stage-v1',
                            'normalized_stage_counts': {stage: sum(int(n) for code, n in zip(raw_codes, counts)
                             if normalized[int(code)]['stage'] == stage) for stage in ('W','N1','N2','N3','REM')}}
        if not valid.any():
            result['errors'].append('labels: no scored epochs')
    except (KeyError, OSError, ValueError) as exc:
        result['errors'].append(f'labels: {exc}')
        starts = None
    for kind in ('heart_rate', 'motion'):
        original = rows = times = None
        try:
            original = read_rows(good[kind], kind)
            result['signals'][kind] = raw_diagnostics(original, labels)
            rows, summary = validate_signal(original, kind)
            summary['validated'] = True
            summary['raw_diagnostics'] = result['signals'][kind]
            del original
            times = rows[:, 0]
            intervals = np.diff(times)
            summary.update(duration_seconds=float(times[-1] - times[0]),
                           interval_seconds={'minimum': float(intervals.min()) if len(intervals) else None,
                                             'median': float(np.median(intervals)) if len(intervals) else None,
                                             'maximum': float(intervals.max()) if len(intervals) else None})
            del intervals
            if starts is not None:
                _, _, coverage = _coverage(times, starts, starts + 30)
                count = coverage['sample_count']
                summary['scored_epoch_coverage'] = {
                    'epochs': len(starts), 'no_samples': int((count == 0).sum()),
                    'sample_count_min': int(count.min()) if len(count) else None,
                    'sample_count_median': float(np.median(count)) if len(count) else None,
                    'sample_count_max': int(count.max()) if len(count) else None,
                    'maximum_gap_including_edges_seconds': float(coverage['maximum_gap_including_edges_seconds'].max()) if len(count) else None,
                    'samples_before_label_range': int(np.searchsorted(times, labels[0, 0])),
                    'samples_after_label_range': int(len(times) - np.searchsorted(times, labels[-1, 0] + 30)),
                    'definition': 'Available samples in [epoch,epoch+30); gaps include edges; no imputation or exclusion'}
                del coverage, count
            result['signals'][kind] = summary
            del times, rows
        except MemoryError:
            raise
        except (KeyError, OSError, ValueError) as exc:
            result['errors'].append(f'{kind}: {exc}')
        finally:
            # Rejected originals must be released before reading the next signal.
            original = rows = times = None
            gc.collect()
    result['valid'] = not result['errors']
    return result


def metadata(root):
    manifest = load_manifest(root / MANIFEST, root)
    plan_path = root / 'data/plans/sleepaccel-expansion-options.json'
    source = json.loads(plan_path.read_text(encoding='utf-8'))
    receipt_path = root / manifest['acquisition']['intake_receipt']
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert sha256(plan_path) == manifest['acquisition']['source_plan_sha256']
    assert sha256(receipt_path) == manifest['acquisition']['intake_receipt_sha256']
    assert receipt['complete'] and not receipt['dry_run']
    assert receipt['archive']['sha256_local'] == manifest['acquisition']['archive_sha256_local']
    assert manifest['subjects']['subjects'] == source['options']['all31']['proposed_split']['subjects']
    split = manifest['subjects']['subjects']
    ids = [value for values in split.values() for value in values]
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [18, 6, 7]
    assert len(ids) == len(set(ids)) == 31
    assert set(source['already_explored_subjects']) <= set(split['train'])
    expected = {'data/raw/sleepaccel/1.0.0/' + f['relative_source_path']: f for f in source['files']}
    intake = {f['path']: f for f in receipt['files']}
    grouped = {subject: {} for subject in ids}
    assert len(manifest['files']) == len(expected) == len(intake) == 93
    for entry in manifest['files']:
        original, admitted = expected[entry['path']], intake[entry['path']]
        assert entry['exact_bytes'] == original['exact_bytes'] == admitted['bytes']
        assert entry['official_checksums']['sha256'] == original['official_checksums']['sha256'] == admitted['sha256']
        assert entry['source'] == original['source'] and entry['source_verified'] is True
        assert entry['subject_id'] in split[entry['split']]
        assert entry['kind'] not in grouped[entry['subject_id']]
        grouped[entry['subject_id']][entry['kind']] = entry
    assert all(set(entries) == {'labels', 'heart_rate', 'motion'} for entries in grouped.values())
    return manifest, grouped


def run(root=ROOT, execute=False):
    root = Path(root)
    manifest, grouped = metadata(root)
    report = {'schema_version': 1, 'dataset': manifest['id'], 'dry_run': not execute,
              'complete': False, 'all_valid': False, 'expected_subjects': 31,
              'audited_subjects': 0, 'subjects': [], 'errors': [],
              'manifest_sha256': sha256(root / MANIFEST),
              'intake_receipt_sha256': manifest['acquisition']['intake_receipt_sha256'],
              'audit_source_sha256': sha256(Path(__file__)),
              'resource_policy': {'workers': 1, 'cpu_threads': 1, 'gpu': False},
              'scope': 'Raw integrity, parser and timestamp validity, label mapping, identical repeats and report-only signal coverage; no training or clinical validation'}
    if not execute:
        return report
    begin = time.monotonic()
    with threadpool_limits(limits=1):
        for index, (subject, entries) in enumerate(grouped.items(), 1):
            print(f'audit {index}/31 subject={subject} start', file=sys.stderr, flush=True)
            try:
                item = audit_subject(root, subject, entries)
            except MemoryError as exc:
                report['errors'].append(f'aborted before completing subject {subject}: {exc}')
                break
            report['subjects'].append(item)
            report['audited_subjects'] += 1
            print(f'audit {index}/31 subject={subject} valid={item["valid"]} errors={len(item["errors"])}', file=sys.stderr, flush=True)
    report['complete'] = report['audited_subjects'] == 31
    report['all_valid'] = report['complete'] and all(item['valid'] for item in report['subjects']) and not report['errors']
    report['elapsed_seconds'] = round(time.monotonic() - begin, 3)
    target = root / REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='verify raw files and write local ignored summary')
    args = parser.parse_args()
    report = run(execute=args.execute)
    print(json.dumps({key: value for key, value in report.items() if key != 'subjects'}, indent=2))
    return 0 if not args.execute or report['all_valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
