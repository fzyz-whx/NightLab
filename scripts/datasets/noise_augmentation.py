"""Prepare one deterministic noisy copy per training target; default is read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from importlib.metadata import version as package_version

import numpy as np
from threadpoolctl import threadpool_limits

from . import noise_features as base
from .cli import ROOT
from .deep_cache import _load, _same, _write_index
from .dense_audio import sha256
from .prepare import output_path

VERSION = 'noise-augmentation-v1'
EXPECTED_TOTAL_ROWS = 3775
CONDITIONS = tuple((kind, snr) for kind in base.KINDS for snr in base.SNRS)


def condition_index(record_id, start_sample):
    """Select on the native sample grid before consulting any target/mask."""
    payload = json.dumps([record_id, int(start_sample), 42]).encode('utf-8')
    return int(hashlib.sha256(payload).hexdigest(), 16) % len(CONDITIONS)


def _sources(root):
    return {**base._sources(root), 'noise_augmentation': sha256(__file__),
            'augmentation_protocol': sha256(root / 'docs/NOISE_AUGMENTATION_PROTOCOL.md')}


def _select(root):
    manifest, corpus, _ = base._select(root, 'test')
    subjects = set(base.FROZEN_SUBJECTS['train'])
    records = [record for record in corpus['records'] if record['record_id'].split('-')[0] in subjects]
    if (len(records) != 6 or {r['record_id'].split('-')[0] for r in records} != subjects
            or sum(record['rows'] for record in records) != EXPECTED_TOTAL_ROWS):
        raise ValueError('exactly six frozen train records and 3775 supervised rows required')
    return manifest, corpus, records


def _allocate(record_id, rate, expected):
    # Hash ALL candidate windows first; labels do not enter this selector.
    choices = np.asarray([condition_index(record_id, round(float(start) * rate))
                          for start in expected['start_seconds']], dtype=np.int16)
    indices = np.flatnonzero(expected['supervised_mask']).astype(np.int64)
    selected = choices[indices]
    rows = len(indices)
    arrays = {key: expected[key][indices].copy() for key in ('y', 'subject_id', 'start_seconds', 'end_seconds')}
    arrays.update(source_window_index=indices, kind=np.asarray([CONDITIONS[i][0] for i in selected]),
                  snr_db=np.asarray([CONDITIONS[i][1] for i in selected], dtype=np.int16),
                  common_gain=np.empty(rows, np.float64), X=np.empty((rows, 67), np.float32),
                  values=np.empty((rows, 32, 100), np.float32))
    return arrays, selected


def _validate(arrays, expected, gains):
    _same(arrays, {key: value for key, value in expected.items() if key not in ('X', 'values', 'common_gain')},
          'augmentation inputs')
    rows = len(expected['y'])
    if set(arrays) != set(expected) or any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('augmentation fields invalid')
    for key, shape in (('X', (rows, 67)), ('values', (rows, 32, 100))):
        if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError('augmentation feature geometry invalid')
    for key, dtype in (('source_window_index', np.int64), ('snr_db', np.int16), ('common_gain', np.float64)):
        if arrays[key].dtype != dtype or arrays[key].shape != (rows,):
            raise ValueError('augmentation dtype/shape invalid')
    if any(arrays[key].dtype.kind != 'U' for key in ('y', 'subject_id', 'kind')):
        raise ValueError('augmentation identity/label strings required')
    expected_gains = np.asarray([gains[(kind, int(snr))] for kind, snr in zip(arrays['kind'], arrays['snr_db'], strict=True)])
    if not np.array_equal(arrays['common_gain'], expected_gains):
        raise ValueError('augmentation common gain differs')


def _fill_condition(observed, calibration, expected, choices, condition, arrays, filled):
    positions = {int(source): row for row, source in enumerate(arrays['source_window_index']) if choices[row] == condition}
    scale,gain = observed._checked_parameters(calibration)
    # Only the unfiltered noisy branch is used for augmentation. Its samples
    # are a pointwise function of the same whole-record arrays/alpha/g, so a
    # selected slice is exactly the streamed branch, without rerunning filters.
    for index,row in positions.items():
        begin = round(float(expected['start_seconds'][index])*observed.source_rate)
        end = round(float(expected['end_seconds'][index])*observed.source_rate)
        if begin < 0 or end <= begin or end > len(observed.clean):
            raise ValueError('augmentation window geometry differs')
        if filled[row]:
            raise ValueError('duplicate augmentation output position')
        original = np.asarray(observed.clean[begin:end],dtype=np.float64)
        interference = np.asarray(observed.noise[begin:end],dtype=np.float64)
        noisy = (gain*(original+scale*interference)).astype(np.float32)
        if not np.isfinite(noisy).all() or np.max(np.abs(noisy)) > 1:
            raise ValueError('augmentation sample clipping or nonfinite values')
        classic,image = base.representations.joint_features(noisy,observed.source_rate)
        arrays['X'][row],arrays['values'][row] = classic,image
        arrays['common_gain'][row] = gain
        filled[row] = True


def _generate(source, rate, record_id, expected, arrays, choices, budget, reuse=False):
    filled = np.zeros(len(choices), bool)
    conditions, kinds = [], {}
    for kind in base.KINDS:
        selected_conditions = [i for i, (name, _) in enumerate(CONDITIONS) if name == kind and np.any(choices == i)]
        if not selected_conditions:
            continue
        if base.noise_audit.available_memory() < budget - source.nbytes:
            raise ValueError('insufficient free RAM before noise allocation')
        seed = base.noise.noise_seed(record_id, 'train', kind)
        interference = base.noise.synthetic_noise(len(source), rate, kind, seed)
        digest = hashlib.sha256(memoryview(interference)).hexdigest()
        kinds[kind] = {'seed': seed, 'noise_sha256': digest}
        observed = base.noise.NoiseRecording(source, interference, rate)
        for index in selected_conditions:
            _, snr = CONDITIONS[index]
            print(f'{record_id} augmentation {kind} SNR {snr}: {int(np.count_nonzero(choices == index))} selected windows',
                  file=sys.stderr,flush=True)
            calibration = observed.calibrate(snr)
            actual = observed.measure_actual_snr(calibration)
            if not np.isfinite(actual) or abs(actual - snr) > 1e-4:
                raise ValueError('actual rounded mixture SNR exceeds tolerance')
            conditions.append({'kind': kind, 'snr_db': snr, 'calibration': calibration,
                               'actual_rounded_analysis_snr_db': actual})
            if not reuse:
                _fill_condition(observed, calibration, expected, choices, index, arrays, filled)
        if hashlib.sha256(memoryview(interference)).hexdigest() != digest:
            raise ValueError('noise input modified')
        del observed, interference
    if not reuse and not filled.all():
        raise ValueError('augmentation output incomplete')
    return kinds, conditions


def prepare_augmentation(root, execute=False):
    root = Path(root).resolve()
    folder = output_path(root, f'data/processed/{base.DATASET}/{VERSION}')
    index = {'dataset': base.DATASET, 'dataset_version': 'V3', 'version': VERSION,
             'complete': False, 'split_policy': None, 'records': [],
             'environment': {'python': platform.python_version(),
                             **{name: package_version(name) for name in ('numpy', 'scipy')}}}
    previous = None
    if execute:
        if (folder / 'corpus.json').exists():
            previous = json.loads((folder / 'corpus.json').read_text(encoding='utf-8'))
        folder.mkdir(parents=True, exist_ok=True)
        # Retain existing checksums while invalidating publication, including on
        # a failed retry. Otherwise a second retry could lose tamper evidence.
        _write_index(folder / 'corpus.json', {**(previous or index), 'complete': False})
    manifest, corpus, records = _select(root)
    sources = _sources(root)
    index.update(split_policy=corpus['split_policy'], source_hashes=sources)
    if not execute:
        return {**index, 'mode': 'augmentation dry-run', 'writes': False, 'network': False,
                'workers': 1, 'numeric_threads': 2,
                'records': [{'record_id': r['record_id'], 'split': 'train', 'rows': r['rows']} for r in records]}
    if previous and previous.get('records') and (previous.get('source_hashes') != sources
            or previous.get('environment') != index['environment'] or previous.get('split_policy') != index['split_policy']):
        raise ValueError('augmentation index provenance differs; refusing overwrite')
    previous_records = {item['record_id']: item for item in (previous or {}).get('records', [])}
    if (len(previous_records) != len((previous or {}).get('records', []))
            or not set(previous_records).issubset({r['record_id'] for r in records})):
        raise ValueError('augmentation index contains duplicate/non-train records')
    index['records'] = list(previous_records.values())
    _write_index(folder / 'corpus.json', index)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for record in records:
            tick = time.perf_counter()
            recording, metadata, provenance, expected, scored = base._verified_context(root, manifest, record)
            identity = record['record_id']
            if provenance['subject_id'] not in base.FROZEN_SUBJECTS['train']:
                raise ValueError('augmentation source is not train')
            rate = int(metadata['source_rate_hz'])
            count = round(rate * metadata['segment_duration_seconds'])
            budget = 2 * count * 4 + 3 * record['rows'] * (67 + 3200) * 4 + 512 * 1024**2 + 2 * 1024**3
            if base.noise_audit.available_memory() < budget:
                raise ValueError('insufficient free RAM for augmentation plus 2GiB reserve')
            source = np.empty(count, np.float32)
            for begin in range(0, count, rate * 60):
                end = min(count, begin + rate * 60)
                source[begin:end] = recording.read_window('Mic', begin / rate, (end - begin) / rate)
            source_hash = hashlib.sha256(memoryview(source)).hexdigest()
            first = float(scored['start_seconds'][0])
            with threadpool_limits(limits=1):
                spot, _ = base.representations.joint_features(source[round(first * rate):round((first + 10) * rate)], rate)
            if not np.array_equal(spot, scored['X'][0]):
                raise ValueError('complete source read differs from frozen scored spot check')
            expected_arrays, choices = _allocate(identity, rate, expected)
            relative = f'data/processed/{base.DATASET}/{VERSION}/{identity}'
            destination = output_path(root, relative)
            reuse = destination.exists()
            old = previous_records.get(identity)
            if reuse:
                if old and (old['metadata_sha256'] != sha256(destination / 'metadata.json')
                            or old['artifact_sha256'] != sha256(destination / 'inputs.npz')):
                    raise ValueError('augmentation indexed checksum differs')
                receipt = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
                if receipt.get('artifact_sha256') != sha256(destination / 'inputs.npz'):
                    raise ValueError('augmentation artifact checksum differs')
                arrays = _load(destination / 'inputs.npz')
            else:
                arrays = expected_arrays
            kinds, conditions = _generate(source, rate, identity, expected, arrays, choices, budget, reuse)
            verified_provenance = {**provenance, 'record_id': identity, 'split': 'train', 'version': VERSION,
                                   'source_hashes': sources, 'environment': index['environment'],
                                   'source_samples_sha256': source_hash, 'original_feature_spotcheck_exact': True}
            counts = {f'{kind}:{snr}': int(np.count_nonzero(choices == i)) for i, (kind, snr) in enumerate(CONDITIONS)}
            gains = {(c['kind'], c['snr_db']): c['calibration']['common_gain'] for c in conditions}
            _validate(arrays, expected_arrays, gains)
            content = {'rows': record['rows'], 'provenance': verified_provenance, 'noise_kinds': kinds,
                       'conditions': conditions, 'condition_counts': counts, 'worker_count': 1, 'numeric_threads': 2,
                       'shapes': {key: list(value.shape) for key, value in arrays.items()}}
            if reuse:
                if any(receipt.get(key) != value for key, value in content.items()):
                    raise ValueError('augmentation provenance/metadata differs; refusing reuse/overwrite')
            else:
                with tempfile.TemporaryDirectory(prefix='.augmentation-', dir=folder) as temporary:
                    staged = Path(temporary) / identity
                    staged.mkdir()
                    np.savez_compressed(staged / 'inputs.npz', **arrays)
                    receipt = {**content, 'artifact_sha256': sha256(staged / 'inputs.npz'),
                               'elapsed_seconds': time.perf_counter() - tick}
                    (staged / 'metadata.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                    staged.rename(destination)
            if hashlib.sha256(memoryview(source)).hexdigest() != source_hash:
                raise ValueError('reference input modified')
            index['records'] = [entry for entry in index['records'] if entry['record_id'] != identity]
            index['records'].append({'record_id': identity, 'split': 'train', 'path': relative, 'rows': record['rows'],
                                     'artifact_sha256': receipt['artifact_sha256'],
                                     'metadata_sha256': sha256(destination / 'metadata.json')})
            _write_index(folder / 'corpus.json', index)
            print(f'{identity}: {record["rows"]} augmented training windows verified',file=sys.stderr,flush=True)
            del arrays, expected_arrays, source
    if _sources(root) != sources or sum(r['rows'] for r in index['records']) != EXPECTED_TOTAL_ROWS:
        raise ValueError('augmentation source changed or output incomplete')
    index.update(complete=True, elapsed_seconds=time.perf_counter() - started)
    _write_index(folder / 'corpus.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare_augmentation(ROOT, args.execute), indent=2))


if __name__ == '__main__':
    main()
