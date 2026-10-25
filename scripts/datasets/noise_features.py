"""Prepare frozen synthetic-noise representations; default is a read-only plan."""
import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import re
import platform
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio import edf, features, noise, noise_windows, reference, representations
from nightlab.audio.features import AudioFeatureConfig, AudioFeatureExtractor
from nightlab.audio.reference import window_target
from nightlab.deep import inputs
from . import cli, deep_cache, dense_audio, noise_audit, prepare, prepare_cohort
from .cli import ROOT, load_manifest
from .deep_cache import _load, _same, _write_index
from .dense_audio import record_context, sha256
from .prepare import output_path

DATASET = 'psg-audio-pilot-10'
EXPECTED_ROWS = 719
PILOT_RECORD = '00000995-100507'
TEST_RECORDS = ('00000999-100507', '00001018-100507')
KINDS = ('white', 'mains_hum')
SNRS = (-5, 0, 5, 10)
FROZEN_SUBJECTS = {
    'train': ['00001010', '00001014', '00000995', '00001016', '00001006', '00001000'],
    'validation': ['00001008', '00001020'], 'test': ['00000999', '00001018']}


def _sources(root):
    modules = {'noise': noise, 'noise_windows': noise_windows, 'representations': representations,
               'features': features, 'deep.inputs': inputs, 'edf': edf, 'reference': reference,
               'dense_audio': dense_audio, 'noise_audit': noise_audit, 'prepare': prepare,
               'deep_cache': deep_cache, 'cli': cli, 'prepare_cohort': prepare_cohort}
    return {**{name: sha256(module.__file__) for name, module in modules.items()},
            'noise_features': sha256(__file__), 'protocol': sha256(root / 'docs/NOISE_PROTOCOL.md'),
            'manifest': sha256(root / f'data/manifests/{DATASET}.json'),
            'classic_corpus': sha256(root / f'data/processed/{DATASET}/corpus.json')}


def _select(root, scope):
    manifest = load_manifest(root / f'data/manifests/{DATASET}.json', root)
    corpus = json.loads((root / f'data/processed/{DATASET}/corpus.json').read_text(encoding='utf-8'))
    records = corpus['records']
    identities = {entry['record_id'] for entry in manifest['files']}
    policy = corpus.get('split_policy', {})
    if (manifest['id'] != DATASET or manifest['version'] != 'V3' or not manifest['license'].get('verified')
            or corpus.get('dataset') != DATASET or corpus.get('version') != 'V3'
            or not corpus.get('complete') or policy != manifest['subjects']
            or policy.get('subjects') != FROZEN_SUBJECTS or policy.get('unit') != 'subject'
            or len(records) != len(identities) or {item['record_id'] for item in records} != identities
            or not all(item['ok'] for item in records)):
        raise ValueError('complete classic corpus with frozen manifest/split required')
    for entry in manifest['files']:
        record_id, subject = entry['record_id'], entry['subject_id']
        if (not re.fullmatch(r'\d{8}-\d{6}', record_id) or record_id.split('-')[0] != subject
                or entry.get('split') not in FROZEN_SUBJECTS
                or subject not in FROZEN_SUBJECTS[entry['split']]):
            raise ValueError('manifest record identity/split differs from frozen policy')
    wanted = (PILOT_RECORD,) if scope == 'pilot' else TEST_RECORDS
    if not set(wanted).issubset(identities):
        raise ValueError('fixed noise records missing from classic corpus')
    selected = [next(item for item in records if item['record_id'] == identity) for identity in wanted]
    return manifest, corpus, selected


def _verified_context(root, manifest, record):
    folder, metadata, recording, events, offset, provenance = record_context(root, manifest, record)
    config = AudioFeatureConfig()
    rate = recording.sample_rate(metadata['channel'])
    duration = recording.duration_seconds
    count = 1 + math.floor((duration - config.window_seconds) / config.hop_seconds + 1e-9)
    if (metadata['channel'] != 'Mic' or metadata['config'] != config.metadata()
            or metadata['feature_names'] != list(AudioFeatureExtractor().feature_names)
            or rate != int(rate) or rate < 16000 or rate % 16000
            or duration != metadata['segment_duration_seconds'] or offset != metadata['segment_offset_seconds']
            or rate != metadata['source_rate_hz'] or count != metadata['candidate_windows']
            or count != EXPECTED_ROWS or duration * rate != round(duration * rate)):
        raise ValueError('source geometry/configuration differs from frozen noise protocol')
    starts = np.arange(count, dtype=np.float64) * config.hop_seconds
    ends = starts + config.window_seconds
    targets, overlaps = zip(*(window_target(offset + left, offset + right, events, config.positive_overlap_fraction)
                              for left, right in zip(starts, ends, strict=True)))
    labels = np.asarray([value or '' for value in targets], dtype=str)
    mask = labels != ''
    _same(_load(folder / 'timeline.npz'), {'start_seconds': starts, 'end_seconds': ends,
          'target': labels, 'target_valid': mask, 'overlap_fraction': np.asarray(overlaps)}, 'classic timeline')
    scored = _load(folder / 'features.npz')
    _same(scored, {'start_seconds': starts[mask], 'end_seconds': ends[mask], 'y': labels[mask],
          'subject_id': np.full(mask.sum(), provenance['subject_id']),
          'record_id': np.full(mask.sum(), record['record_id']),
          'overlap_fraction': np.asarray(overlaps)[mask]}, 'classic scored')
    if (not mask.any() or scored['X'].shape != (mask.sum(), 67) or scored['X'].dtype != np.float32
            or not np.isfinite(scored['X']).all() or metadata['rows'] != int(mask.sum())
            or record['rows'] != int(mask.sum())):
        raise ValueError('classic scored feature geometry differs')
    expected = {'y': labels, 'subject_id': np.full(count, provenance['subject_id']),
                'start_seconds': starts, 'end_seconds': ends, 'supervised_mask': mask}
    return recording, metadata, provenance, expected, scored


def _memory_budget(count, rows, cached_gains=8):
    # At most eight distinct common gains, two clean feature branches per gain,
    # plus four output branches, compression temporaries and bounded signal work.
    matrices = rows * (67 + 32 * 100) * 4
    return 2 * count * 4 + (2 * cached_gains + 8) * matrices + 512 * 1024**2 + 2 * 1024**3


def _validate_arrays(arrays, expected):
    _same(arrays, expected, 'noise inputs')
    rows = len(expected['y'])
    required = set(expected) | {prefix + branch for branch in noise.BRANCHES for prefix in ('X_', 'images_')}
    if (set(arrays) != required or arrays['y'].dtype.kind != 'U' or arrays['subject_id'].dtype.kind != 'U'
            or arrays['supervised_mask'].dtype != np.bool_ or any(value.dtype.hasobject for value in arrays.values())):
        raise ValueError('noise input fields/dtypes invalid')
    for branch in noise.BRANCHES:
        for prefix, shape in (('X_', (rows, 67)), ('images_', (rows, 32, 100))):
            values = arrays[prefix + branch]
            if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError('noise input feature geometry/values invalid')


def _condition_path(root, version, record_id, kind, snr):
    suffix = f'm{abs(snr)}' if snr < 0 else f'p{snr}'
    relative = f'data/processed/{DATASET}/{version}/{record_id}/{kind}_snr_{suffix}'
    return relative, output_path(root, relative)


def _ensure_condition(destination, expected, provenance, generate):
    if destination.exists():
        receipt = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
        if receipt.get('provenance') != provenance or receipt.get('artifact_sha256') != sha256(destination / 'inputs.npz'):
            raise ValueError('noise provenance/checksum differs; refusing reuse/overwrite')
        arrays = _load(destination / 'inputs.npz')
        _validate_arrays(arrays, expected)
        if (receipt.get('rows') != len(expected['y'])
                or receipt.get('shapes') != {key: list(value.shape) for key, value in arrays.items()}):
            raise ValueError('noise receipt row count differs')
        return receipt, arrays
    arrays, statistics = generate()
    _validate_arrays(arrays, expected)
    receipt = {'rows': len(expected['y']), 'provenance': provenance, 'statistics': statistics,
               'worker_count': 1, 'numeric_threads': 2,
               'shapes': {key: list(value.shape) for key, value in arrays.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.noise-features-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'condition'
        staged.mkdir()
        np.savez_compressed(staged / 'inputs.npz', **arrays)
        receipt['artifact_sha256'] = sha256(staged / 'inputs.npz')
        (staged / 'metadata.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staged.rename(destination)
    return receipt, arrays


def _generate_condition(observed, calibration, expected, clean_cache):
    rows = len(expected['y'])
    gain_key = float(calibration['common_gain']).hex()
    cached = clean_cache.get(gain_key)
    arrays = dict(expected)
    for branch in noise.BRANCHES:
        for prefix, shape in (('X_', (rows, 67)), ('images_', (rows, 32, 100))):
            key = prefix + branch
            arrays[key] = cached[key] if cached is not None and branch.startswith('matched_') else np.empty(shape, dtype=np.float32)
    peaks = {name: 0.0 for name in noise.BRANCHES}
    clipping = {name: 0 for name in noise.BRANCHES}
    completed = 0
    for index, window in enumerate(noise_windows.iter_noise_windows(observed, calibration)):
        if index >= rows or window['start_seconds'] != expected['start_seconds'][index] or window['end_seconds'] != expected['end_seconds'][index]:
            raise ValueError('condition window geometry differs from verified timeline')
        for branch in noise.BRANCHES:
            values = window[branch]
            peaks[branch] = max(peaks[branch], float(np.abs(values).max()))
            clipping[branch] += int(np.count_nonzero(np.abs(values) > 1))
            if cached is None or not branch.startswith('matched_'):
                classic, image = representations.joint_features(values, observed.source_rate)
                arrays['X_' + branch][index] = classic
                arrays['images_' + branch][index] = image
        completed = index + 1
        if completed % 100 == 0 or completed == rows:
            print(f'noise feature windows {completed}/{rows}', file=sys.stderr, flush=True)
    if completed != rows or any(clipping.values()) or max(peaks.values()) > 0.980001:
        raise ValueError('condition length/headroom verification failed')
    clean_cache[gain_key] = {key: arrays[key] for key in arrays if key.startswith(('X_matched_', 'images_matched_'))}
    return arrays, {'actual_window_branch_peaks': peaks, 'clipped_window_samples': clipping,
                    'clean_branches_reused': cached is not None, 'common_gain_hex': gain_key}


def prepare_noise(root, scope='pilot', execute=False):
    if scope not in ('pilot', 'test'):
        raise ValueError("scope must be 'pilot' or 'test'")
    root = Path(root).resolve()
    version = 'noise-pilot-v1' if scope == 'pilot' else 'noise-eval-v1'
    folder = output_path(root, f'data/processed/{DATASET}/{version}')
    index = {'dataset': DATASET, 'dataset_version': 'V3', 'scope': scope, 'version': version,
             'complete': False, 'split_policy': None, 'records': [],
             'environment': {'python': platform.python_version(),
                             **{name: package_version(name) for name in ('numpy', 'scipy')}}}
    if execute:
        folder.mkdir(parents=True, exist_ok=True)
        _write_index(folder / 'corpus.json', index)
    manifest, corpus, records = _select(root, scope)
    sources = _sources(root)
    index.update(split_policy=corpus['split_policy'], source_hashes=sources,
                 source_manifest_sha256=sources['manifest'], source_corpus_sha256=sources['classic_corpus'],
                 protocol_sha256=sources['protocol'])
    conditions = [('white', -5)] if scope == 'pilot' else [(kind, snr) for kind in KINDS for snr in SNRS]
    if not execute:
        return {**index, 'mode': 'noise features dry-run', 'writes': False, 'network': False,
                'workers': 1, 'numeric_threads': 2, 'scoring': False,
                'records': [{'record_id': record['record_id'], 'split': 'train' if scope == 'pilot' else 'test',
                             'kind': kind, 'snr_db': snr, 'rows': EXPECTED_ROWS}
                            for record in records for kind, snr in conditions]}
    _write_index(folder / 'corpus.json', index)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for record in records:
            recording, metadata, provenance, expected, scored = _verified_context(root, manifest, record)
            rate, duration = int(metadata['source_rate_hz']), metadata['segment_duration_seconds']
            count = round(rate * duration)
            budget = _memory_budget(count, len(expected['y']))
            if noise_audit.available_memory() < budget:
                raise ValueError(f'insufficient free RAM for reference/noise/cache plus 2GiB reserve: {budget}')
            source = np.empty(count, dtype=np.float32)
            for begin in range(0, count, rate * 60):
                end = min(count, begin + rate * 60)
                source[begin:end] = recording.read_window('Mic', begin / rate, (end - begin) / rate)
            source_hash = hashlib.sha256(memoryview(source)).hexdigest()
            first = float(scored['start_seconds'][0])
            with threadpool_limits(limits=1):
                spot, _ = representations.joint_features(source[round(first * rate):round((first + 10) * rate)], rate)
            if not np.array_equal(spot, scored['X'][0]):
                raise ValueError('complete source read differs from frozen scored spot check')
            clean_cache = {}
            split = 'train' if scope == 'pilot' else 'test'
            for kind in dict.fromkeys(kind for kind, _ in conditions):
                if noise_audit.available_memory() < budget - source.nbytes:
                    raise ValueError('available memory changed before noise allocation')
                seed = noise.noise_seed(record['record_id'], split, kind)
                interference = noise.synthetic_noise(count, rate, kind, seed)
                noise_hash = hashlib.sha256(memoryview(interference)).hexdigest()
                observed = noise.NoiseRecording(source, interference, rate)
                for selected_kind, snr in conditions:
                    if selected_kind != kind:
                        continue
                    print(f"{record['record_id']} {kind} SNR {snr}: calibrate/features", file=sys.stderr, flush=True)
                    calibration = observed.calibrate(snr)
                    actual = observed.measure_actual_snr(calibration)
                    if abs(actual - snr) > 1e-4:
                        raise ValueError('actual rounded mixture SNR exceeds tolerance')
                    condition_provenance = {**provenance, 'source_hashes': sources,
                        'source_samples_sha256': source_hash, 'original_feature_spotcheck_exact': True,
                        'record_id': record['record_id'], 'split': split, 'kind': kind, 'snr_db': snr,
                        'seed': seed, 'noise_sha256': noise_hash, 'calibration': calibration,
                        'actual_rounded_analysis_snr_db': actual, 'version': version,
                        'environment': index['environment']}
                    relative, destination = _condition_path(root, version, record['record_id'], kind, snr)
                    receipt, arrays = _ensure_condition(destination, expected, condition_provenance,
                        lambda: _generate_condition(observed, calibration, expected, clean_cache))
                    gain_key = float(calibration['common_gain']).hex()
                    clean_cache.setdefault(gain_key, {key: arrays[key] for key in arrays
                                                     if key.startswith(('X_matched_', 'images_matched_'))})
                    del arrays
                    index['records'].append({'record_id': record['record_id'], 'split': split, 'kind': kind,
                        'snr_db': snr, 'path': relative, 'artifact_sha256': receipt['artifact_sha256'],
                        'metadata_sha256': sha256(destination / 'metadata.json'), 'rows': receipt['rows']})
                    _write_index(folder / 'corpus.json', index)
                if hashlib.sha256(memoryview(interference)).hexdigest() != noise_hash:
                    raise ValueError('noise input modified')
                del observed, interference
            if hashlib.sha256(memoryview(source)).hexdigest() != source_hash:
                raise ValueError('reference input modified')
            del source, clean_cache
    if _sources(root) != sources:
        raise ValueError('protocol/source changed during preparation')
    if len(index['records']) != len(records) * len(conditions):
        raise ValueError('incomplete condition set')
    index.update(complete=True, elapsed_seconds=time.perf_counter() - started)
    _write_index(folder / 'corpus.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('pilot', 'test'), default='pilot')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare_noise(ROOT, args.scope, args.execute), indent=2))


if __name__ == '__main__':
    main()
