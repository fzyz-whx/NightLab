"""Prepare local deep inputs from verified pilots; dry-run is the default."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.features import AudioFeatureConfig
from nightlab.audio.reference import window_target
from nightlab.deep import inputs
from nightlab.wearable.features import _parse_fields
from .cli import ROOT, load_manifest, raw_path, verify
from .dense_audio import record_context, sha256
from .prepare import output_path
from .prepare_cohort import validate_receipt


VERSION = 'deep-inputs-v1'
DATASETS = {'audio': 'psg-audio-pilot-10', 'wearable': 'sleepaccel-pilot-3'}


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError('object arrays forbidden')
    return arrays


def _same(actual, expected, context):
    for key, value in expected.items():
        if key not in actual or not np.array_equal(actual[key], value):
            raise ValueError(f'{context}: {key} differs from verified source')


def _audio_inputs(root, manifest, record):
    folder, metadata, recording, events, offset, provenance = record_context(root, manifest, record)
    config = AudioFeatureConfig()
    rate = recording.sample_rate(metadata['channel'])
    if (metadata['channel'] != 'Mic' or metadata['config'] != config.metadata() or rate != int(rate) or rate < 8000
            or metadata['segment_duration_seconds'] != recording.duration_seconds
            or metadata['segment_offset_seconds'] != offset or metadata['source_rate_hz'] != int(rate)):
        raise ValueError('audio source geometry/configuration differs')
    count = 1 + math.floor((recording.duration_seconds - config.window_seconds) / config.hop_seconds + 1e-9)
    if count < 1 or count != metadata['candidate_windows']:
        raise ValueError('audio window count differs')
    starts = np.arange(count, dtype=np.float64) * config.hop_seconds
    ends = starts + config.window_seconds
    targets, overlaps = zip(*(window_target(offset + start, offset + end, events,
                                            config.positive_overlap_fraction)
                              for start, end in zip(starts, ends, strict=True)))
    labels = np.asarray([target or '' for target in targets], dtype=str)
    mask = labels != ''
    timeline = _load(folder / 'timeline.npz')
    _same(timeline, {'start_seconds': starts, 'end_seconds': ends, 'target': labels,
                     'target_valid': mask, 'overlap_fraction': np.asarray(overlaps)}, 'classic timeline')
    scored = _load(folder / 'features.npz')
    _same(scored, {'start_seconds': starts[mask], 'end_seconds': ends[mask], 'y': labels[mask],
                   'subject_id': np.full(mask.sum(), provenance['subject_id']),
                   'record_id': np.full(mask.sum(), record['record_id']),
                   'overlap_fraction': np.asarray(overlaps)[mask]}, 'classic scored rows')
    if metadata['rows'] != int(mask.sum()) or record['rows'] != int(mask.sum()):
        raise ValueError('classic audio row count differs')
    expected = {'y': labels, 'supervised_mask': mask, 'start_seconds': starts,
                'end_seconds': ends, 'subject_id': np.full(count, provenance['subject_id'])}

    def generate():
        values = np.empty((count, 32, 100), dtype=np.float32)
        for index, start in enumerate(starts):
            values[index] = inputs.log_mel_image(
                recording.read_window(metadata['channel'], float(start), config.window_seconds), int(rate))
            if (index + 1) % 100 == 0 or index + 1 == count:
                print(f"{record['record_id']} audio windows {index + 1}/{count}", file=sys.stderr, flush=True)
        return {**expected, 'values': values}

    return provenance, expected, generate


def _wearable_inputs(root, manifest, record):
    subject = record['record_id']
    entries = [entry for entry in manifest['files'] if entry.get('subject_id') == subject]
    if len(entries) != 3 or {PurePosixPath(entry['path']).parent.name for entry in entries} != {'heart_rate', 'motion', 'labels'}:
        raise ValueError('one HR, motion and label file required per subject')
    subset = {**manifest, 'id': 'sleepaccel', 'files': entries}
    folder = output_path(root, record['path'])
    metadata = validate_receipt(folder, subset)
    integrity = verify(root, subset)
    previous = {entry['path']: entry['local_checksums'] for entry in metadata['input_integrity']['files']}
    if not integrity['ok'] or any(entry['local_checksums'] != previous[entry['path']] for entry in integrity['files']):
        raise ValueError('raw wearable input differs from official/prepared receipt')
    paths = {PurePosixPath(entry['path']).parent.name: raw_path(root, entry['path']) for entry in entries}
    parsed = {kind: _parse_fields(path.read_text(encoding='utf-8'), kind) for kind, path in paths.items()}
    result = inputs.wearable_epoch_tensors(subject, parsed['heart_rate'], parsed['motion'], parsed['labels'],
                                          repeat_policy='collapse_identical')
    count = len(result['y'])
    expected = {'y': result['y'], 'subject_id': result['subject_id'],
                'start_seconds': result['epoch_start_seconds'],
                'end_seconds': result['epoch_start_seconds'] + 30,
                'supervised_mask': np.ones(count, dtype=bool)}
    _same(_load(folder / 'features.npz'), {key: value for key, value in expected.items()
                                         if key != 'supervised_mask'}, 'classic wearable rows')
    if metadata['rows'] != count or record['rows'] != count:
        raise ValueError('classic wearable row count differs')
    provenance = {'record_id': subject, 'subject_id': subject, 'dataset_version': manifest['version'],
                  'source_features_path': record['path'], 'source_metadata_sha256': sha256(folder / 'metadata.json'),
                  'source_artifact_sha256': metadata['artifact_sha256'],
                  'feature_source_sha256': metadata['feature_source_sha256'],
                  'input_sha256': {entry['path']: entry['local_checksums']['sha256'] for entry in integrity['files']},
                  'repeat_policy': 'collapse_identical'}
    return provenance, expected, lambda: {**expected, 'values': result['values'], 'masks': result['masks']}


def _record_inputs(root, task, manifest, record):
    return (_audio_inputs if task == 'audio' else _wearable_inputs)(root, manifest, record)


def _validate_arrays(arrays, expected, task):
    _same(arrays, expected, 'deep inputs')
    count = len(expected['y'])
    shape = (count, 32, 100) if task == 'audio' else (count, 5, 30)
    if (any(value.dtype.hasobject for value in arrays.values())
            or arrays['values'].shape != shape or arrays['values'].dtype != np.float32
            or arrays['y'].dtype.kind != 'U' or arrays['subject_id'].dtype.kind != 'U'
            or arrays['supervised_mask'].dtype != np.bool_):
        raise ValueError('deep input shape/dtype invalid')
    values = arrays['values']
    if task == 'audio':
        if not np.isfinite(values).all():
            raise ValueError('audio inputs must be finite')
    else:
        masks = arrays['masks']
        if masks.shape != (count, 2, 30) or masks.dtype != np.float32 or not np.isin(masks, (0, 1)).all():
            raise ValueError('wearable masks invalid')
        observed = masks[:, [0, 1, 1, 1, 1], :].astype(bool)
        if not np.isfinite(values[observed]).all() or not np.isnan(values[~observed]).all():
            raise ValueError('wearable values must match observation masks')


def _ensure_record(root, task, manifest, record):
    started = time.perf_counter()
    provenance, expected, generate = _record_inputs(root, task, manifest, record)
    provenance = {**provenance, 'deep_inputs_source_sha256': sha256(inputs.__file__),
                  'deep_cache_source_sha256': sha256(__file__), 'version': VERSION}
    relative = f"data/processed/{manifest['id']}/{VERSION}/{record['record_id']}"
    destination = output_path(root, relative)
    if destination.exists():
        receipt = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
        if receipt.get('provenance') != provenance or receipt.get('artifact_sha256') != sha256(destination / 'inputs.npz'):
            raise ValueError('deep cache provenance/checksum differs; refusing reuse/overwrite')
        arrays = _load(destination / 'inputs.npz')
        _validate_arrays(arrays, expected, task)
        if receipt.get('rows') != len(expected['y']) or receipt.get('shape') != list(arrays['values'].shape):
            raise ValueError('deep cache receipt shape/rows invalid')
    else:
        arrays = generate()
        _validate_arrays(arrays, expected, task)
        receipt = {'rows': len(expected['y']), 'shape': list(arrays['values'].shape),
                   'provenance': provenance, 'elapsed_seconds': time.perf_counter() - started,
                   'worker_count': 1, 'numeric_threads': 2}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.deep-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'cache'
            staged.mkdir()
            np.savez_compressed(staged / 'inputs.npz', **arrays)
            receipt['artifact_sha256'] = sha256(staged / 'inputs.npz')
            (staged / 'metadata.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            staged.rename(destination)
    return {'record_id': record['record_id'], 'path': relative, 'ok': True, 'rows': receipt['rows'],
            'shape': receipt['shape'], 'artifact_sha256': receipt['artifact_sha256']}


def _write_index(path, index):
    with tempfile.TemporaryDirectory(prefix='.index-', dir=path.parent) as temporary:
        staged = Path(temporary) / 'corpus.json'
        staged.write_text(json.dumps(index, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staged.replace(path)


def prepare_deep(root, task, execute=False):
    if task not in DATASETS:
        raise ValueError("task must be 'audio' or 'wearable'")
    root = Path(root)
    dataset = DATASETS[task]
    folder = output_path(root, f'data/processed/{dataset}/{VERSION}')
    if execute:
        folder.mkdir(parents=True, exist_ok=True)
        # Even a failed manifest/classic-corpus check must invalidate old success.
        _write_index(folder / 'corpus.json', {'dataset': dataset, 'task': task, 'version': VERSION,
                                             'complete': False, 'split_policy': None, 'records': []})
    manifest_path = root / f'data/manifests/{dataset}.json'
    corpus_path = output_path(root, f'data/processed/{dataset}/corpus.json')
    manifest = load_manifest(manifest_path, root)
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    records = corpus['records']
    expected = {entry['record_id'] if task == 'audio' else entry['subject_id'] for entry in manifest['files']}
    if (not corpus['complete'] or corpus['split_policy'] != manifest['subjects']
            or corpus.get('dataset') != dataset or corpus.get('version') != manifest['version']
            or len(records) != len(expected) or {record['record_id'] for record in records} != expected
            or not all(record['ok'] for record in records)):
        raise ValueError('complete classic corpus matching manifest and split required')
    pattern = r'\d{8}-\d{6}' if task == 'audio' else r'\d+'
    if any(not re.fullmatch(pattern, record['record_id']) for record in records):
        raise ValueError('invalid record identity')
    if manifest.get('license', {}).get('verified') is not True:
        raise ValueError('verified dataset license required')
    index = {'dataset': dataset, 'task': task, 'version': VERSION, 'complete': False,
             'split_policy': corpus['split_policy'], 'records': [],
             'source_manifest_sha256': sha256(manifest_path), 'source_corpus_sha256': sha256(corpus_path)}
    if not execute:
        return {**index, 'mode': 'deep dry-run', 'network': False, 'writes': False,
                'workers': 1, 'numeric_threads': 2,
                'records': [{'record_id': record['record_id']} for record in records]}
    # Invalidate a former success before checking inputs, including on interruption.
    _write_index(folder / 'corpus.json', index)
    with threadpool_limits(limits=2):
        for record in records:
            result = _ensure_record(root, task, manifest, record)
            index['records'].append(result)
            print(f"{record['record_id']} deep inputs verified: {result['rows']} rows", file=sys.stderr, flush=True)
    index['complete'] = len(index['records']) == len(expected)
    _write_index(folder / 'corpus.json', index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(DATASETS), required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare_deep(ROOT, args.task, args.execute), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
