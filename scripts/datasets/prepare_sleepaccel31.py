"""One-worker v3 cohort preparation; default is metadata-only with no writes."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from threadpoolctl import threadpool_limits
from nightlab.wearable import cleaning_v3, features
from nightlab.deep import inputs
from . import audit_sleepaccel31 as audit
from .cli import ROOT, raw_path

BASE = 'data/processed/sleepaccel-cohort-31'
DEEP = BASE + '/deep-inputs-v3'
FEATURE_VERSION = 'sleepaccel-classic-v3'
DEEP_VERSION = 'deep-inputs-v3'
OLD = {'1066528', '5383425', '8258170'}


def source_hashes(root):
    return {name: audit.sha256(path) for name, path in {
        'manifest': root / audit.MANIFEST, 'cleaning_v3.py': Path(cleaning_v3.__file__),
        'prepare_sleepaccel31.py': Path(__file__), 'features.py': Path(features.__file__),
        'inputs.py': Path(inputs.__file__), 'audit_sleepaccel31.py': Path(audit.__file__),
        'intake_receipt': root / 'data/raw/sleepaccel/archives/intake-receipt.json'}.items()}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise ValueError('object arrays forbidden')
    return result


def assert_equal(actual, expected, context):
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f'{context}: missing {key}')
        nan = value.dtype.kind in 'fc'
        if not np.array_equal(actual[key], value, equal_nan=nan):
            raise ValueError(f'{context}: {key} differs')


def build(subject, canonical):
    hr, motion, labels = (canonical[key] for key in ('heart_rate', 'motion', 'labels'))
    result = features.extract_sleepaccel_features(subject, hr, motion, labels,
                                                  features.FeatureConfig(repeated_block_policy='reject'))
    classic = {'X': result.features.astype(np.float32), 'y': np.asarray(result.stage, dtype=str),
               'subject_id': result.subject_id, 'raw_stage': result.raw_stage,
               'start_seconds': result.epoch_start_seconds, 'end_seconds': result.epoch_end_seconds}
    for signal, fields in result.coverage.items():
        classic.update({f'coverage_{signal}_{name}': value for name, value in fields.items()})
    timeline = {key: (np.asarray(['' if value is None else value for value in values], dtype=str)
                       if key == 'stage' else values) for key, values in result.label_table.items()}
    tensor = inputs.wearable_epoch_tensors(subject, hr, motion, labels, repeat_policy='reject')
    count = len(tensor['y'])
    deep = {'values': tensor['values'], 'masks': tensor['masks'], 'y': tensor['y'],
            'subject_id': tensor['subject_id'], 'start_seconds': tensor['epoch_start_seconds'],
            'end_seconds': tensor['epoch_start_seconds'] + 30, 'supervised_mask': np.ones(count, dtype=bool)}
    assert_equal(deep, {key: classic[key] for key in ('y', 'subject_id', 'start_seconds', 'end_seconds')}, 'classic/deep timeline')
    if any(value.dtype.hasobject for value in (*classic.values(), *timeline.values(), *deep.values())):
        raise ValueError('object arrays forbidden')
    observed = deep['masks'][:, [0, 1, 1, 1, 1], :].astype(bool)
    if not np.isfinite(deep['values'][observed]).all() or not np.isnan(deep['values'][~observed]).all():
        raise ValueError('deep missing-value masks mismatch')
    if not count:
        raise ValueError('no scored epochs')
    return classic, timeline, deep, result.metadata, tensor['metadata']


def legacy_parity(root, subject, classic, deep):
    if subject not in OLD:
        return {'required': False, 'reason': 'subject was not in frozen three-person pilot'}
    old_classic = root / f'data/processed/sleepaccel-pilot-3/{subject}/features-v2'
    old_deep = root / f'data/processed/sleepaccel-pilot-3/deep-inputs-v1/{subject}'
    metadata = json.loads((old_classic / 'metadata.json').read_text(encoding='utf-8'))
    deep_metadata = json.loads((old_deep / 'metadata.json').read_text(encoding='utf-8'))
    if (metadata['artifact_sha256']['features.npz'] != audit.sha256(old_classic / 'features.npz')
            or deep_metadata['artifact_sha256'] != audit.sha256(old_deep / 'inputs.npz')):
        raise ValueError('legacy parity source artifact checksum mismatch')
    assert_equal(load_arrays(old_classic / 'features.npz'),
                 {key: classic[key] for key in ('X', 'y', 'subject_id', 'raw_stage', 'start_seconds', 'end_seconds')},
                 'legacy classical exact parity')
    assert_equal(load_arrays(old_deep / 'inputs.npz'), deep, 'legacy deep exact parity')
    return {'required': True, 'exact': True, 'equal_nan': True,
            'classic_keys': ['X', 'y', 'subject_id', 'raw_stage', 'start_seconds', 'end_seconds'],
            'deep_keys': list(deep), 'legacy_classic_sha256': metadata['artifact_sha256']['features.npz'],
            'legacy_deep_sha256': deep_metadata['artifact_sha256']}


def publish(folder, archives, metadata):
    """Immutable record directories; reuse requires matching provenance and arrays."""
    if folder.exists():
        previous = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        for key in ('provenance', 'rows', 'feature_version', 'version'):
            if previous.get(key) != metadata.get(key):
                raise ValueError('existing record provenance differs; refusing overwrite')
        for name, arrays in archives.items():
            expected_hash = previous['artifact_sha256'] if name == 'inputs.npz' else previous['artifact_sha256'][name]
            if audit.sha256(folder / name) != expected_hash:
                raise ValueError('existing record artifact checksum differs')
            assert_equal(load_arrays(folder / name), arrays, 'existing record exact reuse')
        return previous, 'reused verified v3'
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-v3-', dir=folder.parent) as temp:
        staged = Path(temp) / 'record'
        staged.mkdir()
        for name, arrays in archives.items():
            np.savez_compressed(staged / name, **arrays)
        hashes = {name: audit.sha256(staged / name) for name in archives}
        metadata['artifact_sha256'] = hashes['inputs.npz'] if 'inputs.npz' in hashes else hashes
        (staged / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staged.rename(folder)
    return metadata, 'prepared v3'


def write_index(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.index-v3-', dir=path.parent) as temp:
        staged = Path(temp) / 'corpus.json'
        staged.write_text(json.dumps(index, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        staged.replace(path)


def prepare_one(root, manifest, subject, entries, sources):
    raw = {}
    checks = []
    for kind in ('labels', 'heart_rate', 'motion'):
        entry = entries[kind]
        path = raw_path(root, entry['path'])
        digest = audit.sha256(path)
        if path.stat().st_size != entry['exact_bytes'] or digest != entry['official_checksums']['sha256']:
            raise ValueError(f'{kind}: official raw size/SHA mismatch')
        checks.append({'path': entry['path'], 'actual_bytes': path.stat().st_size,
                       'local_checksums': {'sha256': digest}, 'official_sha256_verified': True})
        raw[kind] = audit.read_rows(path, kind)
    if audit.available_memory() < raw['motion'].nbytes * 8 + 512 * 1024**2:
        raise MemoryError('insufficient RAM reserve for canonicalization')
    canonical = cleaning_v3.canonicalize(raw['heart_rate'], raw['motion'], raw['labels'])
    del raw
    classic, timeline, deep, feature_metadata, tensor_metadata = build(subject, canonical)
    cleaning = canonical['metadata']
    del canonical
    parity = legacy_parity(root, subject, classic, deep)
    provenance = {'source_sha256': sources, 'raw_input_sha256': {item['path']: item['local_checksums']['sha256'] for item in checks},
                  'canonicalization_version': cleaning_v3.VERSION, 'subject_id': subject,
                  'partition': entries['labels']['split'], 'legacy_parity': parity}
    classic_relative = f'{BASE}/{subject}/features-v3'
    common = {'schema_version': 1, 'dataset': manifest['id'], 'dataset_version': manifest['version'],
              'rows': len(classic['y']), 'provenance': provenance,
              'raw_cleaning': cleaning, 'canonical_input_semantics': 'Extractor input statistics refer to canonical cropped rows, not raw files',
              'input_integrity': {'ok': True, 'official_sha256_verified': True, 'files': checks},
              'worker_count': 1, 'numeric_threads': 1, 'legacy_parity': parity}
    classic_metadata = {**common, 'canonical_feature_metadata': feature_metadata,
                        'feature_version': FEATURE_VERSION, 'feature_source_sha256': sources,
                        'feature_names': list(features.FEATURE_NAMES), 'feature_count': len(features.FEATURE_NAMES),
                        'feature_nan_count': int(np.isnan(classic['X']).sum()),
                        'target_counts': dict(Counter(classic['y'].tolist()))}
    _, status = publish(root / classic_relative, {'features.npz': classic, 'timeline.npz': timeline}, classic_metadata)
    deep_relative = f'{DEEP}/{subject}'
    deep_metadata = {**common, 'version': DEEP_VERSION, 'shape': list(deep['values'].shape),
                     'canonical_tensor_metadata': tensor_metadata}
    deep_receipt, deep_status = publish(root / deep_relative, {'inputs.npz': deep}, deep_metadata)
    return ({'record_id': subject, 'path': classic_relative, 'ok': True, 'rows': len(classic['y']), 'status': status,
             'legacy_parity': parity},
            {'record_id': subject, 'path': deep_relative, 'ok': True, 'rows': len(deep['y']),
             'shape': deep_receipt['shape'], 'artifact_sha256': deep_receipt['artifact_sha256'],
             'status': deep_status, 'legacy_parity': parity})


def run(root=ROOT, execute=False):
    root = Path(root)
    # Metadata links to the fixed intake receipt and all93 plan entries are checked
    # before data use; this function does not depend on strict audit validity.
    manifest, grouped = audit.metadata(root)
    if manifest['preparation_policy'].get('version') != cleaning_v3.VERSION:
        raise ValueError('manifest must explicitly authorize cohort31 v3 policy')
    sources = source_hashes(root)
    common = {'dataset': manifest['id'], 'split_policy': manifest['subjects'], 'complete': False,
              'records': [], 'source_sha256': sources, 'errors': [], 'workers': 1, 'numeric_threads': 1}
    classic = {**common, 'version': manifest['version'], 'feature_version': FEATURE_VERSION}
    deep = {**common, 'records': [], 'errors': [], 'task': 'wearable', 'version': DEEP_VERSION}
    if not execute:
        return {'dry_run': True, 'writes': False, 'network': False, 'subjects': list(grouped), 'source_sha256': sources}
    paths = (root / BASE / 'corpus.json', root / DEEP / 'corpus.json')
    for path in paths:
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous.get('source_sha256') != sources:
                raise ValueError('existing corpus source differs; refusing overwrite, use a new version')
    # Invalidate success before processing so any later failure remains incomplete.
    write_index(paths[0], classic)
    write_index(paths[1], deep)
    with threadpool_limits(limits=1):
        for index, (subject, entries) in enumerate(grouped.items(), 1):
            print(f'prepare-v3 {index}/31 subject={subject} start', file=sys.stderr, flush=True)
            try:
                c_record, d_record = prepare_one(root, manifest, subject, entries, sources)
                classic['records'].append(c_record)
                deep['records'].append(d_record)
                print(f'prepare-v3 {index}/31 subject={subject} rows={c_record["rows"]} parity={c_record["legacy_parity"].get("exact", "not-required")}', file=sys.stderr, flush=True)
            except (OSError, ValueError, KeyError, MemoryError) as exc:
                message = f'subject {subject}: {exc}'
                classic['errors'].append(message)
                deep['errors'].append(message)
                print(message, file=sys.stderr, flush=True)
                break
            finally:
                gc.collect()
        complete = len(classic['records']) == len(deep['records']) == len(grouped) == 31 and not classic['errors']
        classic['complete'] = deep['complete'] = complete
        # Detect source drift during a run before publishing success.
        if source_hashes(root) != sources:
            classic['complete'] = deep['complete'] = False
            classic['errors'].append('source changed during preparation')
            deep['errors'].append('source changed during preparation')
        write_index(paths[0], classic)
        write_index(paths[1], deep)
    return {'dry_run': False, 'complete': classic['complete'] and deep['complete'],
            'records': len(classic['records']), 'errors': classic['errors'],
            'classic_corpus': paths[0].relative_to(root).as_posix(),
            'deep_corpus': paths[1].relative_to(root).as_posix()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = run(execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0 if result.get('dry_run') or result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
