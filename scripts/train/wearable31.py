"""Independent v3 cohort loader and fixed-protocol experiment runner.

Private caches, predictions and fitted moments stay under ignored runs/models.
The old three-person runner and source modules are intentionally unchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nightlab.deep import inputs, models, training
from nightlab.deep.training import TrainingConfig, fit_pilot, predict_probabilities
from scripts.datasets import audit_sleepaccel31 as audit
from scripts.datasets import prepare_sleepaccel31 as prepare
from scripts.datasets.cli import ROOT, raw_path
from scripts.datasets.prepare import output_path
from scripts.train import baselines


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_arrays(deep, classic, subject):
    required = {'values', 'masks', 'y', 'subject_id', 'start_seconds', 'end_seconds', 'supervised_mask'}
    if set(deep) != required:
        raise ValueError('unexpected deep fields')
    n = len(deep['y'])
    if (not n or deep['values'].shape != (n, 5, 30) or deep['masks'].shape != (n, 2, 30)
            or deep['supervised_mask'].dtype != np.bool_ or not deep['supervised_mask'].all()
            or deep['supervised_mask'].shape != (n,)
            or not np.isin(deep['y'], training.LABELS['wearable']).all()):
        raise ValueError('deep shapes or supervised targets differ')
    for key in ('y', 'subject_id', 'start_seconds', 'end_seconds'):
        if deep[key].shape != (n,) or not np.array_equal(deep[key], classic[key]):
            raise ValueError('classic/deep identity or timeline differs')
    if (not np.all(deep['subject_id'] == subject) or not np.isfinite(deep['start_seconds']).all()
            or np.any(np.diff(deep['start_seconds']) <= 0)
            or not np.array_equal(deep['end_seconds'], deep['start_seconds'] + 30)):
        raise ValueError('record identity or epoch times invalid')
    masks = deep['masks']
    observed = masks[:, [0, 1, 1, 1, 1], :].astype(bool)
    if (not np.isin(masks, (0, 1)).all() or not np.isfinite(deep['values'][observed]).all()
            or not np.isnan(deep['values'][~observed]).all()):
        raise ValueError('missing values and observation masks differ')


def load_inputs(root=ROOT):
    root = Path(root).resolve()
    manifest, grouped = audit.metadata(root)
    sources = prepare.source_hashes(root)
    classic_path, deep_path = root / prepare.BASE / 'corpus.json', root / prepare.DEEP / 'corpus.json'
    classic, corpus = read_json(classic_path), read_json(deep_path)
    for item in (classic, corpus):
        if (item.get('complete') is not True or item.get('errors') or item['dataset'] != manifest['id']
                or item['source_sha256'] != sources or item['split_policy'] != manifest['subjects']
                or len(item['records']) != 31
                or {r['record_id'] for r in item['records']} != set(grouped)):
            raise ValueError('incomplete, stale, duplicated or foreign cohort/split')
    if classic['feature_version'] != prepare.FEATURE_VERSION or corpus['version'] != prepare.DEEP_VERSION or corpus['task'] != 'wearable':
        raise ValueError('v3 versions required')
    classics = {r['record_id']: r for r in classic['records']}
    loaded, provenance = [], []
    for record in corpus['records']:
        subject = record['record_id']
        entries = grouped[subject]
        raw_hashes = {}
        for entry in entries.values():
            path = raw_path(root, entry['path'])
            digest = audit.sha256(path)
            if path.stat().st_size != entry['exact_bytes'] or digest != entry['official_checksums']['sha256']:
                raise ValueError('raw official size/SHA differs')
            raw_hashes[entry['path']] = digest
        c_record = classics[subject]
        c_folder, folder = output_path(root, c_record['path']), output_path(root, record['path'])
        if c_folder != root / prepare.BASE / subject / 'features-v3' or folder != root / prepare.DEEP / subject:
            raise ValueError('record path and identity differ')
        c_meta, meta = read_json(c_folder / 'metadata.json'), read_json(folder / 'metadata.json')
        for rec, receipt in ((c_record, c_meta), (record, meta)):
            expected = {'source_sha256': sources, 'raw_input_sha256': raw_hashes,
                        'canonicalization_version': 'sleepaccel-canonical-v3', 'subject_id': subject,
                        'partition': entries['labels']['split'], 'legacy_parity': receipt['legacy_parity']}
            if (rec['ok'] is not True or receipt['provenance'] != expected
                    or receipt['dataset'] != manifest['id'] or receipt['rows'] != rec['rows']
                    or not receipt['input_integrity']['ok'] or not receipt['input_integrity']['official_sha256_verified']
                    or rec['legacy_parity'] != receipt['legacy_parity']):
                raise ValueError('record provenance or raw receipt differs')
        if c_meta['feature_version'] != prepare.FEATURE_VERSION or meta['version'] != prepare.DEEP_VERSION:
            raise ValueError('record version differs')
        for name, digest in c_meta['artifact_sha256'].items():
            if audit.sha256(c_folder / name) != digest:
                raise ValueError('classical artifact SHA differs')
        if audit.sha256(folder / 'inputs.npz') != meta['artifact_sha256'] or meta['artifact_sha256'] != record['artifact_sha256']:
            raise ValueError('deep artifact SHA differs')
        c_arrays, arrays = prepare.load_arrays(c_folder / 'features.npz'), prepare.load_arrays(folder / 'inputs.npz')
        validate_arrays(arrays, c_arrays, subject)
        if (len(arrays['y']) != meta['rows'] or list(arrays['values'].shape) != meta['shape']
                or record['shape'] != meta['shape']):
            raise ValueError('metadata tensor shape differs')
        parity = prepare.legacy_parity(root, subject, c_arrays, arrays)
        if parity != meta['legacy_parity']:
            raise ValueError('legacy parity differs')
        loaded.append(arrays)
        provenance.append({'record_id': subject, 'inputs_sha256': meta['artifact_sha256'],
                           'metadata_sha256': audit.sha256(folder / 'metadata.json')})
    corpus['dataset_version'] = manifest['version']
    corpus['verified_corpus_sha256'] = {'classic': audit.sha256(classic_path), 'deep': audit.sha256(deep_path)}
    return corpus, {key: np.concatenate([a[key] for a in loaded]) for key in loaded[0]}, provenance


def run_cnn(corpus, arrays, provenance, name, subset, smoke=False, root=ROOT):
    output, model_folder = root / 'runs' / name, root / 'models' / name
    if output.exists() or model_folder.exists():
        raise ValueError('run outputs already exist')
    state, report, probabilities, features = fit_pilot('wearable', arrays['values'], arrays['masks'], arrays['y'],
        arrays['subject_id'], arrays['supervised_mask'], corpus['split_policy']['subjects'], subset, 'cuda',
        TrainingConfig(cpu_threads=1), smoke, progress=lambda x: print(json.dumps({'run': name, **x}), flush=True))
    model_folder.mkdir(parents=True)
    checkpoint = model_folder / 'weights.pt'
    torch.save(state, checkpoint)
    loaded = models.build_model('wearable', in_channels=features.shape[1]).to(report['device'])
    loaded.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    for item in probabilities.values():
        repeated = predict_probabilities(loaded, features[item['rows']], torch.device(report['device']), 64)
        if not np.array_equal(repeated, item['probabilities']):
            raise ValueError('saved/loaded probabilities differ')
    report.update(dataset=corpus['dataset'], dataset_version=corpus['dataset_version'], input_version=corpus['version'],
        verified_corpus_sha256=corpus['verified_corpus_sha256'], provenance=provenance,
        checkpoint_sha256=audit.sha256(checkpoint), saved_loaded_predictions_exact=True,
        source_sha256={k: audit.sha256(m.__file__) for k, m in [('inputs', inputs), ('models', models), ('training', training)]},
        runner_source_sha256=audit.sha256(__file__))
    output.mkdir(parents=True)
    np.savez_compressed(output / 'predictions.npz', **{f'{part}_{key}': value for part, item in probabilities.items() for key, value in item.items()})
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def compare_cnn(first, second, root=ROOT):
    a, b = (read_json(root / 'runs' / name / 'metrics.json') for name in (first, second))
    def history(report):
        return [{k: v for k, v in item.items() if k != 'elapsed_seconds'} for item in report['history']]
    checks = {'history_exact_excluding_timing': history(a) == history(b), 'metrics_exact': a['models'] == b['models'],
              'best_epoch_exact': a['best_epoch'] == b['best_epoch'], 'checkpoint_bytes_exact': a['checkpoint_sha256'] == b['checkpoint_sha256']}
    left, right = (prepare.load_arrays(root / 'runs' / name / 'predictions.npz') for name in (first, second))
    checks['probabilities_and_rows_exact'] = set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)
    if not all(checks.values()):
        raise ValueError(f'CNN reproducibility failure: {checks}')
    return checks


def run_rf(corpus, subset, name, root=ROOT):
    captured = {}
    original_dump = joblib.dump
    def capture_dump(model, path):
        captured[Path(path).stem] = model
        return original_dump(model, path)
    joblib.dump = capture_dump
    try:
        report = baselines.run(root / prepare.BASE / 'corpus.json', name, root, subset)
    finally:
        joblib.dump = original_dump
    _, x, _, subjects, names, _ = baselines.load_corpus(root / prepare.BASE / 'corpus.json', root)
    if subset != 'all':
        x = x[:, [i for i, field in enumerate(names) if field.startswith('hr_' if subset == 'heart_rate' else 'accel_')]]
    # Compare the actual fitted in-memory estimator to its saved/loaded artifact.
    model_path = root / 'models' / name / 'random_forest.joblib'
    model, loaded = captured['random_forest'], joblib.load(model_path)
    rows = np.isin(subjects, corpus['split_policy']['subjects']['validation'] + corpus['split_policy']['subjects']['test'])
    probabilities = model.predict_proba(x[rows])
    if not np.array_equal(probabilities, loaded.predict_proba(x[rows])):
        raise ValueError('RF deserialization predictions differ')
    np.save(root / 'runs' / name / 'rf-probabilities.npy', probabilities)
    report.update(saved_loaded_predictions_exact=True, checkpoint_sha256=audit.sha256(model_path),
                  verified_corpus_sha256=corpus['verified_corpus_sha256'], runner_source_sha256=audit.sha256(__file__))
    (root / 'runs' / name / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def public_report(report, checks):
    result = copy.deepcopy(report)
    for key in ('normalizer', 'provenance', 'split_policy', 'split_subjects'):
        result.pop(key, None)
    result['subject_counts'] = {'train': 18, 'validation': 6, 'test': 7}
    result['same_seed_rerun'] = checks
    result['scope'] = '31-person subject-held-out research experiment; epoch-pooled metrics; no clinical or real-device validation'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        corpus, arrays, provenance = load_inputs()
        print(json.dumps({'verified': True, 'rows': len(arrays['y']), 'corpus': corpus['verified_corpus_sha256']}), flush=True)
        if args.smoke:
            print(json.dumps(run_cnn(corpus, arrays, provenance, 'wearable31-cnn-smoke-v3', 'all', True)), flush=True)
        elif args.execute:
            for subset, short in [('heart_rate', 'hr'), ('acceleration', 'accel'), ('all', 'fusion')]:
                cnn = f'wearable31-cnn-{short}-v3'
                report = run_cnn(corpus, arrays, provenance, cnn, subset)
                run_cnn(corpus, arrays, provenance, cnn + '-repro', subset)
                checks = compare_cnn(cnn, cnn + '-repro')
                (ROOT / 'docs/results' / f'{cnn}.json').write_text(json.dumps(public_report(report, checks), indent=2) + '\n', encoding='utf-8')
                rf = f'wearable31-rf-{short}-v3'
                report = run_rf(corpus, subset, rf)
                repeated = run_rf(corpus, subset, rf + '-repro')
                checks = {'metrics_exact': report['models'] == repeated['models'],
                          'checkpoint_bytes_exact': report['checkpoint_sha256'] == repeated['checkpoint_sha256'],
                          'probabilities_exact': np.array_equal(np.load(ROOT / 'runs' / rf / 'rf-probabilities.npy'), np.load(ROOT / 'runs' / (rf + '-repro') / 'rf-probabilities.npy'))}
                if not all(checks.values()):
                    raise ValueError(f'RF reproducibility failure: {checks}')
                (ROOT / 'docs/results' / f'{rf}.json').write_text(json.dumps(public_report(report, checks), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
