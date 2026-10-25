"""Evaluate saved CNN probabilities using the fixed continuous-event protocol."""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.events import decode_events, evaluate_events
from nightlab.audio.reference import read_reference
from nightlab.experiments.splits import validate_subject_assignments
from scripts.datasets.cli import ROOT, load_manifest, raw_path
from scripts.datasets.dense_audio import sha256
from scripts.datasets.prepare import output_path
from scripts.evaluate.audio_events import aggregate_events, clipped_references
from scripts.train.cnn import load_inputs


LABEL_ORDER = ['no_target_reference_event', 'reference_respiratory_event']


def validate_predictions(predictions, subjects, assignments, label_order):
    """Require every held-out window, in its exact global corpus row order."""
    if label_order != LABEL_ORDER:
        raise ValueError('unexpected CNN class order')
    subjects = np.asarray(subjects)
    validate_subject_assignments(assignments, subjects)
    expected_keys = {f'{part}_{kind}' for part in ('validation', 'test')
                     for kind in ('rows', 'probabilities')}
    if set(predictions) != expected_keys:
        raise ValueError('expected validation/test rows and probabilities only')
    results = {}
    for part in ('validation', 'test'):
        rows = np.asarray(predictions[f'{part}_rows'])
        probabilities = np.asarray(predictions[f'{part}_probabilities'])
        expected = np.flatnonzero(np.isin(subjects, assignments[part]))
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(expected)
                or not np.array_equal(rows, expected)):
            raise ValueError(f'{part} rows must cover all held-out windows in corpus order')
        if (probabilities.shape != (len(expected), 2) or probabilities.dtype.kind != 'f'
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
            raise ValueError(f'{part} probabilities must be finite normalized N x 2 values')
        results[part] = (rows, probabilities[:, 1])
    return results


def _named_path(root, category, name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
        raise ValueError('portable run/output name required')
    allowed = root / category
    result = allowed / name
    if allowed.resolve() != allowed or not result.resolve().is_relative_to(allowed):
        raise ValueError(f'path escapes repository {category}')
    return result


def run(root, run_name, output_name):
    root = Path(root).resolve()
    source = _named_path(root, 'runs', run_name)
    output = _named_path(root, 'runs', output_name)
    model = _named_path(root, 'models', run_name)
    if output.exists():
        raise ValueError('output exists; refusing overwrite')
    started = time.perf_counter()
    report_path = source / 'metrics.json'
    predictions_path = source / 'predictions.npz'
    checkpoint = model / 'weights.pt'
    for path, allowed in ((report_path, source), (predictions_path, source), (checkpoint, model)):
        if not path.resolve().is_relative_to(allowed.resolve()):
            raise ValueError('source file escapes its run directory')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('smoke_only') is not False or report.get('task') != 'audio'
            or report.get('feature_set') != 'all' or report.get('saved_loaded_predictions_exact') is not True):
        raise ValueError('a completed full audio CNN run with verified saved predictions is required')
    if (report.get('predictions_sha256') != sha256(predictions_path)
            or report.get('checkpoint_sha256') != sha256(checkpoint)):
        raise ValueError('prediction/checkpoint checksum differs from model report')
    corpus, arrays, provenance = load_inputs(root, 'audio')
    if (report.get('provenance') != provenance or report.get('split_policy') != corpus['split_policy']
            or report.get('dataset') != corpus['dataset']):
        raise ValueError('CNN run provenance/split differs from current verified inputs')
    with np.load(predictions_path, allow_pickle=False) as data:
        predictions = {key: data[key] for key in data.files}
    assignments = corpus['split_policy']['subjects']
    validated = validate_predictions(predictions, arrays['subject_id'], assignments, report.get('label_order'))
    manifest_path = root / f"data/manifests/{corpus['dataset']}.json"
    manifest = load_manifest(manifest_path, root)
    dense_records = {record['record_id']: record for record in corpus['records']}
    records = []
    with threadpool_limits(limits=2):
        for part in ('validation', 'test'):
            global_rows, positive = validated[part]
            selected_subjects = set(assignments[part])
            selected = [item for item in provenance if item['record_id'].split('-')[0] in selected_subjects]
            if (len(selected) != len(selected_subjects)
                    or {item['record_id'].split('-')[0] for item in selected} != selected_subjects):
                raise ValueError('expected one complete recording per held-out subject')
            for item in selected:
                record_id = item['record_id']
                left, right = item['row_offset'], item['row_offset'] + item['rows']
                rows = np.arange(left, right)
                positions = np.searchsorted(global_rows, rows)
                if (not len(rows) or np.any(positions >= len(global_rows))
                        or not np.array_equal(global_rows[positions], rows)):
                    raise ValueError('record predictions are incomplete')
                folder = output_path(root, dense_records[record_id]['path'])
                if sha256(folder / 'metadata.json') != item['metadata_sha256']:
                    raise ValueError('deep metadata changed after input verification')
                receipt = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
                context = receipt['provenance']
                rml = [entry for entry in manifest['files']
                       if entry.get('record_id') == record_id and entry['path'].endswith('.rml')]
                if len(rml) != 1:
                    raise ValueError('one verified RML required per record')
                _, _, reference_events = read_reference(raw_path(root, rml[0]['path']))
                duration = context['duration_seconds']
                references, censored, kinds = clipped_references(reference_events, context['offset_seconds'], duration)
                probabilities = positive[positions]
                decoded = decode_events(arrays['start_seconds'][rows], arrays['end_seconds'][rows], probabilities, duration)
                metrics = evaluate_events(decoded['events'], references, duration, censored)
                records.append({'record_id': record_id, 'split': part,
                                'reference_intervals': references, 'reference_censored': censored,
                                'reference_kinds': kinds, 'deep_receipt': receipt,
                                'models': {'cnn': {'window_probabilities': probabilities.tolist(),
                                                   'decoded': decoded, 'metrics': metrics}}})
    models = {'cnn': {part: aggregate_events([record['models']['cnn']['metrics'] for record in records
                                             if record['split'] == part]) for part in ('validation', 'test')}}
    for part in ('validation', 'test'):
        selected = [record for record in records if record['split'] == part]
        models['cnn'][part]['reference_count'] = sum(len(record['reference_intervals']) for record in selected)
        models['cnn'][part]['censored_reference_count'] = sum(sum(record['reference_censored']) for record in selected)
    common = {'schema_version': 1, 'protocol': 'docs/EVENT_PROTOCOL.md',
              'protocol_sha256': sha256(root / 'docs/EVENT_PROTOCOL.md'),
              'evaluation_source_sha256': sha256(__file__),
              'event_source_sha256': sha256(root / 'backend/src/nightlab/audio/events.py'),
              'dataset': corpus['dataset'], 'training_run': run_name,
              'training_metrics_sha256': sha256(report_path),
              'predictions_sha256': sha256(predictions_path), 'checkpoint_sha256': sha256(checkpoint),
              'manifest_sha256': sha256(manifest_path), 'models': models,
              'numeric_threads_max': 2, 'worker_count': 1, 'elapsed_seconds': time.perf_counter() - started,
              'scope': 'fixed pilot event evaluation; no retraining, threshold search or clinical validation'}
    local = {**common, 'split_policy': corpus['split_policy'], 'provenance': provenance, 'records': records}
    public_records = []
    for part in ('validation', 'test'):
        for index, record in enumerate((record for record in records if record['split'] == part), 1):
            public_records.append({'participant': f'{part}-{index:02d}', 'split': part,
                                   'reference_count': len(record['reference_intervals']),
                                   'censored_reference_count': sum(record['reference_censored']),
                                   'models': {'cnn': {key: value for key, value in record['models']['cnn']['metrics'].items()
                                                      if key != 'matches'}}})
    public = {**common, 'records': public_records}
    output.mkdir(parents=True)
    (output / 'events.json').write_text(json.dumps(local, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(public, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    result = run(ROOT, args.run_name, args.output_name)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
