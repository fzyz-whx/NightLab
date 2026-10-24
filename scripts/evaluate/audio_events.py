"""Evaluate frozen owned audio baselines on complete held-out recordings."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.events import decode_events, evaluate_events
from nightlab.audio.reference import TARGET_TYPES
from scripts.datasets.cli import ROOT, load_manifest
from scripts.datasets.dense_audio import ensure_dense, sha256
from scripts.train.baselines import load_corpus


def clipped_references(events, offset, duration):
    intervals, censored, kinds = [], [], []
    for event in events:
        if event.kind not in TARGET_TYPES or event.end <= offset or event.start >= offset+duration:
            continue
        intervals.append([max(0.0,event.start-offset), min(duration,event.end-offset)])
        censored.append(event.start < offset or event.end > offset+duration)
        kinds.append(event.kind)
    return intervals, censored, kinds


def aggregate_events(reports):
    """Micro rates and conditional localization, never match across recordings."""
    if not reports:
        raise ValueError('cannot aggregate an empty split')
    tp, fp, fn = (sum(report[key] for report in reports) for key in ('tp','fp','fn'))
    seconds = sum(report['observed_seconds'] for report in reports)
    if seconds <= 0:
        raise ValueError('positive observation time required')
    localized = sum(report['localization']['matched_uncensored_count'] for report in reports)
    localization = {'matched_uncensored_count':localized}
    for name in ('mean_onset_error_seconds','mean_end_error_seconds',
                 'mean_absolute_onset_error_seconds','mean_absolute_end_error_seconds'):
        localization[name] = (sum(report['localization'][name]*report['localization']['matched_uncensored_count']
                                  for report in reports if report['localization']['matched_uncensored_count'])/localized
                              if localized else None)
    return {'tp':tp, 'fp':fp, 'fn':fn, 'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None, 'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'false_positives_per_hour':fp*3600/seconds, 'observed_seconds':seconds,
            'record_count':len(reports), 'localization':localization}


def load_owned_models(root, baseline_name, baseline_report):
    """Joblib is executable: only admit own previously verified model digests."""
    trusted = json.loads((root/'docs/results/baseline-reproducibility.json').read_text(encoding='utf-8'))
    checks = [check for check in trusted['checks'] if check['run'] == baseline_name
              and check['owned_saved_model_predictions_match_retrained_models']]
    if len(checks) != 1:
        raise ValueError('baseline has no unique owned-model receipt')
    hashes = checks[0]['local_model_sha256']
    if set(hashes) != {'majority','random_forest'} or set(baseline_report['models']) != set(hashes):
        raise ValueError('unexpected model inventory')
    paths = {name:root/'models'/baseline_name/f'{name}.joblib' for name in hashes}
    allowed = (root/'models').resolve()
    if allowed != root.resolve()/'models' or any(not path.resolve().is_relative_to(allowed) for path in paths.values()):
        raise ValueError('owned model path escapes repository')
    # Verify every file before loading any executable serialization.
    if any(sha256(paths[name]) != digest for name,digest in hashes.items()):
        raise ValueError('owned model checksum mismatch; refusing joblib load')
    return {name:joblib.load(path) for name,path in paths.items()}, hashes


def validate_assignments(assignments, subjects):
    if set(assignments) != {'train','validation','test'}:
        raise ValueError('three named subject sets required')
    groups = [set(assignments[name]) for name in ('train','validation','test')]
    if any(not group for group in groups) or any(len(group) != len(assignments[name]) for group,name in
                                                zip(groups,('train','validation','test'),strict=True)):
        raise ValueError('nonempty unique subject sets required')
    if any(groups[a] & groups[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('subject leakage across splits')
    if set.union(*groups) != set(subjects):
        raise ValueError('subject assignment differs from prepared subjects')


def evaluate(root, run_name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',run_name):
        raise ValueError('portable run name required')
    output = root/'runs'/run_name
    allowed = root.resolve()/'runs'
    if not output.resolve().is_relative_to(allowed) or output.exists():
        raise ValueError('output exists or escapes runs; choose a new run name')
    started = time.perf_counter()
    dataset, baseline_name = 'psg-audio-pilot-10', 'psg-rf-pilot-v1'
    manifest_path = root/f'data/manifests/{dataset}.json'
    corpus_path = root/f'data/processed/{dataset}/corpus.json'
    manifest = load_manifest(manifest_path,root)
    corpus,_,_,subjects,names,provenance = load_corpus(corpus_path,root)
    split = corpus['split_policy']['subjects']
    validate_assignments(split, subjects)
    baseline_path = root/f'runs/{baseline_name}/metrics.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if (corpus['dataset'] != dataset or corpus['version'] != manifest['version']
            or corpus['split_policy'] != manifest['subjects']
            or baseline['dataset'] != dataset or baseline['split_policy'] != corpus['split_policy']
            or baseline['provenance'] != provenance or baseline['feature_names'] != names
            or baseline['feature_count'] != len(names)
            or baseline.get('feature_subset','all') != 'all'):
        raise ValueError('baseline/corpus/manifest provenance or split differs')
    with threadpool_limits(limits=2):
        models, model_hashes = load_owned_models(root,baseline_name,baseline)
        records = []
        for part in ('validation','test'):
            selected = [record for record in corpus['records'] if record['record_id'].split('-')[0] in split[part]]
            if (len(selected) != len(split[part]) or
                    {record['record_id'].split('-')[0] for record in selected} != set(split[part])):
                raise ValueError('expected one complete recording per held-out subject')
            for record in selected:
                arrays, receipt, events = ensure_dense(root,manifest,record)
                context = receipt['provenance']
                duration = context['duration_seconds']
                references, censored, kinds = clipped_references(events,context['offset_seconds'],duration)
                result = {'record_id':record['record_id'], 'split':part, 'dense_receipt':receipt,
                          'reference_intervals':references, 'reference_kinds':kinds,
                          'reference_censored':censored, 'models':{}}
                for name,model in models.items():
                    labels = list(model.classes_)
                    if set(labels) != {'no_target_reference_event','reference_respiratory_event'}:
                        raise ValueError('unexpected model classes')
                    probabilities = model.predict_proba(arrays['X'])[:,labels.index('reference_respiratory_event')]
                    decoded = decode_events(arrays['start_seconds'],arrays['end_seconds'],probabilities,duration)
                    metrics = evaluate_events(decoded['events'],references,duration,censored)
                    result['models'][name] = {'window_probabilities':probabilities.tolist(),
                                              'decoded':decoded, 'metrics':metrics}
                records.append(result)
                print(json.dumps({'record_id':record['record_id'], 'split':part,
                                  'reference_count':len(references), 'dense_windows':receipt['rows']}),flush=True)
    report = {'schema_version':1, 'protocol':'docs/EVENT_PROTOCOL.md',
              'protocol_sha256':sha256(root/'docs/EVENT_PROTOCOL.md'),
              'evaluation_source_sha256':sha256(__file__),
              'event_source_sha256':sha256(root/'backend/src/nightlab/audio/events.py'),
              'dataset':dataset, 'dataset_version':corpus['version'], 'split_policy':corpus['split_policy'],
              'manifest_sha256':sha256(manifest_path), 'baseline_run':baseline_name,
              'baseline_feature_subset':'all; exact full feature names and count verified (legacy report may omit feature_subset)',
              'baseline_metrics_sha256':sha256(baseline_path), 'owned_model_sha256':model_hashes,
              'models':{name:{part:aggregate_events([record['models'][name]['metrics'] for record in records
                                                    if record['split'] == part])
                              for part in ('validation','test')} for name in models},
              'records':records, 'numeric_threads_max':2, 'worker_count':1,
              'elapsed_seconds':time.perf_counter()-started,
              'environment':{'python':platform.python_version(),**{name:version(name) for name in
                              ('nightlab','numpy','scipy','scikit-learn','joblib')}},
              'scope':'pilot event evaluation; no retraining, threshold search, population or clinical validation'}
    public = {key:value for key,value in report.items() if key != 'records'}
    public['records'] = [{
        'record_id':record['record_id'], 'split':record['split'],
        'reference_count':len(record['reference_intervals']), 'censored_reference_count':sum(record['reference_censored']),
        'dense_receipt':record['dense_receipt'],
        'models':{name:{key:value for key,value in item['metrics'].items() if key != 'matches'}
                  for name,item in record['models'].items()},
    } for record in records]
    output.mkdir(parents=True)
    (output/'events.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    args = parser.parse_args()
    try:
        report = evaluate(ROOT,args.run_name)
        print(json.dumps({'models':report['models'], 'output':f'runs/{args.run_name}/summary.json'},indent=2))
        return 0
    except (OSError,ValueError,KeyError) as error:
        print(json.dumps({'error':str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
