"""Run fixed classic models only after every planned subject is prepared."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np

from nightlab.experiments.baseline import fit_baselines
from scripts.datasets.cli import ROOT
from scripts.datasets.prepare import output_path


def load_corpus(path, root=ROOT):
    corpus = json.loads(Path(path).read_text(encoding='utf-8'))
    if not corpus.get('complete'):
        raise ValueError('planned cohort is incomplete; refusing evaluation')
    matrices, targets, subjects, provenance, feature_names = [], [], [], [], None
    for record in corpus['records']:
        if not record['ok']:
            raise ValueError('failed record in corpus')
        folder = output_path(root,record['path'])
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        expected = metadata.get('artifact_sha256',{}).get('features.npz')
        digest = hashlib.sha256((folder/'features.npz').read_bytes()).hexdigest()
        if not expected or digest!=expected or not metadata['input_integrity']['ok']:
            raise ValueError('prepared artifact integrity failure')
        if feature_names is not None and feature_names!=metadata['feature_names']:
            raise ValueError('incompatible feature schemas')
        feature_names = metadata['feature_names']
        with np.load(folder/'features.npz',allow_pickle=False) as data:
            x,y,ids = data['X'],data['y'],data['subject_id']
        if x.shape!=(metadata['rows'],len(feature_names)) or y.shape!=(len(x),) or ids.shape!=y.shape:
            raise ValueError('prepared artifact shape mismatch')
        matrices.append(x);targets.append(y);subjects.append(ids)
        provenance.append({'record_id':record['record_id'],'features_path':record['path'],
                           'features_sha256':digest,'feature_version':metadata['feature_version'],
                           'feature_source_sha256':metadata['feature_source_sha256']})
    if not matrices:
        raise ValueError('empty corpus')
    return corpus,np.concatenate(matrices),np.concatenate(targets),np.concatenate(subjects),feature_names,provenance


def run(path, run_name, root=ROOT, feature_set='all'):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',run_name):
        raise ValueError('run name must be a portable identifier')
    corpus,x,y,subjects,names,provenance = load_corpus(path,root)
    wearable = corpus['dataset'].startswith('sleepaccel')
    if feature_set not in {'all','heart_rate','acceleration'} or (not wearable and feature_set!='all'):
        raise ValueError('invalid feature subset for this task')
    if feature_set!='all':
        prefix = 'hr_' if feature_set=='heart_rate' else 'accel_'
        columns = [index for index,name in enumerate(names) if name.startswith(prefix)]
        if not columns:
            raise ValueError('selected modality has no features')
        x = x[:,columns]
        names = [names[index] for index in columns]
    labels = ['W','N1','N2','N3','REM'] if wearable else ['no_target_reference_event','reference_respiratory_event']
    assignments = corpus['split_policy']['subjects']
    run_folder,model_folder = root/'runs'/run_name,root/'models'/run_name
    for folder in (run_folder,model_folder):
        allowed = root/('runs' if folder==run_folder else 'models')
        if not folder.resolve().is_relative_to(allowed):
            raise ValueError('run or model path escapes repository location')
        if folder.exists():
            raise ValueError('run/model output exists; choose a new run name')
    started = time.perf_counter()
    models,results = fit_baselines(x,y,subjects,assignments,labels,
                                   positive_label=None if wearable else 'reference_respiratory_event')
    results.update(dataset=corpus['dataset'],dataset_version=corpus['version'],feature_subset=feature_set,
                   subject_count=len(set(subjects)),feature_count=x.shape[1],feature_names=names,
                   feature_nan_count=int(np.isnan(x).sum()),provenance=provenance,
                   split_policy=corpus['split_policy'],elapsed_seconds=time.perf_counter()-started,
                   environment={'python':platform.python_version(),**{name:version(name) for name in ('nightlab','numpy','scipy','scikit-learn','joblib')}},
                   experiment_scope='pilot held-out subjects; no population or clinical validation')
    run_folder.mkdir(parents=True);model_folder.mkdir(parents=True)
    for name,model in models.items():
        joblib.dump(model,model_folder/f'{name}.joblib')
    (run_folder/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--feature-set',choices=('all','heart_rate','acceleration'),default='all')
    args = parser.parse_args()
    try:
        results = run(Path(args.corpus),args.run_name,feature_set=args.feature_set)
        print(json.dumps({'dataset':results['dataset'],'subject_count':results['subject_count'],
                          'models':results['models'],'output':f'runs/{args.run_name}/metrics.json'},indent=2))
        return 0
    except (OSError,ValueError,KeyError) as error:
        print(json.dumps({'error':str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
