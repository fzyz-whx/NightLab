"""Train a fresh desktop RF from the verified public ten-subject PSG corpus.

Dry-run by default. This creates a local training receipt, not a reproducibility
claim for the historical model. There is deliberately no pickle import option.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.features import AudioFeatureExtractor
from nightlab.services.model_registry import (LABELS, LOCAL_DEPENDENCIES, LOCAL_PREFIX,
    controlled_path, digest, local_run_name, local_source_hashes)
from scripts.datasets.cli import ROOT
from scripts.datasets.prepare import feature_source_hashes
from scripts.datasets.prepare_cohort import validate_receipt
from scripts.train import baselines


CORPUS = 'data/processed/psg-audio-pilot-10/corpus.json'


def validate_corpus(root):
    """Validate task, fixed cohort, source receipts and artifacts before fitting."""
    path = controlled_path(root,CORPUS)
    manifest = json.loads(controlled_path(root,'data/manifests/psg-audio-pilot-10.json').read_text(encoding='utf-8'))
    index = json.loads(path.read_text(encoding='utf-8'))
    for record in index['records']:
        for name in ('metadata.json','features.npz','timeline.npz'):
            controlled_path(root,record['path'],name)
    corpus,x,y,subjects,names,provenance = baselines.load_corpus(path,root)
    expected_records = {entry['record_id'] for entry in manifest['files']}
    records = corpus['records']
    if (manifest['id']!='psg-audio-pilot-10' or corpus['dataset']!=manifest['id']
            or corpus['version']!=manifest['version'] or manifest['license']['verified'] is not True
            or len(expected_records)!=10 or len(records)!=10
            or {record['record_id'] for record in records}!=expected_records
            or corpus['split_policy']!=manifest['subjects']
            or len(set(subjects))!=10 or names!=list(AudioFeatureExtractor().feature_names)
            or set(y)!=set(LABELS) or x.shape[1]!=67):
        raise ValueError('a complete, verified fixed PSG ten-subject classic-feature corpus is required')
    expected_sources = feature_source_hashes('psg-audio')
    for record in records:
        folder = controlled_path(root,record['path'])
        for name in ('metadata.json','features.npz','timeline.npz'):
            controlled_path(root,record['path'],name)
        subset = {**manifest,'id':'psg-audio',
                  'files':[item for item in manifest['files'] if item['record_id']==record['record_id']]}
        metadata = validate_receipt(folder,subset)
        if (metadata['dataset']!='psg-audio' or metadata['feature_names']!=names
                or metadata['feature_source_sha256']!=expected_sources):
            raise ValueError('PSG feature provenance mismatch')
        with np.load(folder/'features.npz',allow_pickle=False) as data:
            if set(data['subject_id'])!={record['record_id'].split('-')[0]}:
                raise ValueError('record subject identity mismatch')
    return path


def train(run_name,execute=False,root=None):
    root = Path(ROOT if root is None else root).resolve()
    run_name = local_run_name(run_name)
    receipt_path = controlled_path(root,'private','app-models',run_name+'.json')
    lock_path = controlled_path(root,'private','app-models',run_name+'.lock')
    outputs = [controlled_path(root,area,run_name) for area in ('runs','models')]
    if any(path.exists() for path in (*outputs,receipt_path,lock_path)):
        raise ValueError('run/model/receipt already exists; choose a new run name')
    plan = {'mode':'execute' if execute else 'dry-run','writes':execute,'network':False,
            'run_name':run_name,'model_id':LOCAL_PREFIX+run_name,'corpus':CORPUS,
            'cpu_threads':2,'training_workers':1,'gpu':False,
            'receipt':receipt_path.relative_to(root).as_posix(),
            'scope':'fresh local classic training; no same-seed replay or clinical validation claim'}
    if not execute:
        return plan
    # Validate before creating output directories. No user-selected model can be registered.
    corpus_path = validate_corpus(root)
    sources = local_source_hashes()
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    lock = lock_path.open('x',encoding='utf-8')
    try:
        with lock:
            try:
                if any(path.exists() for path in (*outputs,receipt_path)):
                    raise ValueError('run/model/receipt already exists')
                with threadpool_limits(limits=2):
                    report = baselines.run(corpus_path,run_name,root=root,feature_set='all')
                if local_source_hashes()!=sources:
                    raise ValueError('training source changed during training; refusing registration')
                receipt = {'schema_version':1,'kind':'owned_audio_rf','run_name':run_name,
                           'model_id':LOCAL_PREFIX+run_name,'dataset':report['dataset'],
                           'model_sha256':digest(controlled_path(root,'models',run_name,'random_forest.joblib')),
                           'metrics_sha256':digest(controlled_path(root,'runs',run_name,'metrics.json')),
                           'feature_names':report['feature_names'],'labels':LABELS,
                           'environment':{name:version(name) for name in LOCAL_DEPENDENCIES},
                           'source_sha256':sources}
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=receipt_path.parent,
                                                 prefix='.'+run_name+'-',suffix='.tmp',delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                # Hard-link publication is atomic and refuses to replace an existing receipt.
                os.link(temporary,receipt_path)
                return {**plan,'status':'trained','model_sha256':receipt['model_sha256']}
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    try:
        result = train(args.run_name,execute=args.execute)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({'error':str(error)},ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
