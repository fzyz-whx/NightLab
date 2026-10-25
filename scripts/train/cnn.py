"""Train fixed small CNN pilots from a complete, verified deep-input corpus."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from nightlab.deep import inputs,models,training
from nightlab.deep.training import fit_pilot, predict_probabilities
from scripts.datasets import deep_cache
from scripts.datasets.cli import ROOT,load_manifest
from scripts.datasets.dense_audio import sha256
from scripts.datasets.prepare import output_path,feature_source_hashes


def load_inputs(root,task):
    dataset = deep_cache.DATASETS[task]
    folder = output_path(root,f'data/processed/{dataset}/{deep_cache.VERSION}')
    corpus = json.loads((folder/'corpus.json').read_text(encoding='utf-8'))
    manifest_path = root/f'data/manifests/{dataset}.json'
    manifest = load_manifest(manifest_path,root)
    corpus['dataset_version'] = manifest['version']
    classic = root/f'data/processed/{dataset}/corpus.json'
    classic_records = {record['record_id']:record for record in json.loads(classic.read_text(encoding='utf-8'))['records']}
    if (not corpus['complete'] or corpus['task'] != task or corpus['dataset'] != dataset
            or corpus['version'] != deep_cache.VERSION or corpus['split_policy'] != manifest['subjects']
            or corpus['source_manifest_sha256'] != sha256(manifest_path)
            or corpus['source_corpus_sha256'] != sha256(classic)):
        raise ValueError('deep corpus differs from current source manifest/classic split')
    expected = {entry['record_id'] if task == 'audio' else entry['subject_id'] for entry in manifest['files']}
    if len(corpus['records']) != len(expected) or {record['record_id'] for record in corpus['records']} != expected:
        raise ValueError('deep corpus does not cover all planned records')
    loaded,provenance,offset = [],[],0
    for record in corpus['records']:
        path = output_path(root,record['path'])
        receipt = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        sources = receipt['provenance']
        current,expected_arrays,_ = deep_cache._record_inputs(root,task,manifest,classic_records[record['record_id']])
        current.update(deep_inputs_source_sha256=sha256(inputs.__file__),
                       deep_cache_source_sha256=sha256(deep_cache.__file__),version=deep_cache.VERSION)
        if (not record['ok'] or receipt['artifact_sha256'] != record['artifact_sha256']
                or receipt['artifact_sha256'] != sha256(path/'inputs.npz')
                or sources != current
                or sources['deep_inputs_source_sha256'] != sha256(inputs.__file__)
                or sources['deep_cache_source_sha256'] != sha256(deep_cache.__file__)
                or sources['feature_source_sha256'] != feature_source_hashes('psg-audio' if task == 'audio' else 'sleepaccel')):
            raise ValueError('deep artifact integrity/source failure')
        with np.load(path/'inputs.npz',allow_pickle=False) as data:
            arrays = {name:data[name] for name in data.files}
        deep_cache._validate_arrays(arrays,expected_arrays,task)
        subject = record['record_id'].split('-')[0]
        if (len(arrays['y']) != receipt['rows'] or len(arrays['y']) != record['rows']
                or not np.all(arrays['subject_id'] == subject) or sources['record_id'] != record['record_id']):
            raise ValueError('deep record identity/shape differs')
        loaded.append(arrays)
        provenance.append({'record_id':record['record_id'], 'path':record['path'],
                           'inputs_sha256':record['artifact_sha256'], 'metadata_sha256':sha256(path/'metadata.json'),
                           'row_offset':offset, 'rows':len(arrays['y'])})
        offset += len(arrays['y'])
    keys = set(loaded[0])
    if any(set(arrays) != keys for arrays in loaded):
        raise ValueError('incompatible deep input fields')
    return corpus,{key:np.concatenate([arrays[key] for arrays in loaded]) for key in keys},provenance


def run(root,task,run_name,feature_set='all',device='cpu',smoke=False,config=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',run_name):
        raise ValueError('portable new run name required')
    root = Path(root).resolve()
    output,model_folder = root/'runs'/run_name,root/'models'/run_name
    for folder,parent in ((output,root/'runs'),(model_folder,root/'models')):
        if folder.exists() or not folder.resolve().is_relative_to(parent):
            raise ValueError('run/model output exists or escapes repository')
    corpus,arrays,provenance = load_inputs(root,task)
    state,report,probabilities,features = fit_pilot(task,arrays['values'],arrays.get('masks'),arrays['y'],
        arrays['subject_id'],arrays['supervised_mask'],corpus['split_policy']['subjects'],feature_set,device,config,smoke,
        progress=lambda item:print(json.dumps(item),flush=True))
    model_folder.mkdir(parents=True)
    checkpoint = model_folder/'weights.pt'
    torch.save(state,checkpoint)
    checksum = sha256(checkpoint)
    if sha256(checkpoint) != checksum:
        raise ValueError('owned checkpoint changed before load')
    loaded = models.build_model(task,in_channels=features.shape[1]).to(report['device'])
    loaded.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    for part,item in probabilities.items():
        repeated = predict_probabilities(loaded,features[item['rows']],torch.device(report['device']),report['config']['batch_size'])
        if not np.array_equal(repeated,item['probabilities']):
            raise ValueError('saved/loaded owned checkpoint predictions differ')
    report.update(dataset=corpus['dataset'], dataset_version=corpus['dataset_version'], input_version=corpus['version'],
                  split_policy=corpus['split_policy'],
                  provenance=provenance, checkpoint_sha256=checksum, saved_loaded_predictions_exact=True,
                  source_sha256={name:sha256(module.__file__) for name,module in
                                 (('models',models),('training',training),('inputs',inputs),('deep_cache',deep_cache))},
                  runner_source_sha256=sha256(__file__))
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz', **{f'{part}_{name}':value for part,item in probabilities.items()
                                                    for name,value in item.items()})
    report['predictions_sha256'] = sha256(output/'predictions.npz')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (model_folder/'metadata.json').write_text(json.dumps({key:report[key] for key in
        ('task','feature_set','label_order','normalizer','checkpoint_sha256','source_sha256','best_epoch')},indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('audio','wearable'),required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--feature-set',choices=('all','heart_rate','acceleration'),default='all')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--smoke',action='store_true',help='only five train batches and validation; no test evaluation')
    args = parser.parse_args()
    report = run(ROOT,args.task,args.run_name,args.feature_set,args.device,args.smoke)
    print(json.dumps({key:report[key] for key in ('models','best_epoch','device','peak_allocated_gpu_bytes','elapsed_seconds','smoke_only')},indent=2))


if __name__ == '__main__':
    main()
