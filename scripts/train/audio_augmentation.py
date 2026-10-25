"""Train the preregistered noisy-copy and equal-row clean-duplicate CNN controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from nightlab.deep import inputs,models,training
from nightlab.deep.training import fit_pilot,predict_probabilities
from scripts.datasets import noise_augmentation as augmentation
from scripts.datasets.dense_audio import sha256
from scripts.datasets.prepare import output_path
from scripts.datasets.cli import ROOT
from scripts.train.cnn import load_inputs


MODES = ('clean_duplicate','noise_augmented')


def combine_training_arrays(base,extra,origins,assignments,mode):
    """Only add each scored train row once; both controls retain identical order."""
    if mode not in MODES:
        raise ValueError('unknown augmentation control')
    origins = np.asarray(origins)
    expected = np.flatnonzero(np.isin(base['subject_id'],assignments['train']) & base['supervised_mask'])
    if (origins.ndim != 1 or origins.dtype.kind not in 'iu' or len(origins) != len(expected)
            or not np.array_equal(np.sort(origins),expected)):
        raise ValueError('one copy of every scored training row, and no other rows, required')
    for key in ('y','subject_id','start_seconds','end_seconds'):
        if not np.array_equal(extra[key],base[key][origins]):
            raise ValueError(f'augmentation {key} differs from its original training row')
    values = extra['values'] if mode == 'noise_augmented' else base['values'][origins]
    if values.shape != (len(origins),32,100) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('invalid selected augmentation image matrix')
    combined = {key:np.concatenate((base[key],extra[key])) for key in ('y','subject_id','start_seconds','end_seconds')}
    combined['values'] = np.concatenate((base['values'],values))
    combined['supervised_mask'] = np.r_[base['supervised_mask'],np.ones(len(origins),dtype=bool)]
    return combined


def load_augmented_inputs(root,mode):
    corpus,base,base_provenance = load_inputs(root,'audio')
    folder = output_path(root,f'data/processed/psg-audio-pilot-10/{augmentation.VERSION}')
    index_path = folder/'corpus.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    sources = augmentation._sources(root)
    _,_,expected_records = augmentation._select(root)
    expected_ids = {record['record_id'] for record in expected_records}
    if (not index['complete'] or index['dataset'] != corpus['dataset']
            or index['version'] != augmentation.VERSION or index['dataset_version'] != 'V3'
            or index['split_policy'] != corpus['split_policy'] or index['source_hashes'] != sources
            or len(index['records']) != 6 or {record['record_id'] for record in index['records']} != expected_ids):
        raise ValueError('complete current six-subject augmentation cache required')
    records_by_id = {item['record_id']:item for item in base_provenance}
    extras,origins,provenance,condition_counts = [],[],[],{}
    for record in sorted(index['records'],key=lambda item:item['record_id']):
        source = records_by_id[record['record_id']]
        left,right = source['row_offset'],source['row_offset']+source['rows']
        original = {key:value[left:right] for key,value in base.items()}
        source_folder = output_path(root,source['path'])
        source_metadata = json.loads((source_folder/'metadata.json').read_text(encoding='utf-8'))['provenance']
        classic_folder = output_path(root,source_metadata['source_features_path'])
        rate = json.loads((classic_folder/'metadata.json').read_text(encoding='utf-8'))['source_rate_hz']
        expected,_ = augmentation._allocate(record['record_id'],rate,original)
        path = output_path(root,record['path'])
        if (record['split'] != 'train' or sha256(path/'metadata.json') != record['metadata_sha256']
                or sha256(path/'inputs.npz') != record['artifact_sha256']):
            raise ValueError('augmentation artifact/index integrity failure')
        receipt = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        detail = receipt['provenance']
        if (detail['source_hashes'] != sources or detail['environment'] != index['environment']
                or detail['record_id'] != record['record_id'] or detail['split'] != 'train'
                or detail['input_sha256'] != source_metadata['input_sha256']
                or detail['source_artifact_sha256'] != source_metadata['source_artifact_sha256']
                or detail['source_metadata_sha256'] != source_metadata['source_metadata_sha256']
                or receipt['artifact_sha256'] != record['artifact_sha256'] or receipt['rows'] != record['rows']):
            raise ValueError('augmentation provenance differs from verified original training data')
        gains = {}
        for condition in receipt['conditions']:
            gain = condition['calibration']['common_gain']
            actual = condition['actual_rounded_analysis_snr_db']
            if (not np.isfinite([actual,gain]).all() or not 0 < gain <= 1
                    or condition['calibration']['target_snr_db'] != condition['snr_db']
                    or abs(actual-condition['snr_db']) > 1e-4):
                raise ValueError('invalid augmentation SNR receipt')
            gains[(condition['kind'],condition['snr_db'])] = gain
        with np.load(path/'inputs.npz',allow_pickle=False) as data:
            arrays = {key:data[key] for key in data.files}
        augmentation._validate(arrays,expected,gains)
        if len(arrays['y']) != record['rows']:
            raise ValueError('augmentation row count mismatch')
        extras.append(arrays)
        origins.append(arrays['source_window_index']+left)
        provenance.append({key:record[key] for key in ('record_id','path','rows','artifact_sha256','metadata_sha256')})
        for key,value in receipt['condition_counts'].items():
            condition_counts[key] = condition_counts.get(key,0)+value
    merged = {key:np.concatenate([item[key] for item in extras]) for key in extras[0]}
    global_origins = np.concatenate(origins)
    combined = combine_training_arrays(base,merged,global_origins,corpus['split_policy']['subjects'],mode)
    if len(global_origins) != 3775 or sum(condition_counts.values()) != 3775:
        raise ValueError('all3775 planned training copies required')
    evidence = {'augmentation_index_sha256':sha256(index_path),'original_provenance':base_provenance,
                'augmentation_provenance':provenance,'condition_counts':condition_counts,
                'original_scored_training_rows':3775,'additional_training_rows':3775,
                'origin_indices_sha256':hashlib.sha256(global_origins.tobytes()).hexdigest(),
                'source_hashes':sources}
    return corpus,combined,evidence


def run(root,mode,run_name,device='cpu',smoke=False):
    root = Path(root).resolve()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',run_name):
        raise ValueError('portable new run name required')
    output,model_folder = root/'runs'/run_name,root/'models'/run_name
    for path,parent in ((output,root/'runs'),(model_folder,root/'models')):
        if path.exists() or not path.resolve().is_relative_to(parent):
            raise ValueError('run/model exists or escapes local directory')
    corpus,arrays,evidence = load_augmented_inputs(root,mode)
    state,report,probabilities,features = fit_pilot('audio',arrays['values'],None,arrays['y'],arrays['subject_id'],
        arrays['supervised_mask'],corpus['split_policy']['subjects'],device=device,smoke=smoke,
        progress=lambda item:print(json.dumps(item),flush=True))
    report.update(mode=mode,dataset=corpus['dataset'],dataset_version=corpus['dataset_version'],
                  input_version=corpus['version'],split_policy=corpus['split_policy'],**evidence,
                  training_protocol_sha256=sha256(root/'docs/NOISE_AUGMENTATION_PROTOCOL.md'),
                  source_sha256={name:sha256(module.__file__) for name,module in
                                 (('models',models),('training',training),('inputs',inputs))},
                  runner_source_sha256=sha256(__file__))
    report['total_optimizer_steps'] = sum(math.ceil(item['training_rows_seen']/report['config']['batch_size']) for item in report['history'])
    report['optimizer_steps_to_selected_epoch'] = sum(math.ceil(item['training_rows_seen']/report['config']['batch_size'])
                                                      for item in report['history'] if item['epoch']<=report['best_epoch'])
    model_folder.mkdir(parents=True)
    checkpoint = model_folder/'weights.pt'
    torch.save(state,checkpoint)
    report['checkpoint_sha256'] = sha256(checkpoint)
    loaded = models.build_model('audio').to(report['device'])
    if sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('owned checkpoint changed before load')
    loaded.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    for item in probabilities.values():
        repeated = predict_probabilities(loaded,features[item['rows']],torch.device(report['device']),report['config']['batch_size'])
        if not np.array_equal(repeated,item['probabilities']):
            raise ValueError('owned checkpoint reload predictions differ')
    report['saved_loaded_predictions_exact'] = True
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',**{f'{part}_{name}':value for part,item in probabilities.items() for name,value in item.items()})
    report['predictions_sha256'] = sha256(output/'predictions.npz')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (model_folder/'metadata.json').write_text(json.dumps({key:report[key] for key in
        ('mode','label_order','normalizer','checkpoint_sha256','source_sha256','best_epoch','training_protocol_sha256')},indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    report = run(ROOT,args.mode,args.run_name,args.device,args.smoke)
    print(json.dumps({key:report[key] for key in ('mode','best_epoch','models','total_optimizer_steps','elapsed_seconds','smoke_only')},indent=2))


if __name__ == '__main__':
    main()
