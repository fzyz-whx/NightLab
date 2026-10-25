"""Check actual seeded reruns and export aggregate reports without learned moments."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


RUNS = ('audio-cnn-pilot-v1','wearable-cnn-fusion-v1','wearable-cnn-hr-v1','wearable-cnn-accel-v1')


def verify_and_export(root):
    checks = []
    public_reports = []
    for name in RUNS:
        repeated = name+'-repro'
        paths = [root/'runs'/run/'metrics.json' for run in (name,repeated)]
        reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
        original,rerun = reports
        for report,run in zip(reports,(name,repeated),strict=True):
            if report['smoke_only'] or not report['saved_loaded_predictions_exact']:
                raise ValueError('a complete formal run with load verification is required')
            if (sha256(root/'models'/run/'weights.pt') != report['checkpoint_sha256']
                    or sha256(root/'runs'/run/'predictions.npz') != report['predictions_sha256']):
                raise ValueError('owned saved run artifacts changed')
        for key in ('models','best_epoch','normalizer','class_weights_training_only','training_class_counts',
                    'split_subjects','config','parameter_count','provenance'):
            if original[key] != rerun[key]:
                raise ValueError(f'{name}: seeded rerun differs in {key}')
        histories = [[{key:value for key,value in item.items() if key != 'elapsed_seconds'}
                      for item in report['history']] for report in reports]
        if histories[0] != histories[1]:
            raise ValueError('seeded training losses/validation history differ')
        with np.load(root/'runs'/name/'predictions.npz',allow_pickle=False) as a, np.load(root/'runs'/repeated/'predictions.npz',allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[key],b[key]) for key in a.files):
                raise ValueError('seeded saved probabilities differ')
        states = [torch.load(root/'models'/run/'weights.pt',map_location='cpu',weights_only=True)
                  for run in (name,repeated)]
        if states[0].keys() != states[1].keys() or any(not torch.equal(states[0][key],states[1][key]) for key in states[0]):
            raise ValueError('seeded checkpoint tensor values differ')
        checks.append({'run':name,'repeated_run':repeated,
                       'metrics_history_probabilities_and_state_tensors_exact':True,
                       'owned_saved_loaded_predictions_exact':True,
                       'original_metrics_sha256':sha256(paths[0]),'repeated_metrics_sha256':sha256(paths[1]),
                       'original_checkpoint_sha256':original['checkpoint_sha256'],
                       'repeated_checkpoint_sha256':rerun['checkpoint_sha256']})
        public = {**rerun,'run_name':repeated,'metrics_source_sha256':sha256(paths[1]),
                  'normalizer':{key:value for key,value in rerun['normalizer'].items() if key not in {'mean','scale'}},
                  'public_export':'aggregate metrics; learned moments, weights and individual predictions remain local',
                  'reproducibility_checked_against':name}
        public_reports.append((root/f'docs/results/{name}.json',public))
    receipt = {'scope':'actual CUDA same-seed reruns on this hardware/software; not cross-platform bitwise guarantee',
               'checks':checks,'metadata_correction':'Initial local runs mislabeled deep-inputs-v1 as dataset_version. Reruns separate source dataset_version from input_version; no model, optimizer, split or score was changed.'}
    outputs = [*public_reports,(root/'docs/results/cnn-reproducibility-v1.json',receipt)]
    if any(path.exists() for path,_ in outputs):
        raise ValueError('public output exists; refusing overwrite')
    for path,report in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'verification/export dry-run','writes':False,'runs':RUNS}))
        return
    print(json.dumps(verify_and_export(ROOT),indent=2))


if __name__ == '__main__':
    main()
