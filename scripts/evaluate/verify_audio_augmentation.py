"""Check real seeded augmentation/control reruns before publishing model receipts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


RUNS = ('audio-clean-duplicate-v1','audio-noise-augmented-v1')


def verify(root):
    checks,outputs = [],[]
    for name in RUNS:
        repeated = name+'-repro'
        paths = [root/'runs'/run/'metrics.json' for run in (name,repeated)]
        reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
        for report,run_name in zip(reports,(name,repeated),strict=True):
            if (report['smoke_only'] or not report['saved_loaded_predictions_exact']
                    or sha256(root/'models'/run_name/'weights.pt') != report['checkpoint_sha256']
                    or sha256(root/'runs'/run_name/'predictions.npz') != report['predictions_sha256']):
                raise ValueError('formal owned run verification failed')
        for key in ('models','normalizer','mode','best_epoch','class_weights_training_only','config',
                    'training_class_counts','augmentation_index_sha256','origin_indices_sha256',
                    'original_provenance','augmentation_provenance','condition_counts','source_hashes',
                    'source_sha256','training_protocol_sha256','total_optimizer_steps','optimizer_steps_to_selected_epoch'):
            if reports[0][key] != reports[1][key]:
                raise ValueError('seeded augmentation/control rerun differs: '+key)
        histories = [[{key:value for key,value in item.items() if key != 'elapsed_seconds'}
                      for item in report['history']] for report in reports]
        if histories[0] != histories[1]:
            raise ValueError('seeded training history differs')
        with np.load(root/'runs'/name/'predictions.npz',allow_pickle=False) as a, np.load(root/'runs'/repeated/'predictions.npz',allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[key],b[key]) for key in a.files):
                raise ValueError('seeded predictions differ')
        states = [torch.load(root/'models'/run/'weights.pt',map_location='cpu',weights_only=True) for run in (name,repeated)]
        if states[0].keys() != states[1].keys() or any(not torch.equal(states[0][key],states[1][key]) for key in states[0]):
            raise ValueError('seeded checkpoint values differ')
        report = reports[1]
        public = {**report,'run_name':repeated,'metrics_source_sha256':sha256(paths[1]),
                  'normalizer':{key:value for key,value in report['normalizer'].items() if key not in ('mean','scale')},
                  'reproducibility_checked_against':name,
                  'public_export':'aggregate metrics; learned moments, weights and per-window predictions remain local'}
        outputs.append((root/f'docs/results/{name}.json',public))
        checks.append({'run':name,'repeat':repeated,'all_metrics_histories_probabilities_and_weight_tensors_exact':True,
                       'original_metrics_sha256':sha256(paths[0]),'repeated_metrics_sha256':sha256(paths[1]),
                       'checkpoint_sha256':report['checkpoint_sha256']})
    receipt = {'scope':'actual same-seed CUDA repeat on this environment, not cross-platform bitwise guarantee','checks':checks}
    outputs.append((root/'docs/results/audio-augmentation-reproducibility-v1.json',receipt))
    if any(path.exists() for path,_ in outputs):
        raise ValueError('public export exists; refusing overwrite')
    for path,report in outputs:
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'augmentation repeat verification dry-run','writes':False,'runs':RUNS}))
        return
    print(json.dumps(verify(ROOT),indent=2))


if __name__ == '__main__':
    main()
