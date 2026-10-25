"""Verify actual frozen inference repeats and export only aggregate noise results."""
import argparse
import json
from pathlib import Path

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


def export(root,original='noise-frozen-models-v1',repeated='noise-frozen-models-v1-repro'):
    paths = [root/'runs'/name/'events.json' for name in (original,repeated)]
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    for key in reports[0]:
        if key != 'elapsed_seconds' and reports[0][key] != reports[1][key]:
            raise ValueError(f'cached frozen-model inference rerun differs in {key}')
    summary = root/'runs'/original/'summary.json'
    public = json.loads(summary.read_text(encoding='utf-8'))
    if 'records' in public or len(public['conditions']) != 32 or public['test_subject_count'] != 2:
        raise ValueError('unexpected or non-aggregate public report')
    for item in public['per_record']:
        if any(key in item for key in ('record_id','reference_intervals','reference_censored')):
            raise ValueError('raw identities/intervals in public record')
        for model in item['models'].values():
            if 'matches' in model['events'] or 'window_probabilities' in model:
                raise ValueError('raw matches/probabilities in public report')
    receipt = {'scope':'actual same-device frozen-model inference on verified cached features; not a second feature build',
               'all_fields_except_elapsed_seconds_exact':True,
               'includes_window_probabilities_bins_event_intervals_matches_classification_and_pooled_metrics':True,
               'original_run_sha256':sha256(paths[0]),'repeated_run_sha256':sha256(paths[1]),
               'public_summary_sha256':sha256(summary),
               'unscaled_classification_matches_frozen':public['unscaled_classification_matches_frozen'],
               'unit_gain_input_control_count':sum(item['unit_gain_inputs_exact'] is True for item in public['input_controls']),
               'condition_record_count':len(public['input_controls']),'pooled_four_branch_conditions':len(public['conditions'])}
    destination = root/'docs/results/noise-frozen-models-v1.json'
    verification = root/'docs/results/noise-frozen-inference-reproducibility-v1.json'
    if destination.exists() or verification.exists():
        raise ValueError('public export exists; refusing overwrite')
    destination.write_bytes(summary.read_bytes())
    verification.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'mode':'noise inference verification/export dry-run','writes':False}))
        return
    print(json.dumps(export(ROOT),indent=2))


if __name__ == '__main__':
    main()
