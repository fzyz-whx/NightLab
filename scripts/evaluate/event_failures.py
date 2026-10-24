"""Describe event mismatches without publishing raw event times or audio."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.datasets.dense_audio import sha256


def describe_failures(predictions, references, metrics):
    predicted = np.asarray(predictions,dtype=float).reshape(-1,2)
    reference = np.asarray(references,dtype=float).reshape(-1,2)
    intersection = np.maximum(0,np.minimum(predicted[:,None,1],reference[None,:,1])
                              - np.maximum(predicted[:,None,0],reference[None,:,0]))
    union = (predicted[:,1]-predicted[:,0])[:,None]+(reference[:,1]-reference[:,0])[None,:]-intersection
    iou = intersection/union
    used_p = {match['prediction_index'] for match in metrics['matches']}
    used_r = {match['reference_index'] for match in metrics['matches']}
    if len(used_p) != metrics['tp'] or len(used_r) != metrics['tp']:
        raise ValueError('matching receipt is not one-to-one')
    def categories(matrix, used):
        result = {'no_reference_overlap':0,'overlap_below_iou_threshold':0,'one_to_one_competition':0}
        for index,row in enumerate(matrix):
            if index in used:
                continue
            best = float(row.max()) if row.size else 0.0
            category = ('no_reference_overlap' if best == 0 else
                        'overlap_below_iou_threshold' if best < .5 else 'one_to_one_competition')
            result[category] += 1
        return result
    fp = categories(iou,used_p)
    fn = categories(iou.T,used_r)
    if sum(fp.values()) != metrics['fp'] or sum(fn.values()) != metrics['fn']:
        raise ValueError('failure categories do not reconcile with metric counts')
    def lengths(values):
        durations = values[:,1]-values[:,0]
        return {'count':len(durations), 'median_seconds':float(np.median(durations)) if durations.size else None,
                'max_seconds':float(durations.max()) if durations.size else None}
    return {'false_positives':fp, 'false_negatives':fn,
            'predictions_overlapping_multiple_references':int(((intersection>0).sum(axis=1)>1).sum()),
            'references_overlapping_multiple_predictions':int(((intersection>0).sum(axis=0)>1).sum()),
            'predicted_duration':lengths(predicted), 'reference_duration':lengths(reference)}


def analyze(input_path, output_path):
    data = json.loads(input_path.read_text(encoding='utf-8'))
    records = []
    for record in data['records']:
        records.append({'record_id':record['record_id'], 'split':record['split'],
                        'models':{name:describe_failures(item['decoded']['events'],record['reference_intervals'],item['metrics'])
                                  for name,item in record['models'].items()}})
    models = {}
    for name in data['models']:
        models[name] = {}
        for part in ('validation','test'):
            selected = [record['models'][name] for record in records if record['split'] == part]
            totals = {kind:{category:sum(item[kind][category] for item in selected)
                            for category in selected[0][kind]} for kind in ('false_positives','false_negatives')}
            totals.update({key:sum(item[key] for item in selected) for key in
                           ('predictions_overlapping_multiple_references','references_overlapping_multiple_predictions')})
            models[name][part] = totals
    report = {'source_run_sha256':sha256(input_path), 'analysis_source_sha256':sha256(__file__),
              'scope':'descriptive interval-overlap errors; no threshold tuning or causal/clinical conclusion',
              'category_interpretation':'FN categories use prediction overlap; no_reference_overlap means no opposite-side interval overlap',
              'models':models, 'records':records}
    if output_path.exists():
        raise ValueError('refusing to overwrite failure report')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input,args.output)['models'],indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
