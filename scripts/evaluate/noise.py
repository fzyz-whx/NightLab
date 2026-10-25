"""Compare frozen RF/CNN models on all preregistered synthetic test conditions."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nightlab.audio.noise import BRANCHES,noise_seed
from nightlab.audio.reference import read_reference
from nightlab.deep.inputs import MaskedChannelStandardizer
from nightlab.deep import inputs as input_module,models as model_module,training as training_module
from nightlab.deep.models import build_model
from nightlab.deep.training import predict_probabilities,select_device
from nightlab.experiments.baseline import classification_metrics
from scripts.datasets import deep_cache,noise_features
from scripts.datasets.cli import ROOT,raw_path
from scripts.datasets.dense_audio import ensure_dense,sha256
from scripts.datasets.prepare import output_path
from scripts.evaluate.audio_events import aggregate_events,clipped_references,load_owned_models
from nightlab.audio.events import decode_events,evaluate_events
from nightlab.audio import events as event_module
from nightlab.experiments import baseline as classification_module
from scripts.evaluate import audio_events as aggregation_module
from scripts.train.baselines import load_corpus


LABELS = ['no_target_reference_event','reference_respiratory_event']


def load_frozen_models(root,device):
    rf_path = root/'runs/psg-rf-pilot-v1/metrics.json'
    rf_report = json.loads(rf_path.read_text(encoding='utf-8'))
    corpus,_,_,_,names,provenance = load_corpus(root/'data/processed/psg-audio-pilot-10/corpus.json',root)
    if (rf_report['dataset'] != corpus['dataset'] or rf_report['dataset_version'] != corpus['version']
            or rf_report['split_policy'] != corpus['split_policy'] or rf_report['provenance'] != provenance
            or rf_report['feature_names'] != names or rf_report['feature_count'] != len(names)
            or len(names) != 67 or rf_report.get('feature_subset','all') != 'all'):
        raise ValueError('frozen RF training provenance/schema differs from current corpus')
    owned,rf_hashes = load_owned_models(root,'psg-rf-pilot-v1',rf_report)
    public_path = root/'docs/results/audio-cnn-pilot-v1.json'
    public = json.loads(public_path.read_text(encoding='utf-8'))
    name = 'audio-cnn-pilot-v1-repro'
    report_path = root/f'runs/{name}/metrics.json'
    metadata_path = root/f'models/{name}/metadata.json'
    weights = root/f'models/{name}/weights.pt'
    if sha256(report_path) != public['metrics_source_sha256'] or sha256(weights) != public['checkpoint_sha256']:
        raise ValueError('frozen owned CNN report/checkpoint differs from public receipt')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    for item in report['provenance']:
        folder = output_path(root,item['path'])
        if sha256(folder/'inputs.npz') != item['inputs_sha256'] or sha256(folder/'metadata.json') != item['metadata_sha256']:
            raise ValueError('frozen CNN training input provenance changed')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (report['smoke_only'] or not report['saved_loaded_predictions_exact']
            or metadata['checkpoint_sha256'] != report['checkpoint_sha256']
            or metadata['normalizer'] != report['normalizer']
            or metadata['label_order'] != LABELS or report['label_order'] != LABELS
            or report['feature_set'] != 'all' or report['task'] != 'audio'):
        raise ValueError('frozen CNN model/normalizer metadata differs')
    if (any(sha256(module.__file__) != report['source_sha256'][name] for name,module in
            (('models',model_module),('inputs',input_module),('training',training_module)))
            or any(version(name) != report['environment'][name] for name in ('torch','numpy','scipy','scikit-learn'))):
        raise ValueError('frozen CNN source/environment differs from training')
    model = build_model('audio').to(device)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.eval()
    normalizer = MaskedChannelStandardizer()
    normalizer.mean_ = np.asarray(metadata['normalizer']['mean'],dtype=np.float64)
    normalizer.scale_ = np.asarray(metadata['normalizer']['scale'],dtype=np.float64)
    if (normalizer.mean_.shape != (32,) or normalizer.scale_.shape != (32,)
            or not np.isfinite(normalizer.mean_).all() or not np.isfinite(normalizer.scale_).all()
            or np.any(normalizer.scale_ <= 0)):
        raise ValueError('invalid frozen CNN moments')
    return owned['random_forest'],model,normalizer,rf_report,report,{
        'random_forest':rf_hashes['random_forest'],'cnn':report['checkpoint_sha256'],
        'rf_metrics_sha256':sha256(rf_path),'owned_model_receipt_sha256':sha256(root/'docs/results/baseline-reproducibility.json'),
        'cnn_public_report_sha256':sha256(public_path),'cnn_metadata_sha256':sha256(metadata_path)}


def validate_index(root):
    path = output_path(root,f'data/processed/{noise_features.DATASET}/noise-eval-v1')/'corpus.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    manifest,classic,records = noise_features._select(root,'test')
    environment = {name:version(name) for name in ('numpy','scipy')}
    sources = noise_features._sources(root)
    expected = {(record['record_id'],kind,snr) for record in records
                for kind in noise_features.KINDS for snr in noise_features.SNRS}
    identities = [(item['record_id'],item['kind'],item['snr_db']) for item in index['records']]
    if (not index['complete'] or index['scope'] != 'test' or index['version'] != 'noise-eval-v1'
            or index['split_policy'] != classic['split_policy'] or index['source_hashes'] != sources
            or index['protocol_sha256'] != sources['protocol'] or len(identities) != len(expected)
            or set(identities) != expected or any(item['split'] != 'test' for item in index['records'])
            or any(index['environment'][name] != value for name,value in environment.items())):
        raise ValueError('complete current fixed test condition set required')
    return index,manifest,records,sources,sha256(path)


def verified_condition(root,entry,base_provenance,expected,sources,environment):
    folder = output_path(root,entry['path'])
    metadata_path,artifact = folder/'metadata.json',folder/'inputs.npz'
    if sha256(metadata_path) != entry['metadata_sha256'] or sha256(artifact) != entry['artifact_sha256']:
        raise ValueError('noise condition metadata/artifact checksum mismatch')
    receipt = json.loads(metadata_path.read_text(encoding='utf-8'))
    provenance = receipt['provenance']
    calibration = provenance['calibration']
    if (any(provenance.get(key) != value for key,value in base_provenance.items())
            or provenance['source_hashes'] != sources or provenance['environment'] != environment
            or provenance['record_id'] != entry['record_id'] or provenance['split'] != 'test'
            or provenance['kind'] != entry['kind'] or provenance['snr_db'] != entry['snr_db']
            or provenance['seed'] != noise_seed(entry['record_id'],'test',entry['kind'])
            or calibration['target_snr_db'] != entry['snr_db']
            or abs(provenance['actual_rounded_analysis_snr_db']-entry['snr_db']) > 1e-4
            or not 0 < calibration['common_gain'] <= 1 or calibration['noise_scale'] <= 0
            or not np.isfinite([calibration['common_gain'],calibration['noise_scale'],
                                provenance['actual_rounded_analysis_snr_db']]).all()
            or receipt['artifact_sha256'] != entry['artifact_sha256'] or receipt['rows'] != len(expected['y'])):
        raise ValueError('noise condition calibration/source differs from fixed protocol')
    with np.load(artifact,allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    noise_features._validate_arrays(arrays,expected)
    return arrays,receipt


def score_probabilities(probabilities,arrays,references,censored,duration):
    probabilities = np.asarray(probabilities,dtype=np.float64)
    if (probabilities.shape != (len(arrays['y']),) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0)|(probabilities > 1))):
        raise ValueError('one finite positive probability per complete window required')
    decoded = decode_events(arrays['start_seconds'],arrays['end_seconds'],probabilities,duration)
    return {'decoded':decoded,'events':evaluate_events(decoded['events'],references,duration,censored),
            'window_probabilities':probabilities.tolist()}


def pooled_scores(items):
    truth = np.concatenate([item['truth'] for item in items])
    probabilities = np.concatenate([item['probabilities'] for item in items])
    predicted = np.concatenate([item['predicted'] for item in items])
    return {'classification':classification_metrics(truth,predicted,LABELS,probabilities,LABELS[1]),
            'events':aggregate_events([item['events'] for item in items])}


def cnn_outputs(bundles,images,device):
    """Apply each frozen model's own training moments, without fitting."""
    return {name:predict_probabilities(bundle['model'],
                torch.from_numpy(bundle['normalizer'].transform(images)[:,None,:,:]),device)
            for name,bundle in bundles.items()}


def evaluate(root,run_name,requested_device='cpu',extra_cnns=None):
    root = Path(root).resolve()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',run_name):
        raise ValueError('portable output name required')
    output = root/'runs'/run_name
    if output.exists() or not output.resolve().is_relative_to(root/'runs'):
        raise ValueError('output exists or escapes local runs')
    started = time.perf_counter()
    torch.set_num_threads(2)
    device = select_device(requested_device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    index,manifest,records,sources,index_sha = validate_index(root)
    rf,cnn,normalizer,rf_report,cnn_report,model_hashes = load_frozen_models(root,device)
    extra_cnns = extra_cnns or {}
    if any(not re.fullmatch(r'[a-z][a-z0-9_]*',name) or name in ('random_forest','cnn') for name in extra_cnns):
        raise ValueError('distinct portable extra CNN names required')
    cnns = {'cnn':{'model':cnn,'normalizer':normalizer,'report':cnn_report},**extra_cnns}
    model_names = ('random_forest',*cnns)
    for name,bundle in extra_cnns.items():
        if bundle['report']['split_policy'] != index['split_policy']:
            raise ValueError('extra CNN and noise corpus use different subject partitions')
        model_hashes[name] = bundle['hashes']
    if rf_report['split_policy'] != index['split_policy'] or cnn_report['split_policy'] != index['split_policy']:
        raise ValueError('models and noise corpus use different subject partitions')
    if list(rf.classes_) != LABELS:
        raise ValueError('unexpected RF class order')
    local_records,collected,controls = [],{},[]
    unscaled = {name:[] for name in model_names}
    with threadpool_limits(limits=2):
        for subject_index,record in enumerate(records,1):
            _,_,base_provenance,expected,_ = noise_features._verified_context(root,manifest,record)
            dense,_,_ = ensure_dense(root,manifest,record)
            deep_folder = output_path(root,f'data/processed/{noise_features.DATASET}/deep-inputs-v1/{record["record_id"]}')
            deep_metadata = json.loads((deep_folder/'metadata.json').read_text(encoding='utf-8'))
            if (sha256(deep_folder/'inputs.npz') != deep_metadata['artifact_sha256']
                    or deep_metadata['provenance']['input_sha256'] != base_provenance['input_sha256']
                    or deep_metadata['provenance']['deep_cache_source_sha256'] != sources['deep_cache']
                    or deep_metadata['provenance']['deep_inputs_source_sha256'] != sources['deep.inputs']):
                raise ValueError('unscaled image control provenance differs')
            with np.load(deep_folder/'inputs.npz',allow_pickle=False) as data:
                unscaled_images = data['values']
            rml = next(entry['path'] for entry in manifest['files'] if entry.get('record_id') == record['record_id'] and entry['path'].endswith('.rml'))
            _,_,events = read_reference(raw_path(root,rml))
            duration = base_provenance['duration_seconds']
            references,censored,_ = clipped_references(events,base_provenance['offset_seconds'],duration)
            base_local = {'record_id':record['record_id'],'participant':f'test-{subject_index:02d}',
                          'kind':'unscaled','snr_db':None,'branch':'unscaled',
                          'reference_intervals':references,'reference_censored':censored,'models':{}}
            original_outputs = {'random_forest':rf.predict_proba(dense['X']),
                                **cnn_outputs(cnns,unscaled_images,device)}
            mask = expected['supervised_mask']
            for name,full in original_outputs.items():
                predicted = np.asarray(LABELS)[full.argmax(axis=1)]
                scored = score_probabilities(full[:,1],expected,references,censored,duration)
                scored['classification'] = classification_metrics(expected['y'][mask],predicted[mask],LABELS,full[mask,1],LABELS[1])
                base_local['models'][name] = scored
                unscaled[name].append({'truth':expected['y'][mask],'predicted':predicted[mask],
                                       'probabilities':full[mask,1],'events':scored['events']})
            local_records.append(base_local)
            for entry in (item for item in index['records'] if item['record_id'] == record['record_id']):
                arrays,receipt = verified_condition(root,entry,base_provenance,expected,sources,index['environment'])
                gain = receipt['provenance']['calibration']['common_gain']
                if gain == 1 and (not np.array_equal(arrays['X_matched_clean'],dense['X'])
                                  or not np.array_equal(arrays['images_matched_clean'],unscaled_images)):
                    raise ValueError('unit-gain reference differs from frozen RF/CNN inputs')
                controls.append({'participant':f'test-{subject_index:02d}','kind':entry['kind'],
                                 'snr_db':entry['snr_db'],'common_gain':gain,
                                 'unit_gain_inputs_exact':True if gain == 1 else None,
                                 'metadata_sha256':entry['metadata_sha256'],'artifact_sha256':entry['artifact_sha256']})
                for branch in BRANCHES:
                    rf_full = rf.predict_proba(arrays['X_'+branch])
                    branch_outputs = {'random_forest':rf_full,**cnn_outputs(cnns,arrays['images_'+branch],device)}
                    local = {'record_id':record['record_id'],'participant':f'test-{subject_index:02d}',
                             'kind':entry['kind'],'snr_db':entry['snr_db'],'branch':branch,
                             'reference_intervals':references,'reference_censored':censored,'models':{}}
                    for name,full in branch_outputs.items():
                        probability = full[:,1]
                        scored = score_probabilities(probability,arrays,references,censored,duration)
                        predicted = np.asarray(LABELS)[full.argmax(axis=1)]
                        mask = arrays['supervised_mask']
                        scored['classification'] = classification_metrics(arrays['y'][mask],predicted[mask],
                                                                          LABELS,np.asarray(probability)[mask],LABELS[1])
                        local['models'][name] = scored
                        key = (entry['kind'],entry['snr_db'],branch,name)
                        collected.setdefault(key,[]).append({'truth':arrays['y'][mask],
                            'probabilities':np.asarray(probability)[mask],
                            'predicted':predicted[mask],'events':scored['events']})
                    local_records.append(local)
                print(json.dumps({'record_id':record['record_id'],'kind':entry['kind'],'snr_db':entry['snr_db'],
                                  'branches_scored':len(BRANCHES),'gain':gain}),flush=True)
    conditions = []
    for kind in noise_features.KINDS:
        for snr_db in noise_features.SNRS:
            for branch in BRANCHES:
                models = {name:pooled_scores(collected[(kind,snr_db,branch,name)]) for name in model_names}
                conditions.append({'kind':kind,'snr_db':snr_db,'branch':branch,'models':models})
    matched = {(item['kind'],item['snr_db']):item for item in conditions if item['branch'] == 'matched_clean'}
    unscaled_scores = {name:pooled_scores(items) for name,items in unscaled.items()}
    for item in conditions:
        for name,scores in item['models'].items():
            baseline = matched[(item['kind'],item['snr_db'])]['models'][name]
            scores['delta_vs_same_gain_reference'] = {
                'macro_f1':scores['classification']['macro_f1']-baseline['classification']['macro_f1'],
                'event_f1':scores['events']['f1']-baseline['events']['f1']}
            scores['delta_vs_unscaled_reference'] = {
                'macro_f1':scores['classification']['macro_f1']-unscaled_scores[name]['classification']['macro_f1'],
                'event_f1':scores['events']['f1']-unscaled_scores[name]['events']['f1']}
    public = {'schema_version':1,'dataset':index['dataset'],'dataset_version':index['dataset_version'],
              'protocol_sha256':index['protocol_sha256'],'event_protocol_sha256':sha256(root/'docs/EVENT_PROTOCOL.md'),
              'input_index_sha256':index_sha,'source_hashes':sources,'evaluation_source_sha256':sha256(__file__),
              'metric_source_hashes':{name:sha256(module.__file__) for name,module in
                                      (('events',event_module),('classification',classification_module),('aggregation',aggregation_module))},
              'model_hashes':model_hashes,'device':str(device),'conditions':conditions,'input_controls':controls,
              'unscaled_reference':unscaled_scores,
              'unscaled_classification_matches_frozen':{
                  'random_forest':unscaled_scores['random_forest']['classification'] == rf_report['models']['random_forest']['test'],
                  **{name:unscaled_scores[name]['classification'] == bundle['report']['models']['cnn']['test']
                     for name,bundle in cnns.items()}},
              'decision_rules':'window labels use model argmax (first class wins ties); event bins use >=0.5 per frozen protocol',
              'test_subject_count':len(records),'distinct_observed_hours':2,
              'scope':'synthetic-interference pilot; repeated conditions on the same two subjects, not independent cohorts or real dorm noise',
              'model_policy':('frozen weights and training moments; no refit or threshold selection'
                              if extra_cnns else 'frozen weights and training moments; no refit, threshold selection or augmented model yet'),
              'elapsed_seconds':time.perf_counter()-started,
              'environment':{name:version(name) for name in ('numpy','scipy','torch','scikit-learn')}}
    public['per_record'] = [{key:value for key,value in record.items() if key in ('participant','kind','snr_db','branch')}
        | {'models':{name:{'classification':item['classification'],
                           'events':{key:value for key,value in item['events'].items() if key != 'matches'}}
                     for name,item in record['models'].items()}} for record in local_records]
    output.mkdir(parents=True)
    (output/'events.json').write_text(json.dumps({**public,'records':local_records},indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    args = parser.parse_args()
    report = evaluate(ROOT,args.run_name,args.device)
    print(json.dumps({'output':f'runs/{args.run_name}/summary.json','conditions':len(report['conditions'])}))


if __name__ == '__main__':
    main()
