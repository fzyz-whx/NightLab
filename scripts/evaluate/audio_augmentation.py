"""Evaluate the two receipt-anchored augmentation controls with the frozen noise suite."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from nightlab.deep import inputs,models,training
from nightlab.deep.inputs import MaskedChannelStandardizer
from scripts.datasets import noise_augmentation as augmentation
from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256
from scripts.datasets.prepare import output_path
from scripts.evaluate import noise


CONTROLS = {
    'cnn_clean_duplicate':('clean_duplicate','audio-clean-duplicate-v1-repro','audio-clean-duplicate-v1'),
    'cnn_noise_augmented':('noise_augmented','audio-noise-augmented-v1-repro','audio-noise-augmented-v1'),
}


def owned_path(root,area,*parts):
    """Reject redirected parents as well as leaf symlinks escaping an owned area."""
    allowed = root/area
    path = allowed.joinpath(*parts).resolve()
    if not allowed.resolve().is_relative_to(allowed) or not path.is_relative_to(allowed):
        raise ValueError('frozen artifact escapes its repository directory')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_training_cache(root,report):
    """Verify receipts and streamed checksums only; never regenerate or fit inputs."""
    folder = output_path(root,'data/processed/psg-audio-pilot-10/deep-inputs-v1')
    owned_path(root,'data/processed',folder.relative_to(root/'data/processed'),'corpus.json')
    corpus = read_json(folder/'corpus.json')
    original = report['original_provenance']
    if (not corpus['complete'] or corpus['split_policy'] != report['split_policy']
            or corpus['dataset'] != report['dataset'] or corpus['version'] != report['input_version']
            or len(original) != len(corpus['records'])
            or len({item['record_id'] for item in original}) != len(original)):
        raise ValueError('original training corpus differs from frozen report')
    by_id = {item['record_id']:item for item in corpus['records']}
    originals,offset = {},0
    for item in original:
        record = by_id.get(item['record_id'])
        path = output_path(root,item['path'])
        for filename in ('inputs.npz','metadata.json'):
            owned_path(root,'data/processed',path.relative_to(root/'data/processed'),filename)
        if (record is None or not record['ok'] or record['path'] != item['path']
                or item['row_offset'] != offset or record['rows'] != item['rows']
                or record['artifact_sha256'] != item['inputs_sha256']
                or sha256(path/'inputs.npz') != item['inputs_sha256']
                or sha256(path/'metadata.json') != item['metadata_sha256']):
            raise ValueError('original training artifact/metadata checksum differs')
        receipt = read_json(path/'metadata.json')
        if receipt['artifact_sha256'] != item['inputs_sha256'] or receipt['rows'] != item['rows']:
            raise ValueError('original training receipt differs')
        originals[item['record_id']] = receipt['provenance']
        offset += item['rows']
    index_path = output_path(root,f'data/processed/psg-audio-pilot-10/{augmentation.VERSION}')/'corpus.json'
    owned_path(root,'data/processed',index_path.relative_to(root/'data/processed'))
    index = read_json(index_path)
    sources = augmentation._sources(root)
    _,_,records = augmentation._select(root)
    identities = [item['record_id'] for item in index['records']]
    if (sha256(index_path) != report['augmentation_index_sha256'] or not index['complete']
            or index['version'] != augmentation.VERSION or index['dataset'] != report['dataset']
            or index['dataset_version'] != report['dataset_version']
            or index['split_policy'] != report['split_policy']
            or index['source_hashes'] != sources or report['source_hashes'] != sources
            or len(identities) != 6 or set(identities) != {item['record_id'] for item in records}
            or any(index['environment'][name] != version(name) for name in ('numpy','scipy'))):
        raise ValueError('current complete train-only augmentation cache required')
    expected = [{key:item[key] for key in ('record_id','path','rows','artifact_sha256','metadata_sha256')}
                for item in sorted(index['records'],key=lambda item:item['record_id'])]
    if report['augmentation_provenance'] != expected or sum(item['rows'] for item in expected) != 3775:
        raise ValueError('augmentation training provenance differs')
    for item in index['records']:
        path = output_path(root,item['path'])
        for filename in ('inputs.npz','metadata.json'):
            owned_path(root,'data/processed',path.relative_to(root/'data/processed'),filename)
        if (item['split'] != 'train' or sha256(path/'inputs.npz') != item['artifact_sha256']
                or sha256(path/'metadata.json') != item['metadata_sha256']):
            raise ValueError('augmentation artifact/metadata checksum differs')
        receipt = read_json(path/'metadata.json')
        detail,original = receipt['provenance'],originals[item['record_id']]
        if (receipt['artifact_sha256'] != item['artifact_sha256'] or receipt['rows'] != item['rows']
                or detail['source_hashes'] != sources or detail['environment'] != index['environment']
                or detail['split'] != 'train' or detail['record_id'] != item['record_id']
                or any(detail[key] != original[key] for key in
                       ('input_sha256','source_artifact_sha256','source_metadata_sha256'))):
            raise ValueError('augmentation source differs from original training data')


def load_frozen_control(root,key,device):
    root = Path(root).resolve()
    if key not in CONTROLS:
        raise ValueError('known frozen augmentation control required')
    mode,run_name,public_name = CONTROLS[key]
    public_path = owned_path(root,'docs','results',public_name+'.json')
    report_path = owned_path(root,'runs',run_name,'metrics.json')
    metadata_path = owned_path(root,'models',run_name,'metadata.json')
    weights = owned_path(root,'models',run_name,'weights.pt')
    public = read_json(public_path)
    if (public['run_name'] != run_name or public['mode'] != mode
            or sha256(report_path) != public['metrics_source_sha256']
            or sha256(weights) != public['checkpoint_sha256']):
        raise ValueError('frozen augmentation report/checkpoint differs from public receipt')
    report,metadata = read_json(report_path),read_json(metadata_path)
    if (report['mode'] != mode or metadata['mode'] != mode or report['smoke_only']
            or not report['saved_loaded_predictions_exact'] or report['task'] != 'audio'
            or report['feature_set'] != 'all' or report['label_order'] != noise.LABELS
            or metadata['label_order'] != noise.LABELS
            or any(metadata[field] != report[field] for field in
                   ('normalizer','checkpoint_sha256','source_sha256','best_epoch','training_protocol_sha256'))
            or report['checkpoint_sha256'] != public['checkpoint_sha256']
            or report['training_protocol_sha256'] != sha256(owned_path(root,'docs','NOISE_AUGMENTATION_PROTOCOL.md'))
            or any(report['source_sha256'][name] != sha256(module.__file__) for name,module in
                   (('models',models),('inputs',inputs),('training',training)))
            or any(report['environment'][name] != version(name) for name in ('torch','numpy','scipy','scikit-learn'))):
        raise ValueError('frozen augmentation metadata/source/protocol differs')
    normalizer = MaskedChannelStandardizer()
    normalizer.mean_ = np.asarray(metadata['normalizer']['mean'],dtype=np.float64)
    normalizer.scale_ = np.asarray(metadata['normalizer']['scale'],dtype=np.float64)
    if (normalizer.mean_.shape != (32,) or normalizer.scale_.shape != (32,)
            or not np.isfinite(normalizer.mean_).all() or not np.isfinite(normalizer.scale_).all()
            or np.any(normalizer.scale_ <= 0)):
        raise ValueError('invalid frozen augmentation training moments')
    verify_training_cache(root,report)
    model = models.build_model('audio').to(device)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.eval()
    return {'model':model,'normalizer':normalizer,'report':report,'hashes':{
        'checkpoint_sha256':public['checkpoint_sha256'],'metrics_sha256':public['metrics_source_sha256'],
        'public_report_sha256':sha256(public_path),'metadata_sha256':sha256(metadata_path),
        'loader_source_sha256':sha256(__file__),
        'augmentation_index_sha256':report['augmentation_index_sha256'],
        'training_protocol_sha256':report['training_protocol_sha256']}}


def run(root,output_name,device='cpu'):
    selected = training.select_device(device)
    bundles = {key:load_frozen_control(root,key,selected) for key in CONTROLS}
    return noise.evaluate(root,output_name,device,extra_cnns=bundles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    args = parser.parse_args()
    report = run(ROOT,args.run_name,args.device)
    print(json.dumps({'output':f'runs/{args.run_name}/summary.json','conditions':len(report['conditions'])}))


if __name__ == '__main__':
    main()
