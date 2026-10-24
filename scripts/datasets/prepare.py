"""Explicit preparation of the audited pilot files into pickle-free artifacts."""
from collections import Counter
from importlib.metadata import version
from importlib.util import find_spec
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time

from .cli import ROOT, raw_path, verify


def output_path(root, value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError('output must be a portable path under data/processed')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('data','processed') or len(path.parts) < 3:
        raise ValueError('output must stay under data/processed')
    root = Path(root).resolve()
    allowed = root/'data/processed'
    if not allowed.resolve().is_relative_to(allowed):
        raise ValueError('processed directory resolves outside repository location')
    result = (root/path).resolve()
    if not result.is_relative_to(allowed.resolve()):
        raise ValueError('output resolves outside processed directory')
    return result


def feature_source_hashes(dataset):
    modules = ('nightlab.audio.edf','nightlab.audio.features','nightlab.audio.reference','nightlab.audio.prepare') if dataset=='psg-audio' else ('nightlab.wearable.features',)
    return {name:hashlib.sha256(Path(find_spec(name).origin).read_bytes()).hexdigest() for name in modules}


def prepare(root, manifest, output=None, execute=False, stop_requested=None):
    if manifest['id'] not in {'psg-audio','sleepaccel'}:
        raise ValueError('preparation currently supports audited PSG-Audio and SleepAccel pilots only')
    output = output or f"data/processed/{manifest['id']}/pilot-features-v1"
    destination = output_path(root, output)
    plan = {'mode':'prepare execute' if execute else 'prepare dry-run', 'dataset':manifest['id'],
            'output':output, 'network':False, 'writes':execute,
            'license':manifest.get('license'),
            'formal_evaluation_eligible':False, 'scope':'audited single-subject pilot; no model evaluation'}
    if not execute:
        return plan
    if stop_requested is not None and stop_requested():
        raise InterruptedError('preparation cancelled')
    if destination.exists():
        raise ValueError('output exists; choose a new --output path, refusing overwrite')
    if manifest.get('license',{}).get('verified') is not True:
        raise ValueError('preparation requires verified dataset license')
    integrity = verify(root, manifest)
    if not integrity['ok']:
        raise ValueError('all pilot input files must pass manifest integrity before preparation')
    import numpy as np
    sources = feature_source_hashes(manifest['id'])
    started = time.perf_counter()
    if manifest['id'] == 'psg-audio':
        from nightlab.audio.prepare import prepare_psg_segment
        edf = next(entry['path'] for entry in manifest['files'] if entry['path'].endswith('.edf'))
        rml = next(entry['path'] for entry in manifest['files'] if entry['path'].endswith('.rml'))
        record_id = PurePosixPath(edf).parent.name
        subject_id = record_id.split('-')[0]
        def progress(done, total):
            if stop_requested is not None and stop_requested():
                raise InterruptedError('preparation cancelled before artifact publication')
            print(f'audio windows {done}/{total}', file=sys.stderr, flush=True)
        prepared = prepare_psg_segment(raw_path(root,edf), raw_path(root,rml),subject_id,record_id,progress=progress)
        arrays = {'X':prepared.features, 'y':prepared.target, 'subject_id':prepared.subject_id,
                  'record_id':prepared.record_id, 'start_seconds':prepared.start_seconds,
                  'end_seconds':prepared.end_seconds, 'overlap_fraction':prepared.overlap_fraction}
        timeline = prepared.all_windows
        metadata = prepared.metadata
    else:
        from nightlab.wearable.features import prepare_sleepaccel_features, FeatureConfig
        paths = {PurePosixPath(entry['path']).parent.name:raw_path(root,entry['path']) for entry in manifest['files']}
        subject_id = paths['labels'].name.split('_')[0]
        policy = manifest.get('preparation_policy',{}).get('repeated_signal_blocks','reject')
        prepared = prepare_sleepaccel_features(subject_id,paths['heart_rate'],paths['motion'],paths['labels'],FeatureConfig(repeated_block_policy=policy))
        arrays = {'X':prepared.features.astype(np.float32), 'y':np.asarray(prepared.stage,dtype=str),
                  'subject_id':prepared.subject_id,'raw_stage':prepared.raw_stage,
                  'start_seconds':prepared.epoch_start_seconds,'end_seconds':prepared.epoch_end_seconds}
        for signal, fields in prepared.coverage.items():
            arrays.update({f'coverage_{signal}_{name}':value for name,value in fields.items()})
        timeline = {key:(np.asarray(['' if value is None else value for value in values],dtype=str)
                         if key=='stage' else values) for key,values in prepared.label_table.items()}
        metadata = {**prepared.metadata,'feature_version':'sleepaccel-classic-v2',
                    'subject_count':1,'formal_evaluation_eligible':False}
    metadata.update(schema_version=1,dataset=manifest['id'],dataset_version=manifest['version'],
                    feature_names=list(prepared.feature_names),target_counts=dict(Counter(arrays['y'].tolist())),
                    rows=len(arrays['y']),feature_count=arrays['X'].shape[1],
                    feature_nan_count=int(np.isnan(arrays['X']).sum()),
                    input_integrity=integrity,feature_source_sha256=sources,elapsed_seconds=time.perf_counter()-started,
                    environment={'python':platform.python_version(),
                                 **{name:version(name) for name in ('nightlab','numpy','scipy','scikit-learn')}})
    if any(np.asarray(values).dtype.hasobject for values in (*arrays.values(),*timeline.values())):
        raise ValueError('object arrays forbidden in prepared artifacts')
    if stop_requested is not None and stop_requested():
        raise InterruptedError('preparation cancelled before artifact publication')
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/'features.npz',**arrays)
    np.savez_compressed(destination/'timeline.npz',**timeline)
    metadata['artifact_sha256'] = {name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
                                   for name in ('features.npz','timeline.npz')}
    (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {**plan,'status':'prepared','rows':metadata['rows'],'feature_count':metadata['feature_count'],
            'target_counts':metadata['target_counts'],'feature_nan_count':metadata['feature_nan_count']}
