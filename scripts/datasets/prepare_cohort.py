"""Prepare each complete, verified record; partial cohorts cannot be evaluated."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import hashlib
import json
from pathlib import PurePosixPath
import re

from .cli import ROOT, load_manifest
from .prepare import output_path, prepare, feature_source_hashes


def validate_receipt(folder, subset):
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    expected = {entry['path']:entry['official_checksums'] for entry in subset['files']}
    observed = {entry['path']:{key:entry['local_checksums'][key] for key in expected[entry['path']]}
                for entry in metadata['input_integrity']['files'] if entry['path'] in expected}
    dataset = subset.get('id','psg-audio')
    feature_version = {'psg-audio':'psg-rms-logmel-v1','sleepaccel':'sleepaccel-classic-v2'}[dataset]
    if observed != expected or metadata.get('feature_version') != feature_version or not metadata['input_integrity']['ok']:
        raise ValueError('existing feature provenance differs from selected record')
    if metadata.get('dataset_version') != subset['version'] or metadata.get('feature_source_sha256') != feature_source_hashes(dataset):
        raise ValueError('existing feature version/source differs; prepare to a new output')
    for name in ('features.npz','timeline.npz'):
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != metadata.get('artifact_sha256',{}).get(name):
            raise ValueError('existing feature artifact checksum mismatch')
    return metadata


def prepare_one(task, stop=None):
    record, subset, relative, pilot = task
    destination = output_path(ROOT,relative)
    if not destination.exists() and record in {'00000995-100507','5383425'} and pilot is not None and pilot.exists():
        metadata = validate_receipt(pilot,subset)
        return {'record_id':record,'ok':True,'path':pilot.relative_to(ROOT).as_posix(),'status':'reused verified pilot','rows':metadata['rows']}
    if destination.exists():
        metadata = validate_receipt(destination,subset)
        return {'record_id':record,'ok':True,'path':relative,'status':'reused verified features','rows':metadata['rows']}
    result = prepare(ROOT,subset,relative,execute=True,stop_requested=stop.is_set if stop is not None else None)
    return {'record_id':record,'ok':True,'path':relative,'status':'prepared','rows':result['rows']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--feature-tag',default='features-v1')
    parser.add_argument('--no-reuse-pilot',action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',args.dataset):
        parser.error('invalid manifest identity')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',args.feature_tag):
        parser.error('invalid feature tag')
    manifest = load_manifest(ROOT/f'data/manifests/{args.dataset}.json',ROOT)
    dataset = 'sleepaccel' if args.dataset.startswith('sleepaccel') else 'psg-audio'
    grouped = {}
    for entry in manifest['files']:
        record = entry.get('subject_id','') if dataset=='sleepaccel' else entry.get('record_id','')
        if not re.fullmatch(r'\d+' if dataset=='sleepaccel' else r'\d{8}-\d{6}',record):
            parser.error('cohort records require fixed source identities')
        grouped.setdefault(record,[]).append(entry)
    tasks = []
    for record,entries in sorted(grouped.items()):
        if dataset=='psg-audio' and (len(entries)!=2 or sum(entry['path'].endswith('.edf') for entry in entries)!=1 or sum(entry['path'].endswith('.rml') for entry in entries)!=1):
            parser.error('one EDF and one RML required per cohort record')
        if dataset=='sleepaccel' and (len(entries)!=3 or {PurePosixPath(entry['path']).parent.name for entry in entries}!={'labels','heart_rate','motion'}):
            parser.error('heart rate, motion and labels required per participant')
        subset = {**manifest,'id':dataset,'files':entries}
        tasks.append((record,subset,f'data/processed/{args.dataset}/{record}/{args.feature_tag}',None if args.no_reuse_pilot else ROOT/f'data/processed/{dataset}/pilot-features-v1'))
    if not args.execute:
        print(json.dumps({'mode':'cohort prepare dry-run','network':False,'writes':False,'records':[task[0] for task in tasks]},indent=2))
        return 0
    results = []
    manager = Manager()
    stop = manager.Event()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    interrupted = False
    pending = {pool.submit(prepare_one,task,stop):task[0] for task in tasks}
    try:
        for future in as_completed(pending):
            try:
                result = future.result()
            except Exception as exc:
                result = {'record_id':pending[future],'ok':False,'error':str(exc)}
            results.append(result)
            print(json.dumps(result),flush=True)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        for future in pending:
            future.cancel()
        print('preparation cancel requested; verified inputs retained',flush=True)
    finally:
        pool.shutdown(wait=True,cancel_futures=interrupted)
        manager.shutdown()
    index = {'dataset':args.dataset,'version':manifest['version'],'split_policy':manifest['subjects'],
             'complete':not interrupted and len(results)==len(tasks) and all(result['ok'] for result in results),
             'interrupted':interrupted,
             'records':sorted(results,key=lambda result:result['record_id'])}
    folder = output_path(ROOT,f'data/processed/{args.dataset}')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'corpus.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    return 130 if interrupted else (0 if index['complete'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
