"""Supplement ambiguous windows for continuous inference; never assign labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio.edf import EDFRecording
from nightlab.audio.features import AudioFeatureConfig, AudioFeatureExtractor
from nightlab.audio.reference import read_reference, window_target
from .cli import ROOT, load_manifest, raw_path, verify
from .prepare import output_path
from .prepare_cohort import validate_receipt


DENSE_VERSION = 'psg-continuous-classic-v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hash():
    return sha256(__file__)


def complete_windows(recording, events, offset, metadata, scored, timeline, subject, record,
                     progress=None):
    """Reuse scored rows exactly; derive and check geometry/targets before reuse."""
    config = AudioFeatureConfig()
    extractor = AudioFeatureExtractor(config)
    if metadata['config'] != config.metadata() or metadata['feature_names'] != list(extractor.feature_names):
        raise ValueError('dense inference requires the frozen classic feature configuration')
    rate = recording.sample_rate(metadata['channel'])
    if rate != int(rate) or rate < 8000:
        raise ValueError('invalid source audio rate')
    duration = recording.duration_seconds
    count = 1 + math.floor((duration-config.window_seconds)/config.hop_seconds+1e-9)
    if count <= 0:
        raise ValueError('recording shorter than feature window')
    starts = np.arange(count, dtype=np.float64)*config.hop_seconds
    ends = starts+config.window_seconds
    labels, fractions = zip(*(window_target(offset+left, offset+right, events,
                                            config.positive_overlap_fraction)
                              for left, right in zip(starts, ends, strict=True)))
    labels = np.asarray([label or '' for label in labels], dtype=str)
    keep = labels != ''
    expected_timeline = {'start_seconds':starts, 'end_seconds':ends, 'target':labels,
                         'target_valid':keep, 'overlap_fraction':np.asarray(fractions)}
    if (count != metadata['candidate_windows'] or int(keep.sum()) != metadata['rows']
            or metadata['segment_duration_seconds'] != duration
            or metadata['segment_offset_seconds'] != offset or int(rate) != metadata['source_rate_hz']):
        raise ValueError('recording metadata differs from feature receipt')
    for name, expected in expected_timeline.items():
        if not np.array_equal(timeline[name], expected):
            raise ValueError(f'timeline {name} differs from source reference/geometry')
    expected_scored = {'start_seconds':starts[keep], 'end_seconds':ends[keep], 'y':labels[keep],
                       'overlap_fraction':np.asarray(fractions)[keep],
                       'subject_id':np.full(keep.sum(), subject), 'record_id':np.full(keep.sum(), record)}
    for name, expected in expected_scored.items():
        if not np.array_equal(scored[name], expected):
            raise ValueError(f'scored {name} differs from timeline or selected identity')
    x = np.asarray(scored['X'])
    if x.shape != (keep.sum(), len(extractor.feature_names)) or not np.isfinite(x).all() or not keep.any():
        raise ValueError('invalid scored feature matrix')
    dense = np.empty((count, x.shape[1]), dtype=np.float32)
    dense[keep] = x
    with threadpool_limits(limits=1):
        check_index = int(np.flatnonzero(keep)[0])
        check = extractor.transform(recording.read_window(metadata['channel'], float(starts[check_index]),
                                                          config.window_seconds), int(rate))
        if not np.array_equal(check, x[0]):
            raise ValueError('recomputed scored feature differs from frozen artifact')
        missing = np.flatnonzero(~keep)
        for done, index in enumerate(missing, 1):
            dense[index] = extractor.transform(recording.read_window(metadata['channel'], float(starts[index]),
                                                                      config.window_seconds), int(rate))
            if progress is not None and (done % 25 == 0 or done == len(missing)):
                progress(done, len(missing))
    return {'X':dense, 'start_seconds':starts, 'end_seconds':ends, 'supervised_mask':keep}, {
        'rows':count, 'reused_scored_rows':int(keep.sum()), 'supplemented_unlabelled_rows':len(missing),
        'spot_check_scored_index':check_index, 'spot_check_exact':True,
    }


def record_context(root, manifest, record):
    entries = [entry for entry in manifest['files'] if entry.get('record_id') == record['record_id']]
    if len(entries) != 2 or {Path(entry['path']).suffix for entry in entries} != {'.edf','.rml'}:
        raise ValueError('one EDF and RML required for selected record')
    subject = record['record_id'].split('-')[0]
    if any(entry.get('subject_id') != subject for entry in entries):
        raise ValueError('manifest subject does not match record')
    subset = {**manifest, 'id':'psg-audio', 'files':entries}
    if not manifest['license'].get('verified'):
        raise ValueError('verified data license required')
    folder = output_path(root, record['path'])
    metadata = validate_receipt(folder, subset)
    integrity = verify(root, subset)
    previous = {entry['path']:entry['local_checksums'] for entry in metadata['input_integrity']['files']}
    if not integrity['ok'] or any(entry['local_checksums'] != previous[entry['path']]
                                  for entry in integrity['files']):
        raise ValueError('raw input differs from official or prepared receipt')
    paths = {Path(entry['path']).suffix:raw_path(root,entry['path']) for entry in entries}
    recording = EDFRecording(paths['.edf'])
    origin, segments, events = read_reference(paths['.rml'])
    offset = (recording.start-origin).total_seconds()
    if (offset, recording.duration_seconds) not in segments:
        raise ValueError('EDF clock/duration does not match RML segment')
    provenance = {'record_id':record['record_id'], 'subject_id':subject,
                  'dataset_version':manifest['version'], 'feature_version':DENSE_VERSION,
                  'source_features_path':record['path'], 'source_metadata_sha256':sha256(folder/'metadata.json'),
                  'source_artifact_sha256':metadata['artifact_sha256'],
                  'feature_source_sha256':metadata['feature_source_sha256'],
                  'dense_source_sha256':source_hash(),
                  'input_sha256':{entry['path']:entry['local_checksums']['sha256'] for entry in integrity['files']},
                  'channel':metadata['channel'], 'duration_seconds':recording.duration_seconds,
                  'offset_seconds':offset, 'feature_names':metadata['feature_names']}
    return folder, metadata, recording, events, offset, provenance


def ensure_dense(root, manifest, record, progress=None):
    """Verify raw files even on reuse, and publish a complete cache atomically."""
    folder, metadata, recording, events, offset, provenance = record_context(root,manifest,record)
    relative = f"data/processed/{manifest['id']}/{record['record_id']}/{DENSE_VERSION}"
    destination = output_path(root,relative)
    if destination.exists():
        receipt = json.loads((destination/'metadata.json').read_text(encoding='utf-8'))
        if receipt.get('provenance') != provenance or receipt.get('artifact_sha256') != sha256(destination/'dense.npz'):
            raise ValueError('dense cache provenance/checksum differs; refusing reuse')
        with np.load(destination/'dense.npz',allow_pickle=False) as data:
            arrays = {key:data[key] for key in data.files}
        expected_rows = metadata['candidate_windows']
        if (arrays['X'].shape != (expected_rows,len(metadata['feature_names']))
                or not np.isfinite(arrays['X']).all() or receipt['rows'] != expected_rows):
            raise ValueError('dense cache shape or values invalid')
        return arrays, receipt, events
    started = time.perf_counter()
    with np.load(folder/'features.npz',allow_pickle=False) as data:
        scored = {key:data[key] for key in data.files}
    with np.load(folder/'timeline.npz',allow_pickle=False) as data:
        timeline = {key:data[key] for key in data.files}
    arrays, counts = complete_windows(recording,events,offset,metadata,scored,timeline,
                                      provenance['subject_id'],record['record_id'],progress)
    receipt = {**counts, 'schema_version':1, 'provenance':provenance,
               'elapsed_seconds':time.perf_counter()-started, 'worker_count':1, 'numeric_threads':1,
               'scope':'continuous inference features only; ambiguous windows remain unlabelled'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dense-',dir=destination.parent) as temporary:
        staged = Path(temporary)/'cache'
        staged.mkdir()
        np.savez_compressed(staged/'dense.npz',**arrays)
        receipt['artifact_sha256'] = sha256(staged/'dense.npz')
        (staged/'metadata.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        staged.rename(destination)
    return arrays, receipt, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='psg-audio-pilot-10',choices=('psg-audio-pilot-10',))
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    manifest = load_manifest(ROOT/f'data/manifests/{args.dataset}.json',ROOT)
    corpus = json.loads((ROOT/f'data/processed/{args.dataset}/corpus.json').read_text(encoding='utf-8'))
    if not corpus['complete'] or corpus['split_policy'] != manifest['subjects']:
        raise ValueError('complete corpus with matching planned split required')
    selected = set(manifest['subjects']['subjects']['validation']+manifest['subjects']['subjects']['test'])
    records = [record for record in corpus['records'] if record['record_id'].split('-')[0] in selected]
    if len(records) != len(selected) or not all(record['ok'] for record in records):
        raise ValueError('missing held-out records')
    if not args.execute:
        print(json.dumps({'mode':'dense dry-run', 'network':False, 'writes':False, 'workers':1,
                          'records':[record['record_id'] for record in records]},indent=2))
        return 0
    for record in records:
        _, receipt, _ = ensure_dense(ROOT,manifest,record,
                                     lambda done,total: print(f"{record['record_id']} supplement {done}/{total}",flush=True))
        print(json.dumps({'record_id':record['record_id'], **{key:receipt[key] for key in
                          ('rows','reused_scored_rows','supplemented_unlabelled_rows','spot_check_exact')}}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
