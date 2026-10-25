"""Audit full-record synthetic mixing on one training record; no model scores."""
import argparse
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nightlab.audio import noise
from nightlab.audio.features import AudioFeatureExtractor
from .cli import ROOT,load_manifest
from .dense_audio import record_context,sha256
from .prepare import output_path


def available_memory():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[
                (name,ctypes.c_uint64) for name in ('total_phys','available_phys','total_page','available_page',
                                                 'total_virtual','available_virtual','extended')]
        status = Status(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('cannot inspect available RAM')
        return int(status.available_phys)
    return int(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))


def peak_working_set():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('length',ctypes.c_uint32),('page_faults',ctypes.c_uint32)]+[
            (name,ctypes.c_size_t) for name in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    counters = Counters(); counters.length = ctypes.sizeof(counters)
    api = ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_uint32]
    if not api(ctypes.c_void_p(-1),ctypes.byref(counters),counters.length):
        raise OSError('cannot inspect process memory')
    return int(counters.peak_ws)


def _select(root,record_id):
    manifest = load_manifest(root/'data/manifests/psg-audio-pilot-10.json',root)
    corpus = json.loads((root/'data/processed/psg-audio-pilot-10/corpus.json').read_text(encoding='utf-8'))
    if not corpus['complete'] or corpus['split_policy'] != manifest['subjects']:
        raise ValueError('complete classic corpus with planned split required')
    if record_id.split('-')[0] not in corpus['split_policy']['subjects']['train']:
        raise ValueError('calibration audit is limited to planned training subjects')
    selected = [record for record in corpus['records'] if record['record_id'] == record_id and record['ok']]
    if len(selected) != 1:
        raise ValueError('one prepared selected training record required')
    record = selected[0]
    metadata = json.loads((output_path(root,record['path'])/'metadata.json').read_text(encoding='utf-8'))
    return manifest,record,metadata


def audit(root,record_id='00000995-100507',execute=False,output=None):
    root = Path(root).resolve()
    manifest,record,metadata = _select(root,record_id)
    relative = output or f'data/processed/noise-audit-v1/{record_id}'
    destination = output_path(root,relative)
    rate,duration = metadata['source_rate_hz'],metadata['segment_duration_seconds']
    count = round(rate*duration)
    if type(rate) is not int or rate < 16000 or rate % 16000 or count <= 0 or count != rate*duration:
        raise ValueError('integral sample geometry at a multiple of 16k required')
    required = 2*count*4+512*1024**2+2*1024**3
    plan = {'mode':'noise calibration audit execute' if execute else 'noise calibration audit dry-run',
            'record_id':record_id, 'split':'train', 'network':False, 'writes':execute,
            'source_rate_hz':rate,'duration_seconds':duration,'sample_count':count,
            'resident_reference_and_noise_bytes':2*count*4,'required_free_ram_bytes':required,
            'workers':1,'numeric_threads_max':2,'output':relative,
            'noise_kinds':['white','mains_hum'],'snr_db':[-5,0,5,10],
            'scope':'mixing/continuity calibration only; no model evaluation or noise performance claim'}
    if not execute:
        return plan
    if destination.exists():
        raise ValueError('audit output exists; choose a new output, refusing overwrite')
    free_before = available_memory()
    if free_before < required:
        raise ValueError(f'insufficient free RAM: need {required}, observed {free_before}')
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        folder,metadata,recording,_,_,provenance = record_context(root,manifest,record)
        if recording.sample_rate(metadata['channel']) != rate or recording.duration_seconds != duration:
            raise ValueError('verified recording geometry differs from plan')
        reference = np.empty(count,dtype=np.float32)
        chunk_seconds = min(60.0,duration)
        block = round(chunk_seconds*rate)
        for begin in range(0,count,block):
            end = min(count,begin+block)
            reference[begin:end] = recording.read_window(metadata['channel'],begin/rate,(end-begin)/rate)
        reference_hash = hashlib.sha256(memoryview(reference)).hexdigest()
        with np.load(folder/'features.npz',allow_pickle=False) as data:
            first = float(data['start_seconds'][0]); old = data['X'][0]
        with threadpool_limits(limits=1):
            recomputed = AudioFeatureExtractor().transform(reference[round(first*rate):round((first+10)*rate)],rate)
        if not np.array_equal(recomputed,old):
            raise ValueError('complete source read differs from frozen feature spot check')
        conditions = []
        for kind in plan['noise_kinds']:
            minimum_after_reference = count*4+512*1024**2+2*1024**3
            if available_memory() < minimum_after_reference:
                raise ValueError('available RAM changed before noise allocation')
            seed = noise.noise_seed(record_id,'train',kind)
            interference = noise.synthetic_noise(count,rate,kind,seed)
            noise_hash = hashlib.sha256(memoryview(interference)).hexdigest()
            print(f'{record_id} {kind}: full recording noise generated; initializing calibration',file=sys.stderr,flush=True)
            observed = noise.NoiseRecording(reference,interference,rate)
            for snr_db in plan['snr_db']:
                tick = time.perf_counter()
                calibration = observed.calibrate(snr_db)
                actual = observed.measure_actual_snr(calibration)
                if abs(actual-snr_db) > 1e-4:
                    raise ValueError('actual rounded mixture SNR exceeds tolerance')
                peaks = {name:0.0 for name in noise.BRANCHES}
                clipped = {name:0 for name in noise.BRANCHES}
                expected_start = 0
                for output_block in observed.iter_blocks(calibration):
                    begin,end = output_block['start_sample'],output_block['end_sample']
                    if begin != expected_start or end <= begin:
                        raise ValueError('branch stream is discontinuous')
                    for name in noise.BRANCHES:
                        values = output_block[name]
                        if values.shape != (end-begin,) or values.dtype != np.float32:
                            raise ValueError('branch sample grid/dtype differs')
                        peaks[name] = max(peaks[name],float(np.abs(values).max()))
                        clipped[name] += int(np.count_nonzero(np.abs(values)>1))
                    expected_start = end
                if expected_start != count or any(clipped.values()) or max(peaks.values())>0.980001:
                    raise ValueError('branch length or headroom verification failed')
                conditions.append({'kind':kind,'seed':seed,'noise_sha256':noise_hash,
                                   'calibration':calibration,'actual_rounded_analysis_snr_db':actual,
                                   'actual_branch_peaks':peaks,'clipped_samples':clipped,
                                   'verified_samples_per_branch':expected_start,'elapsed_seconds':time.perf_counter()-tick})
                print(json.dumps({'kind':kind,'snr_db':snr_db,'actual_snr_db':actual,
                                  'gain':calibration['common_gain'],'peak':max(peaks.values()),
                                  'elapsed_seconds':conditions[-1]['elapsed_seconds']}),file=sys.stderr,flush=True)
            if hashlib.sha256(memoryview(interference)).hexdigest() != noise_hash:
                raise ValueError('noise input was modified')
            del observed,interference
        if hashlib.sha256(memoryview(reference)).hexdigest() != reference_hash:
            raise ValueError('source reference was modified')
    report = {**plan,'conditions':conditions,'provenance':provenance,
              'reference_samples_sha256':reference_hash,'original_feature_spotcheck_exact':True,
              'protocol_sha256':sha256(root/'docs/NOISE_PROTOCOL.md'),
              'noise_source_sha256':sha256(noise.__file__),'audit_source_sha256':sha256(__file__),
              'available_ram_before_bytes':free_before,'peak_working_set_bytes':peak_working_set(),
              'elapsed_seconds':time.perf_counter()-started,
              'environment':{'python':platform.python_version(), **{name:version(name) for name in ('numpy','scipy')}}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.noise-audit-',dir=destination.parent) as temporary:
        staged = Path(temporary)/'audit'
        staged.mkdir()
        (staged/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        staged.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',default='00000995-100507')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = audit(ROOT,args.record,args.execute,args.output)
    print(json.dumps({key:value for key,value in report.items() if key not in ('conditions','provenance')},indent=2))


if __name__ == '__main__':
    main()
