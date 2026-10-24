import hashlib
import json

import numpy as np
import pytest

from nightlab.audio.features import AudioFeatureConfig, AudioFeatureExtractor
from nightlab.audio.reference import ReferenceEvent
from nightlab.audio.events import evaluate_events
from scripts.datasets.dense_audio import complete_windows, ensure_dense
from scripts.datasets.prepare import prepare
from scripts.evaluate.audio_events import (aggregate_events, clipped_references,
                                          load_owned_models, validate_assignments)


class MemoryRecording:
    duration_seconds = 20.0

    def __init__(self):
        self.calls = []

    def sample_rate(self, channel):
        assert channel == 'Mic'
        return 8000

    def read_window(self, channel, start, duration):
        self.calls.append(start)
        t = np.arange(round(duration*8000))/8000+start
        return (.2*np.sin(2*np.pi*(500+start)*t)).astype(np.float32)


def memory_inputs():
    recording = MemoryRecording()
    config = AudioFeatureConfig()
    extractor = AudioFeatureExtractor(config)
    metadata = {'config':config.metadata(), 'feature_names':list(extractor.feature_names), 'channel':'Mic',
                'candidate_windows':3, 'rows':2, 'source_rate_hz':8000,
                'segment_duration_seconds':20.0, 'segment_offset_seconds':100.0}
    scored = {'X':np.stack([extractor.transform(recording.read_window('Mic',start,10),8000) for start in (0,10)]),
              'y':np.array(['reference_respiratory_event','no_target_reference_event']),
              'start_seconds':np.array([0,10]), 'end_seconds':np.array([10,20]),
              'subject_id':np.array(['s','s']), 'record_id':np.array(['r','r']),
              'overlap_fraction':np.array([.6,0])}
    timeline = {'start_seconds':np.array([0,5,10]), 'end_seconds':np.array([10,15,20]),
                'target':np.array(['reference_respiratory_event','','no_target_reference_event']),
                'target_valid':np.array([True,False,True]), 'overlap_fraction':np.array([.6,.3,0])}
    recording.calls.clear()
    return recording, [ReferenceEvent(102,108,'Hypopnea')], 100.0, metadata, scored, timeline, 's','r'


def test_only_missing_windows_and_one_spot_check_are_recomputed():
    inputs = memory_inputs()
    arrays, counts = complete_windows(*inputs)
    assert inputs[0].calls == [0,5]
    assert arrays['X'].shape == (3,67)
    np.testing.assert_array_equal(arrays['X'][[0,2]],inputs[4]['X'])
    expected = AudioFeatureExtractor().transform(inputs[0].read_window('Mic',5,10),8000)
    np.testing.assert_array_equal(arrays['X'][1],expected)
    assert not np.all(arrays['X'][1] == 0)
    assert arrays['supervised_mask'].tolist() == [True,False,True]
    assert 'y' not in arrays  # supplemental rows have no fabricated training label
    assert counts['reused_scored_rows'] == 2 and counts['supplemented_unlabelled_rows'] == 1


@pytest.mark.parametrize('mutation', ['scored_identity','scored_feature','timeline_gap','timeline_target','config','duration'])
def test_inconsistent_reuse_is_rejected(mutation):
    args = memory_inputs()
    if mutation == 'scored_identity':
        args[4]['subject_id'][0] = 'z'
    elif mutation == 'scored_feature':
        args[4]['X'][0,0] += .1
    elif mutation == 'timeline_gap':
        args[5]['start_seconds'][1] = 6
    elif mutation == 'timeline_target':
        args[5]['target'][1] = 'no_target_reference_event'
    elif mutation == 'config':
        args[3]['config']['hop_seconds'] = 10
    else:
        args[3]['segment_duration_seconds'] = 19
    with pytest.raises(ValueError):
        complete_windows(*args)


def one_channel_edf(path):
    fixed = bytearray(b' '*256)
    for start,value in [(0,b'0       '),(168,b'07.05.10'),(176,b'22.00.00'),
                        (184,b'512     '),(236,b'20      '),(244,b'1       '),(252,b'1   ')]:
        fixed[start:start+len(value)] = value
    header = bytearray(b' '*256)
    header[:16] = b'Mic             '
    for offset,value in [(16+80+8+8+8,-32768),(16+80+8+8+8+8,32767),
                         (16+80+8+8+8+8+8+80,8000)]:
        header[offset:offset+8] = str(value).ljust(8).encode()
    t = np.arange(20*8000)/8000
    values = (1000*np.sin(2*np.pi*300*t)).astype('<i2')
    path.write_bytes(fixed+header+values.tobytes())


def prepared_fixture(root):
    record_id = '00000001-100507'
    folder = root/f'data/raw/test/{record_id}'
    folder.mkdir(parents=True)
    edf,rml = folder/'audio.edf',folder/'reference.rml'
    one_channel_edf(edf)
    rml.write_text('''<RML><RecordingStart>2010-05-07T22:00:00</RecordingStart>
<Segment><StartTime>2010-05-07T22:00:00</StartTime><Duration>20</Duration></Segment>
<Event Family="Respiratory" Type="Hypopnea" Start="2" Duration="6"/></RML>''',encoding='utf-8')
    entries = [{'path':path.relative_to(root).as_posix(), 'subject_id':'00000001', 'record_id':record_id,
                'exact_bytes':path.stat().st_size,
                'official_checksums':{'md5':hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest()}}
               for path in (edf,rml)]
    manifest = {'id':'psg-audio','version':'V3','license':{'verified':True},'files':entries,'inventory_complete':False}
    output = f'data/processed/test/{record_id}/features-v1'
    prepare(root,manifest,output,execute=True)
    return {**manifest,'id':'test'}, {'record_id':record_id,'path':output}, edf


def test_real_edf_cache_reuse_and_corruption_detection(tmp_path,monkeypatch):
    manifest,record,edf = prepared_fixture(tmp_path)
    arrays,receipt,events = ensure_dense(tmp_path,manifest,record)
    assert receipt['rows'] == 3 and len(events) == 1
    from nightlab.audio.edf import EDFRecording
    with monkeypatch.context() as context:
        context.setattr(EDFRecording,'read_window',lambda *args: pytest.fail('cache must not recompute features'))
        reused,second,_ = ensure_dense(tmp_path,manifest,record)
        assert second == receipt
        np.testing.assert_array_equal(reused['X'],arrays['X'])
    dense = next((tmp_path/'data/processed/test').glob('*/psg-continuous-classic-v1/dense.npz'))
    original = dense.read_bytes()
    dense.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        ensure_dense(tmp_path,manifest,record)
    dense.write_bytes(original)
    with edf.open('r+b') as stream:
        stream.seek(-2,2)
        stream.write(b'xx')
    with pytest.raises(ValueError,match='raw input'):
        ensure_dense(tmp_path,manifest,record)


def test_clip_references_excludes_touching_and_nontarget_and_marks_truncation():
    events = [ReferenceEvent(90,100,'Hypopnea'),ReferenceEvent(95,105,'Hypopnea'),
              ReferenceEvent(105,110,'CentralApnea'),ReferenceEvent(110,125,'MixedApnea'),
              ReferenceEvent(120,130,'ObstructiveApnea'),ReferenceEvent(105,110,'RERA')]
    intervals,censored,kinds = clipped_references(events,100,20)
    assert intervals == [[0,5],[5,10],[10,20]]
    assert censored == [True,False,True]
    assert kinds == ['Hypopnea','CentralApnea','MixedApnea']


def test_aggregate_counts_time_and_localization_weights():
    a = evaluate_events([[1,11],[50,60]],[[0,10]],3600)
    b = evaluate_events([[10,20],[22,32]],[[10,20],[20,30],[40,50]],10800)
    result = aggregate_events([a,b])
    assert (result['tp'],result['fp'],result['fn']) == (3,1,1)
    assert result['precision'] == result['recall'] == result['f1'] == .75
    assert result['false_positives_per_hour'] == .25  # 4 hours, not two equally weighted rates
    assert result['localization']['mean_onset_error_seconds'] == 1
    assert result['localization']['mean_absolute_end_error_seconds'] == 1
    empty = aggregate_events([evaluate_events([],[],3600)])
    assert empty['precision'] is empty['recall'] is empty['f1'] is None
    assert empty['localization']['mean_onset_error_seconds'] is None


def test_owned_models_all_hashes_checked_before_any_joblib_load(tmp_path,monkeypatch):
    folder = tmp_path/'models/baseline'
    folder.mkdir(parents=True)
    (tmp_path/'docs/results').mkdir(parents=True)
    hashes = {}
    for name in ('majority','random_forest'):
        payload = name.encode()
        (folder/f'{name}.joblib').write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    receipt = {'checks':[{'run':'baseline','owned_saved_model_predictions_match_retrained_models':True,
                          'local_model_sha256':hashes}]}
    (tmp_path/'docs/results/baseline-reproducibility.json').write_text(json.dumps(receipt),encoding='utf-8')
    calls = []
    monkeypatch.setattr('scripts.evaluate.audio_events.joblib.load',lambda path:calls.append(path) or path.stem)
    report = {'models':{name:{} for name in hashes}}
    assert load_owned_models(tmp_path,'baseline',report)[0]['majority'] == 'majority'
    calls.clear()
    (folder/'random_forest.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        load_owned_models(tmp_path,'baseline',report)
    assert calls == []


def test_event_evaluation_refuses_overlapping_or_incomplete_splits():
    valid = {'train':['a'],'validation':['b'],'test':['c']}
    validate_assignments(valid,['a','b','c'])
    for invalid in ({**valid,'test':['b']},{**valid,'test':[]},{**valid,'test':['c','c']},
                    {**valid,'test':['d']}):
        with pytest.raises(ValueError):
            validate_assignments(invalid,['a','b','c'])
