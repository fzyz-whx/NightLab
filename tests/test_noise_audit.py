import json

import numpy as np
import pytest

from nightlab.audio.features import AudioFeatureExtractor
from scripts.datasets import noise_audit


def source_fixture(root,monkeypatch):
    rate,duration = 16000,10
    signal = (.05*np.sin(2*np.pi*500*np.arange(rate*duration)/rate)).astype(np.float32)
    folder = root/'data/processed/fixture/source'
    folder.mkdir(parents=True)
    np.savez(folder/'features.npz',start_seconds=np.array([0.]),X=np.stack([AudioFeatureExtractor().transform(signal,rate)]))
    metadata = {'source_rate_hz':rate,'segment_duration_seconds':duration,'channel':'Mic'}
    manifest = {'id':'fixture'}
    record = {'record_id':'00000995-100507','path':'data/processed/fixture/source'}
    calls = []
    class Recording:
        duration_seconds = duration
        def sample_rate(self,channel):
            return rate
        def read_window(self,channel,start,length):
            calls.append((start,length))
            return signal[round(start*rate):round((start+length)*rate)].copy()
    monkeypatch.setattr(noise_audit,'_select',lambda *_:(manifest,record,metadata))
    monkeypatch.setattr(noise_audit,'record_context',lambda *_:(folder,metadata,Recording(),[],0,{'fixture':True}))
    monkeypatch.setattr(noise_audit,'available_memory',lambda:10*1024**3)
    monkeypatch.setattr(noise_audit,'peak_working_set',lambda:1234)
    (root/'docs').mkdir()
    (root/'docs/NOISE_PROTOCOL.md').write_text('synthetic fixture',encoding='utf-8')
    return calls


def test_default_plan_writes_nothing_and_checks_memory_before_reading(tmp_path,monkeypatch):
    calls = source_fixture(tmp_path,monkeypatch)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*'))
    report = noise_audit.audit(tmp_path)
    assert not report['writes'] and not report['network']
    assert calls == []
    assert before == sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*'))
    monkeypatch.setattr(noise_audit,'available_memory',lambda:1)
    with pytest.raises(ValueError,match='free RAM'):
        noise_audit.audit(tmp_path,execute=True)
    assert calls == []


def test_actual_short_source_audit_has_all_conditions_and_atomic_json(tmp_path,monkeypatch):
    calls = source_fixture(tmp_path,monkeypatch)
    report = noise_audit.audit(tmp_path,execute=True)
    assert calls == [(0.,10.)]
    assert report['original_feature_spotcheck_exact']
    assert len(report['conditions']) == 8
    assert {(item['kind'],item['calibration']['target_snr_db']) for item in report['conditions']} == {
        (kind,snr) for kind in ('white','mains_hum') for snr in (-5,0,5,10)}
    for item in report['conditions']:
        assert not any(item['clipped_samples'].values())
        assert abs(item['actual_rounded_analysis_snr_db']-item['calibration']['target_snr_db']) < 1e-4
        assert item['verified_samples_per_branch'] == 160000
    saved = tmp_path/report['output']/'report.json'
    assert json.loads(saved.read_text(encoding='utf-8')) == report
    assert not list(saved.parent.parent.glob('.noise-audit-*'))
    with pytest.raises(ValueError,match='exists'):
        noise_audit.audit(tmp_path,execute=True)


def test_failed_source_check_publishes_no_success_report(tmp_path,monkeypatch):
    source_fixture(tmp_path,monkeypatch)
    original = noise_audit.record_context
    def changed(*args):
        values = original(*args)
        values[2].read_window = lambda *args:np.zeros(160000,dtype=np.float32)
        return values
    monkeypatch.setattr(noise_audit,'record_context',changed)
    with pytest.raises(ValueError,match='spot check'):
        noise_audit.audit(tmp_path,execute=True)
    assert not (tmp_path/'data/processed/noise-audit-v1').exists()
