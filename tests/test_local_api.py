import csv
import io
from pathlib import Path
import threading
import time

import numpy as np
import pytest

pytest.importorskip('fastapi')
sf = pytest.importorskip('soundfile')
from fastapi.testclient import TestClient
from nightlab.api.app import create_app
from nightlab.services.jobs import audio_windows


PROJECT = Path(__file__).resolve().parents[1]
HEADERS = {'X-Nightlab-Client':'1'}


class FakeRegistry:
    def require(self,model_id):
        if model_id!='audio_rf_pilot_v1': raise ValueError('未知模型')
        return 'synthetic-test-version'
    def list_models(self):
        return [{'id':'audio_rf_pilot_v1','name':'测试控制','kind':'audio','available':True,'version':'test','scope':'mock'}]
    def predict_windows(self,model_id,waves,rate):
        return np.full(len(waves),.9)


def wav_bytes(seconds=12.3):
    buffer = io.BytesIO(); rate = 16000
    sf.write(buffer,.05*np.sin(2*np.pi*400*np.arange(round(rate*seconds))/rate),rate,format='WAV',subtype='PCM_16')
    return buffer.getvalue()


def import_record(client,seconds=12.3):
    response = client.post('/api/records/import',content=wav_bytes(seconds),headers={**HEADERS,'X-Filename':'test.wav','X-Consent':'true'})
    assert response.status_code==201,response.text
    return response.json()


def wait_job(client,job_id,timeout=20):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        response = client.get('/api/jobs/'+job_id); assert response.status_code==200
        job = response.json()
        if job['status'] in ('completed','cancelled','failed'): return job
        time.sleep(.02)
    pytest.fail('local job did not reach a terminal state')


def test_health_security_bad_import_and_zero_orphans(tmp_path):
    app = create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(app) as client:
        assert client.get('/api/health').json()['privacy']=='local'
        assert client.get('/api/records').json()=={'records':[]}
        assert client.post('/api/demo').status_code==403
        assert client.get('/api/health',headers={'host':'evil.example'}).status_code==400
        assert client.post('/api/records/import',content=b'bad',headers=HEADERS).status_code==400
        response = client.post('/api/records/import',content=b'bad',headers={**HEADERS,'X-Consent':'true'})
        assert response.status_code==400
        assert not list(app.state.records.staging_dir.iterdir())
        assert not list(app.state.records.records_dir.iterdir())
        assert client.get('/api/records').json()['records']==[]


def test_demo_media_annotations_wearable_and_physical_delete(tmp_path):
    app = create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(app) as client:
        response = client.post('/api/demo',headers=HEADERS); assert response.status_code==201,response.text
        record = response.json(); uid = record['id']; base = '/api/records/'+uid
        assert record['is_demo'] and record['audio']['duration_seconds']==90
        detail = client.get(base).json(); assert len(detail['events'])==3
        assert all(event['source']=='demo' and event['score'] is None for event in detail['events'])
        part = client.get(base+'/audio',headers={'Range':'bytes=0-127'})
        assert part.status_code==206 and len(part.content)==128 and part.content[:4]==b'RIFF'
        curve = client.get(base+'/waveform?points=25').json()
        assert len(curve['points'])==25 and curve['duration_seconds']==90
        image = client.get(base+'/spectrogram?start=85&duration=10')
        assert image.status_code==200 and image.content[:8]==b'\x89PNG\r\n\x1a\n'
        marker = client.post(base+'/annotations',headers=HEADERS,json={'start_seconds':1,'end_seconds':2,'label':'=1+1'})
        assert marker.status_code==201 and marker.json()['source']=='user' and marker.json()['score'] is None
        assert client.post(base+'/annotations',headers=HEADERS,json={'start_seconds':1,'end_seconds':200,'label':'bad'}).status_code==400
        assert client.post(base+'/annotations',headers=HEADERS,json={'start_seconds':1,'end_seconds':2,'label':'x','source':'model'}).status_code==422
        exported = client.get(base+'/events.csv'); assert exported.status_code==200
        assert "'=1+1" in exported.text
        payload = b'time_seconds,heart_rate_bpm\n0,60\n5,\n10,65\n'
        attached = client.post(base+'/wearable',content=payload,headers={**HEADERS,'Content-Type':'text/csv'})
        assert attached.status_code==200 and attached.json()['heart_rate'][1]['value'] is None
        assert client.get('/api/templates/wearable.csv').status_code==200
        assert client.get('/api/templates/wearable.json').status_code==200
        assert client.request('DELETE',base,headers=HEADERS,json={'confirm':False}).status_code==400
        folder = app.state.records.records_dir/uid
        (folder/'derived').mkdir(); (folder/'derived/own-preview.txt').write_text('fixture')
        deleted = client.request('DELETE',base,headers=HEADERS,json={'confirm':True})
        assert deleted.status_code==200 and deleted.json()['deleted']
        assert not folder.exists() and client.get(base).status_code==404
        with app.state.records.repo._connection() as db:
            for table in ('records','events','wearable','jobs'):
                assert db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]==0


def test_complete_tail_coverage_actual_worker_and_no_personal_accuracy(tmp_path):
    windows = audio_windows(196800,16000)
    assert windows==[(0.,10.),(2.3,12.3)]
    app = create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(app) as client:
        record = import_record(client); base='/api/records/'+record['id']
        submitted = client.post(base+'/analyze',json={'model_id':'audio_rf_pilot_v1'},headers=HEADERS)
        assert submitted.status_code==202
        result = wait_job(client,submitted.json()['id'])
        assert result['status']=='completed' and result['processed_windows']==result['total_windows']==2
        detail = client.get(base).json(); events=detail['events']
        assert len(events)==1 and events[0]['source']=='model' and events[0]['end_seconds']==12.3
        assert events[0]['model_version']=='synthetic-test-version'
        assert 'accuracy' not in detail and detail['wearable'] is None
        short = import_record(client,1)
        assert client.post('/api/records/'+short['id']+'/analyze',json={'model_id':'audio_rf_pilot_v1'},headers=HEADERS).status_code==400


def test_running_cancel_delete_guard_and_atomic_existing_results(tmp_path):
    started,release = threading.Event(),threading.Event()
    class Slow(FakeRegistry):
        def predict_windows(self,model_id,waves,rate):
            started.set(); assert release.wait(10)
            return super().predict_windows(model_id,waves,rate)
    app = create_app(tmp_path,PROJECT,num_threads=1,registry=Slow())
    with TestClient(app) as client:
        try:
            record=import_record(client); base='/api/records/'+record['id']
            old=app.state.records.repo.add_event({'record_id':record['id'],'source':'model','model_id':'previous',
                'start_seconds':0,'end_seconds':1,'label':'previous','score':.6})
            submitted=client.post(base+'/analyze',json={'model_id':'audio_rf_pilot_v1'},headers=HEADERS).json()
            assert started.wait(10)
            assert client.post(base+'/analyze',json={'model_id':'audio_rf_pilot_v1'},headers=HEADERS).status_code==409
            assert client.request('DELETE',base,json={'confirm':True},headers=HEADERS).status_code==409
            assert client.post('/api/jobs/'+submitted['id']+'/cancel',headers=HEADERS).status_code==200
            release.set(); stopped=wait_job(client,submitted['id'])
            assert stopped['status']=='cancelled'
            assert client.get(base).json()['events'][0]['id']==old['id']
            assert client.request('DELETE',base,json={'confirm':True},headers=HEADERS).status_code==200
        finally: release.set()


def test_real_owned_rf_http_file_sql_model_path(tmp_path):
    app = create_app(tmp_path,PROJECT,num_threads=1)
    available = app.state.models.list_models()
    if not next(item for item in available if item['id']=='audio_rf_pilot_v1')['available']:
        app.state.jobs.close(); pytest.skip('own verified model not included in public checkout')
    with TestClient(app) as client:
        record=import_record(client,20)
        response=client.post('/api/records/'+record['id']+'/analyze',json={'model_id':'audio_rf_pilot_v1'},headers=HEADERS)
        assert response.status_code==202,response.text
        result=wait_job(client,response.json()['id'])
        assert result['status']=='completed' and result['processed_windows']==3
        events=client.get('/api/records/'+record['id']).json()['events']
        assert all(event['source']=='model' and 0<=event['score']<=1 for event in events)


def test_readonly_real_experiments_and_no_arbitrary_paths(tmp_path):
    app=create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(app) as client:
        rows=client.get('/api/experiments').json()['experiments']
        assert any(item['id']=='noise_augmentation' for item in rows)
        detail=client.get('/api/experiments/noise_augmentation').json()
        assert len(detail['report']['conditions'])==32
        assert client.get(detail['figures'][0]['url']).content.startswith(b'\x89PNG')
        assert client.get('/api/experiments/not_registered').status_code==404


def test_restart_recovers_stale_jobs_without_publishing_partial_events(tmp_path):
    from uuid import uuid4
    from nightlab.storage.repository import utc_now
    first=create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(first) as client:
        record=import_record(client)
        job_id=str(uuid4())
        first.state.records.repo.create_job({'id':job_id,'record_id':record['id'],'model_id':'audio_rf_pilot_v1',
            'status':'running','created_at':utc_now(),'total_windows':2,'processed_windows':1})
    restarted=create_app(tmp_path,PROJECT,num_threads=1,registry=FakeRegistry())
    with TestClient(restarted) as client:
        job=client.get('/api/jobs/'+job_id).json()
        assert job['status']=='failed' and job['finished_at']
        assert client.get('/api/records/'+record['id']).json()['events']==[]
        submitted=client.post('/api/records/'+record['id']+'/analyze',headers=HEADERS,json={})
        assert submitted.status_code==202
        assert wait_job(client,submitted.json()['id'])['status']=='completed'
