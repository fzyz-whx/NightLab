"""Synthetic EDF/RML integration, never a real-person/clinical acceptance claim."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('soundfile')
from fastapi.testclient import TestClient
from nightlab.api.app import create_app
from nightlab.services.public_examples import PublicExampleService
from nightlab.services.records import RecordService
from scripts.datasets.prepare import prepare


HEADERS = {'X-Nightlab-Client': '1'}
EXAMPLE = 'psg-validation-1008'


def fixture_root(root):
    record_id = '00001008-100507'
    raw = root/'data/raw/psg-audio'
    raw.mkdir(parents=True)
    edf, rml = raw/'sample.edf', raw/'sample.rml'
    fixed = bytearray(b' '*256)
    for start, value in [(0,b'0       '),(168,b'07.05.10'),(176,b'22.00.10'),
                         (184,b'512     '),(236,b'120     '),(244,b'1       '),(252,b'1   ')]:
        fixed[start:start+len(value)] = value
    signal = bytearray(b' '*256)
    for start, value in [(0,b'Mic'),(104,b'-1'),(112,b'1'),(120,b'-32768'),(128,b'32767'),(216,b'8000')]:
        signal[start:start+len(value)] = value
    wave = (1000*np.sin(2*np.pi*400*np.arange(8000)/8000)).astype('<i2')
    edf.write_bytes(fixed+signal+wave.tobytes()*120)
    rml.write_text('''<RML><RecordingStart>2010-05-07T22:00:00</RecordingStart>
      <Segment><StartTime>2010-05-07T22:00:10</StartTime><Duration>120</Duration></Segment>
      <Event Family="Respiratory" Type="Hypopnea" Start="5" Duration="10"/>
      <Event Family="Respiratory" Type="CentralApnea" Start="120" Duration="20"/>
      <Event Family="Respiratory" Type="RERA" Start="50" Duration="10"/></RML>''',encoding='utf-8')
    entries = [dict(path=p.relative_to(root).as_posix(),exact_bytes=p.stat().st_size,
                    official_checksums={'md5':hashlib.md5(p.read_bytes()).hexdigest()},
                    record_id=record_id,subject_id='00001008',split='validation') for p in (edf,rml)]
    split = dict(unit='subject',subjects=dict(train=[],validation=['00001008'],test=[]))
    manifest = dict(schema_version=1,id='psg-audio-pilot-10',version='V3',
                    license={'verified':True,'name':'CC BY 4.0'},sources=[],reported_size={},
                    inventory_complete=False,files=entries,subjects=split)
    manifests = root/'data/manifests'; manifests.mkdir()
    (manifests/'psg-audio-pilot-10.json').write_text(json.dumps(manifest),encoding='utf-8')
    relative = f'data/processed/psg-audio-pilot-10/{record_id}/features-v1'
    prepare(root,{**manifest,'id':'psg-audio'},relative,execute=True)
    corpus = dict(dataset=manifest['id'],version='V3',complete=True,split_policy=split,
                  records=[dict(record_id=record_id,ok=True,path=relative)])
    path = root/'data/processed/psg-audio-pilot-10/corpus.json'
    path.write_text(json.dumps(corpus),encoding='utf-8')
    return edf, path


class SyntheticRegistry:
    def list_models(self): return []
    def require(self, model_id): return 'synthetic-test'
    def predict_windows(self, model_id, waves, rate): return np.full(len(waves), .9)


def test_real_parser_to_api_references_survive_analysis_and_delete(tmp_path):
    fixture_root(tmp_path)
    app = create_app(tmp_path/'private/app',tmp_path,registry=SyntheticRegistry())
    with TestClient(app) as client:
        catalog = client.get('/api/public-examples').json()['examples']
        assert [e['available'] for e in catalog] == [True,False]
        assert client.post(f'/api/public-examples/{EXAMPLE}/import').status_code == 403
        response = client.post(f'/api/public-examples/{EXAMPLE}/import',headers=HEADERS)
        assert response.status_code == 201, response.text
        record = response.json(); record_id = record['id']
        assert not record['is_demo'] and record['start_time'] is None
        assert record['provenance']['split'] == 'validation'
        assert record['audio']['frames'] == 120*8000
        detail = client.get('/api/records/'+record_id).json()
        assert detail['wearable'] is None
        references = detail['events']
        assert [(e['start_seconds'],e['end_seconds']) for e in references] == [(0,5),(110,120)]
        assert all(e['source']=='reference' and e['reference_clipped'] for e in references)
        assert client.delete(f'/api/records/{record_id}/annotations/{references[0]["id"]}',headers=HEADERS).status_code == 400
        assert client.get(f'/api/records/{record_id}/audio',headers={'Range':'bytes=0-43'}).status_code == 206
        assert client.get(f'/api/records/{record_id}/waveform').json()['points']
        job = client.post(f'/api/records/{record_id}/analyze',json={},headers=HEADERS).json()
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            result = client.get('/api/jobs/'+job['id']).json()
            if result['status'] in {'completed','failed'}: break
            time.sleep(.02)
        assert result['status']=='completed'
        events = client.get('/api/records/'+record_id).json()['events']
        assert sum(e['source']=='reference' for e in events)==2
        assert any(e['source']=='model' for e in events)
        assert 'reference' in client.get(f'/api/records/{record_id}/events.csv').text
        assert client.request('DELETE','/api/records/'+record_id,headers=HEADERS,json={'confirm':True}).status_code == 200
        assert client.get('/api/records').json()['records'] == []


@pytest.mark.parametrize('bad', ['raw','split','features'])
def test_import_checks_source_before_creating_record(tmp_path,bad):
    raw, corpus_path = fixture_root(tmp_path)
    if bad=='raw':
        with raw.open('r+b') as stream: stream.seek(520); stream.write(b'bad')
    elif bad=='split':
        corpus=json.loads(corpus_path.read_text()); corpus['split_policy']['subjects']['validation']=[]
        corpus_path.write_text(json.dumps(corpus))
    else:
        features=next((tmp_path/'data/processed').rglob('features.npz'))
        features.write_bytes(b'broken')
    records=RecordService(tmp_path/'private/app')
    with pytest.raises(ValueError): PublicExampleService(tmp_path,records).import_example(EXAMPLE)
    assert records.repo.list_records()==[]
    assert not list(records.records_dir.iterdir()) and not list(records.staging_dir.iterdir())


def test_missing_data_catalog_has_no_side_effect_and_unknown_id_rejected(tmp_path):
    records=RecordService(tmp_path/'app')
    service=PublicExampleService(tmp_path,records)
    assert not any(item['available'] for item in service.list())
    assert not (tmp_path/'data').exists()
    for example_id in ['../../bad','psg-test-999','unknown']:
        with pytest.raises(KeyError): service.import_example(example_id)


def test_record_and_reference_transaction_rolls_back_on_invalid_source(tmp_path):
    import soundfile as sf
    records=RecordService(tmp_path/'app')
    path=records.staging_dir/'fixture.wav'
    sf.write(path,np.zeros(8000),8000)
    with pytest.raises(ValueError):
        records.import_audio(path,'fixture.wav',consent=True,reference_events=[
            dict(start_seconds=0,end_seconds=1,label='x',source='invalid')])
    assert records.repo.list_records()==[]
    assert not list(records.records_dir.iterdir())
