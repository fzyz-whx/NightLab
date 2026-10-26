"""Verify the opt-in Kotlin synthetic WAV/ZIP fixtures through the real desktop API.

No phone capture claim. No server or GPU is started. Temporary app data is removed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

from fastapi.testclient import TestClient
from nightlab.api.app import create_app
from nightlab.services.media import audio_info_validate

ROOT = Path(__file__).resolve().parents[1]


def verify(archive: Path, wav: Path):
    with zipfile.ZipFile(archive) as zipped:
        if any(info.file_size > 1024 * 1024 for info in zipped.infolist()):
            raise ValueError('This verifier only accepts the small synthetic test fixtures')
        metadata = json.loads(zipped.read('session.json'))
        assert metadata['source'] == 'nightlab_android' and metadata['status'] == 'completed'
        assert metadata['device'] == {'manufacturer': 'Example', 'model': 'Phone', 'android_api': 34}
        assert metadata['time_basis']['external_synchronization'] is False
        assert len(metadata['chunks']) == 2
        assert set(zipped.namelist()) == {'session.json', 'chunk_00000.wav', 'chunk_00001.wav'}
        chunks = [(item, zipped.read(item['filename'])) for item in metadata['chunks']]
    for item, content in chunks:
        assert hashlib.sha256(content).hexdigest() == item['sha256']
        assert len(content) == item['bytes'] == 384044
        assert item['frames'] == 192000 and item['duration_seconds'] == 12
    private = ROOT / 'private'
    private.mkdir(exist_ok=True)
    actual_windows = None
    with tempfile.TemporaryDirectory(dir=private) as temporary:
        app = create_app(Path(temporary) / 'app', ROOT, num_threads=1)
        with TestClient(app) as client:
            headers = {'X-Nightlab-Client': '1', 'X-Consent': 'true'}
            imported = []
            for item, content in chunks:
                response = client.post('/api/records/import', content=content, headers={**headers,
                    'X-Filename': item['filename'], 'X-Start-Time': item['start_time']})
                assert response.status_code == 201, response.text
                record = response.json()
                imported.append(record)
                assert record['audio']['frames'] == 192000 and record['audio']['sha256'] == item['sha256']
                assert client.get('/api/records/' + record['id']).json()['wearable'] is None
            assert imported[0]['start_time'] == '2026-10-25T23:59:59+08:00'
            assert imported[1]['start_time'] == '2026-10-26T00:00:11+08:00'
            owned = next(m for m in client.get('/api/models').json()['models'] if m['id'] == 'audio_rf_pilot_v1')
            if owned['available']:
                response = client.post('/api/records/' + imported[0]['id'] + '/analyze', headers=headers,
                                       json={'model_id': 'audio_rf_pilot_v1'})
                assert response.status_code == 202, response.text
                job = response.json()
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    job = client.get('/api/jobs/' + job['id']).json()
                    if job['status'] in ('completed', 'failed', 'cancelled'):
                        break
                    time.sleep(.02)
                assert job['status'] == 'completed' and job['processed_windows'] == 2, job
                actual_windows = 2
            for record in imported:
                assert client.request('DELETE', '/api/records/' + record['id'], headers=headers,
                                      json={'confirm': True}).json()['deleted']
            assert client.get('/api/records').json()['records'] == []
    info = audio_info_validate(wav)
    assert info['frames'] == 16000 and info['duration_seconds'] == 1 and info['sample_rate'] == 16000
    return {'fixture_date': '2026-10-26', 'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'input': 'Actual Kotlin writer/SessionStore fixture export; synthetic PCM, not microphone',
            'zip_chunks': 2, 'chunk_frames': 192000, 'chunk_seconds': 12, 'hashes_and_wav_headers_verified': True,
            'first_start': imported[0]['start_time'], 'second_start': imported[1]['start_time'],
            'desktop_http_imports': 2, 'no_wearable_dependency': True, 'owned_cpu_rf_actual_windows': actual_windows,
            'owned_rf_scope': 'actual model path exercised' if actual_windows else 'not available; no prediction fabricated',
            'confirmed_delete_records_zero': True, 'standalone_kotlin_wav_seconds': 1,
            'phone_microphone_or_screen_off_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = verify(args.archive, args.wav)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
