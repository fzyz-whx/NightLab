"""Mock transport only: never connect to the real SleepAccel source."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.datasets import fetch_sleepaccel_archive as fetcher


def setup(tmp_path, monkeypatch, prefix=b''):
    monkeypatch.setattr(fetcher, 'ARCHIVE_BYTES', 10)
    monkeypatch.setattr(fetcher, 'CHUNK_BYTES', 4)
    monkeypatch.setattr(fetcher.time, 'sleep', lambda _: None)
    plan = {'approved_archive': True, 'selected_option': 'all31', 'archive': {
        'exact_bytes': 10, 'url': 'https://sleepdata.org/fixture.zip',
        'effective_source_evidence': {'etag': '"fixture"'}}}
    path = tmp_path / fetcher.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    target = tmp_path / fetcher.ARCHIVE
    if prefix:
        target.parent.mkdir(parents=True)
        target.write_bytes(prefix)
    return target, plan


def transport(calls, partial=False, alter=None):
    def run(url, host, etag, address, start, end, body, headers):
        calls.append((start, end))
        fields = {'Content-Range': f'bytes {start}-{end}/10', 'Content-Length': str(end-start+1),
                  'Content-Type': 'application/zip', 'ETag': etag}
        status = '206 Partial Content'
        data = b'0123456789'[start:end+1]
        code = 0
        if partial and len(calls) == 1:
            data, code = data[:2], 18
        if alter:
            status, data, code = alter(fields, status, data, code)
        headers.write_bytes(('HTTP/1.1 ' + status + '\r\n' +
                            '\r\n'.join(f'{k}: {v}' for k, v in fields.items()) + '\r\n\r\n').encode())
        body.write_bytes(data)
        return code
    return run


def test_dry_no_write_or_network(tmp_path, monkeypatch):
    target, _ = setup(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = fetcher.fetch(tmp_path, _private_transport=lambda *a: pytest.fail('network'))
    assert result['dry_run'] and result['offset'] == 0 and not target.exists()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('approved,option,confirmed', [(False, 'all31', True),
                                                     (True, 'small', True), (True, 'all31', False)])
def test_authorization(tmp_path, monkeypatch, approved, option, confirmed):
    target, plan = setup(tmp_path, monkeypatch)
    plan.update(approved_archive=approved, selected_option=option)
    (tmp_path / fetcher.PLAN).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='confirmation'):
        fetcher.fetch(tmp_path, execute=True, confirmed_large=confirmed)
    assert not target.exists()


@pytest.mark.parametrize('prefix,partial,ranges', [(b'', False, [(0,3),(4,7),(8,9)]),
                                                 (b'012', False, [(3,6),(7,9)]),
                                                 (b'', True, [(0,3),(2,5),(6,9)])])
def test_ranges_and_resume(tmp_path, monkeypatch, prefix, partial, ranges):
    target, _ = setup(tmp_path, monkeypatch, prefix)
    calls = []
    result = fetcher.fetch(tmp_path, True, True, _private_transport=transport(calls, partial))
    assert target.read_bytes() == b'0123456789' and calls == ranges
    assert result['download_complete'] and not result['member_crc_sha_verified']
    assert result['partial_sha256_local'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not (tmp_path / fetcher.LOCK).exists()
    assert not (tmp_path / 'data/raw/sleepaccel/archives/intake-receipt.json').exists()


@pytest.mark.parametrize('issue', ['status', 'range', 'etag', 'length', 'type', 'encoding',
                                  'overlong', 'short_success', 'curl_error', 'missing'])
def test_invalid_response_never_appended(tmp_path, monkeypatch, issue):
    target, _ = setup(tmp_path, monkeypatch, b'01')
    def alter(fields, status, data, code):
        if issue == 'status': status = '200 OK'
        elif issue == 'range': fields['Content-Range'] = 'bytes 0-3/10'
        elif issue == 'etag': fields['ETag'] = '"other"'
        elif issue == 'length': fields['Content-Length'] = '999'
        elif issue == 'type': fields['Content-Type'] = 'text/html'
        elif issue == 'encoding': fields['Content-Encoding'] = 'gzip'
        elif issue == 'overlong': data += b'x'
        elif issue == 'short_success': data = data[:1]
        elif issue == 'curl_error': code = 22
        elif issue == 'missing': fields.clear()
        return status, data, code
    with pytest.raises((ValueError, RuntimeError)):
        fetcher.fetch(tmp_path, True, True, _private_transport=transport([], alter=alter))
    assert target.read_bytes() == b'01' and not (tmp_path / fetcher.LOCK).exists()


@pytest.mark.parametrize('network_code',[28,35,7,52])
def test_empty_failures_bounded(tmp_path, monkeypatch,network_code):
    target, _ = setup(tmp_path, monkeypatch, b'01')
    calls = []
    def empty(*args):
        calls.append(1)
        return network_code
    with pytest.raises(RuntimeError, match='three'):
        fetcher.fetch(tmp_path, True, True, _private_transport=empty)
    assert len(calls) == 3 and target.read_bytes() == b'01'


def test_observed_duplicate_noncritical_official_headers_are_allowed(tmp_path,monkeypatch):
    target,_ = setup(tmp_path,monkeypatch)
    base = transport([])
    def repeated(*args):
        code = base(*args)
        headers = args[-1]
        data = headers.read_bytes().replace(b'\r\n\r\n',
            b'\r\nStrict-Transport-Security: max-age=31536000\r\nStrict-Transport-Security: max-age=31536000\r\nSet-Cookie: route=a\r\nSet-Cookie: other=b\r\n\r\n')
        headers.write_bytes(data)
        return code
    result = fetcher.fetch(tmp_path,True,True,_private_transport=repeated)
    assert result['download_complete'] and target.read_bytes() == b'0123456789'


def test_duplicate_range_header_is_rejected_without_append(tmp_path,monkeypatch):
    target,_ = setup(tmp_path,monkeypatch,prefix=b'01')
    base = transport([])
    def duplicate(*args):
        code = base(*args)
        headers = args[-1]
        headers.write_bytes(headers.read_bytes().replace(b'\r\n\r\n',b'\r\nContent-Range: bytes 2-5/10\r\n\r\n'))
        return code
    with pytest.raises(ValueError,match='duplicate'):
        fetcher.fetch(tmp_path,True,True,_private_transport=duplicate)
    assert target.read_bytes() == b'01'


def test_temporary_502_retries_without_appending_error_body(tmp_path,monkeypatch):
    target,_ = setup(tmp_path,monkeypatch,prefix=b'01')
    base = transport([]); calls = []
    def response(*args):
        calls.append(1)
        if len(calls)==1:
            args[-1].write_bytes(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nRetry-After: 2\r\n\r\n')
            return 22
        return base(*args)
    result = fetcher.fetch(tmp_path,True,True,_private_transport=response)
    assert result['download_complete'] and target.read_bytes() == b'0123456789'


def test_access_denied_and_long_retry_after_are_not_bypassed(tmp_path,monkeypatch):
    target,_ = setup(tmp_path,monkeypatch,prefix=b'01')
    def response(*args):
        args[-1].write_bytes(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
        return 22
    with pytest.raises(ValueError,match='206'):
        fetcher.fetch(tmp_path,True,True,_private_transport=response)
    assert target.read_bytes() == b'01'
    def defer(*args):
        args[-1].write_bytes(b'HTTP/1.1 503 Unavailable\r\nRetry-After: 120\r\n\r\n')
        return 22
    with pytest.raises(RuntimeError,match='defer'):
        fetcher.fetch(tmp_path,True,True,_private_transport=defer)
    assert target.read_bytes() == b'01'


def test_disk_and_lock(tmp_path, monkeypatch):
    target, _ = setup(tmp_path, monkeypatch, b'01')
    monkeypatch.setattr(fetcher.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='disk'):
        fetcher.fetch(tmp_path, True, True)
    monkeypatch.setattr(fetcher.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**9))
    lock = tmp_path / fetcher.LOCK
    lock.write_bytes(b'other owner')
    with pytest.raises(FileExistsError):
        fetcher.fetch(tmp_path, True, True)
    assert lock.read_bytes() == b'other owner' and target.read_bytes() == b'01'


@pytest.mark.parametrize('url', ['http://sleepdata.org/a', 'https://evil.test/a',
                                'https://a:b@sleepdata.org/a', 'https://sleepdata.org/a?q=1',
                                'https://sleepdata.org/a#x'])
def test_invalid_source(tmp_path, monkeypatch, url):
    _, plan = setup(tmp_path, monkeypatch)
    plan['archive']['url'] = url
    (tmp_path / fetcher.PLAN).write_text(json.dumps(plan))
    with pytest.raises(ValueError): fetcher.fetch(tmp_path)


def test_curl_arguments_no_shell(tmp_path, monkeypatch):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {'shell': False, 'check': False}
        return SimpleNamespace(returncode=28)
    monkeypatch.setattr(fetcher.subprocess, 'run', run)
    code = fetcher._transport('https://sleepdata.org/a', 'sleepdata.org', '"etag"',
                              '2001:db8::1', 2, 5, tmp_path/'body', tmp_path/'headers')
    assert code == 28
    cmd = commands[0]
    assert cmd[cmd.index('--range') + 1] == '2-5'
    assert cmd[cmd.index('--resolve') + 1] == 'sleepdata.org:443:[2001:db8::1]'
    assert cmd[cmd.index('--max-redirs') + 1] == '0'


def test_invalid_ip(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError): fetcher.fetch(tmp_path, resolve_ip='bad\nheader')


def test_symlink_escape(tmp_path, monkeypatch):
    target, _ = setup(tmp_path, monkeypatch)
    outside = tmp_path / 'outside'
    outside.mkdir()
    target.parent.parent.mkdir(parents=True)
    try:
        target.parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation privilege unavailable')
    with pytest.raises(ValueError, match='outside'):
        fetcher.fetch(tmp_path, True, True)
    assert not list(outside.iterdir())
