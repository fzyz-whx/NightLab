"""Sequential bounded-range fetch of the explicitly approved local SleepAccel ZIP."""
import argparse
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

from .cli import ROOT, raw_path
from .sleepaccel_archive import ARCHIVE, PLAN, sha256, _write_receipt

ARCHIVE_BYTES = 576850847
CHUNK_BYTES = 16 * 1024 * 1024
BLOCK = 1024 * 1024
LOCK = 'data/raw/sleepaccel/archives/fetch.lock'
PROGRESS = 'data/raw/sleepaccel/archives/fetch-progress.json'
NETWORK_ERRORS = {18, 28, 56}
EMPTY_RETRY_ERRORS = NETWORK_ERRORS | {5, 6, 7, 35, 52, 55, 92}


def _source(plan, resolve_ip):
    url = plan['archive']['url']
    if not isinstance(url, str) or any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError('source URL contains whitespace or control characters')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in
            {'physionet.org', 'www.physionet.org', 'sleepdata.org'}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.port not in (None, 443)):
        raise ValueError('source must be an approved plain official HTTPS URL')
    etag = plan['archive'].get('effective_source_evidence', {}).get('etag', plan['archive'].get('etag'))
    if not isinstance(etag, str) or not re.fullmatch(r'"[^"\r\n]+"', etag):
        raise ValueError('audited strong ETag required')
    address = str(ipaddress.ip_address(resolve_ip)) if resolve_ip is not None else None
    return url, parsed.hostname, etag, address


def _transport(url, host, etag, address, start, end, body, headers):
    command = ['curl.exe', '--silent', '--show-error', '--proto', '=https', '--http1.1',
               '--tlsv1.2','--tls-max','1.2','--location',
               '--max-redirs', '0', '--fail', '--range', f'{start}-{end}',
               '--max-filesize', str(end - start + 1), '--max-time', '180',
               '--connect-timeout', '15', '--speed-time', '60', '--speed-limit', '1024',
               '--header', 'If-Range: ' + etag, '--header', 'Accept-Encoding: identity',
               '--output', str(body), '--dump-header', str(headers)]
    if address:
        if ':' in address:
            address = '[' + address + ']'
        command += ['--noproxy', '*', '--resolve', f'{host}:443:{address}']
    command.append(url)
    return subprocess.run(command, shell=False, check=False).returncode


def _validate_headers(path, start, end, etag):
    text = path.read_text(encoding='iso-8859-1')
    blocks = re.split(r'\r?\n\r?\n', text)
    responses = [block for block in blocks if block.startswith('HTTP/')]
    if not responses:
        raise ValueError('missing HTTP response headers')
    lines = responses[-1].splitlines()
    if not re.fullmatch(r'HTTP/\S+ 206(?: .*)?', lines[0]):
        raise ValueError('final HTTP response must be 206')
    fields = {}
    critical = {'content-range','content-length','content-type','etag','content-encoding','transfer-encoding'}
    for line in lines[1:]:
        if ':' not in line:
            raise ValueError('invalid HTTP response header')
        key, value = line.split(':', 1)
        key = key.strip().lower()
        if key not in critical:
            continue  # e.g. the observed official response repeats HSTS/cookie headers
        if key in fields:
            raise ValueError('duplicate HTTP response header')
        fields[key] = value.strip()
    if fields.get('content-range') != f'bytes {start}-{end}/{ARCHIVE_BYTES}':
        raise ValueError('Content-Range differs from requested range')
    if fields.get('content-length') != str(end - start + 1):
        raise ValueError('Content-Length differs from requested range')
    if fields.get('content-type', '').split(';')[0].strip().lower() != 'application/zip':
        raise ValueError('response is not application/zip')
    if fields.get('etag') != etag:
        raise ValueError('response ETag differs from audited source')
    if (fields.get('content-encoding', 'identity').lower() != 'identity'
            or fields.get('transfer-encoding', 'identity').lower() != 'identity'):
        raise ValueError('response must use identity transfer')


def _transient_http_delay(path):
    """Only retry observed temporary server failures, respecting Retry-After."""
    if not path.exists():
        return None
    blocks = [part for part in re.split(r'\r?\n\r?\n',path.read_text(encoding='iso-8859-1')) if part.startswith('HTTP/')]
    if not blocks:
        return None
    lines = blocks[-1].splitlines()
    status = re.fullmatch(r'HTTP/\S+ (\d{3})(?: .*)?',lines[0])
    if not status or int(status[1]) not in (502,503,504):
        return None
    values = [line.split(':',1)[1].strip() for line in lines[1:] if line.lower().startswith('retry-after:')]
    if len(values)>1:
        raise ValueError('ambiguous server Retry-After')
    if not values:
        return 3.0
    try:
        delay = float(int(values[0])) if values[0].isdigit() else (parsedate_to_datetime(values[0])-datetime.now(timezone.utc)).total_seconds()
    except (ValueError,TypeError) as error:
        raise ValueError('invalid server Retry-After; prefix preserved') from error
    if delay>60:
        raise RuntimeError(f'server requested Retry-After {delay:.0f}s; prefix preserved, defer this fetch')
    return max(0.0,delay)


def fetch(root, execute=False, confirmed_large=False, resolve_ip=None, *, _private_transport=None):
    """Dry-run is offline and write-free; execute preserves every admitted prefix.

    The lock coordinates only instances of this helper. The owner must stop any
    independently launched downloader before execution.
    """
    root = Path(root).resolve()
    plan_bytes = (root / PLAN).read_bytes()
    plan = json.loads(plan_bytes)
    if plan['archive']['exact_bytes'] != ARCHIVE_BYTES:
        raise ValueError('archive size differs from the fixed approved budget')
    url, host, etag, address = _source(plan, resolve_ip)
    if (root / ARCHIVE).is_symlink():
        raise ValueError('partial archive must not be a file symlink')
    target = raw_path(root, ARCHIVE)
    offset = target.stat().st_size if target.is_file() else 0
    if offset > ARCHIVE_BYTES or (target.exists() and not target.is_file()):
        raise ValueError('invalid existing partial archive')
    probe = target.parent
    while not probe.exists():
        probe = probe.parent
    required = ARCHIVE_BYTES - offset + CHUNK_BYTES + BLOCK
    result = {'source': url, 'etag': etag, 'offset': offset, 'total_bytes': ARCHIVE_BYTES,
              'remaining_bytes': ARCHIVE_BYTES - offset, 'required_free_bytes': required,
              'download_complete': False, 'member_crc_sha_verified': False,
              'dry_run': not execute, 'source_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
              'fetch_source_sha256': sha256(Path(__file__))}
    result['transport_policy'] = 'sequential16MiB; HTTP1.1/TLS1.2 certificate verification; request-only optional DNS pin; no global proxy/DNS changes'
    if not execute:
        return result
    if plan.get('approved_archive') is not True or plan.get('selected_option') != 'all31' or not confirmed_large:
        raise ValueError('approved all31 plan and explicit large-download confirmation required')
    if shutil.disk_usage(probe).free < required:
        raise ValueError('insufficient disk space for remaining archive and temporary chunk')
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = raw_path(root, LOCK)
    # O_EXCL prevents helper concurrency; stale locks need owner inspection.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, 'w') as output:
            output.write(str(os.getpid()))
        if (target.stat().st_size if target.is_file() else 0) != offset:
            raise ValueError('partial changed before lock acquisition')
        digest = hashlib.sha256()
        if target.exists():
            with target.open('rb') as source:
                for data in iter(lambda: source.read(BLOCK), b''):
                    digest.update(data)
        progress = raw_path(root, PROGRESS)
        result['partial_sha256_local'] = digest.hexdigest()
        _write_receipt(progress, result)
        transport = _private_transport or _transport
        failures = 0
        while offset < ARCHIVE_BYTES:
            end = min(offset + CHUNK_BYTES, ARCHIVE_BYTES) - 1
            with tempfile.TemporaryDirectory(prefix='.fetch-', dir=target.parent) as temporary:
                body, headers = Path(temporary) / 'body', Path(temporary) / 'headers'
                code = transport(url, host, etag, address, offset, end, body, headers)
                size = body.stat().st_size if body.exists() else 0
                retry_delay = _transient_http_delay(headers) if code==22 and not size else None
                if retry_delay is not None:
                    failures += 1
                    if failures>=3:
                        raise RuntimeError('three consecutive temporary HTTP server failures; prefix preserved')
                    print('SleepAccel archive: temporary server error; bounded retry',file=sys.stderr,flush=True)
                    time.sleep(retry_delay)
                    continue
                if not size and code in EMPTY_RETRY_ERRORS:
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError('three consecutive empty network failures; prefix preserved')
                    time.sleep(min(failures, 3))
                    continue
                _validate_headers(headers, offset, end, etag)
                if size <= 0 or size > end - offset + 1:
                    raise ValueError('empty or overlong range response')
                if code != 0 and code not in NETWORK_ERRORS:
                    raise RuntimeError(f'curl failed with status {code}; prefix preserved')
                if code == 0 and size != end - offset + 1:
                    raise ValueError('successful response shorter than requested range')
                if raw_path(root, ARCHIVE) != target or (target.stat().st_size if target.exists() else 0) != offset:
                    raise ValueError('partial changed during fetch')
                with target.open('ab') as output, body.open('rb') as source:
                    for data in iter(lambda: source.read(BLOCK), b''):
                        output.write(data)
                        digest.update(data)
                    output.flush()
                    os.fsync(output.fileno())
                offset += size
                failures = 0
                result.update(offset=offset, remaining_bytes=ARCHIVE_BYTES - offset,
                              partial_sha256_local=digest.hexdigest(),
                              download_complete=offset == ARCHIVE_BYTES)
                _write_receipt(progress, result)
                print(f'SleepAccel archive: {offset}/{ARCHIVE_BYTES} bytes', file=sys.stderr, flush=True)
        result['download_complete'] = offset == ARCHIVE_BYTES
        _write_receipt(progress, result)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-large-download', action='store_true')
    parser.add_argument('--resolve-ip')
    args = parser.parse_args()
    print(json.dumps(fetch(ROOT, args.execute, args.confirm_large_download, args.resolve_ip), indent=2))


if __name__ == '__main__':
    main()
