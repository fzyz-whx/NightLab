"""Offline, explicitly approved SleepAccel ZIP intake (one worker, no network)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile

from .cli import ROOT, raw_path

ARCHIVE_BYTES = 576850847
EXPECTED_FILES = 93
PLAN = 'data/plans/sleepaccel-expansion-options.json'
ARCHIVE = 'data/raw/sleepaccel/archives/sleep-accel-1.0.0.zip.part'
RECEIPT = 'data/raw/sleepaccel/archives/intake-receipt.json'
CHECKSUMS = 'data/raw/environment/sleepaccel-SHA256SUMS.txt'
LICENSE = 'data/raw/environment/NSRR-sleepaccel-LICENSE.txt'
LICENSE_SHA256 = '86c0ad300f6d380591298bc03652e30b81f65954bbf22435812bcfd46812ab06'
CHUNK = 1024 * 1024
SUFFIXES = {'heart_rate': '_heartrate.txt', 'motion': '_acceleration.txt',
            'labels': '_labeled_sleep.txt'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(name):
    """Validate even unselected entries, before constructing local paths."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('unsafe ZIP path')
    parts = name.rstrip('/').split('/')
    if any(part in ('', '.', '..') or part.endswith((' ', '.')) for part in parts):
        raise ValueError('unsafe ZIP path')
    if PurePosixPath(name).is_absolute():
        raise ValueError('absolute ZIP path')
    # Windows device names are not portable file names.
    if any(re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', p) for p in parts):
        raise ValueError('Windows device ZIP path')
    return tuple(parts)


def _load_plan(root):
    plan_path = root / PLAN
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    checksums_path = raw_path(root, CHECKSUMS)
    if sha256(checksums_path) != plan['checksum_source_local_sha256']:
        raise ValueError('official checksum evidence changed')
    official = {}
    for line in checksums_path.read_text(encoding='utf-8').splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if name in official or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid official checksum list')
        official[name] = digest
    expected = {}
    for entry in plan['files']:
        rel = entry['relative_source_path']
        parts = _portable(rel)
        kind, filename = entry['kind'], entry['filename']
        if (kind not in SUFFIXES or parts != (kind, filename)
                or not re.fullmatch(r'\d+' + re.escape(SUFFIXES[kind]), filename)):
            raise ValueError('unapproved file kind or filename')
        if (rel in expected or entry['exact_bytes'] <= 0
                or official.get(rel) != entry['official_checksums']['sha256']):
            raise ValueError('duplicate file or official checksum mismatch in plan')
        expected[rel] = entry
    if len(expected) != EXPECTED_FILES:
        raise ValueError('unexpected planned member count')
    return plan, expected, official, hashlib.sha256(plan_bytes).hexdigest()


def _inventory(archive, expected):
    selected, extras, licenses, prefixes, seen = {}, [], [], set(), set()
    infos = archive.infolist()
    for info in infos:
        _portable(info.orig_filename)  # ZipInfo normalizes backslashes/NUL on Windows.
        parts = _portable(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('ZIP symlink rejected')
        if info.filename in seen:
            raise ValueError('duplicate ZIP member')
        seen.add(info.filename)
        if info.is_dir():
            continue
        # Only direct kind/name or one common wrapper directory is allowed.
        rel = '/'.join(parts[-2:])
        if rel in expected:
            if len(parts) not in (2, 3):
                raise ValueError('unexpected ZIP nesting')
            prefix = parts[:-2]
            prefixes.add(prefix)
            if rel in selected:
                raise ValueError('duplicate selected member mapping')
            if info.file_size != expected[rel]['exact_bytes']:
                raise ValueError('ZIP member size mismatch')
            selected[rel] = info
        elif parts[-1] == 'LICENSE.txt':
            licenses.append(info)
        else:
            extras.append({'name': info.filename, 'bytes': info.file_size})
    if set(selected) != set(expected) or len(prefixes) != 1:
        raise ValueError('missing members or inconsistent ZIP wrapper')
    if len(licenses) > 1:
        raise ValueError('ambiguous ZIP license')
    return selected, extras, licenses, len(infos), sum(i.is_dir() for i in infos)


def _consume(archive, info, expected_sha, destination=None):
    digest, count = hashlib.sha256(), 0
    with archive.open(info) as source:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            digest.update(chunk)
            count += len(chunk)
            if count > info.file_size:
                raise ValueError('ZIP member exceeds declared size')
            if destination is not None:
                destination.write(chunk)
    if count != info.file_size or digest.hexdigest() != expected_sha:
        raise ValueError('ZIP member SHA256 mismatch')
    return digest.hexdigest()


def _write_receipt(receipt_path, result):
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.receipt-', dir=receipt_path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, receipt_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def intake_archive(root, archive_path=None, execute=False):
    """Return a metadata-only budget; verify and write only with --execute.

    Caller must stop the downloading process before invoking this function.
    The archive is never renamed, removed, or modified here.
    """
    started = time.monotonic()
    root = Path(root).resolve()
    receipt_path = raw_path(root, RECEIPT)
    intake_source_sha = sha256(Path(__file__))
    if execute:
        # Invalidate an old success before any validation can fail. This attempt
        # remains incomplete if interrupted, while verified raw files can be reused.
        previous_sha = sha256(receipt_path) if receipt_path.is_file() else None
        _write_receipt(receipt_path, {'schema_version': 1, 'complete': False,
                                    'dry_run': False, 'status': 'attempt started; not complete',
                                    'previous_receipt_sha256': previous_sha,
                                    'intake_source_sha256': intake_source_sha})
    plan, expected, official, plan_sha = _load_plan(root)
    if execute and plan.get('approved_archive') is not True:
        raise ValueError('archive intake not explicitly approved')
    if plan['archive']['exact_bytes'] != ARCHIVE_BYTES:
        raise ValueError('archive budget differs from fixed approved size')
    path = raw_path(root, ARCHIVE) if archive_path is None else Path(archive_path)
    if not execute:
        present = []
        for rel, entry in expected.items():
            target = raw_path(root, 'data/raw/sleepaccel/1.0.0/' + rel)
            if target.is_file():
                present.append({'relative_source_path': rel, 'bytes': target.stat().st_size,
                                'expected_bytes': entry['exact_bytes']})
        same_size = {row['relative_source_path'] for row in present
                     if row['bytes'] == row['expected_bytes']}
        additional = sum(entry['exact_bytes'] for rel, entry in expected.items() if rel not in same_size)
        return {'schema_version': 1, 'complete': False, 'dry_run': True,
                'approved_archive': plan.get('approved_archive') is True,
                'archive': {'path': str(path), 'present': path.is_file(),
                            'current_bytes': path.stat().st_size if path.is_file() else None,
                            'approved_exact_bytes': ARCHIVE_BYTES},
                'source_plan_sha256': plan_sha, 'selected_files': len(expected),
                'new_files': len(expected) - len(same_size), 'reused_files': None,
                'local_same_size_candidates': len(same_size), 'local_files': present,
                'additional_raw_bytes_estimate': additional,
                'required_free_bytes_estimate': additional + CHUNK,
                'verification_status': 'metadata only; existing files require execute SHA256 verification'}
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('archive is incomplete or size differs from approval')
    archive_sha = sha256(path)
    targets, reused, additional = {}, [], 0
    for rel, entry in expected.items():
        target = raw_path(root, 'data/raw/sleepaccel/1.0.0/' + rel)
        targets[rel] = target
        if target.exists():
            if (not target.is_file() or target.stat().st_size != entry['exact_bytes']
                    or sha256(target) != entry['official_checksums']['sha256']):
                raise ValueError('existing raw file conflicts; refusing overwrite: ' + rel)
            reused.append(rel)
        else:
            additional += entry['exact_bytes']
    probe = root
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < additional + CHUNK:
        raise ValueError('insufficient free disk for new raw files and receipt')
    rows = []
    with zipfile.ZipFile(path) as archive:
        selected, extras, licenses, member_count, directory_count = _inventory(archive, expected)
        license_evidence = {'archive_member': None, 'status': 'verified external official evidence',
                            'sha256': LICENSE_SHA256}
        if official.get('LICENSE.txt') != LICENSE_SHA256:
            raise ValueError('official license checksum differs from pinned evidence')
        if licenses:
            _consume(archive, licenses[0], official['LICENSE.txt'])
            license_evidence.update(archive_member=licenses[0].filename, status='verified archive license')
        elif sha256(raw_path(root, LICENSE)) != LICENSE_SHA256:
            raise ValueError('external license evidence mismatch')
        for rel, entry in expected.items():
            target = targets[rel]
            digest = entry['official_checksums']['sha256']
            if execute and rel not in reused:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Re-resolve after mkdir to reject any pre-existing escaping symlink.
                if raw_path(root, 'data/raw/sleepaccel/1.0.0/' + rel) != target:
                    raise ValueError('destination changed during intake')
                descriptor, temporary = tempfile.mkstemp(prefix='.intake-', dir=target.parent)
                try:
                    with os.fdopen(descriptor, 'wb') as output:
                        _consume(archive, selected[rel], digest, output)
                        output.flush()
                        os.fsync(output.fileno())
                    # Hard link publication is atomic and fails if target exists.
                    os.link(temporary, target)
                finally:
                    os.unlink(temporary)
            else:
                _consume(archive, selected[rel], digest)
            rows.append({'path': str(target.relative_to(root)).replace('\\', '/'),
                         'archive_member': selected[rel].filename, 'bytes': entry['exact_bytes'],
                         'sha256': digest, 'reused': rel in reused})
            if len(rows) % 10 == 0 or len(rows) == len(expected):
                print(f'SleepAccel intake: verified {len(rows)}/{len(expected)} selected members',
                      file=sys.stderr, flush=True)
    if path.stat().st_size != ARCHIVE_BYTES or sha256(path) != archive_sha:
        raise ValueError('archive changed while intake was running')
    result = {'schema_version': 1, 'complete': bool(execute), 'dry_run': not execute,
              'archive': {'path': str(path), 'bytes': ARCHIVE_BYTES,
                          'sha256_local': archive_sha, 'sha256_status': 'local, not official',
                          'official_source_url': plan['archive']['url']},
              'intake_source_sha256': intake_source_sha,
              'elapsed_seconds': time.monotonic() - started,
              'source_plan_sha256': plan_sha, 'checksum_source_local_sha256': plan['checksum_source_local_sha256'],
              'selected_files': len(rows), 'new_files': len(rows) - len(reused),
              'reused_files': len(reused), 'additional_raw_bytes': additional,
              'zip_member_count': member_count, 'zip_directory_count': directory_count,
              'unselected_extras': extras, 'license': license_evidence, 'files': rows}
    if execute:
        _write_receipt(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = intake_archive(ROOT, execute=args.execute)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
