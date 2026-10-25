"""Tiny synthetic ZIPs only; no real archive or raw participant data."""
import hashlib
import json
from pathlib import Path
import stat
import struct
import warnings
import zipfile

import pytest

from scripts.datasets import sleepaccel_archive as intake


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, monkeypatch, entries=None, payload=b'abc', approved=True):
    rel = 'motion/123_acceleration.txt'
    entries = entries if entries is not None else [('sleep-accel-1.0.0/' + rel, payload)]
    archive = tmp_path / 'fixture.zip'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
            for name, data in entries:
                if isinstance(name, str):
                    info = zipfile.ZipInfo(name)
                    info.filename = name  # Preserve malicious backslashes on Windows.
                else:
                    info = name
                output.writestr(info, data)
    monkeypatch.setattr(intake, 'ARCHIVE_BYTES', archive.stat().st_size)
    monkeypatch.setattr(intake, 'EXPECTED_FILES', 1)
    monkeypatch.setattr(intake, 'LICENSE_SHA256', _hash(b'license'))
    checksums = (_hash(payload) + ' ' + rel + '\n' + _hash(b'license') + ' LICENSE.txt\n').encode()
    for name, data in [(intake.CHECKSUMS, checksums), (intake.LICENSE, b'license')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    plan = {'approved_archive': approved, 'archive': {'exact_bytes': archive.stat().st_size,
                                                  'url': 'https://example.invalid/fixture.zip'},
            'checksum_source_local_sha256': _hash(checksums), 'files': [
                {'kind': 'motion', 'filename': '123_acceleration.txt', 'relative_source_path': rel,
                 'exact_bytes': len(payload), 'official_checksums': {'sha256': _hash(payload)}}]}
    plan_path = tmp_path / intake.PLAN
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    return archive, tmp_path / 'data/raw/sleepaccel/1.0.0' / rel


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    before = _tree(tmp_path)
    result = intake.intake_archive(tmp_path, archive)
    assert result['dry_run'] and not result['complete']
    assert result['selected_files'] == 1 and result['new_files'] == 1
    assert _tree(tmp_path) == before and not target.exists()


def test_dry_run_missing_or_partial_archive(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    archive.unlink()
    result = intake.intake_archive(tmp_path, archive)
    assert result['archive']['present'] is False
    archive.write_bytes(b'partial')
    result = intake.intake_archive(tmp_path, archive)
    assert result['archive']['current_bytes'] == 7 and not target.exists()


def test_execute_reuse_and_license(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch, entries=[
        ('version/motion/123_acceleration.txt', b'abc'), ('version/LICENSE.txt', b'license'),
        ('version/steps/123_steps.txt', b'ignored')])
    result = intake.intake_archive(tmp_path, archive, execute=True)
    assert result['complete'] and target.read_bytes() == b'abc'
    assert result['license']['archive_member'] == 'version/LICENSE.txt'
    assert len(result['unselected_extras']) == 1
    assert not (tmp_path / 'data/raw/sleepaccel/1.0.0/steps').exists()
    old_time = target.stat().st_mtime_ns
    result = intake.intake_archive(tmp_path, archive, execute=True)
    assert result['reused_files'] == 1 and result['new_files'] == 0
    assert target.stat().st_mtime_ns == old_time
    assert json.loads((tmp_path / intake.RECEIPT).read_text())['complete']


@pytest.mark.parametrize('bad', ['../evil', '/evil', r'version\evil', 'C:/evil', 'a/../evil',
                                 'a//evil', 'a/./evil', 'a/NUL.txt'])
def test_unsafe_even_unselected(tmp_path, monkeypatch, bad):
    archive, target = _fixture(tmp_path, monkeypatch, entries=[
        ('version/motion/123_acceleration.txt', b'abc'), (bad, b'x')])
    with pytest.raises(ValueError):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert not target.exists() and not json.loads((tmp_path / intake.RECEIPT).read_text())['complete']


@pytest.mark.parametrize('entries', [
    [('version/motion/123_acceleration.txt', b'abc'), ('version/motion/123_acceleration.txt', b'abc')],
    [('one/motion/123_acceleration.txt', b'abc'), ('two/motion/123_acceleration.txt', b'abc')],
    [('123_acceleration.txt', b'abc')],
    [('too/deep/motion/123_acceleration.txt', b'abc')],
    [('version/motion/123_acceleration.txt', b'abcd')],
    [('version/motion/123_acceleration.txt', b'xyz')],
])
def test_invalid_mapping_size_hash(tmp_path, monkeypatch, entries):
    archive, target = _fixture(tmp_path, monkeypatch, entries=entries)
    with pytest.raises(ValueError):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert not target.exists() and not json.loads((tmp_path / intake.RECEIPT).read_text())['complete']
    assert not list(tmp_path.rglob('.intake-*'))


def test_symlink_rejected(tmp_path, monkeypatch):
    info = zipfile.ZipInfo('link')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive, target = _fixture(tmp_path, monkeypatch, entries=[
        ('version/motion/123_acceleration.txt', b'abc'), (info, b'outside')])
    with pytest.raises(ValueError, match='symlink'):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert not target.exists()


def test_existing_conflict_no_overwrite(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'bad')
    with pytest.raises(ValueError, match='conflicts'):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert target.read_bytes() == b'bad' and not json.loads((tmp_path / intake.RECEIPT).read_text())['complete']


def test_crc_failure_even_reused(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'abc')
    data = bytearray(archive.read_bytes())
    filename_len, extra_len = struct.unpack_from('<HH', data, 26)
    data[30 + filename_len + extra_len] ^= 1
    archive.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile, match='CRC'):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert target.read_bytes() == b'abc' and not json.loads((tmp_path / intake.RECEIPT).read_text())['complete']


def test_approval_and_exact_archive_size(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch, approved=False)
    with pytest.raises(ValueError, match='approved'):
        intake.intake_archive(tmp_path, archive, execute=True)
    monkeypatch.setattr(intake, 'ARCHIVE_BYTES', archive.stat().st_size + 1)
    with pytest.raises(ValueError, match='budget'):
        intake.intake_archive(tmp_path, archive)
    assert not target.exists()


def test_evidence_changed(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    (tmp_path / intake.CHECKSUMS).write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert not target.exists()


def test_destination_symlink_escape(tmp_path, monkeypatch):
    archive, target = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / 'outside'
    outside.mkdir()
    target.parent.parent.mkdir(parents=True)
    try:
        target.parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation privilege unavailable')
    with pytest.raises(ValueError, match='outside'):
        intake.intake_archive(tmp_path, archive, execute=True)
    assert not list(outside.iterdir())


@pytest.mark.parametrize('mutation', ['archive', 'raw', 'source'])
def test_previous_success_invalidated_on_failure(tmp_path, monkeypatch, mutation):
    archive, target = _fixture(tmp_path, monkeypatch)
    intake.intake_archive(tmp_path, archive, execute=True)
    previous = (tmp_path / intake.RECEIPT).read_bytes()
    if mutation == 'archive':
        data = bytearray(archive.read_bytes())
        filename_len, extra_len = struct.unpack_from('<HH', data, 26)
        data[30 + filename_len + extra_len] ^= 1
        archive.write_bytes(data)
    elif mutation == 'raw':
        target.write_bytes(b'bad')
    else:
        (tmp_path / intake.CHECKSUMS).write_bytes(b'changed')
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        intake.intake_archive(tmp_path, archive, execute=True)
    receipt = json.loads((tmp_path / intake.RECEIPT).read_text())
    assert not receipt['complete'] and receipt['previous_receipt_sha256'] == _hash(previous)
