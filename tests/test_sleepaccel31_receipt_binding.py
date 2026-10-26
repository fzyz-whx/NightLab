"""Real intake/binding flow with 93 tiny synthetic files, never participant data."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.datasets import bind_sleepaccel31_receipt as binding
from scripts.datasets import sleepaccel_archive as intake


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    manifest = json.loads((Path(binding.ROOT) / binding.MANIFEST).read_text(encoding='utf-8'))
    plan = json.loads((Path(binding.ROOT) / intake.PLAN).read_text(encoding='utf-8'))
    archive_path = tmp_path / 'actual-local-archive.zip'
    checksums = []
    with zipfile.ZipFile(archive_path, 'w') as archive:
        for entry, planned in zip(manifest['files'], plan['files']):
            assert entry['path'].endswith(planned['relative_source_path'])
            payload = ('synthetic:' + planned['relative_source_path']).encode()
            for item in (entry, planned):
                item['exact_bytes'] = len(payload)
                item['official_checksums'] = {'sha256': digest(payload)}
            checksums.append(digest(payload) + ' ' + planned['relative_source_path'])
            archive.writestr('release/' + planned['relative_source_path'], payload)
        archive.writestr('release/LICENSE.txt', b'synthetic license')
    license_sha = digest(b'synthetic license')
    checksums.append(license_sha + ' LICENSE.txt')
    checksum_path = tmp_path / intake.CHECKSUMS
    checksum_path.parent.mkdir(parents=True)
    checksum_path.write_text('\n'.join(checksums) + '\n', encoding='utf-8')
    plan['checksum_source_local_sha256'] = intake.sha256(checksum_path)
    plan['archive']['exact_bytes'] = archive_path.stat().st_size
    plan_path = tmp_path / intake.PLAN
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    acquisition = manifest['acquisition']
    acquisition.update(source_plan_sha256=intake.sha256(plan_path),
                       checksum_source_local_sha256=intake.sha256(checksum_path),
                       archive_exact_bytes=archive_path.stat().st_size,
                       archive_sha256_local=intake.sha256(archive_path))
    manifest_path = tmp_path / binding.MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    monkeypatch.setattr(intake, 'ARCHIVE_BYTES', archive_path.stat().st_size)
    monkeypatch.setattr(intake, 'LICENSE_SHA256', license_sha)
    monkeypatch.setattr(binding, 'ARCHIVE_SHA256', intake.sha256(archive_path))
    monkeypatch.setattr(binding, 'FROZEN_METADATA_SHA256', binding.metadata_digest(manifest))
    intake.intake_archive(tmp_path, archive_path, execute=True)
    return tmp_path


def receipt(root, mutate):
    path = root / intake.RECEIPT
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_default_validation_writes_nothing(fixture):
    before = tree(fixture)
    report = binding.bind_receipt(fixture)
    assert report['status'] == 'would bind' and report['verified_raw_files'] == 93
    assert report['dry_run'] and not report['changed']
    assert tree(fixture) == before


def test_explicit_binding_changes_only_receipt_field_and_is_auditable(fixture):
    path = fixture / binding.MANIFEST
    before = json.loads(path.read_text())
    report = binding.bind_receipt(fixture, execute=True)
    after = json.loads(path.read_text())
    expected = copy.deepcopy(before)
    expected['acquisition']['intake_receipt_sha256'] = intake.sha256(fixture / intake.RECEIPT)
    assert after == expected and report['changed']
    evidence = json.loads((fixture / binding.PROVENANCE).read_text())
    assert evidence['previous_receipt_sha256'] == before['acquisition']['intake_receipt_sha256']
    assert evidence['public_receipt_sha256'] == binding.PUBLIC_RECEIPT_SHA256
    assert str(fixture) not in json.dumps(evidence)
    # Existing frozen auditor now accepts the actual locally generated receipt.
    from scripts.datasets.audit_sleepaccel31 import metadata
    assert len(metadata(fixture)[1]) == 31


def test_missing_receipt_has_helpful_cli_error_without_traceback(fixture, monkeypatch, capsys):
    (fixture / intake.RECEIPT).unlink()
    monkeypatch.setattr(binding, 'bind_receipt', lambda **kwargs: binding_original(fixture, **kwargs))
    assert binding.main([]) == 2
    error = capsys.readouterr().err
    assert 'local intake receipt missing' in error and '--execute' in error and 'Traceback' not in error


binding_original = binding.bind_receipt


def test_incomplete_receipt_refused(fixture):
    receipt(fixture, lambda row: row.update(complete=False))
    before = tree(fixture)
    with pytest.raises(ValueError, match='incomplete'):
        binding.bind_receipt(fixture, execute=True)
    assert tree(fixture) == before


def test_raw_tamper_refused(fixture):
    path = next((fixture / 'data/raw/sleepaccel/1.0.0').rglob('*.txt'))
    data = path.read_bytes()
    path.write_bytes(b'X' + data[1:])
    with pytest.raises(ValueError, match='actual raw'):
        binding.bind_receipt(fixture, execute=True)


def test_altered_split_refused(fixture):
    path = fixture / binding.MANIFEST
    manifest = json.loads(path.read_text())
    split = manifest['subjects']['subjects']
    split['train'][0], split['test'][0] = split['test'][0], split['train'][0]
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen cohort metadata'):
        binding.bind_receipt(fixture, execute=True)


def test_unsafe_receipt_member_refused(fixture):
    receipt(fixture, lambda value: value['files'][0].update(archive_member='../motion/file.txt'))
    with pytest.raises(ValueError, match='unsafe ZIP path'):
        binding.bind_receipt(fixture, execute=True)


def test_existing_outputs_guard(fixture):
    for rel in ('data/processed/sleepaccel-cohort-31', 'models/custom-name', 'runs/custom-name'):
        path = fixture / rel
        path.mkdir(parents=True)
        sentinel = path / 'existing.bin'
        sentinel.write_bytes(b'frozen')
        before = tree(fixture)
        with pytest.raises(ValueError, match='existing derived outputs'):
            binding.bind_receipt(fixture, execute=True)
        assert tree(fixture) == before
        sentinel.unlink()


def test_equal_receipt_safe_nochange_with_existing_cache(fixture):
    binding.bind_receipt(fixture, execute=True)
    cache = fixture / 'data/processed/sleepaccel-cohort-31/corpus.json'
    cache.parent.mkdir(parents=True)
    cache.write_text('frozen')
    before = tree(fixture)
    report = binding.bind_receipt(fixture, execute=True)
    assert not report['changed'] and report['status'] == 'already bound; no change'
    assert tree(fixture) == before


def test_duplicate_or_mismatched_receipt_target_refused(fixture):
    receipt(fixture, lambda value: value['files'].__setitem__(1, value['files'][0].copy()))
    with pytest.raises(ValueError, match='duplicate receipt target'):
        binding.bind_receipt(fixture, execute=True)


def test_receipt_json_object_required(fixture):
    (fixture / intake.RECEIPT).write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='JSON must be an object'):
        binding.bind_receipt(fixture, execute=True)


def test_manifest_staging_failure_preserves_original_and_no_provenance(fixture, monkeypatch):
    before = tree(fixture)
    def failed_stage(*_args, **_kwargs):
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(binding.tempfile, 'mkstemp', failed_stage)
    with pytest.raises(OSError, match='synthetic disk failure'):
        binding.bind_receipt(fixture, execute=True)
    assert tree(fixture) == before
    assert not (fixture / binding.PROVENANCE).exists()
