"""Bind a verified fresh-machine receipt before cohort preparation (dry by default).

One sequential worker; streaming SHA256 only, no numerical runtime or GPU.
The sole permitted manifest edit is acquisition.intake_receipt_sha256.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from .cli import ROOT, load_manifest, raw_path
from . import sleepaccel_archive as intake

MANIFEST = 'data/manifests/sleepaccel-cohort-31.json'
PROVENANCE = 'private/sleepaccel31-receipt-binding.json'
# Canonical JSON hash of the public manifest with only the receipt digest masked.
# This freezes IDs, split, labels, file hashes, preparation version and all metadata.
FROZEN_METADATA_SHA256 = 'e8292877b3f8ebb5042278ed0f09c26e05502f9c8bb631655fdfd15ef84c0053'
PUBLIC_RECEIPT_SHA256 = '0be9ff8c5f313d3cd2e53f9c3548cbe5f18633e7f9c9b8ebdc794cb271569489'
ARCHIVE_SHA256 = 'f0c1345070823a21e2bcc23a0a2f0cd82b85add5fc7797713dd7ab64a224ff8a'


def metadata_digest(manifest):
    value = copy.deepcopy(manifest)
    value['acquisition']['intake_receipt_sha256'] = '<local-receipt>'
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derived_outputs(root):
    """Conservative guard covering default cohort caches and any local models/runs.

    Runs can use caller-selected names, so any populated models/runs tree blocks
    rebinding. This deliberately restricts this operation to fresh checkouts.
    """
    found = []
    for rel in ('data/processed/sleepaccel-cohort-31', 'models', 'runs'):
        path = root / rel
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            found.append(rel)
    return found


def bind_receipt(root=ROOT, execute=False):
    root = Path(root).resolve()
    manifest_path = root / MANIFEST
    require(not manifest_path.is_symlink() and manifest_path.resolve().is_relative_to(root),
            'manifest must be a local regular repository file')
    original = manifest_path.read_bytes()
    manifest = load_manifest(manifest_path, root)
    require(metadata_digest(manifest) == FROZEN_METADATA_SHA256,
            'frozen cohort metadata changed; only the receipt digest may differ')
    acquisition = manifest['acquisition']
    require(re.fullmatch('[0-9a-f]{64}', acquisition['intake_receipt_sha256']) is not None,
            'invalid current receipt digest')
    plan, expected, _, plan_sha = intake._load_plan(root)
    require(plan_sha == acquisition['source_plan_sha256'] and plan.get('approved_archive') is True,
            'official source plan differs from approved frozen plan')
    require(plan['archive']['exact_bytes'] == acquisition['archive_exact_bytes'] == intake.ARCHIVE_BYTES,
            'approved archive size differs')
    require(manifest['subjects']['subjects'] == plan['options']['all31']['proposed_split']['subjects'],
            'fixed subject split differs')
    receipt_path = raw_path(root, acquisition['intake_receipt'])
    require(receipt_path.is_file(), 'local intake receipt missing; run sleepaccel_archive --execute '
            'on the actual approved archive first; do not copy an old receipt')
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(isinstance(receipt, dict), 'local intake receipt JSON must be an object')
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    require(receipt.get('schema_version') == 1 and receipt.get('complete') is True
            and receipt.get('dry_run') is False, 'local intake receipt is incomplete or dry')
    required = {'archive', 'intake_source_sha256', 'elapsed_seconds', 'source_plan_sha256',
                'checksum_source_local_sha256', 'selected_files', 'new_files', 'reused_files',
                'additional_raw_bytes', 'zip_member_count', 'zip_directory_count',
                'unselected_extras', 'license', 'files'}
    require(required <= receipt.keys(), 'local intake receipt schema is incomplete')
    require(isinstance(receipt['elapsed_seconds'], (int, float))
            and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] >= 0
            and type(receipt['zip_member_count']) is int and receipt['zip_member_count'] >= 93
            and type(receipt['zip_directory_count']) is int
            and 0 <= receipt['zip_directory_count'] <= receipt['zip_member_count'] - 93
            and isinstance(receipt['unselected_extras'], list), 'invalid intake receipt schema values')
    require(receipt['license']['sha256'] == intake.LICENSE_SHA256
            and receipt['license']['status'] in ('verified archive license', 'verified external official evidence'),
            'receipt license evidence differs')
    require(receipt['source_plan_sha256'] == plan_sha
            and receipt['checksum_source_local_sha256'] == plan['checksum_source_local_sha256']
            == acquisition['checksum_source_local_sha256'], 'receipt source evidence differs')
    require(receipt['intake_source_sha256'] == intake.sha256(Path(intake.__file__)),
            'receipt was not generated by the current frozen intake source')
    archive = receipt['archive']
    require(archive['bytes'] == intake.ARCHIVE_BYTES
            and archive['sha256_local'] == acquisition['archive_sha256_local'] == ARCHIVE_SHA256
            and archive['official_source_url'] == plan['archive']['url'],
            'receipt archive identity differs from the approved archive')
    require(isinstance(archive['path'], str) and bool(archive['path']), 'missing actual intake archive path')
    require(receipt['selected_files'] == 93 and len(receipt['files']) == 93,
            'receipt must contain exactly 93 target files')
    admitted = {}
    prefixes = set()
    reused = 0
    additional = 0
    for row in receipt['files']:
        target = raw_path(root, row['path'])
        require(row['path'] not in admitted, 'duplicate receipt target path')
        parts = intake._portable(row['archive_member'])
        rel = '/'.join(parts[-2:])
        require(len(parts) in (2, 3) and rel in expected
                and row['path'] == 'data/raw/sleepaccel/1.0.0/' + rel,
                'receipt archive-member mapping differs')
        prefixes.add(parts[:-2])
        entry = expected[rel]
        require(type(row['reused']) is bool, 'invalid reused flag')
        reused += int(row['reused'])
        additional += 0 if row['reused'] else row['bytes']
        require(row['bytes'] == entry['exact_bytes']
                and row['sha256'] == entry['official_checksums']['sha256'],
                'receipt target size or official SHA256 differs')
        require(target.is_file() and target.stat().st_size == row['bytes']
                and intake.sha256(target) == row['sha256'], 'actual raw size or SHA256 differs')
        admitted[row['path']] = row
    require(len(prefixes) == 1 and len(expected) == 93, 'receipt member wrapper or target count differs')
    require(receipt['reused_files'] == reused and receipt['new_files'] == 93 - reused
            and receipt['additional_raw_bytes'] == additional, 'receipt file counts differ')
    require(set(admitted) == {entry['path'] for entry in manifest['files']},
            'manifest and receipt targets differ')
    for entry in manifest['files']:
        planned = expected[entry['path'].removeprefix('data/raw/sleepaccel/1.0.0/')]
        require(entry['exact_bytes'] == planned['exact_bytes']
                and entry['official_checksums'] == planned['official_checksums']
                and entry['source'] == planned['source'] and entry['source_verified'] is True,
                'manifest official file metadata differs')
    changed = acquisition['intake_receipt_sha256'] != receipt_sha
    blockers = derived_outputs(root) if changed else []
    report = {'schema_version': 1, 'dry_run': not execute, 'changed': False,
              'status': 'would bind' if changed else 'already bound; no change',
              'previous_receipt_sha256': acquisition['intake_receipt_sha256'],
              'actual_receipt_sha256': receipt_sha, 'public_receipt_sha256': PUBLIC_RECEIPT_SHA256,
              'verified_raw_files': 93, 'blocked_by_existing_outputs': blockers,
              'resource_policy': {'workers': 1, 'cpu_threads': 1, 'gpu': False}}
    require(not blockers, 'rebinding denied: existing derived outputs at ' + ', '.join(blockers)
            + '; use a fresh checkout without old caches/models/runs')
    if not execute or not changed:
        return report
    provenance_path = root / PROVENANCE
    require(provenance_path.resolve().is_relative_to(root / 'private'),
            'private provenance path escapes repository private directory')
    require(not provenance_path.exists(), 'private binding provenance already exists; refusing overwrite')
    # Preserve original formatting and every byte except the one SHA field.
    old = acquisition['intake_receipt_sha256'].encode()
    pattern = rb'("intake_receipt_sha256"\s*:\s*")' + old + rb'(")'
    updated, count = re.subn(pattern, lambda m: m[1] + receipt_sha.encode() + m[2], original)
    require(count == 1, 'manifest receipt field must occur exactly once')
    provenance = {**report, 'dry_run': False, 'changed': True, 'status': 'bound',
                  'manifest_before_sha256': hashlib.sha256(original).hexdigest(),
                  'manifest_after_sha256': hashlib.sha256(updated).hexdigest(),
                  'frozen_metadata_sha256': FROZEN_METADATA_SHA256,
                  'scope': 'local receipt binding only; no scientific protocol or score change'}
    require(manifest_path.read_bytes() == original and receipt_path.read_bytes() == receipt_bytes
            and not derived_outputs(root), 'inputs or outputs changed during binding; retry in a quiet fresh checkout')
    staged = None
    provenance_created = False
    try:
        descriptor, staged = tempfile.mkstemp(prefix='.receipt-binding-', dir=manifest_path.parent)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        # Provenance contains hashes only, never receipt content or archive/user paths.
        with provenance_path.open('x', encoding='utf-8') as stream:
            provenance_created = True
            stream.write(json.dumps(provenance, indent=2) + '\n')
        os.replace(staged, manifest_path)
    except OSError:
        if provenance_created:
            provenance_path.unlink()
        raise
    finally:
        if staged and os.path.exists(staged):
            os.unlink(staged)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='bind only in a fresh checkout after actual archive intake')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(bind_receipt(execute=args.execute), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'SleepAccel receipt binding refused: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
