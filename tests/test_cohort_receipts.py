import hashlib
import json

import pytest

from scripts.datasets.prepare import feature_source_hashes
from scripts.datasets.prepare_cohort import validate_receipt


def test_reuse_checks_raw_provenance_source_and_saved_artifact_hashes(tmp_path):
    metadata = {'feature_version':'psg-rms-logmel-v1','dataset_version':'V3',
                'feature_source_sha256':feature_source_hashes('psg-audio'),
                'input_integrity':{'ok':True,'files':[{'path':'data/raw/a.edf','local_checksums':{'md5':'abc'}}]},
                'artifact_sha256':{}}
    subset = {'version':'V3','files':[{'path':'data/raw/a.edf','official_checksums':{'md5':'abc'}}]}
    for name in ['features.npz','timeline.npz']:
        (tmp_path/name).write_bytes(name.encode())
        metadata['artifact_sha256'][name] = hashlib.sha256(name.encode()).hexdigest()
    path = tmp_path/'metadata.json'
    path.write_text(json.dumps(metadata),encoding='utf-8')
    assert validate_receipt(tmp_path,subset)['dataset_version'] == 'V3'
    (tmp_path/'features.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        validate_receipt(tmp_path,subset)
    metadata['feature_source_sha256'] = {}
    path.write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(ValueError,match='version/source'):
        validate_receipt(tmp_path,subset)
