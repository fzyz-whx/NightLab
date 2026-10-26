"""Bounded synthetic checks of the independent cohort audit."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts.datasets import audit_sleepaccel31 as audit


class SleepAccel31AuditTests(unittest.TestCase):
    def fixture(self, root, labels='0 0\n30 4\n60 -1\n', motion=None):
        data = {'labels': labels, 'heart_rate': '0,60\n10,70\n0,60\n10,70\n',
                'motion': motion or '0 1 0 0\n10 1 0 0\n30 1 0 0\n'}
        entries = {}
        for kind, text in data.items():
            path = root / 'data/raw/tiny' / f'{kind}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            entries[kind] = {'path': path.relative_to(root).as_posix(),
                             'exact_bytes': path.stat().st_size,
                             'official_checksums': {'sha256': audit.sha256(path)}}
        return entries

    def test_repeat_coverage_mapping_and_privacy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = audit.audit_subject(root, 'tiny', self.fixture(root))
            self.assertTrue(result['valid'])
            hr = result['signals']['heart_rate']
            self.assertFalse(hr['strict_original_valid'])
            self.assertEqual(hr['identical_blocks'], 2)
            self.assertEqual(hr['collapsed_identical_rows'], 2)
            self.assertEqual(hr['scored_epoch_coverage']['no_samples'], 1)
            self.assertEqual(result['labels']['normalized_stage_counts']['N3'], 1)
            self.assertEqual(result['labels']['unscored_epochs'], 1)
            serialized = json.dumps(result)
            self.assertNotIn('epoch_start', serialized)
            self.assertNotIn('raw_values', serialized)

    def test_nonidentical_reversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-identical'):
            audit.validate_signal(np.array([[0, 60], [10, 70], [0, 61], [10, 70]]), 'heart_rate')

    def test_reversal_diagnostics_distinguish_identical_and_conflicting_times(self):
        rows = np.array([[-5, 1], [0, 2], [10, 3], [-5, 1], [0, 4], [10, 3]])
        original = rows.copy()
        result = audit.raw_diagnostics(rows, np.array([[0, 0], [30, 1]]))
        self.assertEqual(result['negative_time_steps'], 1)
        self.assertEqual(result['reversal_magnitude_seconds']['maximum'], 15)
        self.assertEqual(result['repeated_timestamp_groups']['identical_values'], 2)
        self.assertEqual(result['repeated_timestamp_groups']['conflicting_values'], 1)
        self.assertEqual(result['repeated_timestamp_groups']['conflicting_inside_label_range'], 1)
        self.assertFalse(result['all_reset_blocks_exactly_identical'])
        self.assertEqual([s['rows'] for s in result['strictly_increasing_segments']], [3, 3])
        np.testing.assert_array_equal(rows, original)

    def test_invalid_labels_remain_failure(self):
        for labels in ('0 6\n30 1\n', '0 0\n31 1\n', '0 0\n0 1\n'):
            with self.subTest(labels=labels), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result = audit.audit_subject(root, 'tiny', self.fixture(root, labels=labels))
                self.assertFalse(result['valid'])
                self.assertTrue(any(item.startswith('labels:') for item in result['errors']))

    def test_hash_failure_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = self.fixture(root)
            entries['motion']['official_checksums']['sha256'] = '0' * 64
            result = audit.audit_subject(root, 'tiny', entries)
            self.assertFalse(result['valid'])
            self.assertFalse(result['files']['motion']['official_sha256_verified'])

    def test_memory_guard_and_metadata_only_do_not_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = self.fixture(root)
            with patch.object(audit, 'available_memory', return_value=0):
                with self.assertRaises(MemoryError):
                    audit.read_rows(root / entries['motion']['path'], 'motion')
        manifest = {'id': 'synthetic-cohort', 'acquisition': {'intake_receipt_sha256': 'fixture'}}
        with patch.object(audit, 'metadata', return_value=(manifest, {})), \
             patch.object(Path, 'write_text', side_effect=AssertionError('dry run wrote')):
            result = audit.run()
        self.assertTrue(result['dry_run'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['all_valid'])
        self.assertEqual(result['audited_subjects'], 0)

    def test_partial_abort_is_saved_as_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / audit.MANIFEST).parent.mkdir(parents=True)
            (root / audit.MANIFEST).write_text('{}')
            manifest = {'id': 'test', 'acquisition': {'intake_receipt_sha256': 'x'}}
            with patch.object(audit, 'metadata', return_value=(manifest, {'tiny': {}})), \
                 patch.object(audit, 'audit_subject', side_effect=MemoryError('budget')):
                result = audit.run(root, execute=True)
            self.assertFalse(result['complete'])
            self.assertFalse(result['all_valid'])
            self.assertIn('budget', result['errors'][0])
            self.assertTrue((root / audit.REPORT).is_file())


if __name__ == '__main__':
    unittest.main()
