"""Small synthetic checks; no real data, training or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from nightlab.wearable.cleaning_v3 import canonicalize, VERSION
from nightlab.wearable.features import extract_sleepaccel_features
from nightlab.deep.inputs import wearable_epoch_tensors
from scripts.datasets import prepare_sleepaccel31 as prepare


class SleepAccel31V3Tests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([[0, 4], [30, -1], [60, 5]], dtype=float)
        self.hr = np.array([[0, 60], [10, 70], [30, 80], [60, 90]], dtype=float)
        self.motion = np.array([[0, 1, 0, 0], [10, 2, 0, 0], [30, 3, 0, 0], [60, 4, 0, 0]], dtype=float)

    def test_unsorted_exact_duplicates_and_raw_immutable(self):
        hr = np.vstack((self.hr[::-1], self.hr[1], [-1, 999], [90, 999]))
        motion = np.vstack((self.motion[::-1], self.motion[0]))
        originals = [value.copy() for value in (hr, motion, self.labels)]
        result = canonicalize(hr, motion, self.labels)
        np.testing.assert_array_equal(result['heart_rate'], self.hr)
        np.testing.assert_array_equal(result['motion'], self.motion)
        for value, original in zip((hr, motion, self.labels), originals):
            np.testing.assert_array_equal(value, original)
        self.assertFalse(np.shares_memory(result['labels'], self.labels))
        stats = result['metadata']['signals']['heart_rate']
        self.assertEqual(stats['exact_duplicate_rows_removed'], 1)
        self.assertEqual(stats['unique_samples_cropped'], 2)
        self.assertEqual(stats['raw_samples'], 7)
        self.assertEqual(stats['canonical_samples_after_crop'], 4)
        self.assertGreater(stats['raw_time_reversals'], 0)

    def test_outside_conflict_cropped_inside_conflict_rejected(self):
        result = canonicalize(np.vstack((self.hr, [-1, 1], [-1, 2], [90, 1], [90, 2])), self.motion, self.labels)
        self.assertEqual(result['metadata']['signals']['heart_rate']['conflicting_timestamp_groups_outside_label_interval'], 2)
        self.assertEqual(result['metadata']['signals']['heart_rate']['conflicting_timestamp_groups_inside_label_interval'], 0)
        for conflict in ([0, 999], [30, 999]):
            with self.assertRaisesRegex(ValueError, 'conflicting same-timestamp'):
                canonicalize(np.vstack((self.hr, conflict)), self.motion, self.labels)

    def test_labels_are_never_reordered_or_repaired(self):
        for labels in (self.labels[::-1], [[0, 0], [31, 1]], [[0, 9]], [[0, 1], [0, 2]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                canonicalize(self.hr, self.motion, labels)

    def test_empty_and_nonfinite_rejected(self):
        for hr in ([], [[-1, 1]], [[0, np.nan]], [[0, np.inf]]):
            with self.subTest(hr=hr), self.assertRaises(ValueError):
                canonicalize(hr, self.motion, self.labels)

    def test_old_extractors_exact_parity_and_artifact_semantics(self):
        canonical = canonicalize(self.hr[::-1], self.motion[::-1], self.labels)
        classic, timeline, deep, feature_meta, tensor_meta = prepare.build('tiny', canonical)
        old = extract_sleepaccel_features('tiny', self.hr, self.motion, self.labels)
        old_deep = wearable_epoch_tensors('tiny', self.hr, self.motion, self.labels, repeat_policy='reject')
        np.testing.assert_array_equal(classic['X'], old.features.astype(np.float32))
        np.testing.assert_array_equal(deep['values'], old_deep['values'])
        np.testing.assert_array_equal(deep['masks'], old_deep['masks'])
        self.assertEqual(classic['y'].tolist(), ['N3', 'REM'])
        self.assertEqual(timeline['target_valid'].tolist(), [True, False, True])
        self.assertEqual(timeline['stage'].tolist(), ['N3', '', 'REM'])
        self.assertEqual(deep['values'].shape, (2, 5, 30))
        self.assertTrue(np.isnan(deep['values'][0, 0, 1]))
        self.assertEqual(deep['masks'][0, 0, 1], 0)
        self.assertEqual(feature_meta['repeated_block_policy'], 'reject')
        self.assertEqual(tensor_meta['repeated_block_policy'], 'reject')

    def test_atomic_publish_and_conflicting_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'record'
            meta = {'provenance': {'x': 'sha'}, 'rows': 1, 'feature_version': 'v3'}
            arrays = {'features.npz': {'X': np.array([[np.nan]]), 'y': np.array(['W'])}}
            receipt, _ = prepare.publish(destination, arrays, meta)
            self.assertIn('features.npz', receipt['artifact_sha256'])
            prepare.publish(destination, arrays, {**meta})
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                prepare.publish(destination, arrays, {**meta, 'provenance': {'x': 'different'}})
            with self.assertRaisesRegex(ValueError, 'differs'):
                prepare.publish(destination, {'features.npz': {'X': np.array([[1.]]), 'y': np.array(['W'])}}, meta)

    def test_failed_record_cannot_complete_corpora(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {'id': 'sleepaccel-cohort-31', 'version': 'test', 'subjects': {},
                        'preparation_policy': {'version': VERSION}}
            with patch.object(prepare.audit, 'metadata', return_value=(manifest, {'tiny': {}})), \
                 patch.object(prepare, 'source_hashes', return_value={'fixed': 'sha'}), \
                 patch.object(prepare, 'prepare_one', side_effect=ValueError('bad source')):
                result = prepare.run(root, execute=True)
            self.assertFalse(result['complete'])
            self.assertEqual(result['records'], 0)
            for relative in (prepare.BASE, prepare.DEEP):
                index = json.loads((root / relative / 'corpus.json').read_text())
                self.assertFalse(index['complete'])
                self.assertIn('bad source', index['errors'][0])

    def test_dry_run_never_writes(self):
        manifest = {'id': 'synthetic-cohort', 'version': 'fixture', 'subjects': {},
                    'preparation_policy': {'version': VERSION}}
        with patch.object(prepare.audit, 'metadata', return_value=(manifest, {})), \
             patch.object(prepare, 'source_hashes', return_value={'fixture': 'sha'}), \
             patch.object(Path, 'write_text', side_effect=AssertionError('write')):
            result = prepare.run()
        self.assertTrue(result['dry_run'])
        self.assertFalse(result['writes'])


if __name__ == '__main__':
    unittest.main()
