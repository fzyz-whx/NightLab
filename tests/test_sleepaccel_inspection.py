import unittest
from scripts.datasets.inspect_sleepaccel import inspect_rows, alignment_summary
from scripts.datasets.sleepaccel_labels import normalize_stage


class InspectionTests(unittest.TestCase):
    def test_preserves_unmapped_labels(self):
        result = inspect_rows('0 4\n30 -1\n60 5\n', 'labels')
        self.assertEqual(result['raw_label_counts'], {-1: 1, 4: 1, 5: 1})
        self.assertTrue(result['semantic_mapping_applied'])
        self.assertEqual(result['valid_target_rows'], 2)
        self.assertEqual(result['normalized_stage_counts'], {'N3': 1, 'REM': 1})
        self.assertEqual(result['gap_median_seconds'], 30)

    def test_hr_negative_time_is_valid(self):
        result = inspect_rows('-5,70\n0,71\n5,72', 'heart_rate')
        self.assertEqual(result['before_psg_rows'], 1)

    def test_bad_inputs_rejected(self):
        for text in ('', '0 1', '0 1\n0 2', '30 1\n0 2',
                     '0 1\n30 nan', '0 1 2\n30 2 3', '0 1\n30 1.5'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                inspect_rows(text, 'labels')

    def test_motion_columns(self):
        self.assertEqual(inspect_rows('0 1 2 3\n0.1 2 3 4', 'motion')['columns'], 4)

    def test_stage_mapping_preserves_source_and_masks_unscored(self):
        self.assertEqual(normalize_stage(-1), {'raw_stage': -1, 'stage': None, 'target_valid': False})
        for raw, stage in [(0, 'W'), (1, 'N1'), (2, 'N2'), (3, 'N3'), (4, 'N3'), (5, 'REM')]:
            self.assertEqual(normalize_stage(raw), {'raw_stage': raw, 'stage': stage, 'target_valid': True})
        for raw in (6, -2, True, 4.0, '4'):
            with self.assertRaises(ValueError):
                normalize_stage(raw)

    def test_alignment_crops_outside_points_and_uses_half_open_epochs(self):
        labels = [[0,0],[30,-1],[60,5]]
        result = alignment_summary(labels, {'heart_rate': [[-1,80],[0,70],[29,71],[30,72],[90,73]],
                                            'motion': [[0,1,2,3],[60,1,2,3]]})
        self.assertEqual(result['scored_epochs'], 2)
        self.assertEqual(result['scored_epochs_with_at_least_one_sample_per_signal'], 1)
        self.assertEqual(result['signals']['heart_rate']['scored_epochs_without_samples'], 1)
        self.assertEqual(result['signals']['heart_rate']['samples_before_label_range'], 1)
        self.assertEqual(result['signals']['heart_rate']['samples_after_label_range'], 1)
        self.assertEqual(result['signals']['motion']['samples_per_scored_epoch_min'], 1)

    def test_missing_epoch_grid_is_not_silently_filled(self):
        with self.assertRaises(ValueError):
            alignment_summary([[0,0],[60,1]], {'motion':[[0,1,2,3]]})


if __name__ == '__main__':
    unittest.main()
