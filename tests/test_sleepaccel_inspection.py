import unittest
from scripts.datasets.inspect_sleepaccel import inspect_rows


class InspectionTests(unittest.TestCase):
    def test_preserves_unmapped_labels(self):
        result = inspect_rows('0 4\n30 -1\n60 5\n', 'labels')
        self.assertEqual(result['raw_label_counts'], {-1: 1, 4: 1, 5: 1})
        self.assertFalse(result['semantic_mapping_applied'])
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


if __name__ == '__main__':
    unittest.main()
