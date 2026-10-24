import unittest

import numpy as np

from nightlab.wearable.features import (
    FEATURE_NAMES, FeatureConfig, extract_sleepaccel_features, parse_sleepaccel_rows,
)


class WearableFeatureTests(unittest.TestCase):
    def test_stage_semantics_match_the_dependency_free_auditor(self):
        from scripts.datasets.sleepaccel_labels import normalize_stage
        from nightlab.wearable.features import STAGES
        self.assertEqual(STAGES,{code:normalize_stage(code)['stage'] for code in [-1,0,1,2,3,4,5]})

    def test_manual_math_and_half_open_intervals(self):
        # PSG-outside points would dominate the statistics if accidentally used.
        result = extract_sleepaccel_features(
            "tiny", [[-1, 999], [0, 60], [10, 70], [20, 80], [30, 100], [60, 999]],
            [[-1, 999, 0, 0], [0, 1, 0, 0], [10, 2, 0, 0], [20, 3, 0, 0],
             [30, 4, 0, 0], [60, 999, 0, 0]], [[0, 3], [30, 5]],
        )
        self.assertEqual(result.feature_names, FEATURE_NAMES)
        self.assertEqual(result.subject_id.tolist(), ["tiny", "tiny"])
        self.assertEqual(result.raw_stage.tolist(), [3, 5])
        self.assertEqual(result.stage.tolist(), ["N3", "REM"])
        expected = [3, 70, np.sqrt(200/3), 60, 80, 1, 3,
                    np.sqrt(2/3), 0, 0, 2, np.sqrt(2/3), 1]
        np.testing.assert_allclose(result.features[0], expected)
        self.assertEqual(result.hr_sample_count.tolist(), [3, 1])
        self.assertEqual(result.accel_sample_count.tolist(), [3, 1])
        self.assertEqual(result.features[1, 1], 100)
        self.assertEqual(result.features[1, 2], 0)
        self.assertTrue(np.isnan(result.features[1, 5]))
        for kind in ("heart_rate", "motion"):
            self.assertEqual(result.metadata["signals"][kind]["samples_before_label_range"], 1)
            self.assertEqual(result.metadata["signals"][kind]["samples_after_label_range"], 1)
            coverage = result.coverage[kind]
            self.assertEqual(coverage["maximum_gap_including_edges_seconds"].tolist(), [10, 30])
            self.assertEqual(coverage["leading_gap_seconds"].tolist(), [0, 0])
            self.assertEqual(coverage["trailing_gap_seconds"].tolist(), [10, 30])
            self.assertTrue(np.isnan(coverage["maximum_internal_gap_seconds"][1]))

    def test_unscored_preserved_and_thresholds_only_report(self):
        result = extract_sleepaccel_features(
            "subject", [[7, 50], [35, 900], [70, 80]],
            [[7, 0, 0, 1], [35, 0, 0, 999], [70, 0, 0, 2]],
            [[0, 4], [30, -1], [60, 0], [90, 2]], FeatureConfig(9, 9),
        )
        self.assertEqual(result.epoch_start_seconds.tolist(), [0, 60, 90])
        self.assertEqual(result.raw_stage.tolist(), [4, 0, 2])
        self.assertEqual(result.stage.tolist(), ["N3", "W", "N2"])
        self.assertEqual(result.label_table["target_valid"].tolist(), [True, False, True, True])
        self.assertIsNone(result.label_table["stage"][1])
        self.assertEqual(result.label_table["epoch_start_seconds"].tolist(), [0, 30, 60, 90])
        self.assertEqual(result.metadata["unscored_epochs"], 1)
        self.assertTrue(result.target_valid.all())
        self.assertEqual(result.features.shape, (3, 13))
        self.assertTrue(result.coverage["heart_rate"]["below_configured_min_samples"].all())
        self.assertEqual(result.hr_sample_count.tolist(), [1, 1, 0])
        self.assertTrue(np.isnan(result.features[2, 1:6]).all())
        self.assertTrue(np.isnan(result.features[2, 7:]).all())
        self.assertEqual(result.features[2, 0], 0)
        self.assertEqual(result.features[2, 6], 0)
        self.assertEqual(result.coverage["motion"]["maximum_gap_including_edges_seconds"][2], 30)
        self.assertTrue(np.isnan(result.coverage["motion"]["leading_gap_seconds"][2]))
        # Leading and trailing edges are included even when an epoch is sparse.
        self.assertEqual(result.coverage["heart_rate"]["maximum_gap_including_edges_seconds"][0], 23)

    def test_all_unscored_is_empty_without_fabricated_targets(self):
        result = extract_sleepaccel_features("x", [[0, 70]], [[0, 0, 0, 1]], [[0, -1]])
        self.assertEqual(result.features.shape, (0, len(FEATURE_NAMES)))
        self.assertEqual(result.metadata["scored_epochs"], 0)
        self.assertEqual(result.metadata["unscored_epochs"], 1)

    def test_published_delimiters_and_strict_validation(self):
        np.testing.assert_array_equal(parse_sleepaccel_rows("-1,60\n0,61\n", "heart_rate"),
                                      [[-1, 60], [0, 61]])
        np.testing.assert_array_equal(parse_sleepaccel_rows("0\t1 2 3\n", "motion"), [[0, 1, 2, 3]])
        for kind, bad_inputs in {
            "heart_rate": ["", "0 60", "0,60,1", "0,nan", "0,inf", "0,60\n0,61", "1,60\n0,61", "0,no", "0,60\n\n"],
            "motion": ["", "0,1,2,3", "0 1 2", "0 1 nan 3", "0 1 2 3\n0 2 3 4"],
            "labels": ["", "0 6", "0 -2", "0 1.5", "0 nan", "0 0\n31 1", "0 0\n60 1", "0 0\n0 1"],
        }.items():
            for text in bad_inputs:
                with self.subTest(kind=kind, text=text), self.assertRaises(ValueError):
                    parse_sleepaccel_rows(text, kind)

    def test_array_inputs_are_validated_too(self):
        for labels in ([[0, 6]], [[0, 0], [45, 1]], [[0, 0.5]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                extract_sleepaccel_features("x", [[0, 70]], [[0, 0, 0, 1]], labels)
        with self.assertRaises(ValueError):
            extract_sleepaccel_features("x", [[0, 70], [0, 80]], [[0, 0, 0, 1]], [[0, 0]])
        with self.assertRaises(ValueError):
            FeatureConfig(min_hr_samples=-1)
        with self.assertRaises(ValueError):
            FeatureConfig(min_accel_samples=1.5)

    def test_only_identical_complete_time_reversal_blocks_can_be_collapsed(self):
        heart = [[0,60],[10,80],[0,60],[10,80],[0,60],[10,80]]
        with self.assertRaises(ValueError):
            extract_sleepaccel_features('x',heart,[[0,0,0,1]],[[0,0]])
        config = FeatureConfig(repeated_block_policy='collapse_identical')
        result = extract_sleepaccel_features('x',heart,[[0,0,0,1]],[[0,0]],config)
        self.assertEqual(result.features[0,0],2)
        self.assertEqual(result.features[0,1],70)
        self.assertEqual(result.metadata['signals']['heart_rate']['identical_repeated_blocks'],3)
        self.assertEqual(result.metadata['signals']['heart_rate']['discarded_exact_repeat_samples'],4)
        for invalid in ([[0,60],[10,80],[0,61],[10,80]],[[0,60],[0,60]],[[10,80],[0,60]]):
            with self.subTest(rows=invalid),self.assertRaises(ValueError):
                extract_sleepaccel_features('x',invalid,[[0,0,0,1]],[[0,0]],config)


if __name__ == "__main__":
    unittest.main()
