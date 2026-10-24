import math
import struct
import unittest

from scripts.datasets.inspect_sciencedb import annotation_summary, wav_header_summary


class ScienceDBInspectionTests(unittest.TestCase):
    def document(self):
        return {'record_start': 80000, 'awake_intervals': [[90000, 90030], [90000, 90030]],
                'events': [{'event_type': 'hypo', 'evnet_start': 90000,
                            'event_duration': 30, 'sleep_stage': 'N2'}]}

    def test_cross_midnight_is_not_wrapped_and_conflict_is_explicit(self):
        result = annotation_summary(self.document(), 8000)
        self.assertEqual(result['relative_first_event'], 10000)
        self.assertEqual(result['duplicate_awake_intervals'], 1)
        self.assertEqual(result['events_outside_audio_under_stated_assumption'], 1)
        self.assertEqual(result['alignment_status'], 'conflict')

    def test_bounds_match_does_not_claim_synchronization(self):
        result = annotation_summary(self.document(), 11000)
        self.assertIn('synchronization still unverified', result['alignment_status'])

    def test_span_longer_than_audio_cannot_be_fixed_by_constant_offset(self):
        document = self.document()
        document['events'].append({'event_type': 'osa', 'evnet_start': 110000,
                                   'event_duration': 30, 'sleep_stage': 'N1'})
        result = annotation_summary(document, 10000)
        self.assertEqual(result['annotated_event_span_seconds'], 20030)
        self.assertFalse(result['all_events_fit_under_any_constant_offset'])

    def test_nonfinite_and_nonpositive_duration_rejected(self):
        for value in (math.nan, math.inf, -1, 0, True):
            document = self.document()
            document['events'][0]['event_duration'] = value
            with self.assertRaises(ValueError):
                annotation_summary(document)

    def test_conflicting_start_fields_rejected(self):
        document = self.document()
        document['events'][0]['event_start'] = 80000
        with self.assertRaises(ValueError):
            annotation_summary(document)

    def test_header_can_describe_full_audio_without_reading_it(self):
        fmt = struct.pack('<HHIIHH', 1, 1, 8000, 16000, 2, 16)
        header = b'RIFF' + struct.pack('<I', 16036) + b'WAVEfmt ' + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', 16000)
        result = wav_header_summary(header)
        self.assertEqual(result['duration_seconds'], 1)
        self.assertFalse(result['full_audio_downloaded_or_verified'])
        with self.assertRaises(ValueError):
            wav_header_summary(header[:36])


if __name__ == '__main__':
    unittest.main()
