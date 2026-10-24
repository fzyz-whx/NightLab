import io
import unittest
import xml.etree.ElementTree as ET

from scripts.datasets.inspect_psg import read_header, read_rml


class PSGInspectionTests(unittest.TestCase):
    def test_edf_layout_and_record_sizes(self):
        fixed = bytearray(b' '*256)
        for start, value in [(0,b'0       '),(168,b'07.05.10'),(176,b'22.00.00'),
                             (184,b'768     '),(236,b'2       '),(244,b'1       '),(252,b'2   ')]:
            fixed[start:start+len(value)] = value
        channels = bytearray(b' '*512)
        channels[:16] = b'Mic             '
        channels[16:32] = b'Snore           '
        offset = 2*(16+80+8+8+8+8+8+80)
        channels[offset:offset+16] = b'48000   500     '
        header = read_header(io.BytesIO(fixed+channels))
        self.assertEqual(header['labels'], ['Mic','Snore'])
        self.assertEqual(header['samples'], [48000,500])
        self.assertEqual(header['record_bytes'], 97000)
        with self.assertRaises(ValueError):
            read_header(io.BytesIO(fixed+channels[:100]))

    def test_rml_user_and_machine_stage_changes_are_separate(self):
        xml = '''<PatientStudy><RecordingStart>2010-05-07T21:00:00</RecordingStart>
        <Segment><StartTime>2010-05-07T22:00:00</StartTime><Duration>3600</Duration></Segment>
        <Events><Event Family="Respiratory" Type="Hypopnea" Start="3750" Duration="20"/>
        <Event Family="Cardiac" Type="Tachy" Start="2" Duration="3"/></Events>
        <UserStaging><Stage Start="0" Type="Wake"/></UserStaging>
        <MachineStaging><Stage Start="0" Type="QuietSleep"/></MachineStaging></PatientStudy>'''
        _, segments, events, stages = read_rml(ET.fromstring(xml))
        self.assertEqual(segments, [(3600,3600)])
        self.assertEqual(events, [{'start':3750,'end':3770,'type':'Hypopnea'}])
        self.assertEqual(stages['UserStaging'][0].get('Type'), 'Wake')
        self.assertEqual(stages['MachineStaging'][0].get('Type'), 'QuietSleep')
        with self.assertRaises(ValueError):
            read_rml(ET.fromstring(xml.replace('Duration="20"', 'Duration="nan"')))


if __name__ == '__main__':
    unittest.main()
