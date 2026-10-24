"""PSG respiratory reference intervals, independent of audible sound labels."""
from dataclasses import dataclass
from datetime import datetime
import math
import xml.etree.ElementTree as ET

TARGET_TYPES = frozenset({'Hypopnea', 'ObstructiveApnea', 'CentralApnea', 'MixedApnea'})


@dataclass(frozen=True)
class ReferenceEvent:
    start: float
    end: float
    kind: str


def read_reference(path):
    root = ET.parse(path).getroot()
    tag = lambda element: element.tag.rsplit('}', 1)[-1]
    starts = [element.text for element in root.iter() if tag(element) == 'RecordingStart']
    if len(starts) != 1:
        raise ValueError('expected one RML RecordingStart')
    origin = datetime.fromisoformat(starts[0])
    segments, events = [], []
    for element in root.iter():
        if tag(element) == 'Segment':
            values = {tag(child): child.text for child in element}
            start = (datetime.fromisoformat(values['StartTime'])-origin).total_seconds()
            duration = float(values['Duration'])
            if not all(math.isfinite(value) for value in (start, duration)) or start < 0 or duration <= 0:
                raise ValueError('invalid RML segment interval')
            segments.append((start, duration))
        if tag(element) == 'Event' and element.get('Family') == 'Respiratory':
            start, duration = float(element.get('Start')), float(element.get('Duration'))
            if not all(math.isfinite(value) for value in (start, duration)) or start < 0 or duration <= 0 or not element.get('Type'):
                raise ValueError('invalid respiratory reference interval')
            events.append(ReferenceEvent(start, start+duration, element.get('Type')))
    return origin, tuple(segments), tuple(sorted(events, key=lambda event: event.start))


def union_overlap(start, end, events):
    intervals = sorted((max(start, event.start), min(end, event.end)) for event in events
                       if event.start < end and event.end > start)
    covered, right = 0.0, start
    for left, stop in intervals:
        if stop > right:
            covered += stop - max(left, right)
            right = stop
    return covered


def window_target(start, end, events, positive_fraction=0.5):
    if not 0 < positive_fraction <= 1 or end <= start:
        raise ValueError('invalid target window or overlap threshold')
    overlap = union_overlap(start, end, [event for event in events if event.kind in TARGET_TYPES])
    fraction = overlap/(end-start)
    if fraction >= positive_fraction:
        return 'reference_respiratory_event', fraction
    if union_overlap(start, end, events) == 0:
        return 'no_target_reference_event', 0.0
    return None, fraction  # ambiguous boundary / other respiratory events
