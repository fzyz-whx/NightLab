from nightlab.audio.events import evaluate_events
from scripts.evaluate.event_failures import describe_failures


def test_failure_categories_reconcile_for_overlap_miss_and_one_to_one_competition():
    predictions = [[0,10],[0,9],[20,22],[80,90]]
    references = [[0,10],[20,30],[50,60]]
    metrics = evaluate_events(predictions,references,100)
    result = describe_failures(predictions,references,metrics)
    assert result['false_positives'] == {'no_reference_overlap':1,'overlap_below_iou_threshold':1,'one_to_one_competition':1}
    assert result['false_negatives'] == {'no_reference_overlap':1,'overlap_below_iou_threshold':1,'one_to_one_competition':0}
    assert result['references_overlapping_multiple_predictions'] == 1
    assert result['predicted_duration']['count'] == 4


def test_failure_description_handles_empty_sides_without_nan():
    for predictions,references in [([],[]),([],[[0,10]]),([[0,10]],[])]:
        result = describe_failures(predictions,references,evaluate_events(predictions,references,100))
        assert result['predicted_duration']['median_seconds'] == (10 if predictions else None)
        assert result['reference_duration']['median_seconds'] == (10 if references else None)
