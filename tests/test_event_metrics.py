import numpy as np
import pytest

from nightlab.audio.events import decode_events, evaluate_events


def test_center_averaging_threshold_and_touching_merge():
    result = decode_events([0, 5, 10], [10, 15, 20], [.2, .8, .6], 20)
    assert result['bin_probabilities'] == pytest.approx([.2, .5, .7, .6])
    assert result['bin_starts'] == [0, 5, 10, 15]
    assert result['bin_ends'] == [5, 10, 15, 20]
    assert result['events'] == [[5, 20]]


def test_final_short_bin_and_no_gap_closing():
    result = decode_events([0, 5, 10], [5, 10, 12], [.8, .1, .9], 12)
    assert result['bin_ends'] == [5, 10, 12]
    assert result['events'] == [[0, 5], [10, 12]]
    assert decode_events([0], [2], [.5], 2)['events'] == [[0, 2]]


@pytest.mark.parametrize('starts,ends,probs,duration', [
    ([0], [5], [.8], 10), ([], [], [], 5),
    ([0, 0], [5, 10], [.2, .8], 10),
    ([5, 0], [10, 5], [.2, .8], 10),
    ([-1], [10], [.2], 10), ([0], [11], [.2], 10),
    ([0], [0], [.2], 10), ([0], [10], [1.1], 10),
    ([0], [10], [-.1], 10), ([0], [10], [np.nan], 10),
    ([np.inf], [10], [.2], 10), ([0], [np.nan], [.2], 10),
    ([0], [10], [], 10), ([[0]], [10], [.2], 10),
    ([0], [10], [.2], 0), ([0], [10], [.2], np.inf),
])
def test_invalid_decode_inputs(starts, ends, probs, duration):
    with pytest.raises(ValueError):
        decode_events(starts, ends, probs, duration)


def test_optimal_matching_avoids_greedy_failure():
    result = evaluate_events([[1, 11], [0, 7]], [[0, 10], [4, 14]], 20)
    assert (result['tp'], result['fp'], result['fn']) == (2, 0, 0)
    assert [(m['prediction_index'], m['reference_index']) for m in result['matches']] == [(0, 1), (1, 0)]
    assert [m['iou'] for m in result['matches']] == pytest.approx([7/13, .7])
    assert result['f1'] == 1


def test_matching_maximizes_iou_after_cardinality():
    result = evaluate_events([[0, 10], [2, 12]], [[0, 10], [2, 12]], 12)
    assert [m['iou'] for m in result['matches']] == [1, 1]


def test_censored_detection_and_localization_signed_errors():
    result = evaluate_events([[1, 11], [20, 29]], [[0, 10], [20, 30]], 30, [False, True])
    assert result['tp'] == 2
    assert result['localization'] == {
        'matched_uncensored_count': 1,
        'mean_onset_error_seconds': 1,
        'mean_end_error_seconds': 1,
        'mean_absolute_onset_error_seconds': 1,
        'mean_absolute_end_error_seconds': 1,
    }
    assert result['matches'][1]['onset_error_seconds'] is None
    assert result['matches'][1]['end_absolute_error_seconds'] is None
    negative = evaluate_events([[0, 9]], [[1, 10]], 10)
    assert negative['localization']['mean_onset_error_seconds'] == -1
    assert negative['localization']['mean_absolute_onset_error_seconds'] == 1


@pytest.mark.parametrize('preds,refs,expected', [
    ([], [], (0, 0, 0, None, None, None)),
    ([], [[0, 10]], (0, 0, 1, None, 0, 0)),
    ([[0, 10]], [], (0, 1, 0, 0, None, 0)),
    ([[0, 5]], [[10, 15]], (0, 1, 1, 0, 0, 0)),
])
def test_no_event_edge_cases(preds, refs, expected):
    result = evaluate_events(preds, refs, 1800)
    assert tuple(result[key] for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')) == expected
    assert result['false_positives_per_hour'] == expected[1] * 2
    assert result['localization']['mean_absolute_onset_error_seconds'] is None


def test_iou_threshold_is_inclusive():
    assert evaluate_events([[0, 5]], [[0, 10]], 10)['tp'] == 1
    assert evaluate_events([[0, 4.99]], [[0, 10]], 10)['tp'] == 0


@pytest.mark.parametrize('preds,refs,seconds,flags', [
    ([[0, 0]], [], 10, None), ([[-1, 5]], [], 10, None),
    ([[0, 11]], [], 10, None), ([[0, np.nan]], [], 10, None),
    ([[0, 5, 7]], [], 10, None), ([], [[4, 3]], 10, None),
    ([], [[0, np.inf]], 10, None), ([], [], 0, None),
    ([], [], np.nan, None), ([], [[0, 5]], 10, []),
    ([], [[0, 5]], 10, [1]), ([], [[0, 5]], 10, [[True]]),
])
def test_invalid_metric_inputs(preds, refs, seconds, flags):
    with pytest.raises(ValueError):
        evaluate_events(preds, refs, seconds, flags)
