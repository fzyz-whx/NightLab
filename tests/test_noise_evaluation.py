import json
from importlib.metadata import version

import numpy as np
import pytest

pytest.importorskip('torch')
from nightlab.audio.events import evaluate_events
from scripts.evaluate import noise as evaluator


def test_pooling_uses_all_windows_and_count_weighted_event_hours():
    bg,pos = evaluator.LABELS
    items = [
        {'truth':np.array([bg,pos]),'predicted':np.array([bg,pos]),'probabilities':np.array([.1,.9]),
         'events':evaluate_events([[0,10]],[[0,10]],3600)},
        {'truth':np.array([bg,pos,pos]),'predicted':np.array([pos,bg,pos]),'probabilities':np.array([.6,.4,.8]),
         'events':evaluate_events([[0,10],[30,40]],[[30,40],[60,70]],10800)}]
    result = evaluator.pooled_scores(items)
    assert result['classification']['confusion_matrix'] == [[1,1],[1,2]]
    assert result['events']['tp'] == 2 and result['events']['fp'] == 1 and result['events']['fn'] == 1
    assert result['events']['false_positives_per_hour'] == .25


def test_complete_event_probabilities_keep_ambiguous_windows_and_event_threshold():
    arrays = {'y':np.array(['','reference_respiratory_event','']),
              'start_seconds':np.array([0,5,10]),'end_seconds':np.array([10,15,20])}
    scored = evaluator.score_probabilities([.5,.5,.5],arrays,[[0,20]],[False],20)
    assert scored['decoded']['events'] == [[0,20]]
    assert scored['events']['tp'] == 1  # event >=.5; classifiers retain original argmax tie behavior
    for invalid in ([.5],[-1,.5,.5],[np.nan,.5,.5]):
        with pytest.raises(ValueError):
            evaluator.score_probabilities(invalid,arrays,[],[],20)


def index_fixture(root,monkeypatch):
    records = [{'record_id':identity} for identity in ('00000999-100507','00001018-100507')]
    policy = {'subjects':{'train':['a'],'validation':['b'],'test':['00000999','00001018']}}
    sources = {'protocol':'frozen'}
    monkeypatch.setattr(evaluator.noise_features,'_select',lambda *_:({}, {'split_policy':policy},records))
    monkeypatch.setattr(evaluator.noise_features,'_sources',lambda *_:sources)
    entries = [{'record_id':record['record_id'],'kind':kind,'snr_db':snr,'split':'test'}
               for record in records for kind in evaluator.noise_features.KINDS for snr in evaluator.noise_features.SNRS]
    index = {'complete':True,'scope':'test','version':'noise-eval-v1','split_policy':policy,
             'source_hashes':sources,'protocol_sha256':'frozen','environment':{name:version(name) for name in ('numpy','scipy')},
             'records':entries}
    path = root/'data/processed/psg-audio-pilot-10/noise-eval-v1/corpus.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(index),encoding='utf-8')
    return path,index


def test_model_evaluation_requires_all_fixed_test_conditions(tmp_path,monkeypatch):
    path,index = index_fixture(tmp_path,monkeypatch)
    assert len(evaluator.validate_index(tmp_path)[0]['records']) == 16
    for mutation in ('partial','pilot','duplicate','wrong_split','changed_code','changed_environment'):
        invalid = json.loads(json.dumps(index))
        if mutation == 'partial': invalid['complete'] = False
        elif mutation == 'pilot': invalid['scope'] = 'pilot'
        elif mutation == 'duplicate': invalid['records'][-1] = invalid['records'][0]
        elif mutation == 'wrong_split': invalid['records'][0]['split'] = 'train'
        elif mutation == 'changed_code': invalid['source_hashes'] = {}
        else: invalid['environment']['numpy'] = 'old'
        path.write_text(json.dumps(invalid),encoding='utf-8')
        with pytest.raises(ValueError): evaluator.validate_index(tmp_path)
