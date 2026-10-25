"""Small receipt and evaluator contracts; no training or real datasets required."""
import json
from importlib.metadata import version

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from scripts.evaluate import audio_augmentation as controls
from scripts.evaluate import noise


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def frozen_fixture(root,monkeypatch):
    key = 'cnn_clean_duplicate'
    mode,run,public = controls.CONTROLS[key]
    protocol = root/'docs/NOISE_AUGMENTATION_PROTOCOL.md'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('fixed protocol',encoding='utf-8')
    moments = {'mean':[0.0]*32,'scale':[1.0]*32}
    report = {'mode':mode,'normalizer':moments,'checkpoint_sha256':'unused','best_epoch':1,
              'training_protocol_sha256':controls.sha256(protocol),'label_order':noise.LABELS,
              'task':'audio','feature_set':'all','smoke_only':False,'saved_loaded_predictions_exact':True,
              'source_sha256':{name:controls.sha256(module.__file__) for name,module in
                               (('models',controls.models),('inputs',controls.inputs),('training',controls.training))},
              'environment':{name:version(name) for name in ('torch','numpy','scipy','scikit-learn')},
              'augmentation_index_sha256':'index'}
    weights = root/'models'/run/'weights.pt'
    weights.parent.mkdir(parents=True)
    torch.save({},weights)
    report['checkpoint_sha256'] = controls.sha256(weights)
    metadata = {field:report[field] for field in ('mode','normalizer','checkpoint_sha256','source_sha256',
                                                'best_epoch','training_protocol_sha256','label_order')}
    paths = (root/'runs'/run/'metrics.json',root/'models'/run/'metadata.json',root/'docs/results'/f'{public}.json')
    def save():
        write_json(paths[0],report)
        write_json(paths[1],metadata)
        write_json(paths[2],{'run_name':run,'mode':mode,'metrics_source_sha256':controls.sha256(paths[0]),
                             'checkpoint_sha256':report['checkpoint_sha256']})
    save()
    monkeypatch.setattr(controls,'verify_training_cache',lambda *_:None)
    class Model:
        def to(self,_): return self
        def load_state_dict(self,value): assert value == {}
        def eval(self): return self
    monkeypatch.setattr(controls.models,'build_model',lambda *_:Model())
    monkeypatch.setattr(controls.MaskedChannelStandardizer,'fit',lambda *_:pytest.fail('must not refit'))
    return key,report,metadata,paths,weights,save


def test_frozen_loader_restores_moments_and_uses_safe_checkpoint(tmp_path,monkeypatch):
    key,_,_,_,_,_ = frozen_fixture(tmp_path,monkeypatch)
    actual = torch.load
    calls = []
    def checked(*args,**kwargs):
        calls.append(kwargs)
        return actual(*args,**kwargs)
    monkeypatch.setattr(controls.torch,'load',checked)
    bundle = controls.load_frozen_control(tmp_path,key,torch.device('cpu'))
    assert np.array_equal(bundle['normalizer'].mean_,np.zeros(32))
    assert np.array_equal(bundle['normalizer'].scale_,np.ones(32))
    assert calls == [{'map_location':'cpu','weights_only':True}]


@pytest.mark.parametrize('mutation',['mode','protocol','smoke','moments','invalid_moments','weight','report','class_order'])
def test_frozen_loader_rejects_tampering(tmp_path,monkeypatch,mutation):
    key,report,metadata,paths,weights,save = frozen_fixture(tmp_path,monkeypatch)
    if mutation == 'mode': metadata['mode'] = 'noise_augmented'
    elif mutation == 'protocol':
        report['training_protocol_sha256'] = metadata['training_protocol_sha256'] = 'old'
    elif mutation == 'smoke': report['smoke_only'] = True
    elif mutation == 'moments': metadata['normalizer'] = {'mean':[2.0]*32,'scale':[1.0]*32}
    elif mutation == 'invalid_moments': metadata['normalizer']['scale'][0] = 0
    elif mutation == 'class_order': metadata['label_order'] = list(reversed(noise.LABELS))
    save()
    if mutation == 'weight': weights.write_bytes(b'altered')
    elif mutation == 'report': paths[0].write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError): controls.load_frozen_control(tmp_path,key,torch.device('cpu'))


def test_checkpoint_path_cannot_escape_owned_models(tmp_path):
    with pytest.raises(ValueError): controls.owned_path(tmp_path,'models','..','runs','weights.pt')
    with pytest.raises(ValueError): controls.load_frozen_control(tmp_path,'arbitrary_run','cpu')


@pytest.mark.parametrize('mutation',[None,'base_bytes','augmented_bytes','incomplete','split','sources'])
def test_training_cache_checks_current_original_and_train_only_artifacts(tmp_path,monkeypatch,mutation):
    dataset = 'psg-audio-pilot-10'
    policy = {'subjects':{'train':[f'train{i}' for i in range(6)],'validation':['validation'],'test':['test']}}
    sources,environment = {'protocol':'fixed'},{name:version(name) for name in ('numpy','scipy')}
    records,original,augmented,offset = [],[],[],0
    for i in range(6):
        identity,rows = f'train{i}',630 if i < 5 else 625
        path = f'data/processed/{dataset}/deep-inputs-v1/{identity}'
        provenance = {'input_sha256':'raw','source_artifact_sha256':'classic','source_metadata_sha256':'receipt'}
        folder = tmp_path/path
        write_json(folder/'metadata.json',{'artifact_sha256':'pending','rows':rows,'provenance':provenance})
        (folder/'inputs.npz').write_bytes(b'tiny original fixture')
        checksum = controls.sha256(folder/'inputs.npz')
        write_json(folder/'metadata.json',{'artifact_sha256':checksum,'rows':rows,'provenance':provenance})
        records.append({'record_id':identity,'path':path,'rows':rows,'artifact_sha256':checksum,'ok':True})
        original.append({'record_id':identity,'path':path,'rows':rows,'row_offset':offset,
                         'inputs_sha256':checksum,'metadata_sha256':controls.sha256(folder/'metadata.json')})
        offset += rows
        augpath = f'data/processed/{dataset}/{controls.augmentation.VERSION}/{identity}'
        augfolder = tmp_path/augpath
        augfolder.mkdir(parents=True)
        (augfolder/'inputs.npz').write_bytes(b'tiny augmentation fixture')
        checksum = controls.sha256(augfolder/'inputs.npz')
        write_json(augfolder/'metadata.json',{'artifact_sha256':checksum,'rows':rows,'provenance':{
            **provenance,'record_id':identity,'split':'train','source_hashes':sources,'environment':environment}})
        augmented.append({'record_id':identity,'path':augpath,'rows':rows,'artifact_sha256':checksum,
                          'metadata_sha256':controls.sha256(augfolder/'metadata.json'),'split':'train'})
    corpus = {'complete':True,'split_policy':policy,'dataset':dataset,'version':'deep-inputs-v1','records':records}
    write_json(tmp_path/f'data/processed/{dataset}/deep-inputs-v1/corpus.json',corpus)
    index = {'complete':True,'version':controls.augmentation.VERSION,'dataset':dataset,'dataset_version':'V3',
             'split_policy':policy,'records':augmented,'environment':environment,'source_hashes':sources}
    index_path = tmp_path/f'data/processed/{dataset}/{controls.augmentation.VERSION}/corpus.json'
    if mutation == 'incomplete': index['complete'] = False
    elif mutation == 'split': index['records'][0]['split'] = 'test'
    elif mutation == 'sources': index['source_hashes'] = {'protocol':'old'}
    write_json(index_path,index)
    report = {'split_policy':policy,'dataset':dataset,'dataset_version':'V3','input_version':'deep-inputs-v1',
              'original_provenance':original,'augmentation_index_sha256':controls.sha256(index_path),
              'source_hashes':sources,'augmentation_provenance':[
                  {key:item[key] for key in ('record_id','path','rows','artifact_sha256','metadata_sha256')}
                  for item in augmented]}
    monkeypatch.setattr(controls.augmentation,'_sources',lambda *_:sources)
    monkeypatch.setattr(controls.augmentation,'_select',lambda *_:({},corpus,records))
    monkeypatch.setattr(np,'load',lambda *_args,**_kwargs:pytest.fail('cache verification must not read arrays'))
    if mutation == 'base_bytes': (tmp_path/original[0]['path']/'inputs.npz').write_bytes(b'changed')
    elif mutation == 'augmented_bytes': (tmp_path/augmented[0]['path']/'inputs.npz').write_bytes(b'changed')
    if mutation is None: controls.verify_training_cache(tmp_path,report)
    else:
        with pytest.raises(ValueError): controls.verify_training_cache(tmp_path,report)


def test_extra_cnns_share_every_branch_and_unscaled_scoring(tmp_path,monkeypatch):
    labels = np.asarray(noise.LABELS)
    probabilities = np.asarray([[.5,.5],[.1,.9],[.9,.1]])
    arrays = {'y':labels[[0,1,0]],'supervised_mask':np.array([True,True,False]),
              'start_seconds':np.array([0,5,10]),'end_seconds':np.array([10,15,20])}
    images = np.zeros((3,32,100),np.float32)
    policy = {'subjects':{'train':['train'],'validation':['validation'],'test':['test']}}
    class Normalizer:
        def transform(self,value): return value
    class RF:
        classes_ = labels
        def predict_proba(self,_): return probabilities
    frozen_score = noise.classification_metrics(arrays['y'][:2],labels[[0,1]],noise.LABELS,
                                               probabilities[:2,1],noise.LABELS[1])
    report = {'split_policy':policy,'models':{'cnn':{'test':frozen_score}}}
    rf_report = {'split_policy':policy,'models':{'random_forest':{'test':frozen_score}}}
    extra = {name:{'model':name,'normalizer':Normalizer(),'report':report,'hashes':{'checkpoint':'hash'}}
             for name in controls.CONTROLS}
    calls = []
    def predict(model,features,device):
        calls.append((model,len(features)))
        return probabilities
    monkeypatch.setattr(noise,'predict_probabilities',predict)
    monkeypatch.setattr(noise,'load_frozen_models',lambda *_:(RF(),'cnn',Normalizer(),rf_report,report,{}))
    monkeypatch.setattr(noise.noise_features,'KINDS',('white',))
    monkeypatch.setattr(noise.noise_features,'SNRS',(0,))
    entry = {'record_id':'record','kind':'white','snr_db':0,'metadata_sha256':'hash','artifact_sha256':'hash'}
    index = {'split_policy':policy,'environment':{},'records':[entry],'dataset':'fixture',
             'dataset_version':'v1','protocol_sha256':'hash'}
    manifest = {'files':[{'record_id':'record','path':'fixture.rml'}]}
    provenance = {'duration_seconds':20,'offset_seconds':0,'input_sha256':'hash'}
    monkeypatch.setattr(noise,'validate_index',lambda *_:(index,manifest,[{'record_id':'record'}],
                                                             {'deep_cache':'hash','deep.inputs':'hash'},'hash'))
    monkeypatch.setattr(noise.noise_features,'_verified_context',lambda *_:(None,None,provenance,arrays,None))
    monkeypatch.setattr(noise,'ensure_dense',lambda *_:({'X':np.zeros((3,67))},None,None))
    deep = tmp_path/'data/processed/psg-audio-pilot-10/deep-inputs-v1/record'
    write_json(deep/'metadata.json',{'artifact_sha256':'hash','provenance':{
        'input_sha256':'hash','deep_cache_source_sha256':'hash','deep_inputs_source_sha256':'hash'}})
    np.savez(deep/'inputs.npz',values=images)
    branches = dict(arrays)
    for branch in noise.BRANCHES:
        branches['X_'+branch] = np.zeros((3,67))
        branches['images_'+branch] = images
    monkeypatch.setattr(noise,'verified_condition',lambda *_:(branches,{'provenance':{'calibration':{'common_gain':.8}}}))
    monkeypatch.setattr(noise,'read_reference',lambda *_:(None,None,[]))
    monkeypatch.setattr(noise,'raw_path',lambda *_:tmp_path/'unused')
    monkeypatch.setattr(noise,'clipped_references',lambda *_:([],[],None))
    monkeypatch.setattr(noise,'sha256',lambda *_:'hash')
    default = noise.evaluate(tmp_path,'default')
    calls.clear()
    extended = noise.evaluate(tmp_path,'extended',extra_cnns=extra)
    names = {'random_forest','cnn',*extra}
    assert set(extended['unscaled_reference']) == names
    assert set(extended['unscaled_classification_matches_frozen']) == names
    assert all(extended['unscaled_classification_matches_frozen'].values())
    assert len(calls) == 3*(1+len(noise.BRANCHES))
    assert all(rows == 3 for _,rows in calls)  # ambiguous window retained for events
    assert {item['branch'] for item in extended['conditions']} == set(noise.BRANCHES)
    for before,after in zip(default['conditions'],extended['conditions'],strict=True):
        assert set(after['models']) == names
        for name in ('random_forest','cnn'): assert before['models'][name] == after['models'][name]
        for score in after['models'].values():
            assert score['classification']['confusion_matrix'] == [[1,0],[0,1]]  # first class wins .5 tie
    assert 'augmented model yet' not in extended['model_policy']
    public = (tmp_path/'runs/extended/summary.json').read_text(encoding='utf-8')
    assert 'window_probabilities' not in public and 'reference_intervals' not in public
    local = json.loads((tmp_path/'runs/extended/events.json').read_text(encoding='utf-8'))
    assert all(len(item['window_probabilities']) == 3 for record in local['records'] for item in record['models'].values())
