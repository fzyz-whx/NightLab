"""Local, receipt-anchored audio models. Never accept uploaded pickle/weights."""
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import re

import numpy as np

from nightlab.audio.features import AudioFeatureExtractor


LABELS = ['no_target_reference_event','reference_respiratory_event']
LOCAL_PREFIX = 'local_rf_'
LOCAL_DEPENDENCIES = ('nightlab','numpy','scipy','scikit-learn','joblib')
LOCAL_SOURCES = ('scripts.train.app_audio_rf','scripts.train.baselines',
                 'nightlab.experiments.baseline','nightlab.audio.features')


def local_run_name(name):
    if (not isinstance(name,str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',name)
            or name.lower() in {'con','prn','aux','nul',*[f'com{i}' for i in range(1,10)],
                                *[f'lpt{i}' for i in range(1,10)]}):
        raise ValueError('run name must be a safe single portable name (1-64 characters)')
    return name


def controlled_path(root,*parts):
    """Reject links/junctions even when their destinations remain inside root."""
    root = Path(root).resolve()
    path = root
    for part in parts:
        for component in Path(part).parts:
            if component in {'.','..'} or Path(component).is_absolute() or ':' in component:
                raise ValueError('模型位置不在受控目录')
            path = path/component
            if path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()):
                raise ValueError('模型路径不允许链接或目录联接')
    if not path.resolve().is_relative_to(root):
        raise ValueError('模型位置不在受控目录')
    return path


def local_source_hashes():
    return {name:digest(find_spec(name).origin) for name in LOCAL_SOURCES}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class ModelRegistry:
    def __init__(self,project_root,num_threads=2):
        self.root = Path(project_root).resolve()
        self.num_threads = min(2,max(1,int(num_threads)))
        self._loaded = {}

    def _path(self,area,*parts):
        return controlled_path(self.root,area,*parts)

    @staticmethod
    def _read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def _proof(self,model_id):
        if isinstance(model_id,str) and model_id.startswith(LOCAL_PREFIX):
            return self._local_proof(local_run_name(model_id[len(LOCAL_PREFIX):]))
        if model_id == 'audio_rf_pilot_v1':
            receipt = self._read(self._path('docs','results','baseline-reproducibility.json'))
            checks = [item for item in receipt['checks'] if item['run']=='psg-rf-pilot-v1'
                      and item['owned_saved_model_predictions_match_retrained_models']]
            if len(checks)!=1:
                raise ValueError('缺少自有模型核验记录')
            path = self._path('models','psg-rf-pilot-v1','random_forest.joblib')
            report = self._read(self._path('docs','results','psg-rf-pilot-v1.json'))
            if (report['feature_count']!=67 or report['feature_names']!=list(AudioFeatureExtractor().feature_names)
                    or report['environment']['scikit-learn']!=version('scikit-learn')):
                raise ValueError('模型与特征/依赖版本不匹配')
            sha = checks[0]['local_model_sha256']['random_forest']
            metadata = report
        elif model_id == 'audio_cnn_augmented_v1':
            if find_spec('torch') is None:
                raise ValueError('尚未安装可选神经模型依赖')
            public = self._read(self._path('docs','results','audio-noise-augmented-v1.json'))
            run = 'audio-noise-augmented-v1-repro'
            if public['run_name']!=run or public['mode']!='noise_augmented':
                raise ValueError('神经模型版本不匹配')
            report_path = self._path('runs',run,'metrics.json')
            if digest(report_path)!=public['metrics_source_sha256']:
                raise ValueError('神经模型训练记录校验失败')
            report = self._read(report_path)
            metadata = self._read(self._path('models',run,'metadata.json'))
            if (metadata['normalizer']!=report['normalizer'] or metadata['label_order']!=LABELS
                    or metadata['checkpoint_sha256']!=public['checkpoint_sha256']
                    or not report['saved_loaded_predictions_exact'] or report['smoke_only']):
                raise ValueError('神经模型参数收据不匹配')
            for name,module_name in (('models','nightlab.deep.models'),('inputs','nightlab.deep.inputs'),('training','nightlab.deep.training')):
                if digest(find_spec(module_name).origin)!=report['source_sha256'][name]:
                    raise ValueError('神经模型实现版本发生变化')
            if report['environment']['torch']!=version('torch'):
                raise ValueError('神经模型依赖版本不匹配')
            path = self._path('models',run,'weights.pt')
            sha = public['checkpoint_sha256']
        else:
            raise ValueError('未知的本地模型')
        if digest(path)!=sha:
            raise ValueError('模型文件校验失败，拒绝加载')
        return path,sha,metadata

    def _local_proof(self,run):
        receipt = self._read(controlled_path(self.root,'private','app-models',run+'.json'))
        keys = {'schema_version','kind','run_name','model_id','dataset','model_sha256',
                'metrics_sha256','feature_names','labels','environment','source_sha256'}
        if (not isinstance(receipt,dict) or set(receipt)!=keys or type(receipt['schema_version']) is not int
                or receipt['schema_version']!=1 or receipt['kind']!='owned_audio_rf'
                or receipt['run_name']!=run or receipt['model_id']!=LOCAL_PREFIX+run
                or receipt['dataset']!='psg-audio-pilot-10'
                or receipt['feature_names']!=list(AudioFeatureExtractor().feature_names)
                or receipt['labels']!=LABELS
                or receipt['source_sha256']!=local_source_hashes()
                or receipt['environment']!={name:version(name) for name in LOCAL_DEPENDENCIES}):
            raise ValueError('自有模型收据版本、特征或依赖不匹配')
        for field in ('model_sha256','metrics_sha256'):
            if not isinstance(receipt[field],str) or not re.fullmatch('[0-9a-f]{64}',receipt[field]):
                raise ValueError('自有模型收据摘要无效')
        metrics_path = controlled_path(self.root,'runs',run,'metrics.json')
        path = controlled_path(self.root,'models',run,'random_forest.joblib')
        if digest(metrics_path)!=receipt['metrics_sha256'] or digest(path)!=receipt['model_sha256']:
            raise ValueError('模型文件或训练记录校验失败，拒绝加载')
        report = self._read(metrics_path)
        if (report['dataset']!=receipt['dataset'] or report['subject_count']!=10
                or report['feature_subset']!='all' or report['feature_count']!=67
                or report['feature_names']!=receipt['feature_names']
                or any(report['environment'][name]!=receipt['environment'][name] for name in LOCAL_DEPENDENCIES)
                or any(report['models']['random_forest'][split]['label_order']!=LABELS
                       for split in ('validation','test'))):
            raise ValueError('自有模型训练记录与收据不匹配')
        return path,receipt['model_sha256'],report

    def list_models(self):
        result = []
        entries = [('audio_rf_pilot_v1','声音传统基线'),('audio_cnn_augmented_v1','声音增强CNN（实验）')]
        try:
            folder = controlled_path(self.root,'private','app-models')
            for receipt in sorted(folder.glob('*.json')):
                try:
                    run = local_run_name(receipt.stem)
                    entries.append((LOCAL_PREFIX+run,'本机训练 RF · '+run))
                except ValueError:
                    continue
        except (OSError,ValueError):
            pass
        for model_id,name in entries:
            try:
                _,sha,_ = self._proof(model_id)
                available,reason = True,None
            except (OSError,ValueError,KeyError,TypeError,AttributeError):
                available,reason,sha = False,'本地模型尚未准备或校验未通过',None
            result.append({'id':model_id,'name':name,'kind':'audio','available':available,'reason':reason,
                           'version':sha[:12] if sha else None,
                           'scope':'公开PSG十人试点；个人手机泛化未验证，输出为模型标记，非诊断'})
        return result

    def require(self,model_id):
        _,sha,_ = self._proof(model_id)
        return sha[:12]

    def predict_windows(self,model_id,waves,source_rate):
        path,sha,metadata = self._proof(model_id)
        if model_id=='audio_rf_pilot_v1' or model_id.startswith(LOCAL_PREFIX):
            import joblib
            if sha not in self._loaded:
                model = joblib.load(path)  # only this owned digest, verified before deserialization
                if list(model.classes_)!=LABELS or model.n_features_in_!=67:
                    raise ValueError('模型类别或输入维度不正确')
                self._loaded[sha] = model
            extractor = AudioFeatureExtractor()
            x = np.stack([extractor.transform(wave,source_rate) for wave in waves])
            return self._loaded[sha].predict_proba(x)[:,1]
        import torch
        from nightlab.deep.models import build_model
        from nightlab.deep.inputs import MaskedChannelStandardizer,log_mel_image
        torch.set_num_threads(self.num_threads)
        if sha not in self._loaded:
            model = build_model('audio').to('cpu')
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            model.eval()
            normalizer = MaskedChannelStandardizer()
            normalizer.mean_ = np.asarray(metadata['normalizer']['mean'],dtype=float)
            normalizer.scale_ = np.asarray(metadata['normalizer']['scale'],dtype=float)
            if (normalizer.mean_.shape!=(32,) or normalizer.scale_.shape!=(32,)
                    or not np.isfinite(normalizer.mean_).all() or not np.isfinite(normalizer.scale_).all()
                    or np.any(normalizer.scale_<=0)):
                raise ValueError('模型标准化参数无效')
            self._loaded[sha] = (model,normalizer)
        model,normalizer = self._loaded[sha]
        x = normalizer.transform(np.stack([log_mel_image(wave,source_rate) for wave in waves]))[:,None,:,:]
        with torch.inference_mode():
            return torch.softmax(model(torch.from_numpy(x)),dim=1)[:,1].numpy()
