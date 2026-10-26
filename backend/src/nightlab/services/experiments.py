"""Read only registered aggregate research reports and their own PNG figures."""
import json
from pathlib import Path


CATALOG = {
    'audio_baseline':('声音传统基线','参考呼吸事件窗口与连续定位，十人试点','audio','psg-rf-pilot-v1','psg-rf-pilot-v1'),
    'wearable_baseline':('可穿戴传统基线','HR与加速度五阶段，三人试点','wearable','sleepaccel-rf-pilot-v2','sleepaccel-rf-pilot-v2'),
    'audio_cnn':('声音轻量CNN','随机初始化CNN及负面结果','audio','audio-cnn-pilot-v1','cnn-pilot-v1'),
    'wearable_cnn':('可穿戴轻量CNN','输入消融与N3漏检，三人试点','wearable','wearable-cnn-fusion-v1','cnn-pilot-v1'),
    'noise_frozen':('合成干扰与传统滤波','全部固定SNR、四分支、冻结RF/CNN','noise','noise-frozen-models-v1','noise-frozen-models-v1'),
    'noise_augmentation':('噪声增强与训练量控制','等样本量对照与四模型完整比较','noise','noise-augmentation-comparison-v1','noise-augmentation-comparison-v1'),
    'wearable31_rf':('可穿戴31人RF','独立18/6/7划分，融合输入与真实五阶段结果','wearable','wearable31-rf-fusion-v3','wearable31-v3'),
    'wearable31_cnn':('可穿戴31人CNN','相同划分与输入消融，三个CNN低于对应RF','wearable','wearable31-cnn-fusion-v3','wearable31-v3'),
}


class ExperimentService:
    def __init__(self,project_root):
        self.root = Path(project_root).resolve()

    def _owned(self,path,area):
        allowed = self.root/area
        resolved = path.resolve()
        if allowed.resolve()!=allowed or not resolved.is_relative_to(allowed):
            raise ValueError('实验位置越出受控目录')
        return resolved

    def detail(self,experiment_id):
        if experiment_id not in CATALOG:
            raise KeyError(experiment_id)
        title,description,category,report_name,figure_dir = CATALOG[experiment_id]
        path = self._owned(self.root/f'docs/results/{report_name}.json','docs/results')
        report = json.loads(path.read_text(encoding='utf-8'))
        directory = self._owned(self.root/f'artifacts/experiments/{figure_dir}','artifacts/experiments')
        figures = [{'name':image.name,'url':f'/api/experiments/{experiment_id}/figures/{image.name}'}
                   for image in sorted(directory.glob('*.png')) if self._owned(image,'artifacts/experiments').is_file()]
        return {'id':experiment_id,'title':title,'description':description,'category':category,
                'dataset':report.get('dataset','公开数据试点'),'report':report,'figures':figures}

    def list(self):
        items = []
        for key in CATALOG:
            try:
                detail = self.detail(key)
                items.append({name:value for name,value in detail.items() if name!='report'})
            except FileNotFoundError:
                continue
        return items

    def figure(self,experiment_id,name):
        detail = self.detail(experiment_id)
        if name not in {item['name'] for item in detail['figures']}:
            raise KeyError(name)
        directory = CATALOG[experiment_id][4]
        return self._owned(self.root/f'artifacts/experiments/{directory}/{name}','artifacts/experiments')
