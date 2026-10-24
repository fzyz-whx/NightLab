"""Plot the predeclared wearable modality comparison from saved runs."""
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.datasets.cli import ROOT


def main():
    sources = [('sleepaccel-hr-pilot-v2','仅心率'),('sleepaccel-accel-pilot-v2','仅加速度'),('sleepaccel-rf-pilot-v2','心率+加速度')]
    records = [json.loads((ROOT/f'docs/results/{run}.json').read_text(encoding='utf-8')) for run,_ in sources]
    if any(record['split_subjects']!=records[0]['split_subjects'] or record['config']!=records[0]['config'] for record in records):
        raise ValueError('ablation requires identical subject split and model parameters')
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    figure,axis = plt.subplots(figsize=(9,4.5),layout='constrained')
    positions = np.arange(3)
    for offset,key,title,color in [(-.17,'macro_f1','Macro-F1','#225c80'),(.17,'balanced_accuracy','平衡准确率','#98a9b8')]:
        values = [record['models']['random_forest']['test'][key] for record in records]
        axis.bar(positions+offset,values,.34,label=title,color=color)
        for x,y in zip(positions+offset,values):axis.text(x,y+.012,f'{y:.3f}',ha='center',fontsize=10)
    axis.set_xticks(positions,[title for _,title in sources]);axis.set_ylim(0,1)
    axis.set_ylabel('分数（0–1）');axis.set_title('可穿戴输入消融：冻结参数的测试结果\n3人试点 · 每集合仅1人 · 相同划分与随机森林参数')
    axis.legend(frameon=False);axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    output = ROOT/'artifacts/experiments/sleepaccel-ablation-v2';output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg']:
        figure.savefig(output/f'modality-ablation.{suffix}',dpi=180)
        if suffix=='svg':
            path = output/f'modality-ablation.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(figure)
    (output/'figure-provenance.json').write_text(json.dumps({'metrics_sha256':{run:hashlib.sha256((ROOT/f'docs/results/{run}.json').read_bytes()).hexdigest() for run,_ in sources},'scope':'three-subject pilot only'},indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
