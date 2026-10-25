"""Draw pilot CNN comparisons and a failure matrix from recorded public metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


def plot(output):
    files = {name:ROOT/f'docs/results/{name}.json' for name in (
        'audio-cnn-pilot-v1','wearable-cnn-fusion-v1','wearable-cnn-hr-v1','wearable-cnn-accel-v1',
        'psg-rf-pilot-v1','sleepaccel-rf-pilot-v2','sleepaccel-hr-pilot-v2','sleepaccel-accel-pilot-v2')}
    data = {name:json.loads(path.read_text(encoding='utf-8')) for name,path in files.items()}
    if output.exists():
        raise ValueError('figure output exists')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12,9),layout='constrained')
    audio = data['audio-cnn-pilot-v1']
    history = audio['history']
    axis = axes[0,0]
    axis.plot([item['epoch'] for item in history],[item['validation_macro_f1'] for item in history],
              color='#225c80',marker='o',label='CNN 验证 Macro-F1')
    axis.axvline(audio['best_epoch'],color='#c78349',linestyle='--',label=f"选定 epoch {audio['best_epoch']}")
    axis.set_xlabel('训练 epoch');axis.set_ylabel('验证 Macro-F1');axis.set_ylim(0,1)
    axis.set_title('音频：只由验证集选检查点');axis.legend(frameon=False,fontsize=9)
    axis.grid(alpha=.15)
    axis = axes[1,0]
    positions = np.arange(2)
    for index,(name,model,label) in enumerate((('psg-rf-pilot-v1','random_forest','RF'),('audio-cnn-pilot-v1','cnn','CNN'))):
        scores = data[name]['models'][model]['test']
        values = [scores[key] for key in ('macro_f1','pr_auc_average_precision')]
        x = positions+(index-.5)*.32
        axis.bar(x,values,.32,label=label,color=('#97a8b9','#225c80')[index])
        for pos,value in zip(x,values,strict=True):
            axis.text(pos,value+.015,f'{value:.3f}',ha='center',fontsize=10)
    axis.set_xticks(positions,['Macro-F1','AP']);axis.set_ylim(0,1)
    axis.set_title('音频测试：2人 / 明确标签窗口');axis.legend(frameon=False)
    axis.set_ylabel('分数');axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    axis = axes[0,1]
    positions = np.arange(3)
    for index,(model,names,label) in enumerate((
        ('random_forest',('sleepaccel-hr-pilot-v2','sleepaccel-accel-pilot-v2','sleepaccel-rf-pilot-v2'),'RF'),
        ('cnn',('wearable-cnn-hr-v1','wearable-cnn-accel-v1','wearable-cnn-fusion-v1'),'CNN'))):
        values = [data[name]['models'][model]['test']['macro_f1'] for name in names]
        x = positions+(index-.5)*.32
        axis.bar(x,values,.32,label=label,color=('#97a8b9','#225c80')[index])
        for pos,value in zip(x,values,strict=True):
            axis.text(pos,value+.015,f'{value:.3f}',ha='center',fontsize=10)
    axis.set_xticks(positions,['仅心率','仅加速度','融合']);axis.set_ylim(0,1)
    axis.set_ylabel('测试 Macro-F1');axis.set_title('可穿戴输入消融：测试仅1人')
    axis.legend(frameon=False);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    axis = axes[1,1]
    scores = data['wearable-cnn-fusion-v1']['models']['cnn']['test']
    counts = np.asarray(scores['confusion_matrix'])
    fractions = counts/counts.sum(axis=1,keepdims=True)
    chart = axis.imshow(fractions,vmin=0,vmax=1,cmap='Blues')
    labels = scores['label_order']
    axis.set_xticks(range(5),labels);axis.set_yticks(range(5),labels)
    axis.set_xlabel('预测阶段');axis.set_ylabel('参考阶段')
    axis.set_title('可穿戴融合 CNN：N3未检出')
    for row in range(5):
        for column in range(5):
            axis.text(column,row,str(counts[row,column]),ha='center',va='center',
                      color='white' if fractions[row,column]>.55 else '#172b4d')
    fig.colorbar(chart,ax=axis,shrink=.8,label='每参考类别内比例')
    fig.suptitle('固定协议的轻量 CNN 试点 · 单种子42\n数据量小、受试者差异大，不支持人群或医学结论',fontsize=15)
    for suffix in ('png','svg'):
        path = output/f'cnn-comparison.{suffix}'
        fig.savefig(path,dpi=170)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (output/'figure-provenance.json').write_text(json.dumps({
        'sources':{path.relative_to(ROOT).as_posix():sha256(path) for path in files.values()},
        'plot_source_sha256':sha256(__file__),'scope':'actual recorded metrics; no manual score edits'},indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'artifacts'):
        parser.error('output must stay under artifacts')
    plot(output)
    print('plotted',args.output)


if __name__ == '__main__':
    main()
