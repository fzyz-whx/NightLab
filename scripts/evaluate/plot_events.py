"""Plot frozen event rates and mismatch categories from real recorded results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


def plot_events(metrics_path,failures_path,output):
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    failures = json.loads(failures_path.read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('figure output exists; refusing overwrite')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'], 'axes.unicode_minus':False,
                         'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig,axes = plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    x = np.arange(2)
    for axis,(key,label) in zip(axes[:2],(('f1','事件 F1'),('false_positives_per_hour','误报事件 / 观测小时')),strict=True):
        for index,(name,title) in enumerate((('majority','多数类'),('random_forest','随机森林'))):
            values = [metrics['models'][name][part][key] for part in ('validation','test')]
            axis.bar(x+(index-.5)*.32,values,.32,label=title,color=('#97a8b9','#225c80')[index])
            for position,value in zip(x+(index-.5)*.32,values,strict=True):
                if value is not None:
                    axis.annotate(f'{value:.3f}' if key == 'f1' else f'{value:g}',(position,value),
                                  xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        axis.set_xticks(x,['验证 2人 / 2小时','测试 2人 / 2小时'])
        axis.set_ylabel(label)
        axis.grid(axis='y',alpha=.15)
        axis.set_axisbelow(True)
        axis.set_ylim(0,1 if key == 'f1' else max(1,metrics['models']['random_forest']['test'][key]*1.25))
    axes[0].legend(frameon=False)
    categories = [('no_reference_overlap','无相交区间','#97a8b9'),
                  ('overlap_below_iou_threshold','有相交但 IoU < 0.5','#c78349'),
                  ('one_to_one_competition','一对一匹配竞争','#225c80')]
    bottom = np.zeros(2)
    test = failures['models']['random_forest']['test']
    for key,label,color in categories:
        values = np.array([test[kind][key] for kind in ('false_positives','false_negatives')])
        axes[2].bar(x,values,bottom=bottom,label=label,color=color,width=.5)
        for pos,value,base in zip(x,values,bottom,strict=True):
            if value:
                axes[2].text(pos,base+value/2,str(value),ha='center',va='center',color='white')
        bottom += values
    axes[2].set_xticks(x,['误报 FP','漏检 FN'])
    axes[2].set_ylabel('事件数量')
    axes[2].set_title('随机森林测试集失败类型')
    axes[2].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
    fig.suptitle('PSG-Audio 连续事件评价 · 固定阈值0.5 / 一对一 IoU≥0.5\n每集合仅2名受试者，非人群或医学验证',fontsize=14)
    for suffix in ('png','svg'):
        path = output/f'event-evaluation.{suffix}'
        fig.savefig(path,dpi=170)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (output/'figure-provenance.json').write_text(json.dumps({
        'metrics_path':metrics_path.relative_to(ROOT).as_posix(), 'metrics_sha256':sha256(metrics_path),
        'failures_path':failures_path.relative_to(ROOT).as_posix(), 'failures_sha256':sha256(failures_path),
        'plot_source_sha256':sha256(__file__), 'scope':'actual computed event metrics; no manually edited scores'},indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--failures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'artifacts'):
        parser.error('figure output must stay under artifacts')
    plot_events((ROOT/args.metrics).resolve(),(ROOT/args.failures).resolve(),output)
    print(json.dumps({'status':'plotted','output':args.output.as_posix()}))


if __name__ == '__main__':
    main()
