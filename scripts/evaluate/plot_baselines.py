"""Draw standalone scientific plots from recorded metrics, never edited scores."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.datasets.cli import ROOT


def plot_metrics(source, output):
    source,output = Path(source),Path(output)
    content = source.read_bytes()
    result = json.loads(content)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'], 'axes.unicode_minus':False,
                         'font.size':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    models = [('majority','多数类基线'),('random_forest','随机森林')]
    labels = result['models']['random_forest']['test']['label_order']
    display = {'no_target_reference_event':'无目标参考事件','reference_respiratory_event':'参考呼吸事件'}
    label_text = [display.get(label,label) for label in labels]
    test_subjects = len(result['split_subjects']['test'])
    wearable = result['dataset'].startswith('sleepaccel')
    unit = '个30秒epoch' if wearable else '个有明确标签的10秒窗（步长5秒）'
    subtitle = f"留出受试者 {test_subjects} 人 · {result['split_rows']['test']} {unit} · 试点结果"
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for axis,(key,title) in zip(axes,models):
        scores = result['models'][key]['test']
        counts = np.asarray(scores['confusion_matrix'])
        totals = counts.sum(axis=1,keepdims=True)
        fractions = np.divide(counts,totals,out=np.zeros_like(counts,dtype=float),where=totals>0)
        image = axis.imshow(fractions,vmin=0,vmax=1,cmap='Blues')
        axis.set_xticks(range(len(labels)),label_text,rotation=15,ha='right')
        axis.set_yticks(range(len(labels)),label_text)
        axis.set_xlabel('预测阶段' if wearable else '预测窗口类别');axis.set_ylabel('参考阶段' if wearable else '参考窗口类别')
        axis.set_title(f"{title}\nMacro-F1={scores['macro_f1']:.3f}")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column,row,f'{counts[row,column]}\n{fractions[row,column]:.1%}',ha='center',va='center',
                          color='white' if fractions[row,column]>.55 else '#172b4d')
    fig.colorbar(image,ax=axes,label='每个参考类别内的比例',shrink=.75)
    fig.suptitle(f'测试集混淆矩阵\n{subtitle}',fontsize=14)
    for suffix in ('png','svg'):
        fig.savefig(output/f'confusion-matrix.{suffix}',dpi=180)
        if suffix=='svg':
            path = output/f'confusion-matrix.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    metrics = [('macro_f1','Macro-F1'),('balanced_accuracy','平衡准确率')]
    if 'pr_auc_average_precision' in result['models']['random_forest']['test']:
        metrics.append(('pr_auc_average_precision','PR-AUC（平均精度）'))
    fig,axis = plt.subplots(figsize=(9,4.5),layout='constrained')
    positions = np.arange(len(metrics));width=.32
    for index,(key,title) in enumerate(models):
        values = [result['models'][key]['test'].get(name) for name,_ in metrics]
        for slot,value in enumerate(values):
            if value is None:
                continue
            x = positions[slot]+(index-.5)*width
            axis.bar(x,value,width,label=title if slot==0 else None,color=('#97a8b9','#225c80')[index])
            axis.text(x,value+.018,f'{value:.3f}',ha='center',fontsize=10)
    axis.set_xticks(positions,[title for _,title in metrics]);axis.set_ylim(0,1.06)
    axis.set_ylabel('分数（0–1）');axis.legend(frameon=False)
    axis.set_title(f'冻结参数后的测试集比较\n{subtitle}')
    axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    for suffix in ('png','svg'):
        fig.savefig(output/f'baseline-comparison.{suffix}',dpi=180)
        if suffix=='svg':
            path = output/f'baseline-comparison.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (output/'figure-provenance.json').write_text(json.dumps({'metrics_path':source.relative_to(ROOT).as_posix(),
        'metrics_sha256':hashlib.sha256(content).hexdigest(),'source':'actual recorded metrics; no manual score edits',
        'scope':result['experiment_scope']},indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'artifacts'):
        parser.error('figure output must stay under artifacts')
    plot_metrics(ROOT/args.metrics,output)
    print(json.dumps({'status':'plotted','output':args.output}))


if __name__ == '__main__':
    main()
