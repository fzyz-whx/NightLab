"""Compare all frozen models under the same synthetic corruption and filter."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.datasets.cli import ROOT
from scripts.datasets.dense_audio import sha256


def plot(source,output):
    data = json.loads(source.read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('figure output exists')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    models = [('random_forest','RF 原始训练','#7b8d9f','x','--'),
              ('cnn','CNN 原始训练','#225c80','o',':'),
              ('cnn_clean_duplicate','CNN 原始重复对照','#c78349','s','--'),
              ('cnn_noise_augmented','CNN 噪声增强','#26796d','^','-')]
    images = []
    for branch,label in (('noisy','加噪'),('filtered_noisy','加噪并滤波')):
        fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for column,(kind,noise_name) in enumerate((('white','合成白噪声'),('mains_hum','合成工频'))):
            rows = sorted((item for item in data['conditions'] if item['kind'] == kind and item['branch'] == branch),
                          key=lambda item:item['snr_db'])
            for row,(category,metric,title) in enumerate((('classification','macro_f1','窗口 Macro-F1'),('events','f1','事件 F1'))):
                axis = axes[row,column]
                for key,name,color,marker,style in models:
                    axis.plot([item['snr_db'] for item in rows],[item['models'][key][category][metric] for item in rows],
                              label=name,color=color,marker=marker,linestyle=style,markersize=5,
                              markerfacecolor='none' if marker in ('o','s') else color)
                axis.set_xticks([-5,0,5,10]);axis.set_ylim(0,1)
                axis.set_xlabel('整段16k分析频带的新增干扰 SNR / dB');axis.set_ylabel(title)
                axis.set_title(f'{noise_name} · {title}');axis.grid(alpha=.15)
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
        fig.suptitle(f'原始训练、等样本量对照与噪声增强 · {label}\n同两名测试受试者 / 两小时，早停不同且不支持人群或医学结论',fontsize=13)
        for suffix in ('png','svg'):
            path = output/f'{branch}-model-comparison.{suffix}'
            fig.savefig(path,dpi=170)
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
            images.append(path.name)
        plt.close(fig)
    (output/'figure-provenance.json').write_text(json.dumps({'metrics_path':source.relative_to(ROOT).as_posix(),
        'metrics_sha256':sha256(source),'plot_source_sha256':sha256(__file__),
        'scope':'actual recorded metrics, every model and all fixed SNR values; no manual score edits',
        'outputs':images},indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    destination = (ROOT/args.output).resolve()
    if not destination.is_relative_to(ROOT/'artifacts'):
        parser.error('output must stay under artifacts')
    plot((ROOT/args.metrics).resolve(),destination)
    print('plotted',args.output)


if __name__ == '__main__':
    main()
