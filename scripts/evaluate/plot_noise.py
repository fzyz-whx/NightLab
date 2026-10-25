"""Plot every preregistered synthetic-noise branch from recorded measurements."""
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
        raise ValueError('figure output exists; refusing overwrite')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    branches = [('matched_clean','同增益原始参考','#7b8d9f','--'),
                ('matched_filtered_clean','同增益滤波参考','#54856e',':'),
                ('noisy','加噪','#c78349','-'),('filtered_noisy','加噪并滤波','#225c80','-')]
    kinds = [('white','合成白噪声'),('mains_hum','合成工频干扰')]
    models = [('random_forest','RF'),('cnn','CNN')]
    outputs = []
    for category,key,title,ylim in [('classification','macro_f1','窗口 Macro-F1',(0,1)),
                                   ('events','f1','事件 F1',(0,1)),
                                   ('events','false_positives_per_hour','误报事件 / 观测小时',None)]:
        fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for row,(model,label) in enumerate(models):
            for column,(kind,kind_label) in enumerate(kinds):
                axis = axes[row,column]
                for branch,branch_label,color,style in branches:
                    rows = sorted((item for item in data['conditions'] if item['kind'] == kind and item['branch'] == branch),
                                  key=lambda item:item['snr_db'])
                    axis.plot([item['snr_db'] for item in rows],[item['models'][model][category][key] for item in rows],
                              label=branch_label,color=color,linestyle=style,marker='o',markersize=4)
                axis.set_xticks([-5,0,5,10]);axis.set_xlabel('整段16k分析频带的新增干扰 SNR / dB')
                axis.set_ylabel(title);axis.set_title(f'{label} · {kind_label}');axis.grid(alpha=.15)
                if ylim: axis.set_ylim(*ylim)
                else: axis.set_ylim(bottom=0)
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
        fig.suptitle(f'固定模型的合成干扰对照 · {title}\n同两名测试受试者 / 两小时，不代表真实宿舍噪声或独立重复样本',fontsize=13)
        filename = {'macro_f1':'window-macro-f1','f1':'event-f1','false_positives_per_hour':'false-positives-per-hour'}[key]
        for suffix in ('png','svg'):
            path = output/f'{filename}.{suffix}'
            fig.savefig(path,dpi=170)
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
            outputs.append(path.name)
        plt.close(fig)
    (output/'figure-provenance.json').write_text(json.dumps({'metrics_path':source.relative_to(ROOT).as_posix(),
        'metrics_sha256':sha256(source),'plot_source_sha256':sha256(__file__),
        'scope':'actual recorded metrics; all preregistered branches and both fixed models','outputs':outputs},indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'artifacts'):
        parser.error('figure output must stay under artifacts')
    plot((ROOT/args.metrics).resolve(),output)
    print('plotted',args.output)


if __name__ == '__main__':
    main()
