"""Render only real public aggregate v3 results; no private rows or fitted moments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.datasets.cli import ROOT


def run(root=ROOT):
    folder = root / 'artifacts/experiments/wearable31-v3'
    folder.mkdir(parents=True, exist_ok=True)
    reports = {(model, subset): json.loads((root / 'docs/results' / f'wearable31-{model}-{subset}-v3.json').read_text())
               for model in ('rf', 'cnn') for subset in ('hr', 'accel', 'fusion')}
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    xs = np.arange(3)
    for offset, model, label in [(-.18, 'rf', 'Random forest'), (.18, 'cnn', 'Small CNN')]:
        scores = [reports[model, subset]['models']['random_forest' if model == 'rf' else 'cnn']['test']['macro_f1']
                  for subset in ('hr', 'accel', 'fusion')]
        bars = ax.bar(xs + offset, scores, .34, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set(xticks=xs, xticklabels=['Heart rate', 'Acceleration', 'Fusion'], ylim=(0, .55),
           ylabel='Test Macro-F1 (fixed 5 stages)', title='SleepAccel31 v3: seven held-out test subjects')
    ax.legend()
    fig.savefig(folder / 'test-macro-f1.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for row, model in enumerate(('rf', 'cnn')):
        for col, subset in enumerate(('hr', 'accel', 'fusion')):
            report = reports[model, subset]
            metric = report['models']['random_forest' if model == 'rf' else 'cnn']['test']
            counts = np.array(metric['confusion_matrix'])
            normalized = counts / counts.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
            for y in range(5):
                for x in range(5):
                    ax.text(x, y, f'{normalized[y,x]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if normalized[y,x] > .6 else 'black')
            ax.set(xticks=range(5), yticks=range(5), xticklabels=metric['label_order'], yticklabels=metric['label_order'],
                   xlabel='Predicted', ylabel='Reference', title=f'{model.upper()} / {subset}')
    fig.suptitle('SleepAccel31 v3 test confusion matrices (row-normalized; counts in public JSON)')
    fig.savefig(folder / 'test-confusion-matrices.png', dpi=160)
    plt.close(fig)
    return folder


if __name__ == '__main__':
    print(run())
