"""Build README figures from recorded metrics and saved development predictions."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from voc_detection.project import load_config,make_loader
from voc_detection.visualization import plot_category_examples


def main():
    output=ROOT/'reports/figures'
    history=pd.read_csv(ROOT/'reports/experiments.csv')
    baseline=history.iloc[:4]
    panels=[('Baseline experiments\nEvaluation: 320/640 pixels · 150 proposals',
             ['Epoch 1','Epoch 2','Epoch 3 control','Epoch 3, lower LR'],baseline.dev_map50.tolist()),
            ('GPU continuations\nEvaluation: 384/768 pixels · 300 proposals',
             ['Epoch 4, no flips','Epoch 4, flips','Epoch 5, mixed scale','Epoch 6, mixed scale','Epoch 7, fixed 384'],
             [json.loads((ROOT/'reports/optimization'/name/'summary.json').read_text())['map50']
              for name in ['epoch4_control_384','epoch4_flip_384','epoch5_multiscale_384','epoch6_multiscale_384','epoch7_fixed384']])]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for ax,(title,labels,values) in zip(axes,panels):
        for i,value in enumerate(values):
            ax.plot(value*100,i,'o',color='#168b77',markersize=9)
            ax.text(value*100+.12,i,f'{value:.2%}',va='center',fontsize=10)
        ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis()
        ax.set(xlim=(68,76),xlabel='Development mAP@0.5 (%) · zoomed axis',title=title)
        ax.grid(axis='x',color='#e1e7e5')
        ax.set_axisbelow(True)
    fig.savefig(output/'experiment_comparison.png',dpi=150)
    plt.close(fig)
    records=torch.load(ROOT/'reports/evaluation/dev/predictions.pt',map_location='cpu',weights_only=True)
    dataset,_=make_loader(load_config(),'dev')
    fig=plot_category_examples(records,dataset,'bottle',threshold=.5,limit=1)
    fig.savefig(output/'readme_error_example.png',dpi=120)
    plt.close(fig)
    print('Generated experiment comparison and one ground-truth/prediction example.')


if __name__=='__main__':main()
