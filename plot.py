"""Regenerate a compact signed-network figure directly from numeric results."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',nargs='?',default='results/demo');a=parser.parse_args()
    p=Path(a.output);p=p if p.is_absolute() else ROOT/p
    z=np.load(p/'point.npz');c=json.loads((p/'manifest.json').read_text())['countries']
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,key,title in zip(axes,['B','peer','contextual'],['Contemporaneous B0','Lagged peer beta1 G','Contextual rho H']):
        matrix=z[key];v=max(float(abs(matrix).max()),1e-12)
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-v,vmax=v)
        ax.set_xticks(range(len(c)),c,rotation=90);ax.set_yticks(range(len(c)),c)
        ax.set_xlabel('Source');ax.set_ylabel('Receiver');ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('Ten-country NVAR: point estimates')
    fig.savefig(p/'networks.pdf');fig.savefig(p/'networks.png',dpi=160)

if __name__=='__main__':main()
