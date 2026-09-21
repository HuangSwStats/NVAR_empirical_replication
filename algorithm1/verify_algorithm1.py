"""Replay the two selected ranks, threshold rule and Algorithm 1 objective."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from data_io import load_panel
from rank_selection import prepare,fit_penalties
from mean_model import prep

def main():
    p=argparse.ArgumentParser();p.add_argument('output',nargs='?',default='results/selected');a=p.parse_args()
    out=Path(a.output);out=out if out.is_absolute() else ROOT/out
    m=json.loads((out/'manifest.json').read_text())
    for group in ['source_hashes','data_hashes']:
        for name,digest in m[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    Y,X,countries,_=load_panel();target,Ux,Uz,_,_=prepare(Y,X)
    z=fit_penalties(target,Ux,Uz,*m['penalties'])
    assert z['converged'] and (z['rank_g'],z['rank_h'])==(2,1)
    assert (z['penalized_rank_g'],z['penalized_rank_h'])==(2,1)
    assert np.count_nonzero(z['singular_g']>m['penalties'][0])==2
    assert np.count_nonzero(z['singular_h']>m['penalties'][1])==1
    f=np.load(out/'algorithm1_fit.npz');response,_,_=prep(Y,X,1)
    assert np.linalg.matrix_rank(f['Pi_G'],tol=1e-8)==2
    assert np.linalg.matrix_rank(f['Pi_H'],tol=1e-8)==1
    assert np.allclose(f['F'].T@f['F']/len(f['F']),np.eye(1),atol=1e-10)
    assert np.allclose(f['Pi_G']*f['beta'][0],f['peer'])
    assert np.allclose(f['Pi_H']*f['rho'][0],f['contextual'])
    reconstructed=f['peer']@Y[:,:-1]+f['contextual']@X[:,0,1:]+f['Lambda']@f['F'].T
    assert np.allclose(reconstructed,f['fitted'],atol=1e-10)
    loss=float(np.sum((response-f['fitted'])**2)/response.shape[1])
    assert np.isclose(loss,m['mean_objective'],rtol=1e-10)
    print(json.dumps(dict(status='PASS',ranks=[2,1],penalties=m['penalties'],
        threshold_singular_G=z['singular_g'][:4].tolist(),
        threshold_singular_H=z['singular_h'][:3].tolist(),
        mean_objective=loss,source_data_hashes_match=True),indent=2))

if __name__=='__main__':main()
