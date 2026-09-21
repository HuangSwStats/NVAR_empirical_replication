"""Independent replay checks of fitted products, constraints, generator, and basic intervals."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,csv,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
from data_io import load_panel
from estimation import generate
import structural as S

def main():
    a=argparse.ArgumentParser();a.add_argument('output',nargs='?',default='results/demo');args=a.parse_args()
    out=Path(args.output);out=out if out.is_absolute() else ROOT/out
    manifest=json.loads((out/'manifest.json').read_text())
    for group in ['source_hashes','data_hashes']:
        for name,digest in manifest[group].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    Y,X,C,dates=load_panel();p=dict(np.load(out/'point.npz'));N=len(C);B=p['B'];A=np.eye(N)-B
    order=p['order'];lower=B[np.ix_(order,order)]
    assert np.max(abs(np.triu(lower)))<1e-12
    assert np.linalg.norm(B,2)<=S.CAP+1e-7 and np.linalg.norm(B)<=S.FROB+1e-7
    assert S.TMIN<=p['tau']<=S.TMAX and 0<=p['omega']<=S.WMAX
    assert np.allclose(p['Lam']@p['F'].T,p['common'])
    assert np.allclose(p['F'].T@p['F']/p['F'].shape[0],np.eye(1))
    assert np.allclose(A@p['Pi_G']*p['beta'][0],p['peer'])
    assert np.allclose(A@p['Pi_H']*p['rho'][0],p['contextual'])
    assert np.linalg.matrix_rank(p['Pi_G'],tol=1e-8)<=2
    assert np.linalg.matrix_rank(p['Pi_H'],tol=1e-8)<=1
    assert np.allclose([np.linalg.norm(B),np.linalg.norm(p['peer']),np.linalg.norm(p['contextual'])],p['theta'][:3])
    inv=np.linalg.inv(A);Cstr=float(p['tau'])*inv@inv.T
    assert np.allclose(A@Cstr@A.T,float(p['tau'])*np.eye(N),atol=1e-10)
    base={'mean':p,**p}
    seed=manifest['seed']+100000;ys=generate(base,Y,X,seed)
    manual=np.zeros_like(Y);manual[:,0]=Y[:,0];rng=np.random.default_rng(seed)
    for t in range(1,Y.shape[1]):
        eta=rng.standard_normal(N)
        manual[:,t]=p['Pi_G']@manual[:,t-1]*p['beta'][0]+p['Pi_H']@X[:,0,t]*p['rho'][0]
        manual[:,t]+=np.sqrt(p['omega'])*p['common'][:,t-1]+np.sqrt(p['tau'])*inv@eta
    assert np.allclose(ys,manual,atol=1e-11)
    files=sorted(out.glob('draw_*.npz'));D=np.load(out/'bootstrap.npz')['draws']
    assert len(files)==len(D)==manifest['draws_requested']
    for f,row in zip(files,D):
        z=np.load(f);bo=z['B'];oo=z['order']
        assert np.allclose(z['theta'],row) and z['lambda0']==p['lambda0']
        assert np.max(abs(np.triu(bo[np.ix_(oo,oo)])))<1e-12
        assert np.linalg.norm(bo,2)<=S.CAP+1e-7 and np.linalg.norm(bo)<=S.FROB+1e-7
    if len(D)>=2:
        rows=list(csv.DictReader((out/'strengths.csv').open()))
        for j,row in enumerate(rows):
            expected=2*p['theta'][j]-np.quantile(D[:,j],[.975,.025])
            assert np.allclose(expected,[float(row['basic_lower']),float(row['basic_upper'])])
    # Check the two analytic covariance gradients independently by central differences.
    b=np.zeros((3,3));b[1,0]=.12;b[2,1]=-.08
    l=np.array([.4,.2,.3]);L=np.outer(l,l);ss=np.eye(3)+.3*np.ones((3,3))
    _,g=S.fro_objective(ss,L,b,.7,.9);errors=[]
    for i,j in [(1,0),(2,0),(2,1)]:
        plus=b.copy();minus=b.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
        numerical=(S.fro_objective(ss,L,plus,.7,.9)[0]-S.fro_objective(ss,L,minus,.7,.9)[0])/2e-6
        errors.append(abs(numerical-g[i,j]))
    assert max(errors)<1e-6
    print(json.dumps({'status':'PASS','draws_checked':len(D),'data_shape':list(Y.shape),
        'responses':Y.shape[1]-1,'structural_covariance_identity':True,
        'independent_recursive_generation_max_error':float(np.max(abs(ys-manual))),
        'frobenius_gradient_max_error':max(errors),'source_data_hashes_match':True},indent=2))

if __name__=='__main__':main()
