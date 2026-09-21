"""Standalone runner. Each run needs a fresh output directory; never consumes old caches."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, argparse, json, hashlib, platform, time, csv, traceback
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import scipy
from data_io import load_panel
from estimation import fit, generate, GRID

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_fit(path, z):
    f=z['mean']
    np.savez_compressed(path, B=z['B'], G=z['G'], H=z['H'], G0=z['G0'],
        peer=z['peer'], contextual=z['contextual'], theta=z['theta'],
        omega=z['omega'], tau=z['tau'], lambda0=z['lambda0'], order=z['order'],
        Pi_G=f['Pi_G'], Pi_H=f['Pi_H'], beta=f['beta'], rho=f['rho'],
        common=f['common'], Lam=f['Lam'], F=f['F'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/run')
    parser.add_argument('--draws', type=int, default=20)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--random-orders', type=int, default=256)
    args=parser.parse_args()
    if args.draws<0 or args.random_orders<1: parser.error('draws >= 0 and random-orders >= 1 required')
    out=Path(args.output)
    if not out.is_absolute(): out=ROOT/out
    if out.exists(): parser.error('Output exists. Choose a NEW directory to prevent accidental cache reuse.')
    out.mkdir(parents=True)
    Y,X,C,dates=load_panel()
    config=dict(p=1,rG=2,rH=1,rf=1,rank_selection='fixed by design',seed=args.seed,
        draws_requested=args.draws,random_orders=args.random_orders,lambda_grid=GRID,
        bootstrap='structural Gaussian; sqrt(omega) common; fixed original-sample lambda',
        countries=C,level_period=['2000-01','2024-06'],change_period=[dates[0],dates[-1]],
        N=Y.shape[0],changes=Y.shape[1],responses=Y.shape[1]-1,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*.py'))},
        data_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'data').glob('*')) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.time()
    base=fit(Y,X,seed=args.seed,random_orders=args.random_orders)
    if base['radius']>=1: raise RuntimeError('Unstable fitted recursion; refusing bootstrap generation')
    save_fit(out/'point.npz',base)
    fields=['lambda0','edges','bic','penalized_converged','penalized_iterations','objective_min','objective_max']
    with (out/'selection_path.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+['qml_success','qml_feasible']);w.writeheader()
        for a in base['path']: w.writerow({**{k:a[k] for k in fields},'qml_success':a['fit']['success'],'qml_feasible':a['fit']['feasible']})
    print('POINT',base['theta'][:3], 'lambda',base['lambda0'],'seconds',round(time.time()-start,2),flush=True)
    draws=[]; diagnostics=[]; failures=[]
    for b in range(args.draws):
        try:
            Ys=generate(base,Y,X,args.seed+100000+b)
            z=fit(Ys,X,seed=args.seed+200000+b,random_orders=args.random_orders,fixed_lambda=base['lambda0'])
            assert len(z['path'])==1 and z['lambda0']==base['lambda0']
            draws.append(z['theta']);save_fit(out/f'draw_{b:04d}.npz',z)
            diagnostics.append(dict(draw=b,generation_seed=args.seed+100000+b,fit_seed=args.seed+200000+b,
                norm_boundary=z['norm_boundary'],penalized_converged=z['penalized_converged'],
                mean_iterations=z['mean']['iterations'],orders_retained=z['orders_retained']))
        except Exception:
            failures.append(b);(out/f'failure_{b:04d}.txt').write_text(traceback.format_exc())
        print('DRAW',b+1,'/',args.draws,'successes',len(draws),'seconds',round(time.time()-start,2),flush=True)
    width=len(base['theta']);D=np.asarray(draws).reshape(-1,width)
    np.savez_compressed(out/'bootstrap.npz',draws=D,theta=base['theta'])
    (out/'bootstrap_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    names=['beta0','beta1','rho']
    with (out/'strengths.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['target','estimate','basic_lower','basic_upper','bootstrap_sd'])
        for j,name in enumerate(names):
            if len(D)>=2 and not failures:
                lo,hi=2*base['theta'][j]-np.quantile(D[:,j],[.975,.025])
                w.writerow([name,base['theta'][j],lo,hi,D[:,j].std(ddof=1)])
            else:w.writerow([name,base['theta'][j],'','',''])
    summary=dict(status='complete' if not failures else 'failed_draws',successful_draws=len(D),failed_draws=failures,
        strengths=base['theta'][:3].tolist(),selected_lambda=base['lambda0'],radius=base['radius'],
        selected_edges=int(base['path'][np.argmin([p['bic'] if p['fit']['success'] and p['fit']['feasible'] else np.inf for p in base['path']])]['edges']),
        order=[C[i] for i in base['order']],point_norm_boundary=base['norm_boundary'],
        point_penalized_converged=base['penalized_converged'],mean_iterations=base['mean']['iterations'],
        seconds=time.time()-start,interval_use='demonstration only; small draw count' if args.draws<300 else 'Monte Carlo intervals; theorem assumptions not verified')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if failures: raise SystemExit('Failed draws recorded. No confidence intervals reported; inspect failures.')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
