"""Estimate fitted-signal ranks and run Algorithm 1."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,csv,hashlib,json,platform,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from data_io import load_panel
from rank_selection import prepare,fit_penalties
from mean_model import prep,fit_model_ii,companion_radius

MULTIPLIERS=[0.4,0.6,0.8,1.0,1.2,1.4,1.44,1.48,1.5,1.52,1.56,1.6,1.8,2.0,2.4,2.8,3.2]
SELECTED_MULTIPLIERS=(1.44,1.2)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='results/selected')
    args=p.parse_args();out=Path(args.output)
    if not out.is_absolute():out=ROOT/out
    if out.exists():p.error('Choose a new output directory')
    out.mkdir(parents=True)
    Y,X,countries,dates=load_panel()
    target,Ux,Uz,scale_g,scale_h=prepare(Y,X)
    rows=[];fits={}
    for multiplier_g in MULTIPLIERS:
        for multiplier_h in MULTIPLIERS:
            lambda_g=multiplier_g*scale_g;lambda_h=multiplier_h*scale_h
            z=fit_penalties(target,Ux,Uz,lambda_g,lambda_h)
            key=(multiplier_g,multiplier_h)
            fits[key]=z
            rows.append(dict(multiplier_g=multiplier_g,multiplier_h=multiplier_h,
                lambda_g=lambda_g,lambda_h=lambda_h,rank_g=z['rank_g'],rank_h=z['rank_h'],
                penalized_rank_g=z['penalized_rank_g'],penalized_rank_h=z['penalized_rank_h'],
                converged=z['converged'],iterations=z['iterations'],final_change=z['final_change'],
                objective=z['objective']))
    with (out/'penalty_grid.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected=next(r for r in rows if (r['multiplier_g'],r['multiplier_h'])==SELECTED_MULTIPLIERS)
    z=fits[SELECTED_MULTIPLIERS]
    if not z['converged']:
        raise RuntimeError('Rank-selection iteration did not converge')
    if (z['rank_g'],z['rank_h']) != (2,1) or (z['penalized_rank_g'],z['penalized_rank_h']) != (2,1):
        raise RuntimeError('Selected penalties did not produce rank (2,1)')
    n=len(countries)
    response,lags,covariates=prep(Y,X,1)
    fit=fit_model_ii(response,lags,covariates,2,1,1,max_iter=3000)
    if not fit['converged'] or fit['worst_increase']>1e-7:
        raise RuntimeError('Algorithm 1 did not meet its convergence check')
    rank_refit=(int(np.linalg.matrix_rank(fit['Pi_G'],tol=1e-8)),
                int(np.linalg.matrix_rank(fit['Pi_H'],tol=1e-8)))
    if rank_refit!=(2,1):raise RuntimeError(f'Algorithm 1 refit ranks are {rank_refit}')
    peer=fit['Pi_G']*fit['beta'][0]
    contextual=fit['Pi_H']*fit['rho'][0]
    fitted=peer@Y[:,:-1]+contextual@X[:,0,1:]+fit['common']
    loss=float(np.sum((response-fitted)**2)/response.shape[1])
    if not np.isclose(loss,fit['objective'],rtol=1e-10):raise RuntimeError('Objective replay failed')
    np.savez_compressed(out/'algorithm1_fit.npz',Pi_G=fit['Pi_G'],Pi_H=fit['Pi_H'],
        beta=fit['beta'],rho=fit['rho'],peer=peer,contextual=contextual,
        Lambda=fit['Lam'],F=fit['F'],common=fit['common'],fitted=fitted,
        rank_signal_G=z['fitted_g'],rank_signal_H=z['fitted_h'],rank_factor=z['factor'],
        partial_singular_G=z['singular_g'],partial_singular_H=z['singular_h'])
    manifest=dict(status='complete',selected_ranks=[z['rank_g'],z['rank_h']],
        penalized_fitted_ranks=[z['penalized_rank_g'],z['penalized_rank_h']],
        algorithm1_refit_ranks=list(rank_refit),penalty_multipliers=[selected['multiplier_g'],selected['multiplier_h']],
        penalties=[selected['lambda_g'],selected['lambda_h']],threshold_scale=[scale_g,scale_h],
        penalty_candidates=len(rows),
        rank_iterations=z['iterations'],rank_final_change=z['final_change'],
        rank_objective=z['objective'],mean_objective=fit['objective'],mean_iterations=fit['iterations'],
        mean_worst_block_increase=fit['worst_increase'],companion_radius=float(companion_radius(fit['Pi_G'],fit['beta'])),
        countries=countries,changes=Y.shape[1],responses=response.shape[1],period=[dates[0],dates[-1]],
        python=platform.python_version(),numpy=np.__version__,
        source_hashes={str(q.relative_to(ROOT)):sha(q) for q in sorted(ROOT.rglob('*.py'))},
        data_hashes={str(q.relative_to(ROOT)):sha(q) for q in sorted((ROOT/'data').glob('*.csv'))})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
