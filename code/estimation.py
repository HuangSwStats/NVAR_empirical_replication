"""One-factor, one-lag empirical estimation with fixed ranks (2,1)."""
import numpy as np
import mean_model as M
import structural as S

GRID = [0., .01, .02, .04, .06, .08, .10, .12, .15, .20, .30]

def fit(Y, X, seed=0, random_orders=256, fixed_lambda=None):
    Ye, YL, XN = M.prep(Y, X, 1)
    f = M.fit_model_ii(Ye, YL, XN, 2, 1, 1, max_iter=3000)
    if not f['converged'] or f['worst_increase'] > 1e-7:
        raise RuntimeError('Mean solver did not converge monotonically')
    # Positive reference-product convention; fitted products are unchanged.
    for matrix, coef in [('Pi_G', 'beta'), ('Pi_H', 'rho')]:
        if f[coef][0] < 0:
            f[matrix] = -f[matrix]
            f[coef] = -f[coef]
    residual = Ye - f['Pi_G'] @ np.einsum('tip,p->it', YL, f['beta'])
    residual -= f['Pi_H'] @ np.einsum('tid,d->it', XN, f['rho'])
    covariance = residual @ residual.T / Ye.shape[1]
    loading_outer = f['Lam'] @ f['Lam'].T
    orders = S.order_candidates(covariance, loading_outer, seed, random_orders)
    grid = GRID if fixed_lambda is None else [float(fixed_lambda)]
    candidates = {lam: [] for lam in grid}
    for screening_score, order, initial in orders:
        B = initial
        for lam in grid:
            z = S.proximal_fit(covariance, loading_outer, order, lam, B, maxiter=1200)
            B = z['B']
            candidates[lam].append((z, order, screening_score))
    path = []
    for lam in grid:
        # Approximate min of the penalized criterion over screened orders.
        z, order, screening_score = min(candidates[lam], key=lambda t: t[0]['objective'])
        support = abs(z['B']) > S.THRESHOLD
        q = S.qml_fit(covariance, loading_outer, support, z)
        objectives = [a[0]['objective'] for a in candidates[lam]]
        path.append(dict(lambda0=float(lam), edges=int(support.sum()),
                         bic=float(Ye.shape[1]*q['q']+(support.sum()+2)*np.log(Ye.shape[1])),
                         fit=q, order=order, support=support,
                         penalized_converged=bool(z['converged']),
                         penalized_iterations=int(z['iterations']),
                         objective_min=float(min(objectives)), objective_max=float(max(objectives)),
                         screening_score=float(screening_score)))
    valid = [p for p in path if p['fit']['success'] and p['fit']['feasible']]
    if not valid:
        raise RuntimeError('No converged feasible covariance QML refit')
    chosen = min(valid, key=lambda p: p['bic'])
    q = chosen['fit']; B = q['B']; A = np.eye(len(B))-B
    peer = A @ f['Pi_G'] * f['beta'][0]
    contextual = A @ f['Pi_H'] * f['rho'][0]
    strengths = np.array([np.linalg.norm(B), np.linalg.norm(peer), np.linalg.norm(contextual)])
    if np.any(strengths[1:] < 1e-12):
        raise RuntimeError('Degenerate structural reporting map')
    theta = np.r_[strengths, peer.ravel(), contextual.ravel(), B.ravel()]
    return dict(mean=f, B=B, G0=B/strengths[0] if strengths[0]>0 else np.zeros_like(B),
                G=peer/strengths[1], H=contextual/strengths[2], peer=peer, contextual=contextual,
                theta=theta, omega=q['w'], tau=q['tau'], lambda0=chosen['lambda0'],
                order=chosen['order'], path=path, orders_retained=len(orders),
                n=Ye.shape[1], radius=float(M.companion_radius(f['Pi_G'], f['beta'])),
                norm_boundary=bool(np.linalg.norm(B,2)>S.CAP-1e-5 or np.linalg.norm(B)>S.FROB-1e-5),
                penalized_converged=chosen['penalized_converged'])

def generate(base, Y, X, seed):
    """Structural bootstrap with fixed factors, covariates and initial lag."""
    f = base['mean']
    multiplier = np.sqrt(base['tau']) * np.linalg.inv(np.eye(Y.shape[0])-base['B'])
    common = np.sqrt(base['omega']) * f['common']
    return M.simulate(f['Pi_G'], f['Pi_H'], f['beta'], f['rho'], common,
                      multiplier, Y, X, 1, np.random.default_rng(seed))
