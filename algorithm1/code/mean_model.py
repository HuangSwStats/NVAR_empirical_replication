"""Reduced-rank alternating least squares and PCA with p=1."""
import numpy as np

def prep(Y, X, p):
    """Yeff (N,n); YL (n,N,p); XN (n,N,d)."""
    N, T = Y.shape
    idx = np.arange(p, T)
    YL = np.stack([np.column_stack([Y[:, t - j] for j in range(1, p + 1)])
                   for t in idx])
    XN = np.stack([X[:, :, t] for t in idx])
    return Y[:, idx], YL, XN

def pca_common(resid, r_f):
    """Rank-r_f PCA of the N x n residual; F'F/n = I."""
    if r_f <= 0:
        return np.zeros_like(resid), None, None
    N, n = resid.shape
    E = resid.T                                   # n x N
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    F = np.sqrt(n) * U[:, :r_f]                   # n x r_f
    Lam = E.T @ F / n                             # N x r_f
    return Lam @ F.T, Lam, F

def rr_update(R, Z, rank):
    """P_r(S_ab S_bb^{-1} S_ba) S_ab S_bb^{-1}."""
    n = R.shape[1]
    S_ab = R @ Z.T / n
    S_bb = Z @ Z.T / n
    Sbi = np.linalg.inv(S_bb)
    tgt = S_ab @ Sbi @ S_ab.T
    tgt = (tgt + tgt.T) / 2
    w, V = np.linalg.eigh(tgt)
    U = V[:, np.argsort(w)[::-1][:int(rank)]]
    return U @ U.T @ S_ab @ Sbi

def companion_radius(Pi_G, beta):
    N, p = Pi_G.shape[0], len(beta)
    trans = np.hstack([b * Pi_G for b in beta])
    if p == 1:
        return np.abs(np.linalg.eigvals(trans)).max()
    comp = np.vstack([trans, np.hstack([np.eye(N * (p - 1)),
                                        np.zeros((N * (p - 1), N))])])
    return np.abs(np.linalg.eigvals(comp)).max()

def fit_distinct(Yeff, YL, XN, r_G, r_H, r_f=1, tol=1e-6, max_iter=500,
                 gamma0=0.1, warm=None):
    n, N, p = YL.shape
    d = XN.shape[2]
    if warm is None:
        beta = np.full(p, gamma0); rho = np.full(d, gamma0)
        Pi_G = np.zeros((N, N)); Pi_H = np.zeros((N, N))
        common, Lam, F = pca_common(Yeff, r_f)
    else:
        beta = warm["beta"].copy(); rho = warm["rho"].copy()
        Pi_G = warm["Pi"].copy(); Pi_H = warm["Pi"].copy()
        common, Lam, F = warm["common"], warm["Lam"], warm["F"]

    def obj(Pi_G, Pi_H, beta, rho, common, z, w):
        return ((Yeff - Pi_G @ z - Pi_H @ w - common) ** 2).sum() / n

    z = np.einsum('tip,p->it', YL, beta)
    w = np.einsum('tid,d->it', XN, rho)
    cur = obj(Pi_G, Pi_H, beta, rho, common, z, w)
    worst_increase = -np.inf
    conv = False
    for it in range(1, max_iter + 1):
        yc = Yeff - common
        z = np.einsum('tip,p->it', YL, beta)
        w = np.einsum('tid,d->it', XN, rho)

        Pi_G = rr_update(yc - Pi_H @ w, z, r_G)
        s1 = obj(Pi_G, Pi_H, beta, rho, common, z, w)
        Pi_H = rr_update(yc - Pi_G @ z, w, r_H)
        s2 = obj(Pi_G, Pi_H, beta, rho, common, z, w)

        KG = np.einsum('ij,tjp->tip', Pi_G, YL)
        KH = np.einsum('ij,tjd->tid', Pi_H, XN)
        K = np.concatenate([KG, KH], axis=2).reshape(n * N, p + d)
        gam = np.linalg.pinv(K.T @ K) @ (K.T @ yc.T.reshape(n * N))
        beta, rho = gam[:p], gam[p:]
        z = np.einsum('tip,p->it', YL, beta)
        w = np.einsum('tid,d->it', XN, rho)
        s3 = obj(Pi_G, Pi_H, beta, rho, common, z, w)

        common, Lam, F = pca_common(Yeff - Pi_G @ z - Pi_H @ w, r_f)
        s4 = obj(Pi_G, Pi_H, beta, rho, common, z, w)

        worst_increase = max(worst_increase, s1 - cur, s2 - s1, s3 - s2, s4 - s3)
        done = abs(s4 - cur) / max(1.0, cur) < tol
        cur = s4
        if done:
            conv = True
            break

    cG, cH = np.linalg.norm(Pi_G), np.linalg.norm(Pi_H)
    return dict(Pi_G=Pi_G / cG, Pi_H=Pi_H / cH, beta=cG * beta, rho=cH * rho,
                common=common, Lam=Lam, F=F, objective=cur,
                rmse=np.sqrt(cur / N), iterations=it, converged=conv,
                worst_increase=worst_increase)

def simulate(Pi_G, Pi_H, beta, rho, common, chol, Y_init, X, p, rng):
    """Recursive draw using the pseudo-sample's own lags."""
    N, T = Y_init.shape[0], X.shape[2]
    Ys = np.zeros((N, T))
    Ys[:, :p] = Y_init[:, :p]
    for t in range(p, T):
        lagged = np.column_stack([Ys[:, t - j] for j in range(1, p + 1)])
        Ys[:, t] = (Pi_G @ lagged @ beta + Pi_H @ X[:, :, t] @ rho
                    + common[:, t - p] + chol @ rng.standard_normal(N))
    return Ys

def fit_common(Yeff, YL, XN, rank, r_f=1, tol=1e-6, max_iter=500):
    """Single shared network (Model I); also the warm start for Model II."""
    n, N, p = YL.shape
    d = XN.shape[2]
    gam = np.full(p + d, 0.1)
    Pi = np.zeros((N, N))
    common, Lam, F = pca_common(Yeff, r_f)
    J = np.concatenate([YL, XN], axis=2)              # n x N x (p+d)

    def obj(Pi, gam, common):
        q = np.einsum('tik,k->it', J, gam)
        return ((Yeff - Pi @ q - common) ** 2).sum() / n

    cur = obj(Pi, gam, common)
    conv = False
    for it in range(1, max_iter + 1):
        yc = Yeff - common
        q = np.einsum('tik,k->it', J, gam)
        Pi = rr_update(yc, q, rank)
        K = np.einsum('ij,tjk->tik', Pi, J).reshape(n * N, p + d)
        gam = np.linalg.pinv(K.T @ K) @ (K.T @ yc.T.reshape(n * N))
        q = np.einsum('tik,k->it', J, gam)
        common, Lam, F = pca_common(Yeff - Pi @ q, r_f)
        new = obj(Pi, gam, common)
        done = abs(new - cur) / max(1.0, cur) < tol
        cur = new
        if done:
            conv = True
            break
    s = np.linalg.norm(Pi)
    return dict(Pi=Pi / s, beta=s * gam[:p], rho=s * gam[p:], common=common,
                Lam=Lam, F=F, objective=cur, rmse=np.sqrt(cur / N),
                iterations=it, converged=conv)

def fit_model_ii(Yeff, YL, XN, r_G, r_H, r_f=1, tol=1e-6, max_iter=500):
    """Three deterministic starts; retain the lowest objective."""
    cands = []
    for g0 in (0.10, 0.25):
        cands.append(fit_distinct(Yeff, YL, XN, r_G, r_H, r_f, tol, max_iter,
                                  gamma0=g0))
    cm = fit_common(Yeff, YL, XN, max(r_G, r_H), r_f, tol, max_iter)
    cands.append(fit_distinct(Yeff, YL, XN, r_G, r_H, r_f, tol, max_iter,
                              warm=cm))
    best = min(cands, key=lambda f: f["objective"])
    return best
