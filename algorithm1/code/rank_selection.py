"""Factor-adjusted nuclear-norm selection of two fitted-signal ranks."""
import numpy as np
from mean_model import prep

def pca_component(matrix, rank=1):
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]

def soft_singular(matrix, penalty):
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u * np.maximum(singular - penalty, 0)) @ vt

def design_basis(design):
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    if singular[0] == 0:
        raise ValueError('Degenerate design')
    return u[:, singular > singular[0] * 1e-10]

def prepare(Y, X):
    response, lags, covariates = prep(Y, X, 1)
    target = response.T
    Ux = design_basis(lags[:, :, 0])
    Uz = design_basis(covariates[:, :, 0])
    factor = pca_component(target)
    sigma = float(np.sqrt(np.mean((target - factor) ** 2)))
    scale_g = sigma * (np.sqrt(Ux.shape[1]) + np.sqrt(target.shape[1]))
    scale_h = sigma * (np.sqrt(Uz.shape[1]) + np.sqrt(target.shape[1]))
    return target, Ux, Uz, scale_g, scale_h

def fit_penalties(target, Ux, Uz, lambda_g, lambda_h, tolerance=1e-6, max_iter=1000):
    """Alternate exact conditional fitted-block SVT and rank-one PCA updates."""
    Sg = np.zeros_like(target)
    Sh = np.zeros_like(target)
    factor = pca_component(target)
    for iteration in range(1, max_iter + 1):
        new_g = Ux @ soft_singular(Ux.T @ (target - Sh - factor), lambda_g)
        new_h = Uz @ soft_singular(Uz.T @ (target - new_g - factor), lambda_h)
        new_factor = pca_component(target - new_g - new_h)
        change = (np.linalg.norm(new_g-Sg) + np.linalg.norm(new_h-Sh)
                  + np.linalg.norm(new_factor-factor))
        Sg, Sh, factor = new_g, new_h, new_factor
        if change < tolerance:
            break
    partial_g = Ux @ (Ux.T @ (target - Sh - factor))
    partial_h = Uz @ (Uz.T @ (target - Sg - factor))
    singular_g = np.linalg.svd(partial_g, compute_uv=False)
    singular_h = np.linalg.svd(partial_h, compute_uv=False)
    rank_g = int(np.count_nonzero(singular_g > lambda_g))
    rank_h = int(np.count_nonzero(singular_h > lambda_h))
    penalized_rank_g = int(np.linalg.matrix_rank(Sg, tol=1e-8))
    penalized_rank_h = int(np.linalg.matrix_rank(Sh, tol=1e-8))
    objective = (0.5 * np.sum((target-Sg-Sh-factor)**2)
                 + lambda_g*np.linalg.svd(Sg,compute_uv=False).sum()
                 + lambda_h*np.linalg.svd(Sh,compute_uv=False).sum())
    return dict(rank_g=rank_g,rank_h=rank_h,
                penalized_rank_g=penalized_rank_g,penalized_rank_h=penalized_rank_h,
                iterations=iteration,converged=change<tolerance,final_change=float(change),
                objective=float(objective),singular_g=singular_g,singular_h=singular_h,
                fitted_g=Sg,fitted_h=Sh,factor=factor)
