"""Background-suppressed kNN graph: the target kNN graph built on local background residuals.

For each target point x_i: its k_loc nearest background points (exact, Euclidean), their mean mu_i and the top r
principal directions V_i of those neighbors centered at mu_i (SVD). The residual is
    r_i = (x_i - mu_i) - V_i V_i^T (x_i - mu_i)
(r = 0: local mean subtraction only). The graph is common.knn.knn_graph on the residuals, symmetrized exactly as every
earlier graph. Only the attraction changes; xi_B and the initialization stay on the raw data.
"""
from dataclasses import dataclass

import numpy as np

from common.knn import knn_graph, knn_indices

R_MAX = 10


@dataclass
class LocalBackground:
    k_loc: int
    mu: np.ndarray  # (n_T, D) local background means
    V: np.ndarray  # (n_T, D, r_max) top local background directions, orthonormal columns
    neighbors: np.ndarray  # (n_T, k_loc) background indices


def local_background(X_T, X_B, k_loc, r_max=R_MAX):
    """Each target's k_loc nearest background points, their mean, and their top min(r_max, k_loc - 1) directions."""
    X_T, X_B = np.asarray(X_T, dtype=np.float64), np.asarray(X_B, dtype=np.float64)
    nbrs = knn_indices(X_T, k_loc, Y=X_B)
    r_max = min(r_max, k_loc - 1)
    mu = np.empty_like(X_T)
    V = np.empty((len(X_T), X_T.shape[1], r_max))
    for i in range(len(X_T)):  # one target at a time: (k_loc, D) in memory, not (n_T, k_loc, D)
        N = X_B[nbrs[i]]
        mu[i] = N.mean(axis=0)
        _, _, vt = np.linalg.svd(N - mu[i], full_matrices=False)
        V[i] = vt[:r_max].T
    return LocalBackground(k_loc=k_loc, mu=mu, V=V, neighbors=nbrs)


def residuals(X_T, local, r):
    """(x_i - mu_i) minus its projection on the top r local background directions (sign-invariant)."""
    if r > local.V.shape[2]:
        raise ValueError(f"r = {r} exceeds the {local.V.shape[2]} stored directions")
    d = np.asarray(X_T, dtype=np.float64) - local.mu
    if r == 0:
        return d
    V = local.V[:, :, :r]
    return d - np.einsum("ndr,nr->nd", V, np.einsum("ndr,nd->nr", V, d))


def residual_graph(X_T, X_B, k_loc, r, k=15):
    """The symmetrized kNN graph (k = 15) on the residuals, and the residuals."""
    R = residuals(X_T, local_background(X_T, X_B, k_loc), r)
    return knn_graph(R, k=k), R
