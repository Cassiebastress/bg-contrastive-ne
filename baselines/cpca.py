"""Contrastive PCA (Abid et al. 2018) with the reference package's automatic alpha selection (Phase_5_Plan.md §2).

Mirrors `contrastive` 1.2.0 (`CPCA(standardize=False)`): target and background each centered by their own mean,
covariances X^T X / (n - 1), directions = top eigenvectors of C_T - alpha C_B. Alpha grid {0} + logspace(-1, 3, 40).
Affinity between two alphas = product of the cosines of the principal angles between the column spaces of the
projected target data (QR of X_T V(alpha), 2 components), as in the package. Spectral clustering of the alphas into 4
groups; the group containing alpha = 0 is dropped, every other group contributes its exemplar (largest summed affinity
within the group), and alpha = 0 is always added back.

Differences from the package, for determinism only: symmetric eigensolver (`eigh`) with each direction's sign fixed so
its largest-magnitude entry is positive, and an explicit `random_state` for the spectral clustering. One alpha is then
picked without labels: the representative with the largest target-to-background variance ratio in its 2-D subspace.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import SpectralClustering

ALPHAS = np.concatenate(([0.0], np.logspace(-1, 3, 40)))
N_CLUSTERS = 4  # the package's default n_alphas_to_return


@dataclass
class CPCAFit:
    X_T: np.ndarray  # centered target, float64
    C_T: np.ndarray
    C_B: np.ndarray


def fit(X_T, X_B):
    """Center each dataset by its own mean and compute its covariance X^T X / (n - 1)."""
    T = np.asarray(X_T, dtype=np.float64)
    B = np.asarray(X_B, dtype=np.float64)
    T, B = T - T.mean(axis=0), B - B.mean(axis=0)
    return CPCAFit(X_T=T, C_T=T.T @ T / (len(T) - 1), C_B=B.T @ B / (len(B) - 1))


def directions(f, alpha, n_components):
    """Top n_components eigenvectors of C_T - alpha C_B (descending eigenvalue), signs fixed; shape (D, n)."""
    w, v = np.linalg.eigh(f.C_T - alpha * f.C_B)
    v = v[:, np.argsort(-w, kind="stable")[:n_components]]
    signs = np.sign(v[np.abs(v).argmax(axis=0), np.arange(v.shape[1])])
    return v * signs


def project(f, alpha, n_components):
    """The centered target projected onto the top n_components cPCA directions."""
    return f.X_T @ directions(f, alpha, n_components)


def affinity_matrix(f, alphas=ALPHAS):
    """Package affinity: product of the two principal-angle cosines between projected 2-D target subspaces."""
    qs = [np.linalg.qr(project(f, a, 2))[0] for a in alphas]
    k = len(alphas)
    A = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            s = np.linalg.svd(qs[i].T @ qs[j], compute_uv=False)
            A[i, j] = A[j, i] = s[0] * s[1]
    return np.nan_to_num(A)


def representative_alphas(f, seed, alphas=ALPHAS, n_clusters=N_CLUSTERS):
    """The package's procedure: spectral clustering of alphas, exemplar of each cluster without alpha = 0, plus 0."""
    A = affinity_matrix(f, alphas)
    labels = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=seed).fit(A).labels_
    reps = []
    for c in range(n_clusters):
        idx = np.flatnonzero(labels == c)
        if 0 not in idx:
            reps.append(alphas[idx[np.argmax(A[np.ix_(idx, idx)].sum(axis=0))]])
    return np.concatenate(([0.0], np.sort(reps))), labels


def variance_ratio(f, alpha):
    """Target-to-background variance in the 2-D cPCA subspace: tr(V^T C_T V) / tr(V^T C_B V)."""
    V = directions(f, alpha, 2)
    return float(np.trace(V.T @ f.C_T @ V) / np.trace(V.T @ f.C_B @ V))


def select_alpha(f, seed):
    """Label-free choice among the representatives: the largest variance ratio (ties -> smaller alpha)."""
    reps, labels = representative_alphas(f, seed)
    ratios = [variance_ratio(f, a) for a in reps]
    best = max(range(len(reps)), key=lambda i: (ratios[i], -reps[i]))
    return float(reps[best]), [float(a) for a in reps], ratios, labels
