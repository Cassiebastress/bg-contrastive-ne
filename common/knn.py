"""Deterministic kNN graphs (exact search; see Initial_Project_Plan.md, Reproducibility rules)."""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def knn_indices(X, k, Y=None):
    """Exact Euclidean k nearest neighbors of each row of X among the rows of Y (default: X itself, excluding self).

    Returns an (n_X, k) int array. Ties are broken by sklearn's deterministic brute-force ordering.
    """
    if Y is None:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X)
        idx = nn.kneighbors(X, return_distance=False)
        # drop each point itself; with exact duplicates it need not be in column 0
        self_mask = idx == np.arange(len(X))[:, None]
        no_self = ~self_mask.any(axis=1)
        self_mask[no_self, -1] = True
        return idx[~self_mask].reshape(len(X), k)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(Y)
    return nn.kneighbors(X, return_distance=False)


def knn_graph(X, k=15):
    """Binary kNN graph symmetrized by union, the same structure contrastive-ne builds with annoy
    (third_party/contrastive-ne/src/cne/_cne.py:423-430), but from exact neighbors."""
    n = len(X)
    idx = knn_indices(X, k)
    rows = np.repeat(np.arange(n), k)
    A = sp.coo_matrix((np.ones(n * k), (rows, idx.ravel())), shape=(n, n)).tocsr()
    A = A.maximum(A.T).tocsr()
    A.sort_indices()
    return A
