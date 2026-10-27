"""Phase 5a metrics for benchmarks whose signal may differ between background clusters.

- within_cluster_accuracy (WC): signal kNN accuracy computed inside each true background cluster separately (the same
  classifier as signal accuracy: k = 15, stratified 5-fold CV, seed 0), size-weighted over clusters. It measures local
  signal separation whatever the global layout. On high-D X_T it is the "signal detectability" diagnostic.
- coherence: whether "same signal" points the same way in every background cluster. For cell (k, g),
  d_kg = mean of X over the cell - mean of X over cluster k; coherence is the mean over g and over cluster pairs
  k != k' of cos(d_kg, d_k'g). Additive designs give about 1; independent cluster-specific signal maps give about 0.

Labels are ground truth, used for evaluation only.
"""
import numpy as np

from eval.metrics import knn_accuracy


def within_cluster_accuracy(X, y_s, y_b, k=15):
    clusters = np.unique(y_b)
    sizes = np.array([(y_b == c).sum() for c in clusters])
    accs = np.array([knn_accuracy(X[y_b == c], y_s[y_b == c], k=k) for c in clusters])
    return float((sizes * accs).sum() / sizes.sum())


def coherence(X, y_s, y_b):
    X = np.asarray(X, dtype=np.float64)
    clusters, groups = np.unique(y_b), np.unique(y_s)
    d = np.zeros((len(clusters), len(groups), X.shape[1]))
    for i, c in enumerate(clusters):
        mean_c = X[y_b == c].mean(axis=0)
        for j, g in enumerate(groups):
            d[i, j] = X[(y_b == c) & (y_s == g)].mean(axis=0) - mean_c
    unit = d / np.linalg.norm(d, axis=2, keepdims=True)
    cos = np.einsum("igD,jgD->ijg", unit, unit)  # (cluster, cluster, group)
    off = ~np.eye(len(clusters), dtype=bool)
    return float(cos[off].mean())
