"""Phase 5a metrics for benchmarks whose signal may differ between background clusters.

- within_cluster_accuracy (WC): signal kNN accuracy computed inside each true background cluster separately (the same
  classifier as signal accuracy: k = 15, stratified 5-fold CV, seed 0), size-weighted over clusters. It measures local
  signal separation whatever the global layout. On high-D X_T it is the "signal detectability" diagnostic.
- coherence: whether "same signal" points the same way in every background cluster. For cell (k, g),
  d_kg = mean of X over the cell - mean of X over cluster k; coherence is the mean over g and over cluster pairs
  k != k' of cos(d_kg, d_k'g). Additive designs give about 1; independent cluster-specific signal maps give about 0.
- signal_background_geometry (data diagnostic, clean generator components): the fraction of the clean signal's
  variance inside span(B), and the background variance along the signal's top principal directions (those explaining
  90% of its variance), globally over all of X_B vs the mean within background cluster. cPCA can only use directions
  with low global background variance; xi_B, built from background neighborhoods, effectively uses directions with low
  local background variance. A large global/local ratio marks where (c) is predicted to have an advantage.

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


def signal_background_geometry(signal, B, X_B, y_b_background, var_frac=0.9):
    S = signal - signal.mean(axis=0)
    total = float((S ** 2).sum())
    _, sv, Vt = np.linalg.svd(S, full_matrices=False)
    cum = np.cumsum(sv ** 2) / (sv ** 2).sum()
    n = int(np.searchsorted(cum, var_frac) + 1)
    P = np.asarray(X_B, dtype=np.float64) @ Vt[:n].T
    global_var = float(P.var(axis=0).sum())
    local_var = float(np.mean([P[y_b_background == c].var(axis=0).sum() for c in np.unique(y_b_background)]))
    return {"frac_signal_var_in_span_B": float(((S @ B) ** 2).sum() / total), "n_signal_directions": n,
            "bg_global_var": global_var, "bg_local_var": local_var, "bg_global_local_ratio": global_var / local_var}
