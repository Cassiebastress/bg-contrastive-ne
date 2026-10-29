"""Local (cluster-then-cPCA) baseline for Design L (Phase 5b extension; results/phase5b/predictions.md, P6-P9).

1. Label-free version: k-means on X_B for K in {2, ..., 10} (n_init = 10, random_state = seed); K with the largest
   silhouette on X_B (all points), ties -> smaller K. Each target point goes to the nearest centroid (Euclidean).
   Oracle version: the true background clusters of X_B and X_T instead.
2. cPCA within each cluster (`baselines.cpca` unchanged): the cluster's target and background points, each centered by
   its own mean; label-free alpha per cluster (`select_alpha`); the top k directions give local coordinates.
3. One joint representation Z = [local coordinates | s * one-hot(cluster)], with s = 10 R and R the largest distance
   of a target point from its cluster's center in the local coordinates. Different clusters are then at least
   s * sqrt(2) apart and points within a cluster at most 2 R, so no kNN edge crosses clusters (checked).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from baselines import cpca
from common.knn import knn_indices

K_RANGE = range(2, 11)
SCALE = 10.0
MIN_CLUSTER_POINTS = 50  # smaller target or background parts make the per-cluster covariance ill-posed


def select_clusters(X_B, seed, k_range=K_RANGE):
    """k-means on X_B for each K; the K with the largest silhouette (ties -> smaller K). Labels, centers and info."""
    X = np.asarray(X_B, dtype=np.float64)
    fits, sil = {}, {}
    for K in k_range:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X)
        fits[K], sil[K] = km, float(silhouette_score(X, km.labels_))
    K = max(k_range, key=lambda k: (sil[k], -k))
    return fits[K].labels_, fits[K].cluster_centers_, {"K": int(K), "silhouettes": {str(k): v for k, v in sil.items()}}


def assign(X_T, centers):
    """Nearest centroid for each target point (Euclidean, full dimension)."""
    return knn_indices(np.asarray(X_T, dtype=np.float64), 1, Y=np.asarray(centers, dtype=np.float64))[:, 0]


def local_alphas(X_T, X_B, lab_T, lab_B, seed):
    """Per cluster: the cPCA fit and its label-free alpha."""
    out = {}
    for j in np.unique(lab_B):
        f = cpca.fit(X_T[lab_T == j], X_B[lab_B == j])
        alpha, reps, ratios, _ = cpca.select_alpha(f, seed)
        out[int(j)] = {"fit": f, "alpha": alpha, "representatives": reps, "variance_ratios": ratios}
    return out


def representation(X_T, lab_T, per_cluster, k, k_graph=15):
    """Z = [local k coordinates | s * one-hot(cluster)] for every target point, and its scale and kNN check."""
    clusters = sorted(per_cluster)
    local = np.zeros((len(X_T), k))
    for j in clusters:
        f = per_cluster[j]["fit"]
        local[lab_T == j] = cpca.project(f, per_cluster[j]["alpha"], k)  # f.X_T is the cluster's centered target
    R = float(np.linalg.norm(local, axis=1).max())
    s = SCALE * R
    onehot = (lab_T[:, None] == np.array(clusters)[None, :]).astype(np.float64)
    Z = np.hstack([local, s * onehot])
    nbrs = knn_indices(Z, k_graph)
    cross = float((lab_T[nbrs] != lab_T[:, None]).mean())
    return Z, {"k": k, "R": R, "s": s, "cross_cluster_knn_frac": cross}


def check_sizes(lab_T, lab_B):
    """Clusters with fewer than MIN_CLUSTER_POINTS target or background points (empty list if none)."""
    small = []
    for j in np.unique(np.concatenate([lab_T, lab_B])):
        n_t, n_b = int((lab_T == j).sum()), int((lab_B == j).sum())
        if min(n_t, n_b) < MIN_CLUSTER_POINTS:
            small.append({"cluster": int(j), "n_T": n_t, "n_B": n_b})
    return small


def clusters_for(ds, version, seed):
    """Target and background cluster labels for the 'labelfree' or 'oracle' version, with diagnostics."""
    t = ds.truth
    if version == "oracle":
        return np.asarray(t["y_b"]), np.asarray(t["y_b_background"]), {"K": int(len(np.unique(t["y_b"])))}
    lab_B, centers, info = select_clusters(ds.X_B, seed)
    lab_T = assign(ds.X_T, centers)
    info.update({"ARI_background": float(adjusted_rand_score(t["y_b_background"], lab_B)),
                 "ARI_target": float(adjusted_rand_score(t["y_b"], lab_T))})  # evaluation only
    return lab_T, lab_B, info
