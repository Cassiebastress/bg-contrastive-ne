"""Evaluation metrics for embeddings. Ground-truth labels are used here only, never by the method."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from common.knn import knn_graph, knn_indices


def knn_accuracy(embd, labels, k=15, n_splits=5, seed=0):
    """Mean kNN classification accuracy (k neighbors, stratified n_splits-fold CV) of labels in the embedding."""
    clf = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, embd, labels, cv=cv)))


def pair_label_agreement(P, labels, head_weights=None):
    """Expected fraction of pairs sharing a label under a pair distribution: Σ_i w_i Σ_j P_ij [labels_i == labels_j].

    P is a sparse matrix whose non-empty rows are distributions over tails. Heads with empty rows are dropped and the
    head weights (default uniform) renormalized over the rest. Exact, no sampling.
    """
    P = P.tocoo()
    same = np.zeros(P.shape[0])
    np.add.at(same, P.row, P.data * (labels[P.row] == labels[P.col]))
    nonempty = np.zeros(P.shape[0], dtype=bool)
    nonempty[P.row] = True
    w = np.ones(P.shape[0]) if head_weights is None else np.asarray(head_weights, dtype=float)
    w = w * nonempty
    return float((w * same).sum() / w.sum())


def uniform_pair_agreement(labels):
    """Fraction of distinct pairs (i != j) sharing a label: Σ_c n_c (n_c - 1) / (n (n - 1))."""
    counts = np.bincount(labels).astype(float)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def knn_recall(embd, reference, k=15):
    """Mean over points of |N_k(embd) ∩ N_k(reference)| / k, with exact neighbors in both spaces."""
    a, b = knn_indices(embd, k), knn_indices(reference, k)
    return float((a[:, :, None] == b[:, None, :]).any(axis=2).mean())


def pair_distance_ranks(X):
    """Ranks (average ties) of all pairwise Euclidean distances, in pdist order; float64."""
    return rankdata(pdist(np.asarray(X, dtype=np.float64)))


def distance_spearman(embd, reference_ranks):
    """Spearman correlation between all pairwise distances of embd and precomputed reference distance ranks."""
    return float(np.corrcoef(pair_distance_ranks(embd), reference_ranks)[0, 1])


def component_stats(embd, k=15, small=50):
    """Connected components of the k-NN graph (union-symmetrized) of an embedding: their number and the fraction of
    points in components with fewer than `small` points."""
    n_comp, labels = connected_components(knn_graph(embd, k=k), directed=False)
    sizes = np.bincount(labels)
    return {"n_components": int(n_comp), "frac_in_small_components": float((sizes[labels] < small).mean())}
