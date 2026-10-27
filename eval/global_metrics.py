"""Global layout metrics (Phase_5_Plan.md §1): do same-signal points from different background clusters end up together?

Labels are ground truth, used for evaluation only. k = 15 and exact neighbors, as in Phase 4.
- G1: mean fraction of a point's k nearest neighbors with the same signal label and a different background label.
- G2: background kNN accuracy (eval.metrics.knn_accuracy on y_b); drops toward chance if background islands merge.
- G3: over the (signal x background) cells, the fraction whose nearest other cell centroid shares the signal label,
  shares the background label, or neither.
"""
import numpy as np
from scipy.spatial.distance import cdist

from common.knn import knn_indices
from eval.metrics import knn_accuracy

K = 15


def cross_cluster_same_signal_rate(X, y_s, y_b, k=K):
    """G1: mean over points of the fraction of k nearest neighbors with the same y_s and a different y_b."""
    nb = knn_indices(X, k)
    return float(((y_s[nb] == y_s[:, None]) & (y_b[nb] != y_b[:, None])).mean())


def g1_chance(y_s, y_b):
    """G1 for random neighbors: the exact fraction of distinct pairs (i != j) with the same y_s and a different y_b."""
    n = len(y_s)
    same_s = np.bincount(y_s).astype(float)
    cells = np.unique(np.stack([y_s, y_b], axis=1), axis=0, return_counts=True)[1].astype(float)
    pairs = (same_s * (same_s - 1)).sum() - (cells * (cells - 1)).sum()
    return float(pairs / (n * (n - 1)))


def cell_layout(X, y_s, y_b):
    """G3: for each (y_s, y_b) cell, the nearest other cell centroid (exact ties go to the lower cell index) and whether
    it shares the signal label, the background label, or neither. Returns the three fractions and the cell count."""
    cells = np.unique(np.stack([y_s, y_b], axis=1), axis=0)  # lexicographic order fixes the cell index
    centroids = np.array([X[(y_s == s) & (y_b == b)].mean(axis=0) for s, b in cells])
    D = cdist(centroids, centroids)
    np.fill_diagonal(D, np.inf)
    nearest = cells[D.argmin(axis=1)]  # argmin returns the first (lowest-index) minimum
    shares_signal = (nearest[:, 0] == cells[:, 0]) & (nearest[:, 1] != cells[:, 1])
    shares_background = (nearest[:, 1] == cells[:, 1]) & (nearest[:, 0] != cells[:, 0])
    return {"G3_shares_signal": float(shares_signal.mean()), "G3_shares_background": float(shares_background.mean()),
            "G3_neither": float((~shares_signal & ~shares_background).mean()), "n_cells": int(len(cells))}


def global_scores(X, y_s, y_b, k=K):
    """G1, G2 and G3 for one representation (2-D embedding or high-D data)."""
    return {"G1": cross_cluster_same_signal_rate(X, y_s, y_b, k), "G2": knn_accuracy(X, y_b, k=k),
            **cell_layout(X, y_s, y_b)}
