"""Diagnostics of xi_B against ground truth (Phase 2; reused for the Phase 4 confirmation and stress datasets).

Ground truth is used here only, never by the method. `diagnose` returns label agreement of xi_B pairs (exact
expectations with heads weighted by target-graph degree), background geometry, overlap with target kNN edges, list
statistics, and the informativeness gates.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from eval.metrics import pair_label_agreement, uniform_pair_agreement

GATES = {"min_same_b": 0.90, "max_same_s_excess": 0.10}


def weighted_row_mean(P, values, w):
    """Σ_i w_i Σ_j P_ij values_ij / Σ_i w_i over heads with non-empty rows; values aligned with P's (CSR) data."""
    heads = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    row_sum = np.bincount(heads, weights=P.data * values, minlength=P.shape[0])
    w = w * (np.diff(P.indptr) > 0)
    return float((w * row_sum).sum() / w.sum())


def same_label_bg_distance(Bg, y):
    """Mean ||Bg_i - Bg_j|| over all distinct pairs sharing a label (exact)."""
    total, count = 0.0, 0
    for c in np.unique(y):
        d = pdist(Bg[y == c])
        total, count = total + d.sum(), count + len(d)
    return total / count


def diagnose(ds, xi, G):
    t = ds.truth
    y_b, y_s, Bg = t["y_b"], t["y_s"], t["Bg_T"]
    deg = np.diff(G.indptr).astype(float)
    P = xi.to_csr()
    heads = np.repeat(np.arange(xi.n), xi.row_lengths())
    tails = xi.indices
    lengths = xi.row_lengths()

    G_rows = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
    on_edge = np.asarray(G[heads, tails]).ravel() > 0
    bgdist = np.linalg.norm(Bg[heads] - Bg[tails], axis=1)
    same_s_xi = pair_label_agreement(P, y_s, head_weights=deg)
    s_unif = uniform_pair_agreement(y_s)
    covered = np.asarray(P[G_rows, G.indices]).ravel() > 0

    row = {
        "same_b_xi": pair_label_agreement(P, y_b, head_weights=deg), "same_s_xi": same_s_xi,
        "same_s_excess": same_s_xi - s_unif,
        "same_b_xi_uniform_heads": pair_label_agreement(P, y_b),
        "same_s_xi_uniform_heads": pair_label_agreement(P, y_s),
        "same_b_uniform": uniform_pair_agreement(y_b), "same_s_uniform": s_unif,
        "same_b_knn": float(np.mean(y_b[G_rows] == y_b[G.indices])),
        "same_s_knn": float(np.mean(y_s[G_rows] == y_s[G.indices])),
        "len_min": int(lengths.min()), "len_median": float(np.median(lengths)),
        "len_p90": float(np.percentile(lengths, 90)), "len_max": int(lengths.max()),
        "n_empty_rows": xi.params["n_empty_rows"],
        "frac_tie_split_rows": xi.params["n_tie_split_rows"] / xi.n,
        "frac_capped_rows": xi.params["n_capped_rows"] / xi.n,
        "bgdist_xi": weighted_row_mean(P, bgdist, deg),
        "bgdist_same_b_random": same_label_bg_distance(Bg, y_b),
        "bgdist_knn": float(np.linalg.norm(Bg[G_rows] - Bg[G.indices], axis=1).mean()),
        "spearman_jaccard_bgdist": float(spearmanr(xi.jaccard, bgdist).statistic),
        "frac_knn_edges_in_xi": float(covered.mean()),
        "xi_mass_on_knn": weighted_row_mean(P, on_edge, deg),
        "xi_mass_on_knn_same_s": weighted_row_mean(P, on_edge & (y_s[heads] == y_s[tails]), deg),
        "xi_mass_on_knn_diff_s": weighted_row_mean(P, on_edge & (y_s[heads] != y_s[tails]), deg),
        "sha256": xi.sha256(),
    }
    row["gate_same_b"] = row["same_b_xi"] >= GATES["min_same_b"]
    row["gate_same_s"] = row["same_s_excess"] <= GATES["max_same_s_excess"]
    return row, lengths
