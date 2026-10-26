"""Background-informed negative distribution xi_B over target pairs (Plan §3).

For each target point i, N_B(i) is its k_B nearest background points (exact kNN). Target points j sharing at least
one background neighbor with i are candidates, scored by the Jaccard overlap s_B(i, j) of N_B(i) and N_B(j). Each row
keeps its top-L candidates, with ties at the cut kept up to 2L entries (beyond that, a seeded random draw within the
tied block, keyed by the row). Row probabilities are s_B normalized per row. Target kNN edges are not excluded.
"""
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from common.knn import knn_indices

TIE_RULE = "keep ties at the top-L cut up to 2L; beyond, random draw in the tied block (SeedSequence(seed, (row,)))"


@dataclass
class XiB:
    indptr: np.ndarray  # int64 (n_T + 1,)
    indices: np.ndarray  # int64, target indices, ascending within each row
    jaccard: np.ndarray  # float64, s_B of each entry
    probs: np.ndarray  # float64, s_B normalized per row
    params: dict  # settings and build statistics

    ARRAYS = ("indptr", "indices", "jaccard", "probs")

    @property
    def n(self):
        return len(self.indptr) - 1

    def row_lengths(self):
        return np.diff(self.indptr)

    def to_csr(self, values="probs"):
        """The distribution as an n_T x n_T scipy CSR matrix with the given per-entry values."""
        return sp.csr_matrix((getattr(self, values), self.indices, self.indptr), shape=(self.n, self.n))

    def sha256(self):
        h = hashlib.sha256()
        for name in self.ARRAYS:
            a = getattr(self, name)
            h.update(f"{name}:{a.dtype.str}:{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez(path, **{name: getattr(self, name) for name in self.ARRAYS}, params=json.dumps(self.params))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(**{name: f[name] for name in cls.ARRAYS}, params=json.loads(str(f["params"])))


def top_l_with_ties(inter, L, seed, row):
    """Boolean mask of the candidates a row keeps, given their integer overlaps |N_B(i) ∩ N_B(j)|.

    Keeps everything if there are at most L candidates. Otherwise, with c the L-th largest overlap, keeps all
    candidates with overlap > c and the tied block (overlap == c), capped at 2L entries in total; when capped, the
    kept block members are a random draw without replacement from a stream keyed by (seed, row).
    Returns (mask, tie_split, capped).
    """
    keep = np.ones(len(inter), dtype=bool)
    if len(inter) <= L:
        return keep, False, False
    c = np.partition(inter, len(inter) - L)[len(inter) - L]  # L-th largest
    above, tied = inter > c, inter == c
    n_above = int(above.sum())
    tie_split = n_above + int(tied.sum()) > L
    if n_above + tied.sum() <= 2 * L:
        return above | tied, tie_split, False
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(row,)))
    pick = rng.choice(np.flatnonzero(tied), size=2 * L - n_above, replace=False)
    keep = above.copy()
    keep[pick] = True
    return keep, True, True


def background_neighbor_overlaps(X_T, X_B, k_B):
    """n_T x n_T int64 CSR of |N_B(i) ∩ N_B(j)| over candidate pairs (a shared background neighbor), no diagonal."""
    n_T, n_B = len(X_T), len(X_B)
    nb = knn_indices(X_T, k_B, Y=X_B)
    M = sp.csr_matrix((np.ones(n_T * k_B, dtype=np.int64), (np.repeat(np.arange(n_T), k_B), nb.ravel())),
                      shape=(n_T, n_B))
    inter = (M @ M.T).tocsr()  # inverted index: rows i, j meet only through shared background points
    inter = (inter - sp.diags(inter.diagonal(), dtype=np.int64)).tocsr()
    inter.eliminate_zeros()
    inter.sort_indices()
    return inter


def build_xi_b(X_T, X_B, k_B=15, L=30, seed=0):
    """Build xi_B from target X_T and background X_B (same feature space, identical preprocessing)."""
    if not 1 <= k_B <= len(X_B):
        raise ValueError(f"k_B must be in [1, n_B={len(X_B)}], got {k_B}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    inter = background_neighbor_overlaps(X_T, X_B, k_B)
    n_T = len(X_T)

    indptr = np.zeros(n_T + 1, dtype=np.int64)
    indices, jaccard, probs = [], [], []
    n_tie_split = n_capped = 0
    for i in range(n_T):
        lo, hi = inter.indptr[i], inter.indptr[i + 1]
        cand, overlap = inter.indices[lo:hi], inter.data[lo:hi]
        keep, tie_split, capped = top_l_with_ties(overlap, L, seed, i)
        n_tie_split += tie_split
        n_capped += capped
        j, s = cand[keep], overlap[keep] / (2 * k_B - overlap[keep])
        indices.append(j)
        jaccard.append(s)
        probs.append(s / s.sum() if len(s) else s)
        indptr[i + 1] = indptr[i] + len(j)

    lengths = np.diff(indptr)
    params = {"k_B": k_B, "L": L, "cap": 2 * L, "seed": seed, "n_T": n_T, "n_B": len(X_B), "tie_rule": TIE_RULE,
              "n_entries": int(indptr[-1]), "n_empty_rows": int((lengths == 0).sum()),
              "n_tie_split_rows": int(n_tie_split), "n_capped_rows": int(n_capped)}
    return XiB(indptr=indptr, indices=np.concatenate(indices).astype(np.int64),
               jaccard=np.concatenate(jaccard).astype(np.float64), probs=np.concatenate(probs).astype(np.float64),
               params=params)
