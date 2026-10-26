"""Tests for the background-informed negative distribution xi_B (method/xi_b.py)."""
import numpy as np
import pytest

from common.knn import knn_graph
from data.synthetic import MAIN_RHO, make_dataset
from eval.metrics import pair_label_agreement
from method.xi_b import XiB, background_neighbor_overlaps, build_xi_b, top_l_with_ties


def _reference(X_T, X_B, k_B, L, seed):
    """All-pairs brute force from Python sets, with the same tie rule and keyed RNG as build_xi_b."""
    d = ((X_T[:, None, :].astype(np.float64) - X_B[None].astype(np.float64)) ** 2).sum(-1)
    nb = [set(np.argsort(row, kind="stable")[:k_B].tolist()) for row in d]
    rows = []
    for i in range(len(X_T)):
        cand = [(j, len(nb[i] & nb[j])) for j in range(len(X_T)) if j != i and nb[i] & nb[j]]
        j = np.array([c[0] for c in cand], dtype=np.int64)
        inter = np.array([c[1] for c in cand], dtype=np.int64)
        if len(j) > L:
            c = sorted(inter.tolist(), reverse=True)[L - 1]
            above, tied = inter > c, inter == c
            keep = above | tied
            if keep.sum() > 2 * L:
                rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(i,)))
                keep = above.copy()
                keep[rng.choice(np.flatnonzero(tied), size=2 * L - above.sum(), replace=False)] = True
            j, inter = j[keep], inter[keep]
        s = inter / (2 * k_B - inter)
        rows.append((j, s, s / s.sum() if len(s) else s))
    return rows


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 4)).astype(np.float32), rng.standard_normal((50, 4)).astype(np.float32)


def test_matches_brute_force(small):
    X_T, X_B = small
    xi = build_xi_b(X_T, X_B, k_B=5, L=8, seed=7)
    ref = _reference(X_T, X_B, k_B=5, L=8, seed=7)
    assert xi.params["n_capped_rows"] > 0 and xi.params["n_tie_split_rows"] > 0  # the test exercises both paths
    for i, (j, s, p) in enumerate(ref):
        lo, hi = xi.indptr[i], xi.indptr[i + 1]
        assert np.array_equal(xi.indices[lo:hi], j)
        assert np.array_equal(xi.jaccard[lo:hi], s)
        assert np.array_equal(xi.probs[lo:hi], p)


def test_csr_invariants(small):
    X_T, X_B = small
    L = 8
    xi = build_xi_b(X_T, X_B, k_B=5, L=L, seed=0)
    assert xi.indptr.dtype == np.int64 and xi.indices.dtype == np.int64
    assert xi.jaccard.dtype == np.float64 and xi.probs.dtype == np.float64
    assert xi.indptr[0] == 0 and np.all(np.diff(xi.indptr) >= 0) and xi.indptr[-1] == len(xi.indices)
    assert np.all(xi.jaccard > 0) and np.all(xi.jaccard <= 1)
    lengths = xi.row_lengths()
    assert lengths.max() <= 2 * L
    for i in range(xi.n):
        lo, hi = xi.indptr[i], xi.indptr[i + 1]
        j = xi.indices[lo:hi]
        assert i not in j and np.all(np.diff(j) > 0)
        if hi > lo:
            assert abs(xi.probs[lo:hi].sum() - 1) < 1e-12
    # each row keeps at least min(L, candidates) entries and nothing below its L-th largest overlap
    inter = background_neighbor_overlaps(X_T, X_B, k_B=5)
    for i in range(xi.n):
        overlap = inter.data[inter.indptr[i]:inter.indptr[i + 1]]
        assert lengths[i] >= min(L, len(overlap))
        if len(overlap) > L:
            cut = np.sort(overlap)[::-1][L - 1]
            kept = xi.jaccard[xi.indptr[i]:xi.indptr[i + 1]]
            assert kept.min() >= cut / (10 - cut)


def test_tie_rule_keeps_ties_below_cap():
    inter = np.array([5, 4, 3, 3, 3, 3, 2, 1])
    keep, split, capped = top_l_with_ties(inter, L=3, seed=0, row=0)
    assert np.array_equal(keep, inter >= 3) and split and not capped
    keep, split, capped = top_l_with_ties(inter, L=2, seed=0, row=0)
    assert np.array_equal(keep, inter >= 4) and not split and not capped
    keep, split, capped = top_l_with_ties(inter[:3], L=5, seed=0, row=0)  # at most L candidates: keep all
    assert keep.all() and not split and not capped


def test_tie_rule_cap():
    inter = np.array([9, 8] + [3] * 20 + [1, 1])
    L = 4
    keep, split, capped = top_l_with_ties(inter, L=L, seed=0, row=5)
    assert split and capped and keep.sum() == 2 * L
    assert keep[:2].all() and not keep[-2:].any()  # all above-cut entries kept; extras only from the tied block
    again, _, _ = top_l_with_ties(inter, L=L, seed=0, row=5)
    assert np.array_equal(keep, again)
    picks = {tuple(np.flatnonzero(top_l_with_ties(inter, L=L, seed=s, row=5)[0])) for s in range(5)}
    picks |= {tuple(np.flatnonzero(top_l_with_ties(inter, L=L, seed=0, row=r)[0])) for r in range(5)}
    assert len(picks) > 1  # the draw depends on the seed and the row


def test_deterministic_and_background_permutation_invariant(small):
    X_T, X_B = small
    a = build_xi_b(X_T, X_B, k_B=5, L=8, seed=3)
    b = build_xi_b(X_T, X_B, k_B=5, L=8, seed=3)
    perm = np.random.default_rng(1).permutation(len(X_B))
    c = build_xi_b(X_T, X_B[perm], k_B=5, L=8, seed=3)
    assert a.sha256() == b.sha256() == c.sha256()


def test_keeps_target_edges_and_empty_rows():
    rng = np.random.default_rng(2)
    X_B = np.vstack([rng.standard_normal((20, 3)), 100 + rng.standard_normal((5, 3))]).astype(np.float32)
    X_T = rng.standard_normal((10, 3)).astype(np.float32)
    X_T[1] = X_T[0] + 1e-3  # a target kNN edge whose endpoints share all background neighbors
    X_T = np.vstack([X_T, np.full((1, 3), 100, dtype=np.float32)])  # only target near the far background points
    xi = build_xi_b(X_T, X_B, k_B=5, L=3, seed=0)
    G = knn_graph(X_T, k=1)
    assert G[0, 1] == 1
    lo, hi = xi.indptr[0], xi.indptr[1]
    assert 1 in xi.indices[lo:hi] and xi.jaccard[lo:hi][list(xi.indices[lo:hi]).index(1)] == 1.0
    assert xi.row_lengths()[-1] == 0 and xi.params["n_empty_rows"] == 1


def test_save_load_roundtrip(small, tmp_path):
    X_T, X_B = small
    xi = build_xi_b(X_T, X_B, k_B=5, L=8, seed=0)
    xi.save(tmp_path / "xi.npz")
    back = XiB.load(tmp_path / "xi.npz")
    assert back.sha256() == xi.sha256() and back.params == xi.params


def test_bad_arguments(small):
    X_T, X_B = small
    with pytest.raises(ValueError):
        build_xi_b(X_T, X_B, k_B=len(X_B) + 1)
    with pytest.raises(ValueError):
        build_xi_b(X_T, X_B, L=0)


def test_easy_pairs_share_background():
    ds = make_dataset("easy", n_T=1000, n_B=1000, seed=0, signal_to_background=MAIN_RHO["easy"])
    xi = build_xi_b(ds.X_T, ds.X_B, seed=0)
    deg = np.diff(knn_graph(ds.X_T, k=15).indptr)
    assert pair_label_agreement(xi.to_csr(), ds.truth["y_b"], head_weights=deg) >= 0.9


def test_overlaps_are_integer(small):
    X_T, X_B = small
    inter = background_neighbor_overlaps(X_T, X_B, k_B=5)
    assert inter.dtype == np.int64 and inter.diagonal().max() == 0 and inter.data.min() >= 1
