"""Tests for the background-suppressed graph (method/residual_graph.py)."""
import numpy as np
import pytest

from method import residual_graph as rg


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((120, 20)), rng.standard_normal((150, 20))


def test_r0_is_local_mean_subtraction(data):
    X_T, X_B = data
    local = rg.local_background(X_T, X_B, 15)
    assert np.array_equal(rg.residuals(X_T, local, 0), X_T - local.mu)
    assert np.allclose(local.mu, X_B[local.neighbors].mean(axis=1))


def test_residuals_orthogonal_to_their_directions(data):
    X_T, X_B = data
    local = rg.local_background(X_T, X_B, 30)
    R = rg.residuals(X_T, local, 5)
    assert np.abs(np.einsum("nd,ndr->nr", R, local.V[:, :, :5])).max() < 1e-10
    assert np.allclose(np.einsum("ndr,nds->nrs", local.V, local.V), np.eye(local.V.shape[2]))


def test_sign_flips_do_not_change_residuals(data):
    X_T, X_B = data
    local = rg.local_background(X_T, X_B, 30)
    flipped = rg.LocalBackground(local.k_loc, local.mu, local.V * np.where(np.arange(10) % 2, -1, 1),
                                 local.neighbors)
    assert np.allclose(rg.residuals(X_T, local, 5), rg.residuals(X_T, flipped, 5), atol=1e-12)


def test_rebuild_bit_identical(data):
    X_T, X_B = data
    G1, R1 = rg.residual_graph(X_T, X_B, 30, 2)
    G2, R2 = rg.residual_graph(X_T, X_B, 30, 2)
    assert np.array_equal(R1, R2) and (G1 != G2).nnz == 0


def test_r_capped_at_k_loc_minus_one(data):
    X_T, X_B = data
    local = rg.local_background(X_T, X_B, 5)
    assert local.V.shape[2] == 4
    with pytest.raises(ValueError):
        rg.residuals(X_T, local, 5)


def test_toy_recovers_offset_orthogonal_to_local_planes():
    """Background on 4 well-separated local 2-D planes; target = background point + an offset (length 0.5) orthogonal
    to its plane. With r = 2 the residual recovers the offset; with r = 0 an in-plane error larger than the offset
    itself remains (the local mean removes only part of the in-plane position)."""
    rng = np.random.default_rng(1)
    D, per = 30, 400
    X_B, planes, centers = [], [], []
    for c in range(4):
        center = 100 * rng.standard_normal(D)
        P = np.linalg.qr(rng.standard_normal((D, 2)))[0]
        X_B.append(center + rng.standard_normal((per, 2)) * 3 @ P.T)
        planes.append(P)
        centers.append(center)
    X_B = np.vstack(X_B)
    cluster = rng.integers(4, size=200)
    offsets = np.empty((200, D))
    X_T = np.empty((200, D))
    for i, c in enumerate(cluster):
        o = rng.standard_normal(D)
        o -= planes[c] @ (planes[c].T @ o)
        offsets[i] = 0.5 * o / np.linalg.norm(o)
        X_T[i] = centers[c] + planes[c] @ (rng.standard_normal(2) * 3) + offsets[i]
    local = rg.local_background(X_T, X_B, 100)
    err2 = np.linalg.norm(rg.residuals(X_T, local, 2) - offsets, axis=1).mean()
    err0 = np.linalg.norm(rg.residuals(X_T, local, 0) - offsets, axis=1).mean()
    assert err2 < 0.05 and err0 > 0.5
