"""Tests for the cPCA baseline (baselines/cpca.py)."""
import numpy as np

from baselines.cpca import ALPHAS, directions, fit, project, representative_alphas, select_alpha, variance_ratio

rng = np.random.default_rng(0)


def _toy(n=600):
    """Target: a high-variance shared background direction (dim 0) plus a 2-group signal on dim 1; background: dim 0
    only. Small isotropic noise everywhere."""
    groups = rng.integers(2, size=n)
    X_T = 0.1 * rng.standard_normal((n, 4))
    X_T[:, 0] += 10 * rng.standard_normal(n)
    X_T[:, 1] += 3.0 * (2 * groups - 1)
    X_B = 0.1 * rng.standard_normal((n, 4))
    X_B[:, 0] += 10 * rng.standard_normal(n)
    return X_T, X_B, groups


def test_alpha_zero_is_pca():
    X_T, X_B, _ = _toy()
    f = fit(X_T, X_B)
    V = directions(f, 0.0, 2)
    _, _, vt = np.linalg.svd(X_T - X_T.mean(axis=0), full_matrices=False)
    # same 2-D subspace as the top-2 principal components (projector equality)
    assert np.allclose(V @ V.T, vt[:2].T @ vt[:2], atol=1e-8)


def test_large_alpha_removes_background_direction():
    X_T, X_B, groups = _toy()
    f = fit(X_T, X_B)
    assert abs(directions(f, 0.0, 1)[0, 0]) > 0.99  # PCA: the background direction dominates
    v = directions(f, 10.0, 1)[:, 0]
    assert abs(v[1]) > 0.99 and abs(v[0]) < 0.05  # cPCA: the signal direction
    z = project(f, 10.0, 1)[:, 0]
    assert abs(z[groups == 1].mean() - z[groups == 0].mean()) > 5


def test_signs_fixed_and_repeatable():
    X_T, X_B, _ = _toy()
    f = fit(X_T, X_B)
    a, b = directions(f, 2.0, 3), directions(fit(X_T, X_B), 2.0, 3)
    assert np.array_equal(a, b)
    assert np.all(a[np.abs(a).argmax(axis=0), np.arange(3)] > 0)


def test_variance_ratio_by_hand():
    X_T, X_B, _ = _toy()
    f = fit(X_T, X_B)
    V = directions(f, 5.0, 2)
    by_hand = np.trace(V.T @ np.cov(X_T, rowvar=False) @ V) / np.trace(V.T @ np.cov(X_B, rowvar=False) @ V)
    assert np.isclose(variance_ratio(f, 5.0), by_hand)


def test_selection_deterministic_and_well_formed():
    X_T, X_B, _ = _toy()
    f = fit(X_T, X_B)
    first, second = select_alpha(f, seed=0), select_alpha(fit(X_T, X_B), seed=0)
    assert first[0] == second[0] and first[1] == second[1]
    reps, labels = representative_alphas(f, seed=0)
    assert reps[0] == 0.0 and set(reps) <= set(ALPHAS) and len(labels) == len(ALPHAS)
    assert first[0] in first[1]
