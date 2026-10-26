"""Tests for the synthetic target/background generator (data/synthetic.py)."""
import numpy as np
import pytest

from data.synthetic import DEFAULTS, LEVELS, make_dataset

N = 2000  # smaller than the default for speed; structure checks do not depend on n


@pytest.fixture(scope="module", params=LEVELS)
def ds(request):
    return make_dataset(request.param, n_T=N, n_B=N + 100, seed=0, signal_to_background=0.3)


def _within_cluster_linear_r2(z, Y, labels):
    """Mean over clusters of the R^2 of an affine fit of Y on z within each cluster."""
    r2 = []
    for k in np.unique(labels):
        m = labels == k
        Z = np.c_[z[m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(Z, Y[m], rcond=None)
        r2.append(1 - ((Y[m] - Z @ coef) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
    return float(np.mean(r2))


def test_shapes_dtypes_labels(ds):
    assert ds.X_T.shape == (N, 50) and ds.X_B.shape == (N + 100, 50)
    assert ds.X_T.dtype == np.float32 and ds.X_B.dtype == np.float32
    t = ds.truth
    assert set(np.unique(t["y_b"])) == set(range(DEFAULTS["K_b"]))
    assert set(np.unique(t["y_b_background"])) == set(range(DEFAULTS["K_b"]))
    assert set(np.unique(t["y_s"])) == set(range(DEFAULTS["K_s"]))


@pytest.mark.parametrize("level", LEVELS)
def test_deterministic(level):
    a = make_dataset(level, n_T=500, n_B=500, seed=3, signal_to_background=0.4)
    b = make_dataset(level, n_T=500, n_B=500, seed=3, signal_to_background=0.4)
    c = make_dataset(level, n_T=500, n_B=500, seed=4, signal_to_background=0.4)
    assert np.array_equal(a.X_T, b.X_T) and np.array_equal(a.X_B, b.X_B)
    assert not np.array_equal(a.X_T, c.X_T)


@pytest.mark.parametrize("level", LEVELS)
def test_rho_scales_only_the_signal(level):
    a = make_dataset(level, n_T=500, n_B=500, seed=0, signal_to_background=1.0)
    b = make_dataset(level, n_T=500, n_B=500, seed=0, signal_to_background=2.0)
    assert np.array_equal(b.truth["S_T"], 2 * a.truth["S_T"])
    assert np.array_equal(a.truth["Bg_T"], b.truth["Bg_T"])
    assert np.array_equal(a.X_B, b.X_B)


def test_n_B_does_not_change_target():
    a = make_dataset("easy", n_T=500, n_B=500, seed=0)
    b = make_dataset("easy", n_T=500, n_B=900, seed=0)
    assert np.array_equal(a.X_T, b.X_T)


def test_decomposition(ds):
    t = ds.truth
    resid = ds.X_T.astype(np.float64) - t["S_T"] - t["Bg_T"]
    assert abs(resid.std() - DEFAULTS["noise_std"]) < 0.005
    # the background has no signal component: with A orthogonal to the background (easy), X_B along A is noise only
    if ds.params["level"] == "easy":
        assert abs((ds.X_B @ t["A"]).std() - DEFAULTS["noise_std"]) < 0.01


def test_simplex_centers(ds):
    assert ds.params["min_center_distance_b"] == pytest.approx(DEFAULTS["edge_b"])
    assert ds.params["min_center_distance_s"] == pytest.approx(DEFAULTS["edge_s"])


def test_easy_maps():
    t = make_dataset("easy", n_T=200, n_B=200).truth
    A, B = t["A"], t["B"]
    np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(B.T @ B, np.eye(5), atol=1e-6)
    np.testing.assert_allclose(B.T @ A, 0, atol=1e-6)


def test_medium_maps():
    ds = make_dataset("medium", n_T=200, n_B=200)
    A, B = ds.truth["A"], ds.truth["B"]
    np.testing.assert_allclose(A.T @ A, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(B.T @ B, np.eye(5), atol=1e-6)
    assert np.linalg.norm(B.T @ A) ** 2 / 3 == pytest.approx(DEFAULTS["overlap"], abs=1e-6)


def test_hard_map_is_nonlinear_without_collapse():
    ds = make_dataset("hard", n_T=N, n_B=N, seed=0)
    t = ds.truth
    np.testing.assert_allclose(t["A"].T @ t["A"], np.eye(3), atol=1e-6)
    # total background variance matches an isometric linear map of the same latent
    assert t["Bg_T"].var(0).sum() == pytest.approx(t["z_b"].var(0).sum(), rel=0.05)
    # genuinely nonlinear within clusters (global R^2 is uninformative: 6 centers in 5-D are always affine-fit)
    assert _within_cluster_linear_r2(t["z_b"], t["Bg_T"], t["y_b"]) <= 0.9
    # within-cluster background variation does not collapse (linear levels: d_b * sigma_b^2 = 5)
    within = np.mean([t["Bg_T"][t["y_b"] == k].var(0).sum() for k in range(DEFAULTS["K_b"])])
    assert within >= 0.5 * DEFAULTS["d_b"] * DEFAULTS["sigma_b"] ** 2


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_dataset("impossible")
    with pytest.raises(ValueError):
        make_dataset("easy", not_a_param=1)
    with pytest.raises(ValueError):
        make_dataset("easy", K_s=5)  # simplex centers need K = d + 1
