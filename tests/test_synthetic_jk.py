"""Tests for the Phase 5a generators (data/synthetic_j.py, data/synthetic_k.py)."""
import numpy as np
import pytest

from data.synthetic import make_dataset
from data.synthetic_j import SIGNAL_TOTAL_VAR, make_j
from data.synthetic_k import make_k

N = 800


@pytest.fixture(scope="module")
def easy():
    return make_dataset("easy", n_T=N, n_B=N, seed=3, signal_to_background=0.2)


def test_j_theta0_is_easy_rescaled(easy):
    ds = make_j(seed=3, rho=0.7, theta=0, n_T=N, n_B=N)
    assert np.array_equal(ds.X_B, easy.X_B)
    assert np.array_equal(ds.truth["Bg_T"], easy.truth["Bg_T"])
    ratio = ds.truth["signal_T"] / easy.truth["S_T"]
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-12)  # exactly proportional to easy's S_T
    assert np.isclose(ratio.flat[0], 0.7 / 0.2 / ds.params["map_info"]["scale_A"])
    assert np.isclose(ds.params["map_info"]["scale_A"] ** 2, SIGNAL_TOTAL_VAR, rtol=0.1)


def test_j_theta0_independent_of_projection_and_gain():
    ref = make_j(seed=1, rho=0.8, theta=0, n_T=N, n_B=N)
    for projected in (True, False):
        for gain in (2.0, 6.0):
            ds = make_j(seed=1, rho=0.8, theta=0, projected=projected, gain=gain, n_T=N, n_B=N)
            assert np.array_equal(ds.X_T, ref.X_T) and np.array_equal(ds.X_B, ref.X_B)


@pytest.mark.parametrize("theta", [30, 60, 90])
def test_j_signal_strength_fixed_and_projection(theta):
    for projected in (True, False):
        ds = make_j(seed=0, rho=1.0, theta=theta, projected=projected, gain=2.0, n_T=N, n_B=N)
        sig, B = ds.truth["signal_T"], ds.truth["B"]
        # unit-variance parts mixed by (cos, sin): total variance 1 + 2 cos sin <As~, Delta~>, so between 0 and 2
        assert 0 < sig.var(axis=0).sum() < 2
        if projected:  # A is orthogonal to span(B), and so is the projected Delta
            assert np.abs(sig @ B).max() < 1e-10
        else:
            assert np.abs(sig @ B).max() > 1e-3 and ds.params["map_info"]["frac_Delta_var_in_span_B"] > 0
        np.testing.assert_array_equal(ds.X_B, make_dataset("easy", n_T=N, n_B=N, seed=0).X_B)


def test_j_theta90_is_pure_interaction():
    ds = make_j(seed=0, rho=1.0, theta=90, projected=True, gain=2.0, n_T=N, n_B=N)
    assert np.isclose(ds.truth["signal_T"].var(axis=0).sum(), 1.0)
    # the interaction vanishes at the reference signal s0 = 0: Delta(z_b, 0) = 0 is built in, so X_B has no signal
    assert "scale_Delta" in ds.params["map_info"]


def test_j_gain_changes_nonlinearity():
    lo = make_j(seed=0, theta=90, gain=2.0, n_T=N, n_B=N).params["map_info"]
    hi = make_j(seed=0, theta=90, gain=6.0, n_T=N, n_B=N).params["map_info"]
    assert hi["frac_abs_preactivation_gt_2"] > lo["frac_abs_preactivation_gt_2"]


def test_k_background_is_easy_and_maps_orthogonal(easy):
    ds = make_k(seed=3, rho=0.2, n_T=N, n_B=N)
    assert np.array_equal(ds.X_B, easy.X_B)
    assert np.array_equal(ds.truth["Bg_T"], easy.truth["Bg_T"])
    A_k, B = ds.truth["A_k"], ds.truth["B"]
    for A in A_k:
        assert np.allclose(A.T @ A, np.eye(3), atol=1e-12) and np.abs(B.T @ A).max() < 1e-12
    S = ds.truth["S_T"]
    for c in range(len(A_k)):  # each cluster's signal lives in its own subspace
        rows = ds.truth["y_b"] == c
        resid = S[rows] - S[rows] @ A_k[c] @ A_k[c].T
        assert np.abs(resid).max() < 1e-12


def test_generators_deterministic():
    assert np.array_equal(make_j(seed=2, theta=60, n_T=N, n_B=N).X_T, make_j(seed=2, theta=60, n_T=N, n_B=N).X_T)
    assert np.array_equal(make_k(seed=2, n_T=N, n_B=N).X_T, make_k(seed=2, n_T=N, n_B=N).X_T)
