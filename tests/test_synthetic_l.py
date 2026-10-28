"""Tests for the Phase 5b generator (data/synthetic_l.py) and the Design L projections (common/benchmark_l.py)."""
import numpy as np
import pytest

from common.benchmark_l import clean_reveals, projections, score
from data.synthetic_l import N_BLOCKS, SIGMA_W, make_l
from eval.benchmark_metrics import g1_k200, per_cluster_wc, within_cluster_accuracy
from eval.global_metrics import cross_cluster_same_signal_rate

N = 900


@pytest.fixture(scope="module", params=["cyclic", "between"])
def ds(request):
    return make_l(request.param, seed=1, rho=0.3, n_T=N, n_B=N)


def test_orthogonal_blocks(ds):
    t = ds.truth
    if ds.params["variant"] == "cyclic":
        basis = np.hstack([t["B"]] + list(t["Q"]))
    else:
        basis = np.hstack([t["B"], t["V"]])
        assert np.allclose(t["W"].T @ t["W"], np.eye(3)) and np.abs(t["V"].T @ t["W"]).max() < 1e-12
        assert np.allclose(t["B"] @ (t["B"].T @ t["W"]), t["W"])  # W lies inside U
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)


def test_exact_balance_same_in_target_and_background(ds):
    counts_T = np.bincount(ds.truth["y_b"])
    counts_B = np.bincount(ds.truth["y_b_background"])
    assert np.array_equal(counts_T, counts_B) and counts_T.max() - counts_T.min() <= 1


def test_signal_location_and_background_without_signal(ds):
    t = ds.truth
    for k in range(6):
        S = t["S_T"][t["y_b"] == k]
        M = t["Q"][k] if ds.params["variant"] == "cyclic" else t["W"]
        assert np.abs(S - S @ M @ M.T).max() < 1e-12  # cluster k's signal lies only in its own block / in W
    # X_B = centers + nuisance + noise: removing the cluster mean and the nuisance leaves only noise (std 0.1)
    X_B, y = ds.X_B.astype(np.float64), t["y_b_background"]
    for k in range(6):
        R = X_B[y == k] - X_B[y == k].mean(axis=0)
        N_k = t["Q"][(k + 1) % 6] if ds.params["variant"] == "cyclic" else t["V"]
        R = R - R @ N_k @ N_k.T
        assert abs(R.std() - 0.1) < 0.01


def test_cyclic_nuisance_is_next_clusters_signal():
    ds = make_l("cyclic", seed=0, rho=0.3, n_T=N, n_B=N)
    t = ds.truth
    for k in range(6):
        bg = t["Bg_T"][t["y_b"] == k]
        bg = bg - bg.mean(axis=0)
        Qn = t["Q"][(k + 1) % 6]
        assert np.abs(bg - bg @ Qn @ Qn.T).max() < 1e-10  # within-cluster spread lies in Q_{k+1}
        assert abs(np.sqrt((bg @ Qn).var(axis=0).mean()) - SIGMA_W) < 0.15


def test_deterministic():
    a, b = make_l("cyclic", seed=4, rho=0.2, n_T=N, n_B=N), make_l("cyclic", seed=4, rho=0.2, n_T=N, n_B=N)
    assert np.array_equal(a.X_T, b.X_T) and np.array_equal(a.X_B, b.X_B)


def test_projections_and_clean_reveals(ds):
    projs = list(projections(ds))
    assert len(projs) == (2 ** N_BLOCKS if ds.params["variant"] == "cyclic" else 2)
    for _, _, basis in projs:
        assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    assert clean_reveals([0, 2, 4]) == 3 and clean_reveals(range(6)) == 0 and clean_reveals([0]) == 1
    assert max(clean_reveals([i for i in range(6) if m >> i & 1]) for m in range(64)) == 3


def test_metric_helpers(ds):
    t = ds.truth
    accs, sizes = per_cluster_wc(ds.X_T, t["y_s"], t["y_b"])
    assert np.isclose((sizes * accs).sum() / sizes.sum(), within_cluster_accuracy(ds.X_T, t["y_s"], t["y_b"]))
    assert g1_k200(ds.X_T, t["y_s"], t["y_b"]) == cross_cluster_same_signal_rate(ds.X_T, t["y_s"], t["y_b"], k=200)
    s = score(ds.X_T[:, :2], t)
    assert set(f"WC_c{i}" for i in range(6)) <= set(s)
