"""Tests for the local (cluster-then-cPCA) baseline (baselines/local_cpca.py)."""
import numpy as np
import pytest

from baselines import cpca
from baselines import local_cpca as lc
from data.synthetic_l import make_l

N = 900


@pytest.fixture(scope="module", params=["cyclic", "between"])
def ds(request):
    return make_l(request.param, seed=1, rho=0.2, n_T=N, n_B=N)


def test_labelfree_clustering_finds_the_six_clusters(ds):
    lab_T, lab_B, info = lc.clusters_for(ds, "labelfree", seed=1)
    assert info["K"] == 6 and info["ARI_background"] == 1.0 and info["ARI_target"] == 1.0
    assert set(info["silhouettes"]) == {str(k) for k in lc.K_RANGE}
    assert lc.check_sizes(lab_T, lab_B) == []


@pytest.mark.parametrize("version", ["labelfree", "oracle"])
def test_representation_shape_and_no_cross_cluster_edges(ds, version):
    lab_T, lab_B, info = lc.clusters_for(ds, version, seed=1)
    per = lc.local_alphas(ds.X_T, ds.X_B, lab_T, lab_B, seed=1)
    for k in (3, 10):
        Z, rep = lc.representation(ds.X_T, lab_T, per, k)
        assert Z.shape == (N, k + info["K"])
        assert rep["cross_cluster_knn_frac"] == 0.0
        assert rep["s"] == lc.SCALE * rep["R"]


def test_per_cluster_alpha_is_select_alpha_on_that_cluster(ds):
    lab_T, lab_B, _ = lc.clusters_for(ds, "oracle", seed=1)
    per = lc.local_alphas(ds.X_T, ds.X_B, lab_T, lab_B, seed=1)
    j = 2
    f = cpca.fit(ds.X_T[lab_T == j], ds.X_B[lab_B == j])
    assert per[j]["alpha"] == cpca.select_alpha(f, 1)[0]


def test_local_coordinates_are_the_cluster_projection(ds):
    lab_T, lab_B, _ = lc.clusters_for(ds, "oracle", seed=1)
    per = lc.local_alphas(ds.X_T, ds.X_B, lab_T, lab_B, seed=1)
    Z, _ = lc.representation(ds.X_T, lab_T, per, 3)
    j = 4
    assert np.array_equal(Z[lab_T == j, :3], cpca.project(per[j]["fit"], per[j]["alpha"], 3))


def test_deterministic(ds):
    out = []
    for _ in range(2):
        lab_T, lab_B, _ = lc.clusters_for(ds, "labelfree", seed=1)
        per = lc.local_alphas(ds.X_T, ds.X_B, lab_T, lab_B, seed=1)
        out.append(lc.representation(ds.X_T, lab_T, per, 5)[0])
    assert np.array_equal(out[0], out[1])


def test_check_sizes_flags_small_clusters():
    lab_T = np.array([0] * 60 + [1] * 10)
    lab_B = np.array([0] * 60 + [1] * 60)
    assert lc.check_sizes(lab_T, lab_B) == [{"cluster": 1, "n_T": 10, "n_B": 60}]
