"""Tests for the background-negative term (method/bg_term.py) and its trainer hook (Plan §4)."""
import sys

import cne
import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from common.embed import fit_neg, pca_init, stage2, two_stage
from common.knn import knn_graph
from common.repro import setup_determinism
from data.synthetic import MAIN_RHO, make_dataset
from method.bg_term import BackgroundTerm
from method.xi_b import XiB, build_xi_b

cne_core = sys.modules["cne.cne"]
N2_TEST = 8  # short Stage 2 fits; bit-identity does not depend on length


@pytest.fixture(autouse=True)
def determinism():
    setup_determinism(0)


@pytest.fixture(scope="module")
def setup():
    setup_determinism(0)
    ds = make_dataset("easy", n_T=600, n_B=600, seed=0, signal_to_background=MAIN_RHO["easy"])
    G = knn_graph(ds.X_T, k=15)
    stage1 = fit_neg(pca_init(ds.X_T), G, seed=0, n_epochs=20)
    return {"G": G, "stage1": stage1["embd"].astype(np.float32), "xi": build_xi_b(ds.X_T, ds.X_B, seed=0)}


def _criterion(aggregation="sum", negative_samples=5, spec_param=1.0):
    return cne_core.ContrastiveLoss(negative_samples=negative_samples, loss_mode="neg",
                                    spec_param=torch.tensor(spec_param), eps=torch.tensor(1.0),
                                    clamp_low=1e-4, clamp_high=1.0, loss_aggregation=aggregation)


def _tiny_xi(rows, n):
    """XiB with the given per-row tail lists and uniform probabilities."""
    indptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
    probs = np.concatenate([np.full(len(r), 1.0 / len(r)) for r in rows if len(r)])
    return XiB(indptr=indptr, indices=np.concatenate([np.array(r, dtype=np.int64) for r in rows]),
               jaccard=probs.copy(), probs=probs, params={"n_T": n})


# 1. bit-identity when the term is off
def test_off_matches_plain_package(setup):
    G, init, xi = setup["G"], setup["stage1"], setup["xi"]
    plain = cne.CNE(loss_mode="neg", seed=0, n_epochs=N2_TEST, early_exaggeration=False).fit_transform(
        init=init.copy(), graph=G)
    off = [{"bg_mode": "none"}, {"bg_mode": "xi_b", "m_B": 0, "xi": xi}, {"bg_mode": "uniform_extra", "m_B": 0}]
    for kwargs in off:
        out = stage2(init, G, seed=0, n2=N2_TEST, **kwargs)
        assert np.array_equal(out["embd"], plain), kwargs


def test_two_stage_starts_from_stage1(setup):
    G = setup["G"]
    s1, s2 = two_stage(setup["stage1"], G, seed=0, n1=3, n2=3, bg_mode="xi_b", m_B=5, xi=setup["xi"])
    assert np.array_equal(s2["start"], s1["embd"].astype(np.float32))
    assert all(v == 0 for v in s1["log"]["bg_share"]) and all(v > 0 for v in s2["log"]["bg_share"])


# 2. the global torch RNG is untouched by the sampler
def test_global_rng_untouched(setup):
    xi = setup["xi"]
    model = torch.nn.Embedding.from_pretrained(torch.from_numpy(setup["stage1"])).requires_grad_(True)
    item, neigh = torch.arange(0, 40), torch.arange(40, 80)
    features = model(torch.cat([item, neigh]))
    criterion = _criterion()
    criterion(features, force_resample=True)
    for hook in [BackgroundTerm("xi_b", 5, n=xi.n, xi=xi), BackgroundTerm("uniform_extra", 5, n=xi.n)]:
        state = torch.get_rng_state()
        hook(item, neigh, features, model, criterion, 0)
        assert torch.equal(state, torch.get_rng_state())


# 3. the xi_B sampler follows probs
def test_xi_sampler_frequencies(setup):
    xi = setup["xi"]
    hook = BackgroundTerm("xi_b", 1, n=xi.n, xi=xi, seed=0)
    for i in [0, 17, 311, 599]:
        lo, hi = xi.indptr[i], xi.indptr[i + 1]
        tails, valid = hook.sample(torch.full((200_000,), i))
        assert valid.all()
        pos = {j: k for k, j in enumerate(xi.indices[lo:hi])}
        assert set(tails.unique().tolist()) <= set(pos)  # never leaves its row
        counts = np.bincount([pos[t] for t in tails.ravel().tolist()], minlength=hi - lo)
        assert chisquare(counts, xi.probs[lo:hi] * counts.sum()).pvalue > 1e-4


def test_empty_rows_masked():
    xi = _tiny_xi([[1, 2], [], [0]], n=3)
    hook = BackgroundTerm("xi_b", 4, n=3, xi=xi)
    tails, valid = hook.sample(torch.tensor([0, 1, 2]))
    assert valid.tolist() == [True, False, True]
    assert set(tails[0].tolist()) <= {1, 2} and tails[2].tolist() == [0] * 4
    model = torch.nn.Embedding.from_pretrained(torch.randn(3, 2)).requires_grad_(True)
    item, neigh = torch.tensor([1]), torch.tensor([0])
    criterion = _criterion(negative_samples=1)
    criterion.neigh_inds = torch.tensor([[1, 1]])
    out = hook(item, neigh, model(torch.cat([item, neigh])), model, criterion, 0)
    assert float(out.detach()) == 0.0 and hook.log[0]["n_masked_heads"] == 1


# 4. uniform_extra: never the head, uniform over the rest
def test_uniform_sampler():
    hook = BackgroundTerm("uniform_extra", 1, n=50, seed=0)
    tails, valid = hook.sample(torch.full((200_000,), 7))
    assert valid.all() and not (tails == 7).any()
    counts = np.bincount(tails.ravel().numpy(), minlength=50)
    assert chisquare(np.delete(counts, 7)).pvalue > 1e-4


# 5. value and sign of the term
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_value_and_gradient_sign(aggregation):
    Y = torch.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [0.7, -0.4]])
    model = torch.nn.Embedding.from_pretrained(Y.clone()).requires_grad_(True)
    xi = _tiny_xi([[3], [0], [0], [0]], n=4)
    hook = BackgroundTerm("xi_b", 2, gamma=1.0, n=4, xi=xi)
    item, neigh = torch.tensor([0]), torch.tensor([1])
    criterion = _criterion(aggregation, negative_samples=1)
    criterion.neigh_inds = torch.tensor([[1, 1]])  # b = 1, m = 1
    out = hook(item, neigh, model(torch.cat([item, neigh])), model, criterion, 0)
    d2 = float(((Y[0] - Y[3]) ** 2).sum())
    by_hand = 2 * -np.log(1 - 1 / (1 + 1.0 * (d2 + 1.0)))
    if aggregation == "mean":
        by_hand /= 1 * (1 + 1)  # b (m + 1)
    assert np.isclose(float(out.detach()), by_hand, rtol=1e-6)
    out.backward()
    g = model.weight.grad
    assert torch.all(g[[1, 2]] == 0)  # rows not sampled get no gradient
    assert float(g[0] @ (Y[0] - Y[3])) < 0 and float(g[3] @ (Y[3] - Y[0])) < 0  # descent pushes the pair apart


# 6. one background negative counts exactly like one uniform in-batch negative
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_background_negative_weighted_like_uniform_negative(aggregation):
    Y = torch.tensor([[0.3, -0.2], [1.1, 0.4]])
    item, neigh = torch.tensor([0]), torch.tensor([1])
    criterion = _criterion(aggregation, negative_samples=2)

    model = torch.nn.Embedding.from_pretrained(Y.clone()).requires_grad_(True)
    features = model(torch.cat([item, neigh]))
    criterion.neigh_inds = torch.tensor([[1, 1, 1]])  # both uniform negatives at the positive tail's slot: m = 2
    L2 = criterion(features, force_resample=False)
    criterion.neigh_inds = torch.tensor([[1]])  # attraction only
    L0 = criterion(features, force_resample=False)
    contribution = (L2 - L0) / 2 if aggregation == "sum" else (L2 - L0 / 3) / 2
    (g_uniform,) = torch.autograd.grad(contribution, model.weight)

    model_b = torch.nn.Embedding.from_pretrained(Y.clone()).requires_grad_(True)
    features_b = model_b(torch.cat([item, neigh]))
    criterion.neigh_inds = torch.tensor([[1, 1, 1]])  # the hook sees the same m
    hook = BackgroundTerm("xi_b", 1, gamma=1.0, n=2, xi=_tiny_xi([[1], [0]], n=2))
    bg = hook(item, neigh, features_b, model_b, criterion, 0)  # its one tail is row 1: the same point, same phi
    (g_bg,) = torch.autograd.grad(bg, model_b.weight)

    np.testing.assert_allclose(float(bg.detach()), float(contribution.detach()), rtol=1e-6)
    np.testing.assert_allclose(g_bg.numpy(), g_uniform.numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(g_bg.norm()), float(g_uniform.norm()), rtol=1e-6)


# 7. logging matches the package's own batch losses
@pytest.mark.parametrize("kwargs", [{"bg_mode": "none"}, {"bg_mode": "xi_b", "m_B": 5}])
def test_logged_components_match_package_loss(setup, kwargs):
    out = stage2(setup["stage1"], setup["G"], seed=0, n2=N2_TEST, xi=setup["xi"], **kwargs)
    log = out["log"]
    package = [sum(bl) for bl in out["batch_losses"]]
    logged = [a + u + w for a, u, w in zip(log["attraction"], log["uniform_repulsion"], log["bg_weighted"])]
    np.testing.assert_allclose(logged, package, rtol=1e-5)
    assert log["epoch"] == list(range(N2_TEST)) and sum(log["n_nonfinite_batches"]) == 0
    assert all(e == setup["G"].nnz for e in log["n_edges"])


# 8. determinism and effect
def test_deterministic_and_distinct(setup):
    G, init, xi = setup["G"], setup["stage1"], setup["xi"]
    a = stage2(init, G, seed=0, n2=N2_TEST, bg_mode="xi_b", m_B=5, xi=xi)["embd"]
    b = stage2(init, G, seed=0, n2=N2_TEST, bg_mode="xi_b", m_B=5, xi=xi)["embd"]
    u = stage2(init, G, seed=0, n2=N2_TEST, bg_mode="uniform_extra", m_B=5)["embd"]
    none = stage2(init, G, seed=0, n2=N2_TEST)["embd"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, u) and not np.array_equal(a, none)
    assert np.isfinite(a).all() and np.isfinite(u).all()


def test_bad_arguments(setup):
    with pytest.raises(ValueError):
        BackgroundTerm("other")
    with pytest.raises(ValueError):
        BackgroundTerm("xi_b", 5, n=600)  # no xi
    with pytest.raises(ValueError):
        BackgroundTerm("xi_b", 5, n=10, xi=setup["xi"])  # wrong size
    with pytest.raises(ValueError):
        BackgroundTerm("uniform_extra", 2.5, n=10)
    with pytest.raises(ValueError):
        BackgroundTerm("uniform_extra", 2, gamma=-1, n=10)
