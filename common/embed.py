"""Neg-t-SNE fitting helpers shared by all phases: explicit init, single fits, and the two-stage schedule (Plan §4)."""
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

import cne
from cne._cne import rescale

from method.bg_term import BackgroundTerm

cne_core = sys.modules["cne.cne"]  # module holding ContrastiveEmbedding and new_lr

N1, N2 = 250, 500
N_SCALE_PAIRS = 100_000
# Explicit device: the package's "auto" would pick CUDA on a GPU node, where method/bg_term.py (CPU tensors) is
# untested. On the Mac "auto" already resolved to "cpu", so the default changes nothing there.
DEVICE = os.environ.get("CNE_DEVICE", "cpu")


def rescale_init(Y):
    """Rescale a 2-D layout with the package's own helper so its first column has std 1; float32."""
    return rescale(np.array(Y, dtype=np.float32), target_std=1.0).astype(np.float32)


def pca_init(X):
    """First 2 PCs of X (deterministic full SVD), rescaled so PC1 has std 1 (paper Supp. K.2); float32."""
    return rescale_init(PCA(n_components=2, svd_solver="full").fit_transform(X))


def scale_pairs(n, seed, n_pairs=N_SCALE_PAIRS):
    """Fixed uniform sample of pairs i != j for the embedding-scale diagnostic (its own numpy stream)."""
    rng = np.random.default_rng([seed, 2])
    i = rng.integers(n, size=n_pairs)
    j = rng.integers(n - 1, size=n_pairs)
    return torch.from_numpy(i), torch.from_numpy(j + (j >= i))


def fit_neg(init, graph, seed, batch_hook=None, **kwargs):
    """Fit non-parametric Neg-t-SNE from `init` on `graph`, with an optional trainer `batch_hook`.

    Returns the embedding, the per-epoch learning rates actually set by the package, the embedding weights
    at the start of training (read via the package's epoch -1 callback), the runtime, the resolved params, and a
    per-epoch `scale` log (epoch -1 = start): mean Cauchy kernel phi = 1 / (1 + d^2) over a fixed pair sample, its
    all-pairs sum estimate, the embedding std, and whether the embedding is finite.
    """
    lrs, start = [], {}
    n = len(init)
    pair_i, pair_j = scale_pairs(n, seed)
    scale = {"epoch": [], "mean_phi": [], "sum_phi_est": [], "embd_std": [], "finite": []}
    orig_new_lr = cne_core.new_lr

    def recording_new_lr(*args, **kw):
        lr = orig_new_lr(*args, **kw)
        lrs.append(lr)
        return lr

    def callback(epoch, model, *_):
        w = model.weight.detach()
        if epoch == -1:
            start["weight"] = w.cpu().numpy().copy()
        with torch.no_grad():
            mean_phi = float((1 / (1 + ((w[pair_i] - w[pair_j]) ** 2).sum(1))).mean())
        scale["epoch"].append(epoch)
        scale["mean_phi"].append(mean_phi)
        scale["sum_phi_est"].append(mean_phi * n * (n - 1))
        scale["embd_std"].append(float(w.std(dim=0).mean()))
        scale["finite"].append(bool(torch.isfinite(w).all()))

    cne_core.new_lr = recording_new_lr
    try:
        embedder = cne.CNE(loss_mode="neg", seed=seed, callback=callback, save_freq=1, batch_hook=batch_hook,
                           **{"device": DEVICE, **kwargs})
        t0 = time.perf_counter()  # pauses while the machine sleeps (mach_absolute_time on macOS)
        embd = embedder.fit_transform(init=init, graph=graph)
        runtime = time.perf_counter() - t0
    finally:
        cne_core.new_lr = orig_new_lr

    c = embedder.cne
    params = {
        "k": embedder.k, "seed": seed, "n_epochs": c.n_epochs, "batch_size": c.batch_size,
        "negative_samples": c.negative_samples, "optimizer": c.optimizer, "learning_rate": c.learning_rate,
        "momentum": c.momentum, "anneal_lr": c.anneal_lr, "lr_min_factor": c.lr_min_factor,
        "spec_param": float(c.spec_param), "neg_spec": c.neg_spec, "s": c.s, "Z_bar": c.Z_bar, "eps": c.eps,
        "clamp_low": c.clamp_low, "clamp_high": c.clamp_high, "loss_aggregation": c.loss_aggregation,
        "early_exaggeration": c.early_exaggeration, "force_resample": c.force_resample,
        "clip_grad": c.clip_grad, "device": c.device,
    }
    out = {"embd": embd, "lrs": lrs, "start": start["weight"], "runtime_s": runtime, "params": params, "scale": scale,
           "batch_losses": [[float(v) for v in bl] for bl in c.losses]}
    if isinstance(batch_hook, BackgroundTerm):
        out["params"]["background"] = batch_hook.params()
        out["log"] = batch_hook.epoch_log()
    return out


def stage2(stage1_embd, graph, seed, n2=N2, bg_mode="none", m_B=0, gamma=1.0, xi=None, **kwargs):
    """Stage 2 alone: a new fit for n2 epochs from the Stage 1 embedding (float32, unchanged), fresh LR annealing,
    early exaggeration off, and the requested background term (logging only for bg_mode="none")."""
    hook = BackgroundTerm(bg_mode, m_B, gamma, n=len(stage1_embd), xi=xi, seed=seed)
    return fit_neg(np.asarray(stage1_embd, dtype=np.float32), graph, seed, batch_hook=hook, n_epochs=n2,
                   early_exaggeration=False, **kwargs)


def two_stage(init, graph, seed, n1=N1, n2=N2, bg_mode="none", m_B=0, gamma=1.0, xi=None, **stage2_kwargs):
    """Stage 1: plain Neg-t-SNE for n1 epochs from `init` (logging only). Stage 2: `stage2` from its embedding with
    the requested background term and `stage2_kwargs`."""
    stage1 = fit_neg(init, graph, seed, batch_hook=BackgroundTerm("none"), n_epochs=n1)
    return stage1, stage2(stage1["embd"], graph, seed, n2=n2, bg_mode=bg_mode, m_B=m_B, gamma=gamma, xi=xi,
                          **stage2_kwargs)
