"""Neg-t-SNE fitting helpers shared by all phases: explicit init, single fits, and the two-stage schedule (Plan §4)."""
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

import cne
from cne._cne import rescale

cne_core = sys.modules["cne.cne"]  # module holding ContrastiveEmbedding and new_lr

N1, N2 = 250, 500


def rescale_init(Y):
    """Rescale a 2-D layout with the package's own helper so its first column has std 1; float32."""
    return rescale(np.array(Y, dtype=np.float32), target_std=1.0).astype(np.float32)


def pca_init(X):
    """First 2 PCs of X (deterministic full SVD), rescaled so PC1 has std 1 (paper Supp. K.2); float32."""
    return rescale_init(PCA(n_components=2, svd_solver="full").fit_transform(X))


def fit_neg(init, graph, seed, **kwargs):
    """Fit non-parametric Neg-t-SNE from `init` on `graph`.

    Returns the embedding, the per-epoch learning rates actually set by the package, the embedding weights
    at the start of training (read via the package's epoch -1 callback), the runtime, and the resolved params.
    """
    lrs, start = [], {}
    orig_new_lr = cne_core.new_lr

    def recording_new_lr(*args, **kw):
        lr = orig_new_lr(*args, **kw)
        lrs.append(lr)
        return lr

    def callback(epoch, model, *_):
        if epoch == -1:
            start["weight"] = model.weight.detach().cpu().numpy().copy()

    cne_core.new_lr = recording_new_lr
    try:
        embedder = cne.CNE(loss_mode="neg", seed=seed, callback=callback, **kwargs)
        t0 = time.time()
        embd = embedder.fit_transform(init=init, graph=graph)
        runtime = time.time() - t0
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
    return {"embd": embd, "lrs": lrs, "start": start["weight"], "runtime_s": runtime, "params": params}


def two_stage(init, graph, seed, n1=N1, n2=N2, **stage2_kwargs):
    """Stage 1: plain Neg-t-SNE for n1 epochs from `init`. Stage 2: a new fit for n2 epochs from the Stage 1
    embedding (float32, unchanged), fresh LR annealing, early exaggeration off, plus `stage2_kwargs`."""
    stage1 = fit_neg(init, graph, seed, n_epochs=n1)
    stage1_embd = stage1["embd"].astype(np.float32)
    stage2 = fit_neg(stage1_embd, graph, seed, n_epochs=n2, early_exaggeration=False, **stage2_kwargs)
    return stage1, stage2
