"""Phase 0 sanity run: plain Neg-t-SNE on a 10k MNIST subset, plus a two-stage (Stage 1 -> Stage 2) dry run.

Run from anywhere with the cne-bg env:  python scripts/phase0_mnist_sanity.py
Outputs go to results/phase0/.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torchvision
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cne  # noqa: E402
from cne._cne import rescale  # noqa: E402
from eval.metrics import knn_accuracy  # noqa: E402

cne_core = sys.modules["cne.cne"]  # module holding ContrastiveEmbedding and new_lr

UPSTREAM_COMMIT = "d5f0b530f35be4d6247300a601fb5a860bc0a1c8"
OUT = ROOT / "results" / "phase0"
DATA = ROOT / "data"
PAPER_FIG = ROOT / "third_party" / "contrastive-ne" / "figures" / "negtsne_mnist.png"

SEED = 0
N_SUBSET = 10_000
N_PCA = 50
N1, N2 = 250, 500


def load_mnist_subset():
    """All 70k MNIST digits, flattened to raw float32 pixels; random N_SUBSET subset (seed SEED)."""
    parts = [torchvision.datasets.MNIST(DATA, train=t, download=True) for t in (True, False)]
    x = np.concatenate([p.data.numpy() for p in parts]).reshape(-1, 28 * 28).astype(np.float32)
    y = np.concatenate([p.targets.numpy() for p in parts])
    idx = np.random.default_rng(SEED).choice(len(x), N_SUBSET, replace=False)
    return x[idx], y[idx]


def fit_neg(init, graph, **kwargs):
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
        embedder = cne.CNE(loss_mode="neg", seed=SEED, callback=callback, **kwargs)
        t0 = time.time()
        embd = embedder.fit_transform(init=init, graph=graph)
        runtime = time.time() - t0
    finally:
        cne_core.new_lr = orig_new_lr

    c = embedder.cne
    params = {
        "k": embedder.k, "seed": SEED, "n_epochs": c.n_epochs, "batch_size": c.batch_size,
        "negative_samples": c.negative_samples, "optimizer": c.optimizer, "learning_rate": c.learning_rate,
        "momentum": c.momentum, "anneal_lr": c.anneal_lr, "lr_min_factor": c.lr_min_factor,
        "spec_param": float(c.spec_param), "neg_spec": c.neg_spec, "s": c.s, "Z_bar": c.Z_bar, "eps": c.eps,
        "clamp_low": c.clamp_low, "clamp_high": c.clamp_high, "loss_aggregation": c.loss_aggregation,
        "early_exaggeration": c.early_exaggeration, "force_resample": c.force_resample,
        "clip_grad": c.clip_grad, "device": c.device,
    }
    return {"embd": embd, "lrs": lrs, "start": start["weight"], "runtime_s": runtime, "params": params}


def scatter(ax, embd, y, title):
    ax.scatter(*embd.T, c=y, s=1.0, alpha=0.5, cmap="tab10", edgecolor="none", rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def save_embedding_plot(embd, y, title, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter(ax, embd, y, title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    x, y = load_mnist_subset()
    x50 = PCA(n_components=N_PCA, random_state=SEED).fit_transform(x).astype(np.float32)

    # kNN graph: package code path (annoy, k=15, symmetrized), built once on the 50-dim PCA and reused everywhere,
    # because the annoy index is unseeded. n_epochs=0 builds the graph without training.
    graph_builder = cne.CNE(loss_mode="neg", seed=SEED, n_epochs=0)
    graph_builder.fit(x50)
    G = graph_builder.neighbor_mat.tocsr()
    sp.save_npz(OUT / "knn_graph.npz", G)
    deg = np.diff(G.indptr)
    graph_stats = {
        "n": G.shape[0], "directed_edges": int(G.nnz), "degree_min": int(deg.min()),
        "degree_mean": float(deg.mean()), "degree_max": int(deg.max()),
        "symmetric": bool((G != G.T).nnz == 0), "weight_values": np.unique(G.data).tolist(),
        "self_loops": int(G.diagonal().sum()),
    }

    # Init: first 2 PCs (columns of the PCA-50 output), rescaled with the package's own helper so PC1 has std 1.
    pca_init = rescale(x50[:, :2].copy(), target_std=1.0).astype(np.float32)
    np.save(OUT / "pca_init.npy", pca_init)

    # Plain run: package defaults apart from the explicit init and graph.
    plain = fit_neg(pca_init, G)
    # Same run with early exaggeration off: identical iff early exaggeration is a no-op with neg defaults.
    plain_no_ee = fit_neg(pca_init, G, early_exaggeration=False)

    # Two-stage dry run.
    stage1 = fit_neg(pca_init, G, n_epochs=N1)
    stage1_embd = stage1["embd"].astype(np.float32)
    stage2_a = fit_neg(stage1_embd, G, n_epochs=N2, early_exaggeration=False)
    stage2_b = fit_neg(stage1_embd, G, n_epochs=N2, early_exaggeration=False)

    for name, run in [("plain", plain), ("stage1", stage1), ("stage2", stage2_a)]:
        np.save(OUT / f"{name}_embd.npy", run["embd"])
    np.save(OUT / "labels.npy", y)

    acc = {name: knn_accuracy(run["embd"], y) for name, run in
           [("plain", plain), ("stage1", stage1), ("stage2", stage2_a)]}

    def lr_resets(lrs):
        return [i for i in range(1, len(lrs)) if lrs[i] > lrs[i - 1]]

    checks = {
        "a_early_exaggeration_noop": bool(np.array_equal(plain["embd"], plain_no_ee["embd"])),
        "b_lr_resets_within_fit": {"plain": lr_resets(plain["lrs"]), "stage1": lr_resets(stage1["lrs"]),
                                   "stage2": lr_resets(stage2_a["lrs"])},
        "c_stage2_start_equals_stage1_output": bool(np.array_equal(stage2_a["start"], stage1_embd)),
        "c_stage2_start_dtype": str(stage2_a["start"].dtype),
        "stage1_output_dtype": str(stage1["embd"].dtype),
        "stage2_first_lr": stage2_a["lrs"][0],
        "stage2_lr_first_last": [stage2_a["lrs"][0], stage2_a["lrs"][-1]],
        "stage2_rerun_bit_identical": bool(np.array_equal(stage2_a["embd"], stage2_b["embd"])),
        "plain_start_equals_pca_init": bool(np.array_equal(plain["start"], pca_init)),
    }

    def scale(e):
        return {"std_dim0": float(e[:, 0].std()), "std_dim1": float(e[:, 1].std())}

    git_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {
        "upstream_commit": UPSTREAM_COMMIT, "project_git_head": git_head, "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(), "numpy": np.__version__,
        "data": {"dataset": "MNIST train+test (70k)", "subset": N_SUBSET, "subset_seed": SEED,
                 "pixels": "raw float32 0-255, flattened", "graph_input": f"PCA-{N_PCA} of pixels"},
        "init": {"used": "first 2 PCs, rescaled so PC1 std = 1 (cne._cne.rescale, target_std=1.0), float32",
                 "v0.4.0_default_would_have_been": "init=None with X given -> 'pca', rescaled to PC1 std 1e-4 "
                                                   "(_cne.py:470-494); docstring at _cne.py:384 says std 1"},
        "graph": graph_stats,
        "runs": {name: {"params": run["params"], "runtime_s": run["runtime_s"], "scale": scale(run["embd"]),
                        "lr_first": run["lrs"][0], "lr_last": run["lrs"][-1], "n_lr_values": len(run["lrs"])}
                 for name, run in [("plain", plain), ("plain_no_early_exaggeration", plain_no_ee),
                                   ("stage1", stage1), ("stage2", stage2_a), ("stage2_rerun", stage2_b)]},
        "knn_accuracy_k15_5fold": acc,
        "checks": checks,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figures.
    save_embedding_plot(plain["embd"], y, f"Neg-t-SNE, 10k MNIST (kNN acc {acc['plain']:.3f})", OUT / "plain.png")
    save_embedding_plot(stage2_a["embd"], y, f"Stage 2 ({N1}+{N2} epochs, kNN acc {acc['stage2']:.3f})",
                        OUT / "stage2.png")

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(plt.imread(PAPER_FIG))
    axes[0].axis("off")
    axes[0].set_title("Package README figure (full 70k MNIST)")
    scatter(axes[1], plain["embd"], y, f"Plain, {plain['params']['n_epochs']} epochs (acc {acc['plain']:.3f})")
    scatter(axes[2], stage1["embd"], y, f"Stage 1, {N1} epochs (acc {acc['stage1']:.3f})")
    scatter(axes[3], stage2_a["embd"], y, f"Stage 2, +{N2} epochs (acc {acc['stage2']:.3f})")
    fig.savefig(OUT / "comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(N1), stage1["lrs"], label="Stage 1")
    ax.plot(np.arange(N1, N1 + N2), stage2_a["lrs"], label="Stage 2")
    ax.plot(np.arange(len(plain["lrs"])), plain["lrs"], "--", label="Plain (single fit)")
    ax.axvline(len(plain["lrs"]) // 3, color="grey", lw=0.8, ls=":", label="n_epochs // 3 (plain)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.legend()
    fig.savefig(OUT / "lr_schedule.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"knn_accuracy": acc, "checks": checks, "graph": graph_stats}, indent=2))


if __name__ == "__main__":
    main()
