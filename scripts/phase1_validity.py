"""Phase 1 validity sweep: is each synthetic dataset a valid test, i.e. does plain Neg-t-SNE organize X_T by background?

For each level and signal_to_background (rho), embed X_T with the Phase 4 baseline (Stage 1 + Stage 2, no background
term) and measure kNN accuracy of the background labels y_b and signal labels y_s. The chosen rho per level is the
largest valid one on seed 0, then confirmed on more seeds.

Run with the cne-bg env:  python scripts/phase1_validity.py
Outputs go to results/phase1/.
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.embed import N1, N2, pca_init, two_stage
from common.knn import knn_graph
from common.repro import environment_info, setup_determinism
from data.synthetic import DEFAULTS, LEVELS, make_dataset
from eval.metrics import knn_accuracy

OUT = ROOT / "results" / "phase1"

RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
SWEEP_SEED = 0
CONFIRM_SEEDS = [0, 1, 2]
K_GRAPH = 15
CRITERION = {"min_emb_acc_b": 0.90, "max_emb_acc_s": 0.50, "min_oracle_acc_s": 0.95}
CHANCE = {"y_b": 1 / DEFAULTS["K_b"], "y_s": 1 / DEFAULTS["K_s"]}

CSV_FIELDS = ["level", "rho", "seed", "emb_acc_b", "emb_acc_s", "stage1_acc_b", "stage1_acc_s",
              "highd_acc_b", "highd_acc_s", "oracle_acc_s", "valid"]


def is_valid(row):
    return (row["emb_acc_b"] >= CRITERION["min_emb_acc_b"] and row["emb_acc_s"] <= CRITERION["max_emb_acc_s"]
            and row["oracle_acc_s"] >= CRITERION["min_oracle_acc_s"])


def evaluate(level, rho, seed):
    """Generate one dataset, embed it with the baseline schedule, and score it."""
    setup_determinism(seed)
    t0 = time.time()
    ds = make_dataset(level, seed=seed, signal_to_background=rho)
    y_b, y_s = ds.truth["y_b"], ds.truth["y_s"]
    # D = 50, so a PCA-50 step would only rotate the data: build the graph and init on X_T directly
    G = knn_graph(ds.X_T, k=K_GRAPH)
    stage1, stage2 = two_stage(pca_init(ds.X_T), G, seed)
    embd = stage2["embd"]
    row = {
        "level": level, "rho": rho, "seed": seed,
        "emb_acc_b": knn_accuracy(embd, y_b), "emb_acc_s": knn_accuracy(embd, y_s),
        "stage1_acc_b": knn_accuracy(stage1["embd"], y_b), "stage1_acc_s": knn_accuracy(stage1["embd"], y_s),
        "highd_acc_b": knn_accuracy(ds.X_T, y_b), "highd_acc_s": knn_accuracy(ds.X_T, y_s),
        "oracle_acc_s": knn_accuracy(ds.truth["S_T"], y_s),
    }
    row["valid"] = is_valid(row)
    extra = {"runtime_s": time.time() - t0, "embd_std": embd.std(axis=0).tolist(),
             "finite": bool(np.isfinite(embd).all()), "generator": ds.params}
    print(f"{level:6s} rho={rho:<4} seed={seed}  emb y_b {row['emb_acc_b']:.3f}  emb y_s {row['emb_acc_s']:.3f}  "
          f"oracle {row['oracle_acc_s']:.3f}  valid={row['valid']}  ({extra['runtime_s']:.0f}s)", flush=True)
    return row, extra, embd, ds


def scatter(ax, embd, y, title):
    ax.scatter(*embd.T, c=y, s=1.0, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, extras, kept = [], {}, {}

    # sweep on seed 0
    for level in LEVELS:
        for rho in RHOS:
            row, extra, embd, ds = evaluate(level, rho, SWEEP_SEED)
            rows.append(row)
            extras[f"{level}/rho={rho}/seed={SWEEP_SEED}"] = extra
            kept[(level, rho, SWEEP_SEED)] = (embd, ds)

    # choose the largest valid rho per level, then confirm on the other seeds
    chosen = {}
    for level in LEVELS:
        valid = [r["rho"] for r in rows if r["level"] == level and r["valid"]]
        chosen[level] = max(valid) if valid else None
    for level, rho in chosen.items():
        if rho is None:
            continue
        for seed in CONFIRM_SEEDS:
            if seed == SWEEP_SEED:
                continue
            row, extra, embd, ds = evaluate(level, rho, seed)
            rows.append(row)
            extras[f"{level}/rho={rho}/seed={seed}"] = extra

    with open(OUT / "validity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    per_level = {}
    for level in LEVELS:
        sweep = [r for r in rows if r["level"] == level and r["seed"] == SWEEP_SEED]
        valid_idx = [i for i, r in enumerate(sweep) if r["valid"]]
        rho = chosen[level]
        confirm = [r for r in rows if r["level"] == level and r["rho"] == rho] if rho is not None else []
        per_level[level] = {
            "chosen_rho": rho,
            "valid_rhos_seed0": [sweep[i]["rho"] for i in valid_idx],
            # the valid rhos should form one contiguous range; flag it if not
            "valid_set_contiguous": not valid_idx or valid_idx == list(range(valid_idx[0], valid_idx[-1] + 1)),
            "confirmed_on_all_seeds": bool(confirm) and all(r["valid"] for r in confirm),
            "confirm_rows": confirm,
        }

    summary = {
        "environment": environment_info(),
        "criterion": CRITERION, "chance": CHANCE, "rhos": RHOS, "sweep_seed": SWEEP_SEED,
        "confirm_seeds": CONFIRM_SEEDS, "knn_graph": f"exact, k={K_GRAPH}, on X_T (D=50, no PCA step)",
        "init": "PCA of X_T (full SVD), PC1 std 1, float32",
        "schedule": {"stage1_epochs": N1, "stage2_epochs": N2, "stage2": "early_exaggeration=False, no background"},
        "generator_defaults": DEFAULTS, "per_level": per_level, "runs": extras,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # validity curves
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
    for ax, level in zip(axes, LEVELS):
        sweep = [r for r in rows if r["level"] == level and r["seed"] == SWEEP_SEED]
        rhos = [r["rho"] for r in sweep]
        ax.plot(rhos, [r["emb_acc_b"] for r in sweep], "o-", color="C0", label="embedding, y_b")
        ax.plot(rhos, [r["emb_acc_s"] for r in sweep], "o-", color="C1", label="embedding, y_s")
        ax.plot(rhos, [r["highd_acc_b"] for r in sweep], "--", color="C0", alpha=0.6, label="high-D X_T, y_b")
        ax.plot(rhos, [r["highd_acc_s"] for r in sweep], "--", color="C1", alpha=0.6, label="high-D X_T, y_s")
        ax.axhline(CHANCE["y_b"], color="C0", lw=0.8, ls=":")
        ax.axhline(CHANCE["y_s"], color="C1", lw=0.8, ls=":")
        ax.axhline(CRITERION["min_emb_acc_b"], color="grey", lw=0.8)
        ax.axhline(CRITERION["max_emb_acc_s"], color="grey", lw=0.8)
        if chosen[level] is not None:
            ax.axvline(chosen[level], color="k", lw=1, ls="--", label=f"chosen rho = {chosen[level]}")
        ax.set_title(level)
        ax.set_xlabel("signal_to_background (rho)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("kNN accuracy (k=15, 5-fold)")
    axes[-1].legend(fontsize=7, loc="center right")
    fig.savefig(OUT / "validity_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # embeddings at the chosen rho (seed 0), colored by background and by signal
    fig, axes = plt.subplots(len(LEVELS), 2, figsize=(8, 4 * len(LEVELS)))
    for ax_row, level in zip(axes, LEVELS):
        rho = chosen[level]
        if rho is None:
            for ax in ax_row:
                ax.axis("off")
                ax.set_title(f"{level}: no valid rho")
            continue
        embd, ds = kept[(level, rho, SWEEP_SEED)]
        r = next(r for r in rows if r["level"] == level and r["rho"] == rho and r["seed"] == SWEEP_SEED)
        scatter(ax_row[0], embd, ds.truth["y_b"], f"{level}, rho={rho}: colored by y_b (acc {r['emb_acc_b']:.2f})")
        scatter(ax_row[1], embd, ds.truth["y_s"], f"{level}, rho={rho}: colored by y_s (acc {r['emb_acc_s']:.2f})")
    fig.savefig(OUT / "embeddings_chosen.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # raw PCA view: does X_B cover X_T's background?
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 5))
    for ax, level in zip(axes, LEVELS):
        rho = chosen[level] if chosen[level] is not None else RHOS[0]
        _, ds = kept[(level, rho, SWEEP_SEED)]
        pca = PCA(n_components=2, svd_solver="full").fit(ds.X_T)
        xb, xt = pca.transform(ds.X_B), pca.transform(ds.X_T)
        ax.scatter(*xt.T, s=1, c=ds.truth["y_b"], cmap="tab10", vmin=0, vmax=9, edgecolor="none", alpha=0.5,
                   rasterized=True)
        ax.scatter(*xb.T, s=0.5, c="black", edgecolor="none", alpha=0.25, rasterized=True)
        ax.set_title(f"{level}, rho={rho}: PCA of X_T (color y_b), X_B black on top", fontsize=9)
        ax.set_aspect("equal")
    fig.savefig(OUT / "raw_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for level in LEVELS:
        np.save(OUT / f"embd_{level}_chosen_seed{SWEEP_SEED}.npy",
                kept[(level, chosen[level], SWEEP_SEED)][0] if chosen[level] is not None else np.empty((0, 2)))
    print(json.dumps({lvl: {k: v for k, v in d.items() if k != "confirm_rows"} for lvl, d in per_level.items()},
                     indent=2))


if __name__ == "__main__":
    main()
