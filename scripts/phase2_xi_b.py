"""Phase 2 diagnostics: is xi_B informative on the main datasets?

For each level and seed (main datasets only; the stress datasets are not touched until the end of Phase 4), build
xi_B with the default settings and measure, with ground truth used for evaluation only:
- list sizes, empty rows, rows where the top-L cut split a tie, and rows where the 2L cap triggered;
- the expected fraction of xi_B pairs sharing y_b / y_s (exact expectation, heads weighted by target-graph degree),
  against uniform random pairs (= the uniform_extra control) and target kNN edges;
- background geometry: distances ||Bg_i - Bg_j|| of xi_B pairs vs random same-y_b pairs and target kNN edges;
- overlap with the target kNN graph.
Plus an informational k_B x L grid on seed 0. The defaults are not changed by it.

Run with the cne-bg env:  python scripts/phase2_xi_b.py
Outputs go to results/phase2/.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.knn import knn_graph
from common.repro import environment_info, setup_determinism
from data.synthetic import LEVELS, MAIN_RHO, make_dataset
from eval.metrics import pair_label_agreement, uniform_pair_agreement
from eval.xi_b_diag import GATES, diagnose
from method.xi_b import build_xi_b

OUT = ROOT / "results" / "phase2"
SEEDS = [0, 1, 2]
K_GRAPH = 15
K_B, L = 15, 30
GRID = {"k_B": [5, 15, 30], "L": [15, 30, 60]}
COLORS = {"xi_B": "#2a78d6", "target kNN edges": "#eb6834", "uniform pairs": "#8c8b85"}

CSV_FIELDS = ["level", "seed", "rho", "same_b_xi", "same_s_xi", "same_s_excess", "same_b_xi_uniform_heads",
              "same_s_xi_uniform_heads", "same_b_uniform", "same_s_uniform", "same_b_knn", "same_s_knn",
              "gate_same_b", "gate_same_s", "len_min", "len_median", "len_p90", "len_max", "n_empty_rows",
              "frac_tie_split_rows", "frac_capped_rows", "bgdist_xi", "bgdist_same_b_random", "bgdist_knn",
              "spearman_jaccard_bgdist", "frac_knn_edges_in_xi", "xi_mass_on_knn", "xi_mass_on_knn_same_s",
              "xi_mass_on_knn_diff_s", "sha256"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lengths_by_run = [], {}
    for level in LEVELS:
        for seed in SEEDS:
            setup_determinism(seed)
            ds = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level])
            xi = build_xi_b(ds.X_T, ds.X_B, k_B=K_B, L=L, seed=seed)
            G = knn_graph(ds.X_T, k=K_GRAPH)
            diag, lengths = diagnose(ds, xi, G)
            rows.append({"level": level, "seed": seed, "rho": MAIN_RHO[level], **diag})
            lengths_by_run[(level, seed)] = lengths
            xi.save(OUT / f"xi_b_{level}_seed{seed}.npz")
            r = rows[-1]
            print(f"{level:6s} seed={seed}  same y_b {r['same_b_xi']:.3f}  same y_s {r['same_s_xi']:.3f} "
                  f"(excess {r['same_s_excess']:+.3f})  len median {r['len_median']:.0f} max {r['len_max']}  "
                  f"capped {r['frac_capped_rows']:.3f}  gates {r['gate_same_b']}/{r['gate_same_s']}", flush=True)

    # informational sensitivity grid, seed 0 (defaults are not chosen from it)
    grid = []
    for level in LEVELS:
        setup_determinism(0)
        ds = make_dataset(level, seed=0, signal_to_background=MAIN_RHO[level])
        G = knn_graph(ds.X_T, k=K_GRAPH)
        deg = np.diff(G.indptr).astype(float)
        for k_B in GRID["k_B"]:
            for L_ in GRID["L"]:
                xi = build_xi_b(ds.X_T, ds.X_B, k_B=k_B, L=L_, seed=0)
                P, lengths = xi.to_csr(), xi.row_lengths()
                same_s = pair_label_agreement(P, ds.truth["y_s"], head_weights=deg)
                grid.append({"level": level, "k_B": k_B, "L": L_,
                             "same_b_xi": pair_label_agreement(P, ds.truth["y_b"], head_weights=deg),
                             "same_s_xi": same_s, "same_s_excess": same_s - uniform_pair_agreement(ds.truth["y_s"]),
                             "len_median": float(np.median(lengths)), "len_max": int(lengths.max()),
                             "n_empty_rows": xi.params["n_empty_rows"],
                             "frac_capped_rows": xi.params["n_capped_rows"] / xi.n})

    with open(OUT / "xi_b_diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / "sensitivity_seed0.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)

    all_pass = all(r["gate_same_b"] and r["gate_same_s"] for r in rows)
    per_level = {level: {"same_s_excess": [r["same_s_excess"] for r in rows if r["level"] == level],
                         "gates_pass_all_seeds": all(r["gate_same_b"] and r["gate_same_s"]
                                                     for r in rows if r["level"] == level)}
                 for level in LEVELS}
    summary = {
        "environment": environment_info(), "main_rho": MAIN_RHO, "seeds": SEEDS,
        "xi_b": {"k_B": K_B, "L": L, "cap": 2 * L, "knn": "exact, X_T to X_B, raw features (no preprocessing)"},
        "target_graph": f"exact, k={K_GRAPH}, on X_T", "head_weights": "target-graph degree", "gates": GATES,
        "all_gates_pass": all_pass, "per_level": per_level,
        "stress_datasets": "not touched; xi_B for them is built only at the end of Phase 4",
        "runs": rows, "sensitivity_seed0": grid,
    }
    (OUT / "xi_b_diagnostics.json").write_text(json.dumps(summary, indent=2))

    # list-length histograms, one panel per level, seeds overlaid
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 3.5), sharey=True)
    bins = np.arange(0, 2 * L + 2) - 0.5
    for ax, level in zip(axes, LEVELS):
        for seed, color in zip(SEEDS, ["#2a78d6", "#eb6834", "#1baf7a"]):
            ax.hist(lengths_by_run[(level, seed)], bins=bins, histtype="step", lw=1.5, color=color,
                    label=f"seed {seed}")
        ax.axvline(L, color="#52514e", lw=0.8, ls="--")
        ax.axvline(2 * L, color="#52514e", lw=0.8, ls=":")
        ax.set_title(f"{level}: xi_B list length (L = {L} dashed, cap 2L dotted)", fontsize=9)
        ax.set_xlabel("entries per row")
    axes[0].set_ylabel("target points")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "list_sizes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # label agreement: xi_B vs target kNN edges vs uniform pairs, per level, min-max over seeds
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    keys = {"xi_B": "xi", "target kNN edges": "knn", "uniform pairs": "uniform"}
    x = np.arange(len(LEVELS))
    width = 0.26
    for ax, lab in zip(axes, ["b", "s"]):
        for k, (name, suffix) in enumerate(keys.items()):
            vals = np.array([[r[f"same_{lab}_{suffix}"] for r in rows if r["level"] == level] for level in LEVELS])
            mean = vals.mean(axis=1)
            err = np.vstack([mean - vals.min(axis=1), vals.max(axis=1) - mean])
            ax.bar(x + (k - 1) * width, mean, width * 0.92, yerr=err, color=COLORS[name], label=name,
                   error_kw={"lw": 1, "ecolor": "#52514e"})
        ax.set_xticks(x, LEVELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"fraction of pairs sharing y_{lab}" + (" (gate: xi_B >= 0.90)" if lab == "b" else
                                                             " (gate: xi_B <= uniform + 0.10)"), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fraction (mean, min-max over seeds)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "precision.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"all_gates_pass": all_pass, "per_level": per_level}, indent=2))


if __name__ == "__main__":
    main()
