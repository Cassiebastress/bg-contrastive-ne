"""MNIST-on-grass tables and figures, generated from results/mnist_grass/ (no number typed by hand).

Tables: confirmation seeds (mean [min, max] and per seed), the four pre-registered comparisons on digit kNN accuracy
(secondary metrics alongside, no verdict), T3's alpha per seed, the selection-seed appendix, and the Phase B
Mac-vs-cluster claim. Figures (confirmation seed 3): the five runs colored by digit and by mean grass brightness, and
the 2-D cPCA projections at every alpha representative. Needs data/processed/ (run on the cluster).

    python scripts/mnist_grass_report.py
Outputs: results/mnist_grass/tables.md, comparisons.json, embeddings_seed3.png, cpca_alphas_seed3.png
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

from common.experiment import paired
from data import mnist_grass as mg

OUT = ROOT / "results" / "mnist_grass"
RUNS_DIR = OUT / "runs"
FIG_SEED = 3
METRICS = [("digit_acc", "digit kNN accuracy", True), ("recall_digits", "recall vs clean digits", True),
           ("recall_grass", "recall vs grass patches (lower is better)", False)]
NAMES = {"a": "(a) Neg-t-SNE, background term off", "T1": "T1 t-SNE", "T2": "T2 PCA-50 → t-SNE",
         "T3": "T3 cPCA (label-free α, 10 comp.) → t-SNE"}


def read(name):
    with open(OUT / name) as f:
        return [r for r in csv.DictReader(f) if r["rep"] in ("", "0")]


def label(run):
    return f"(c) ξ_B negatives, m_B = {run[3:]}" if run.startswith("c_m") else NAMES[run]


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def run_table(rows, runs, per_seed=False):
    seeds = sorted({int(r["seed"]) for r in rows})
    head = "| run | " + " | ".join(n for _, n, _ in METRICS) + " | flagged seeds |"
    out = [head, "|---" * (len(METRICS) + 2) + "|"]
    for run in runs:
        rr = sorted((r for r in rows if r["run"] == run), key=lambda r: int(r["seed"]))
        flagged = sum(r["flag_collapse"] == "True" or r["flag_fragment"] == "True" for r in rr)
        out.append(f"| {label(run)} | " + " | ".join(fmt([float(r[m]) for r in rr]) for m, _, _ in METRICS)
                   + f" | {flagged}/{len(rr)} |")
    if per_seed:
        out += ["", "Per seed, digit kNN accuracy:", "", "| run | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                "|---" * (len(seeds) + 1) + "|"]
        for run in runs:
            v = {int(r["seed"]): float(r["digit_acc"]) for r in rows if r["run"] == run}
            out.append(f"| {label(run)} | " + " | ".join(f"{v[s]:.3f}" for s in seeds) + " |")
    return "\n".join(out)


def comparisons(rows, c):
    out = ["| comparison | per-seed difference in digit kNN accuracy ((c) − other) | mean | verdict "
           "| recall vs digits, mean difference | recall vs grass, mean difference |", "|---|---|---|---|---|---|"]
    res = []
    for y in ("T1", "T2", "T3", "a"):
        cy, yc = paired(rows, c, y, "digit_acc"), paired(rows, y, c, "digit_acc")
        verdict = ("(c) better, consistent across 3 paired seeds" if cy["consistent"] else
                   f"{label(y)} better, consistent across 3 paired seeds" if yc["consistent"] else "not consistent")
        sec = {m: paired(rows, c, y, m)["mean_diff"] for m in ("recall_digits", "recall_grass")}
        res.append({"x": c, "y": y, "per_seed": cy["per_seed"], "mean": cy["mean_diff"],
                    "c_better_consistent": cy["consistent"], "other_better_consistent": yc["consistent"],
                    "secondary_mean_diff": sec})
        out.append(f"| (c) vs {label(y)} | {', '.join(f'{d + 0.0:+.3f}' for d in cy['per_seed'])} "
                   f"| {cy['mean_diff'] + 0.0:+.3f} | {verdict} | {sec['recall_digits'] + 0.0:+.3f} "
                   f"| {sec['recall_grass'] + 0.0:+.3f} |")
    return "\n".join(out), res


def alpha_table(rows):
    out = ["| seed | label-free α (T3) |", "|---|---|"]
    out += [f"| {r['seed']} | {float(r['alpha']):.4g} |" for r in sorted(rows, key=lambda r: int(r["seed"]))
            if r["run"] == "T3"]
    return "\n".join(out)


def phase_b():
    v = json.loads((OUT / "validation" / "validation.json").read_text())
    m = v["smallest_phase5b_per_seed_margin"]
    return (f"Phase B (descriptive): the largest Mac-vs-cluster difference in WC was "
            f"{v['max_abs_WC_diff_mac_vs_cluster']:.3f}, below the smallest per-seed margin among the Phase 5b "
            f"verdicts ({m['abs_WC_diff']:.3f}; L-{m['variant']}, (c) vs {m['vs']}): "
            f"{'yes' if v['diff_below_smallest_margin'] else 'NO'}. Gate (two cluster runs bit-identical): "
            f"{'passed' if v['gate_passed'] else 'FAILED'}.")


def figures(c):
    d = mg.load(mg.processed_path(FIG_SEED, ROOT))
    y, brightness = d["y"], d["grass_T"].mean(axis=1)
    runs = [c, "a", "T1", "T2", "T3"]
    fig, axes = plt.subplots(len(runs), 2, figsize=(6, 3 * len(runs)))
    for ax_row, run in zip(axes, runs):
        e = np.load(RUNS_DIR / f"confirmation_seed{FIG_SEED}_{run}.npy")
        rec = json.loads((RUNS_DIR / f"confirmation_seed{FIG_SEED}_{run}.json").read_text())
        ax_row[0].scatter(*e.T, c=y, s=0.6, cmap="coolwarm", vmin=0, vmax=1, edgecolor="none", rasterized=True)
        sc = ax_row[1].scatter(*e.T, c=brightness, s=0.6, cmap="viridis", edgecolor="none", rasterized=True)
        ax_row[0].set_title(f"{label(run)}\nby digit (blue 0, red 1): acc {rec['digit_acc']:.3f}", fontsize=7)
        ax_row[1].set_title(f"{label(run)}\nby mean grass brightness: grass recall {rec['recall_grass']:.3f}",
                            fontsize=7)
        fig.colorbar(sc, ax=ax_row[1], fraction=0.04)
        for ax in ax_row:
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle(f"MNIST-on-grass, confirmation seed {FIG_SEED}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / f"embeddings_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    with np.load(RUNS_DIR / f"confirmation_seed{FIG_SEED}_T3_cpca2d.npz") as f:
        proj = {k: f[k] for k in f.files}
    chosen = json.loads((RUNS_DIR / f"confirmation_seed{FIG_SEED}_T3.json").read_text())["alpha"]
    fig, axes = plt.subplots(1, len(proj), figsize=(3 * len(proj), 3))
    for ax, (a, Z) in zip(np.atleast_1d(axes), sorted(proj.items(), key=lambda kv: float(kv[0]))):
        ax.scatter(*Z.T, c=y, s=0.6, cmap="coolwarm", vmin=0, vmax=1, edgecolor="none", rasterized=True)
        ax.set_title(f"α = {float(a):.3g}" + (" (label-free choice)" if np.isclose(float(a), chosen) else ""),
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"2-D cPCA projections at each α representative, seed {FIG_SEED} (blue 0, red 1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"cpca_alphas_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    c = f"c_m{sel['best_m_B']}"
    conf, selr = read("runs_confirmation.csv"), read("runs_selection.csv")
    recheck = json.loads((OUT / "recheck.json").read_text())
    cmp_md, cmp_res = comparisons(conf, c)
    runs = [c, "a", "T1", "T2", "T3"]
    sel_runs = [r for r in dict.fromkeys(x["run"] for x in selr) if r.startswith("c_m")] + ["a", "T1", "T2", "T3"]
    sel_runs = sorted(sel_runs, key=lambda r: (not r.startswith("c_m"), int(r[3:]) if r.startswith("c_m") else 0))
    parts = ["# MNIST-on-grass tables", "",
             f"## Confirmation seeds ({', '.join(sorted({r['seed'] for r in conf}))}): mean [min, max]", "",
             run_table(conf, runs, per_seed=True), "",
             "## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)", "", cmp_md, "",
             "Secondary columns: mean paired difference (c) − other; for grass recall, negative means (c) keeps "
             "less grass structure.", "",
             f"Determinism: (c) on seed {recheck['seed']} rerun: "
             f"{'bit-identical' if recheck['identical'] else 'NOT identical'}.", "",
             "## T3: label-free α per seed", "", alpha_table(conf + selr), "",
             "## Appendix: selection seeds 0–2, every run", "",
             f"Selection rule: {sel['rule']}. Chosen: m_B = {sel['best_m_B']}.", "", run_table(selr, sel_runs), "",
             "## Cluster validation", "", phase_b()]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    (OUT / "comparisons.json").write_text(json.dumps({"c": c, "comparisons": cmp_res}, indent=2))
    figures(c)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
