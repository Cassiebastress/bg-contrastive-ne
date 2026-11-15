"""Single-cell RNA-seq phase: tables and figures, generated from results/scrna/ (no number typed by hand).

Tables: every run's label kNN accuracy, ARI, background-structure retention and fragmentation over seeds 0-9; the
pre-registered comparisons (r_a vs each baseline, consistent across 10 paired seeds) and the ceiling rule; the BC-NE
sensitivity grid; cPCA with alpha chosen using labels; the larger background; the checks and runtimes.
Figures: the seed-0 embeddings colored by label and by sequencing depth; per-seed accuracy and ARI; the sensitivity
grid; accuracy against alpha; the replication of the paper's figure. Needs the embeddings and data/processed/
(cluster) for the figures.

    python scripts/scrna_report.py            # everything
    python scripts/scrna_report.py --tables   # tables only
Outputs: results/scrna/tables.md, tables.json and the figures below.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.experiment import paired  # noqa: E402
from scripts.scrna_run import K_LOC, OUT, PAPER_ALPHAS, R, RUNS_DIR, load_data  # noqa: E402

MAIN_RUNS = ("r_a", "a", "T1", "T2", "U2", "T3", "cPCA2D", "PCA2D")
BASELINES = MAIN_RUNS[1:]
LABELS = {"r_a": "BC-NE (r_a)", "a": "plain Neg-t-SNE (a)", "T1": "t-SNE (T1)", "T2": "PCA-50 → t-SNE (T2)",
          "U2": "PCA-50 → UMAP (U2)", "T3": "cPCA → t-SNE (T3)", "cPCA2D": "cPCA-2D", "PCA2D": "PCA-2D",
          "r_a_bg12": "BC-NE, donors 1 + 2", "T3_bg12": "cPCA → t-SNE, donors 1 + 2",
          "cPCA2D_bg12": "cPCA-2D, donors 1 + 2"}
LABEL_COLORS = np.array(["#2a78d6", "#eb6834"])  # pre blue, post orange (the project's reference slots 1-2)
INK, MUTED, RULE = "#1f1f1e", "#6b6a66", "#d9d8d3"
CEILING = 0.95
SENS_K, SENS_R = (10, 30, 60), (0, 2, 5)
FIG_SEED = 0


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for m in ("label_acc", "ari", "bg_retention", "runtime_s", "frac_in_small_components", "alpha"):
            if m in r:
                r[m] = float(r[m]) if r[m] not in ("", None) else None
    return rows


def vals(rows, run, m):
    return [r[m] for r in rows if r["run"] == run]


def fmt(v, digits=3):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.{digits}f} [{v.min():.{digits}f}, {v.max():.{digits}f}]"


def run_table(rows, runs):
    out = ["| run | label kNN accuracy | ARI | background-structure retention | fragmented seeds |",
           "|---|---|---|---|---|"]
    for run in runs:
        frag = sum(str(r["flag_fragment"]) == "True" for r in rows if r["run"] == run)
        out.append(f"| {LABELS.get(run, run)} | {fmt(vals(rows, run, 'label_acc'))} | {fmt(vals(rows, run, 'ari'))} "
                   f"| {fmt(vals(rows, run, 'bg_retention'))} | {frag} of {len(vals(rows, run, 'ari'))} |")
    return "\n".join(out)


def comparisons(rows):
    out = ["| comparison | per-seed differences: min, max | mean | verdict |", "|---|---|---|---|"]
    res = []
    for y in BASELINES:
        xy, yx = paired(rows, "r_a", y, "label_acc"), paired(rows, y, "r_a", "label_acc")
        v = ("r_a better, consistent across 10 paired seeds" if xy["consistent"] else
             f"{y} better, consistent across 10 paired seeds" if yx["consistent"] else "not consistent")
        res.append({**xy, "verdict": v})
        out.append(f"| r_a vs {LABELS[y]} | {min(xy['per_seed']):+.4f}, {max(xy['per_seed']):+.4f} "
                   f"| {xy['mean_diff']:+.4f} | {v} |")
    return "\n".join(out), res


def sensitivity(rows, m):
    out = ["| k_loc \\ r | " + " | ".join(str(r) for r in SENS_R) + " |", "|---" * (len(SENS_R) + 1) + "|"]
    grid = {}
    for k in SENS_K:
        cells = []
        for r in SENS_R:
            run = "r_a" if (k, r) == (K_LOC, R) else f"sens_k{k}_r{r}"
            grid[(k, r)] = float(np.mean(vals(rows, run, m)))
            cells.append(f"{grid[(k, r)]:.3f}")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out), grid


def oracle_curves(orows):
    curves = {}
    for run in ("cPCA2D", "T3"):
        alphas = sorted({r["alpha"] for r in orows if r["run"] == run})
        curves[run] = {a: {m: float(np.mean([r[m] for r in orows if r["run"] == run and r["alpha"] == a]))
                           for m in ("label_acc", "ari")} for a in alphas}
    return curves


def oracle_table(rows, curves, pre):
    out = ["| run | α chosen with labels (best mean accuracy) | its mean accuracy | label-free α | its mean accuracy |",
           "|---|---|---|---|---|"]
    label_free = sorted({v["alpha"] for v in pre["label_free_alpha"].values()})
    for run, c in curves.items():
        best = max(c, key=lambda a: (c[a]["label_acc"], -a))
        out.append(f"| {LABELS[run]} | {best:.3g} | {c[best]['label_acc']:.3f} | "
                   f"{', '.join(f'{a:.3g}' for a in label_free)} | {np.mean(vals(rows, run, 'label_acc')):.3f} |")
    return "\n".join(out)


def background_table(rows):
    out = ["| run | donor 1 (1985 cells) | donors 1 + 2 (4457 cells) |", "|---|---|---|"]
    for run in ("r_a", "T3", "cPCA2D"):
        out.append(f"| {LABELS[run]} | {fmt(vals(rows, run, 'label_acc'))} | "
                   f"{fmt(vals(rows, f'{run}_bg12', 'label_acc'))} |")
    return "\n".join(out)


def runtime_table(rows):
    procs = {"BC-NE (r_a)": "r_a", "plain Neg-t-SNE (a)": "a", "t-SNE (T1)": "T1", "PCA-50 → t-SNE (T2)": "T2",
             "PCA-50 → UMAP (U2)": "U2", "cPCA fit + T3 + cPCA-2D + PCA-2D": "T3"}
    out = ["| process | median runtime (s) | range |", "|---|---|---|"]
    for label, run in procs.items():
        t = vals(rows, run, "runtime_s")
        out.append(f"| {label} | {np.median(t):.0f} | {min(t):.0f}–{max(t):.0f} |")
    return "\n".join(out)


def tables():
    rows, orows = read(OUT / "runs.csv"), read(OUT / "oracle.csv")
    pre = json.loads((OUT / "prechecks.json").read_text())
    recheck = json.loads((OUT / "recheck.json").read_text())
    cmp_md, cmp = comparisons(rows)
    t1, ra = np.mean(vals(rows, "T1", "label_acc")), np.mean(vals(rows, "r_a", "label_acc"))
    ceiling = bool(t1 >= CEILING and ra >= CEILING)
    sens_acc, grid_acc = sensitivity(rows, "label_acc")
    sens_ari, grid_ari = sensitivity(rows, "ari")
    curves = oracle_curves(orows)
    md = ["## Every run, seeds 0–9: mean [min, max]", "", run_table(rows, MAIN_RUNS), "",
          "## Pre-registered comparisons (label kNN accuracy; differences are r_a − baseline)", "", cmp_md, "",
          f"Ceiling rule (mean T1 and mean r_a both ≥ {CEILING}): T1 {t1:.3f}, r_a {ra:.3f} → "
          + ("**applies**: the dataset cannot discriminate between the methods." if ceiling else "does not apply."),
          "", "## BC-NE sensitivity (descriptive): mean label kNN accuracy", "", sens_acc, "",
          "Mean ARI:", "", sens_ari, "",
          "## cPCA with α chosen using labels (descriptive; an upper bound, not a fair baseline)", "",
          oracle_table(rows, curves, pre), "",
          "## Larger background (descriptive): label kNN accuracy", "", background_table(rows), "",
          "## Checks", "",
          f"Seed-0 reruns bit-identical: {'yes, every run' if recheck['all_identical'] else 'NO'} "
          f"({', '.join(r for r, ok in recheck['identical'].items() if not ok) or 'no exceptions'}).", "",
          "## Runtimes (one thread, whole process)", "", runtime_table(rows)]
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "tables.json").write_text(json.dumps({
        "comparisons": cmp, "ceiling": {"T1": t1, "r_a": ra, "applies": ceiling},
        "sensitivity": {"label_acc": {f"{k},{r}": v for (k, r), v in grid_acc.items()},
                        "ari": {f"{k},{r}": v for (k, r), v in grid_ari.items()}},
        "oracle_curves": {run: {str(a): v for a, v in c.items()} for run, c in curves.items()}}, indent=2))
    return rows, curves, grid_acc, grid_ari, pre


LINEAR = ("cPCA2D", "PCA2D")  # 2-D projections: a few extreme cells stretch the axes, so these zoom to the bulk


def scatter(ax, e, c, order, central=False, **kw):
    """Points in a fixed random draw order. central=True: axes limited to the 0.5-99.5 percentile box of each axis
    (stated in the panel title), which keeps the bulk of a linear projection readable."""
    e = e[order]
    ax.scatter(*e.T, c=c[order] if isinstance(c, np.ndarray) else c, s=0.6, edgecolor="none", rasterized=True, **kw)
    if central:
        lo, hi = np.percentile(e, 0.5, axis=0), np.percentile(e, 99.5, axis=0)
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    else:
        ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def embedding_figures(rows, d):
    y, depth = d["y"], d["log10_total_T"]
    order = np.random.default_rng(0).permutation(len(y))
    rec = {r["run"]: r for r in rows if r["seed"] == FIG_SEED}
    for kind in ("label", "depth"):
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        for ax, run in zip(axes.flat, MAIN_RUNS):
            e = np.load(RUNS_DIR / f"seed{FIG_SEED}_{run}.npy")
            central = run in LINEAR
            if kind == "label":
                scatter(ax, e, LABEL_COLORS[y], order, central=central)
            else:
                scatter(ax, e, depth, order, central=central, cmap="viridis")
            ax.set_title(f"{LABELS[run]}{' (central 99%)' if central else ''}\nacc {rec[run]['label_acc']:.3f} · "
                         f"ARI {rec[run]['ari']:.3f}", fontsize=9)
        if kind == "label":
            handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c, label=lab)
                       for c, lab in zip(LABEL_COLORS, ("pre-transplant", "post-transplant"))]
            fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=10)
        else:
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(depth.min(), depth.max()))
            fig.colorbar(sm, ax=axes, fraction=0.02, label="log10 total counts per cell")
        fig.suptitle(f"Seed {FIG_SEED}, colored by {'sample' if kind == 'label' else 'sequencing depth'}",
                     fontsize=10, color=MUTED)
        if kind == "label":
            fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(OUT / f"embeddings_seed{FIG_SEED}_{kind}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def per_seed_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(MAIN_RUNS))
    for ax, (m, title) in zip(axes, (("label_acc", "label kNN accuracy"), ("ari", "ARI (k-means, k = 2)"))):
        for i, run in enumerate(MAIN_RUNS):
            v = vals(rows, run, m)
            color = "#2a78d6" if run == "r_a" else INK
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=12, color=color, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=color, lw=2, zorder=4)
        ax.set_xticks(x, [LABELS[r].split(" (")[0] for r in MAIN_RUNS], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color=RULE, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("per seed (dots) and mean (bar)", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "per_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sensitivity_figure(grid_acc, grid_ari):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (grid, title) in zip(axes, ((grid_acc, "mean label kNN accuracy"), (grid_ari, "mean ARI"))):
        A = np.array([[grid[(k, r)] for r in SENS_R] for k in SENS_K])
        im = ax.imshow(A, cmap="Blues", aspect="auto")
        for i in range(len(SENS_K)):
            for j in range(len(SENS_R)):
                ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if A[i, j] > A.min() + 0.6 * np.ptp(A) else INK,
                        fontweight="bold" if (SENS_K[i], SENS_R[j]) == (K_LOC, R) else "normal")
        ax.set_xticks(range(len(SENS_R)), [str(r) for r in SENS_R])
        ax.set_yticks(range(len(SENS_K)), [str(k) for k in SENS_K])
        ax.set_xlabel("r (local background directions removed)", fontsize=9)
        ax.set_ylabel("k_loc", fontsize=9)
        ax.set_title(title + " (default in bold)", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def alpha_figure(rows, curves, pre):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for run, color in (("cPCA2D", "#1baf7a"), ("T3", "#6250d6")):
        c = curves[run]
        a = np.array([k for k in c if k > 0])
        ax.plot(a, [c[k]["label_acc"] for k in a], color=color, lw=2, marker="o", ms=3, label=LABELS[run])
        if 0.0 in c:
            ax.scatter([a.min() / 2], [c[0.0]["label_acc"]], color=color, marker="s", s=25, zorder=4)
    ax.axhline(np.mean(vals(rows, "r_a", "label_acc")), color="#2a78d6", lw=1.5, ls="--", label="BC-NE (mean)")
    for alpha in sorted({v["alpha"] for v in pre["label_free_alpha"].values()}):
        ax.axvline(alpha, color=MUTED, lw=1, ls=":")
        ax.text(alpha, ax.get_ylim()[0], " label-free α", rotation=90, va="bottom", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("α (squares at the left: α = 0)", fontsize=9)
    ax.set_ylabel("mean label kNN accuracy, seeds 0–9", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def replication_figure(d, pre):
    y = d["y"]
    order = np.random.default_rng(0).permutation(len(y))
    acc = pre["replication"]["label_acc"]
    with np.load(RUNS_DIR / "replication.npz") as f:
        proj = {k: f[k] for k in f.files}
    keys = [f"{a:.6g}" for a in PAPER_ALPHAS]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4))
    for ax, k in zip(axes, keys):
        scatter(ax, proj[k], LABEL_COLORS[y], order, central=True)
        ax.set_title(f"α = {float(k):.3g}\nacc {acc[k]:.3f}", fontsize=9.5)
    fig.suptitle("cPCA-2D on the paper's preprocessing (its α values; blue pre, orange post; central 99% of cells)",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, curves, grid_acc, grid_ari, pre = tables()
    if "--tables" in sys.argv:
        return
    d = load_data()
    embedding_figures(rows, d)
    per_seed_figure(rows)
    sensitivity_figure(grid_acc, grid_ari)
    alpha_figure(rows, curves, pre)
    replication_figure(d, pre)


if __name__ == "__main__":
    main()
