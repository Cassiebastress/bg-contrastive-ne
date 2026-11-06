"""MNIST-on-grass 0-9 tables and figures, generated from results/mnist_grass_0to9/ (no number typed by hand).

Tables: confirmation seeds (mean [min, max] and per seed, ARI included), the six pre-registered comparisons on digit
kNN accuracy (secondary metrics alongside, no verdict), the descriptive comparisons, per-digit accuracy, T3's alpha per
seed, the determinism checks, the 0/1 side-by-side (stored cluster results; 0/1 ARI post hoc) and the selection-seed
appendix. Figures (confirmation seed 3):
- the seven runs colored by digit, styled after Damrich et al. (2023), Fig. 1: tab10 in digit order, one scatter per
  run drawn in a fixed random order, a scale bar, serif captions and a two-column legend. The tab10 colors are a
  deliberate match to that paper; they fail the dataviz palette validator (some pairs, e.g. green/orange, cannot be told
  apart by colorblind readers), so the next figure is the accessible view;
- small multiples by digit (one run per row, one digit per panel, in one blue over the other points in gray), which
  carry identity by position, not hue;
- the seven runs colored by mean grass brightness (one-hue blue ramp);
- the row-normalized confusion matrices of (r_c), (r_a) and T3.
The blue colors come from the dataviz reference palette and pass its validator. Needs data/processed/ and the
embeddings (cluster).

    python scripts/mnist_grass_0to9_report.py
Outputs: results/mnist_grass_0to9/tables.md, comparisons.json, embeddings_seed3.png, embeddings_by_digit_seed3.png,
embeddings_grass_seed3.png, confusion_seed3.png
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.experiment import paired  # noqa: E402
from data import mnist_grass as mg  # noqa: E402

NAME = "mnist_grass_0to9"
OUT = ROOT / "results" / NAME
RUNS_DIR = OUT / "runs"
OLD_RAW, OLD_RES = ROOT / "results" / "mnist_grass", ROOT / "results" / "bsgraph"
FIG_SEED = 3
DIGITS = range(10)
HIGHLIGHT, CONTEXT = "#2a78d6", "#d9d8d3"  # reference blue (validator: all checks pass) over a recessive gray
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                       "#184f95", "#0d366b"])  # one-hue sequential ramp, light -> dark
METRICS = [("digit_acc", "digit kNN accuracy"), ("recall_digits", "recall vs clean digits"),
           ("recall_grass", "recall vs grass (lower is better)"), ("ari", "ARI (k-means, k = 10)")]
NAMES = {"a": "(a) raw graph, background term off", "r_a": "(r_a) residual graph, background term off",
         "T1": "T1 t-SNE", "T2": "T2 PCA-50 → t-SNE", "T3": "T3 cPCA → t-SNE"}


def label(run):
    if run.startswith("r_c_m"):
        return f"(r_c) residual graph + ξ_B, m_B = {run[5:]}"
    if run.startswith("c_m"):
        return f"(c) raw graph + ξ_B, m_B = {run[3:]}"
    return NAMES[run]


def short(run):
    base = run.split("_m")[0]
    return base if base.startswith("T") else f"({base})"


def read(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["rep"] in ("", "0")]


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def run_table(rows, runs, per_seed=False):
    out = ["| run | " + " | ".join(n for _, n in METRICS) + " | flagged seeds |", "|---" * (len(METRICS) + 2) + "|"]
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        flagged = sum(r.get("flag_collapse") == "True" or r.get("flag_fragment") == "True" for r in rr)
        out.append(f"| {label(run)} | " + " | ".join(fmt([float(r[m]) for r in rr]) for m, _ in METRICS)
                   + f" | {flagged}/{len(rr)} |")
    if per_seed:
        seeds = sorted({int(r["seed"]) for r in rows})
        for metric, name in (("digit_acc", "digit kNN accuracy"), ("ari", "ARI")):
            out += ["", f"Per seed, {name}:", "", "| run | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                    "|---" * (len(seeds) + 1) + "|"]
            for run in runs:
                v = {int(r["seed"]): float(r[metric]) for r in rows if r["run"] == run}
                out.append(f"| {label(run)} | " + " | ".join(f"{v[s]:.3f}" for s in seeds) + " |")
    return "\n".join(out)


def compare(rows, pairs, verdicts=True):
    out = ["| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean "
           "difference | recall vs grass, mean difference | ARI, mean difference |", "|---|---|---|---|---|---|---|"]
    res = []
    for x, y in pairs:
        xy, yx = paired(rows, x, y, "digit_acc"), paired(rows, y, x, "digit_acc")
        verdict = (f"{short(x)} better, consistent across 3 paired seeds" if xy["consistent"] else
                   f"{short(y)} better, consistent across 3 paired seeds" if yx["consistent"] else "not consistent")
        sec = {m: paired(rows, x, y, m)["mean_diff"] for m in ("recall_digits", "recall_grass", "ari")}
        res.append({"x": x, "y": y, "per_seed": xy["per_seed"], "mean": xy["mean_diff"],
                    "x_better_consistent": xy["consistent"], "y_better_consistent": yx["consistent"],
                    "secondary_mean_diff": sec})
        out.append(f"| {label(x)} vs {label(y)} | {', '.join(f'{d + 0.0:+.3f}' for d in xy['per_seed'])} "
                   f"| {xy['mean_diff'] + 0.0:+.3f} | {verdict if verdicts else '(descriptive) ' + verdict} "
                   f"| {sec['recall_digits'] + 0.0:+.3f} | {sec['recall_grass'] + 0.0:+.3f} "
                   f"| {sec['ari'] + 0.0:+.3f} |")
    return "\n".join(out), res


def run_json(stage, seed, run):
    return json.loads((RUNS_DIR / f"{stage}_seed{seed}_{run}.json").read_text())


def per_digit_table(rows, runs):
    seeds = sorted({int(r["seed"]) for r in rows})
    out = ["| run | " + " | ".join(str(d) for d in DIGITS) + " |", "|---" * 11 + "|"]
    for run in runs:
        acc = np.mean([run_json("confirmation", s, run)["per_digit_acc"] for s in seeds], axis=0)
        out.append(f"| {label(run)} | " + " | ".join(f"{a:.3f}" for a in acc) + " |")
    return "\n".join(out)


def alpha_table(rows):
    out = ["| seed | label-free α (T3) |", "|---|---|"]
    out += [f"| {r['seed']} | {float(r['alpha']):.4g} |" for r in sorted(rows, key=lambda r: int(r["seed"]))
            if r["run"] == "T3"]
    return "\n".join(out)


def side_by_side(conf, rc, c):
    """0-9 next to the stored 0/1 cluster results (confirmation seeds), with the 0/1 ARI computed post hoc."""
    old_c = f"c_m{json.loads((OLD_RAW / 'selection.json').read_text())['best_m_B']}"
    old_rc = f"r_c_m{json.loads((OLD_RES / 'selection.json').read_text())['best_m_B']}"
    old = read(OLD_RAW / "runs_confirmation.csv") + read(OLD_RES / "runs_confirmation.csv")
    with open(OUT / "ari_01_posthoc.csv") as f:
        ari01 = [r for r in csv.DictReader(f) if r["stage"] == "confirmation"]
    pairs = [(rc, old_rc), ("r_a", "r_a"), (c, old_c), ("a", "a"), ("T1", "T1"), ("T2", "T2"), ("T3", "T3")]
    out = ["| run | m_B (0–9 / 0/1) | digit kNN accuracy, 0–9 | digit kNN accuracy, 0/1 | ARI, 0–9 "
           "| ARI, 0/1 (post hoc) | recall vs grass, 0–9 | recall vs grass, 0/1 |", "|---|---|---|---|---|---|---|---|"]
    for new, prev in pairs:
        n = [r for r in conf if r["run"] == new]
        o = [r for r in old if r["run"] == prev]
        oa = [float(r["ari"]) for r in ari01 if r["run"] == prev]
        mb = f"{new.split('_m')[1]} / {prev.split('_m')[1]}" if "_m" in new else "—"
        out.append(f"| {short(new)} | {mb} | {fmt([float(r['digit_acc']) for r in n])} "
                   f"| {fmt([float(r['digit_acc']) for r in o])} | {fmt([float(r['ari']) for r in n])} | {fmt(oa)} "
                   f"| {fmt([float(r['recall_grass']) for r in n])} | {fmt([float(r['recall_grass']) for r in o])} |")
    return "\n".join(out)


def by_digit_figure(runs, y):
    fig, axes = plt.subplots(len(runs), len(DIGITS), figsize=(1.3 * len(DIGITS), 1.55 * len(runs)))
    for i, run in enumerate(runs):
        e = np.load(RUNS_DIR / f"confirmation_seed{FIG_SEED}_{run}.npy")
        rec = run_json("confirmation", FIG_SEED, run)
        for d, ax in zip(DIGITS, axes[i]):
            ax.scatter(*e[y != d].T, color=CONTEXT, s=0.3, edgecolor="none", rasterized=True)
            ax.scatter(*e[y == d].T, color=HIGHLIGHT, s=0.6, edgecolor="none", rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(f"digit {d}", fontsize=8)
        axes[i, 0].set_ylabel(f"{short(run)}\nacc {rec['digit_acc']:.3f}\nARI {rec['ari']:.3f}", fontsize=7,
                              rotation=0, ha="right", va="center", labelpad=4)
    fig.suptitle(f"MNIST-on-grass 0–9, confirmation seed {FIG_SEED}: each digit (blue) against all other images "
                 "(gray)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"embeddings_by_digit_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def grass_figure(runs, brightness):
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6.2))
    lo, hi = brightness.min(), brightness.max()
    for ax, run in zip(axes.flat, runs):
        e = np.load(RUNS_DIR / f"confirmation_seed{FIG_SEED}_{run}.npy")
        rec = run_json("confirmation", FIG_SEED, run)
        sc = ax.scatter(*e.T, c=brightness, s=0.6, cmap=BLUE_RAMP, vmin=lo, vmax=hi, edgecolor="none",
                        rasterized=True)
        ax.set_title(f"{label(run)}\ngrass recall {rec['recall_grass']:.3f}", fontsize=7)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes, fraction=0.02, label="mean grass-patch brightness")
    fig.suptitle(f"MNIST-on-grass 0–9, confirmation seed {FIG_SEED}: colored by the grass behind each digit",
                 fontsize=9)
    fig.savefig(OUT / f"embeddings_grass_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def confusion_figure(rc):
    panels = [rc, "r_a", "T3"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    for ax, run in zip(axes, panels):
        cm = np.array(run_json("confirmation", FIG_SEED, run)["confusion"], dtype=float)
        cm /= cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm, cmap=BLUE_RAMP, vmin=0, vmax=1)
        for i in DIGITS:
            for j in DIGITS:
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > 0.45 else "#0b0b0b")
        ax.set_xticks(DIGITS)
        ax.set_yticks(DIGITS)
        ax.set_xlabel("predicted digit", fontsize=8)
        ax.set_ylabel("true digit", fontsize=8)
        ax.set_title(f"{label(run)}", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, label="fraction of the true digit")
    fig.suptitle(f"kNN confusion matrices (row-normalized, %), confirmation seed {FIG_SEED}", fontsize=9)
    fig.savefig(OUT / f"confusion_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def round_length(width):
    """A round scale-bar length (1, 2 or 5 x 10^k) close to a quarter of the panel width."""
    target = width / 4
    base = 10.0 ** np.floor(np.log10(target))
    return min((m * base for m in (1, 2, 5, 10)), key=lambda v: abs(v - target))


def paper_figure(runs, y):
    """All digits in one scatter per run, colored like Damrich et al. (2023), Fig. 1 (tab10 in digit order)."""
    colors = plt.get_cmap("tab10").colors
    order = np.random.default_rng(0).permutation(len(y))  # fixed draw order: no digit painted last on purpose
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
        fig, axes = plt.subplots(2, 4, figsize=(13, 7.2))
        for ax, run in zip(axes.flat, runs):
            e = np.load(RUNS_DIR / f"confirmation_seed{FIG_SEED}_{run}.npy")[order]
            rec = run_json("confirmation", FIG_SEED, run)
            ax.scatter(*e.T, c=np.asarray(colors)[y[order]], s=1, edgecolor="none", rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")  # same box for every panel, so the captions line up
            ax.axis("off")
            length = round_length(np.ptp(e[:, 0]))
            ax.add_artist(AnchoredSizeBar(ax.transData, length, f"{length:g}", "lower right", pad=0.1, sep=3,
                                          borderpad=0.2, frameon=False, size_vertical=0, fontproperties={"size": 9},
                                          label_top=False))
            caption = label(run).replace("ξ_B", r"$\xi_B$").replace("m_B", r"$m_B$")
            ax.text(0.5, -0.03, f"{caption}\ndigit kNN accuracy {rec['digit_acc']:.3f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10)
        legend_ax = axes.flat[len(runs)]
        legend_ax.axis("off")
        handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=colors[d], label=str(d)) for d in DIGITS]
        legend_ax.legend(handles=handles, ncol=2, frameon=False, loc="center", fontsize=12, handletextpad=0.6,
                         columnspacing=2.5, labelspacing=0.9)
        fig.suptitle(f"MNIST-on-grass 0–9, confirmation seed {FIG_SEED}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.5)
        fig.savefig(OUT / f"embeddings_seed{FIG_SEED}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def figures(runs, rc):
    d = mg.load(mg.processed_path(FIG_SEED, ROOT, name=NAME))
    paper_figure(runs, d["y"])
    by_digit_figure(runs, d["y"])
    grass_figure(runs, d["grass_T"].mean(axis=1))
    confusion_figure(rc)


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    c, rc = f"c_m{sel['c']['best_m_B']}", f"r_c_m{sel['r_c']['best_m_B']}"
    conf, selr = read(OUT / "runs_confirmation.csv"), read(OUT / "runs_selection.csv")
    rechecks = [json.loads((OUT / f"recheck_{s}.json").read_text()) for s in ("selection", "confirmation")]
    runs = [rc, "r_a", c, "a", "T1", "T2", "T3"]
    pre_md, pre = compare(conf, [(rc, "T1"), (rc, "T2"), (rc, "T3"), ("r_a", "T3"), (rc, c), (rc, "r_a")])
    desc_md, desc = compare(conf, [("a", "T1"), (c, "T1"), (c, "T2"), (c, "T3")], verdicts=False)
    sel_runs = ([f"r_c_m{m}" for m in sorted(map(int, sel["r_c"]["means"]))] + ["r_a"]
                + [f"c_m{m}" for m in sorted(map(int, sel["c"]["means"]))] + ["a", "T1", "T2", "T3"])
    parts = ["# MNIST-on-grass 0–9 tables", "",
             "## Confirmation seeds (3, 4, 5): mean [min, max]", "",
             "Chance digit kNN accuracy is 0.10. ARI is secondary, with no verdicts.", "",
             run_table(conf, runs, per_seed=True), "",
             "## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)", "",
             "Differences are first run − second run; secondary columns are mean paired differences, without "
             "verdicts.", "", pre_md, "",
             "## Descriptive comparisons (no verdicts)", "", desc_md, "",
             "## Per-digit kNN accuracy (confirmation seeds, mean)", "", per_digit_table(conf, runs), "",
             "## Determinism", "",
             *[f"- {label(r['run'])} on seed {r['seed']} ({r['stage']}) rerun: "
               f"{'bit-identical' if r['identical'] else 'NOT identical'}." for r in rechecks], "",
             "## T3: label-free α per seed", "", alpha_table(conf + selr), "",
             "## Beside the 0/1 runs (descriptive; confirmation seeds 3–5, all cluster results)", "",
             "0/1 values are the stored MNIST-on-grass and background-suppressed graph confirmation runs, each method "
             "at the m_B selected in its own phase. Chance is 0.50 there and 0.10 here, so accuracies are not "
             "directly comparable. The 0/1 ARI (k-means, k = 2) was computed post hoc from the saved embeddings "
             "(`ari_01_posthoc.csv`); it was not pre-registered.", "", side_by_side(conf, rc, c), "",
             "## Appendix: selection seeds 0–2, every run", "",
             f"Selection rule: {sel['rule']}. Chosen: m_B = {sel['c']['best_m_B']} for (c), "
             f"m_B = {sel['r_c']['best_m_B']} for (r_c).", "", run_table(selr, sel_runs)]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    (OUT / "comparisons.json").write_text(json.dumps({"c": c, "r_c": rc, "preregistered": pre, "descriptive": desc},
                                                     indent=2))
    figures(runs, rc)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
