"""Single-stage vs two-stage (r_a): tables and figures, generated from results/single_stage/ (no number typed by hand).

Tables: the reuse check of the stored two-stage runs, the optimization-noise reference and tolerances, then per dataset
the confirmation seeds 3-5 (per seed, the difference, whether it is within tolerance, and mean [min, max]), the
agreement between the two embeddings (kNN overlap at 15, Procrustes disparity), the runtimes, and seeds 0-2 as a
secondary table. Figures (seed 3, per dataset): both embeddings side by side, colored by digit and by mean grass
brightness; and the per-epoch learning rate of each schedule. Digit colors: 0/1 in the reference blue and orange (the
pair passes the dataviz validator); 0-9 in tab10 in digit order, as in Damrich et al. (2023), Fig. 1, by choice (it
fails the validator; see REPORT_mnist_grass_0to9.md for the accessible view). Grass: the one-hue blue ramp. Needs
data/processed/ and the embeddings (cluster).

    python scripts/single_stage_report.py
Outputs: results/single_stage/tables.md, embeddings_{01,0to9}_seed3.png, learning_rate.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import mnist_grass as mg  # noqa: E402
from scripts.mnist_grass_0to9_report import BLUE_RAMP, round_length  # noqa: E402
from scripts.single_stage_run import DATASETS, OUT, PRIMARY, RECHECK_SEED, new, stored  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference palette slots 1-2 (validator: all checks pass)
VERSIONS = (("two_stage", "two-stage (r_a), stored"), ("single", "r_a-single"))
METRIC_NAMES = {"digit_acc": "digit kNN accuracy", "recall_grass": "recall vs grass (lower is better)",
                "recall_digits": "recall vs clean digits", "ari": "ARI (k-means)"}


def fmt(v, spec=".3f"):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():{spec}} [{v.min():{spec}}, {v.max():{spec}}]"


def yes(b):
    return "yes" if b else "**no**"


def reuse_table(noise_ref):
    out = ["| dataset | seed | stored at | Slurm job | code unchanged since | .npy SHA-256 matches | digit accuracy "
           "reproduced |", "|---|---|---|---|---|---|---|"]
    for d, info in DATASETS.items():
        for r in noise_ref[d]["reuse"]["stored_runs"]:
            out.append(f"| {info['label']} | {r['seed']} | `{r['commit']}` | {r['slurm_job']} | "
                       f"{yes(r['code_unchanged'])} | {yes(r['npy_sha_matches'])} | {yes(r['digit_acc_reproduced'])} |")
    out += ["", "Seed-3 two-stage rerun, bit-identical to the stored embedding: " + ", ".join(
        f"{info['label']} {yes(noise_ref[d]['reuse']['rerun_seed3_identical'])}" for d, info in DATASETS.items()) + "."]
    return "\n".join(out)


def noise_table(noise_ref):
    out = ["| dataset | training seed | digit kNN accuracy | recall vs grass |", "|---|---|---|---|"]
    for d, info in DATASETS.items():
        for r in noise_ref[d]["noise"]["runs"]:
            out.append(f"| {info['label']} | {r['train_seed']} | {r['digit_acc']:.4f} | {r['recall_grass']:.4f} |")
    out += ["", "| dataset | noise_acc | tolerance, accuracy | noise_grass | tolerance, grass recall |",
            "|---|---|---|---|---|"]
    for d, info in DATASETS.items():
        n = noise_ref[d]["noise"]
        out.append(f"| {info['label']} | {n['noise_acc']:.4f} | {n['tol_acc']:.4f} | {n['noise_grass']:.4f} | "
                   f"{n['tol_grass']:.4f} |")
    return "\n".join(out)


def seed_table(rows, tol):
    out = ["| seed | digit acc, two-stage | digit acc, single | Δ | within ±{:.4f} | grass recall, two-stage | grass "
           "recall, single | Δ | within ±{:.4f} | matches closely |".format(tol["acc"], tol["grass"]),
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['seed']} | {r['two_stage']['digit_acc']:.4f} | {r['single']['digit_acc']:.4f} | "
                   f"{r['delta']['digit_acc']:+.4f} | {yes(r['within_acc'])} | {r['two_stage']['recall_grass']:.4f} | "
                   f"{r['single']['recall_grass']:.4f} | {r['delta']['recall_grass']:+.4f} | {yes(r['within_grass'])} "
                   f"| {yes(r['matches'])} |")
    return "\n".join(out)


def summary_table(rows):
    out = ["| metric | two-stage (r_a) | r_a-single | Δ (single − two-stage) |", "|---|---|---|---|"]
    for m, name in METRIC_NAMES.items():
        out.append(f"| {name} | {fmt([r['two_stage'][m] for r in rows])} | {fmt([r['single'][m] for r in rows])} | "
                   f"{fmt([r['delta'][m] for r in rows], '+.4f')} |")
    out.append(f"| kNN overlap at 15, single vs two-stage | | | {fmt([r['knn_overlap'] for r in rows])} |")
    out.append(f"| Procrustes disparity (0 = same shape) | | | {fmt([r['procrustes'] for r in rows])} |")
    out.append(f"| runtime, s | {fmt([r['runtime_two_stage'] for r in rows], '.0f')} | "
               f"{fmt([r['runtime_single'] for r in rows], '.0f')} | |")
    return "\n".join(out)


def tables(res, noise_ref):
    md = ["# Single-stage vs two-stage (r_a): tables (generated)", "",
          "## Reuse check of the stored two-stage runs", "", reuse_table(noise_ref), "",
          "## Optimization-noise reference (seed 3, two-stage)", "", noise_table(noise_ref)]
    for d, info in DATASETS.items():
        r = res[d]
        main = [x for x in r["seeds"] if x["seed"] in PRIMARY]
        rest = [x for x in r["seeds"] if x["seed"] not in PRIMARY]
        md += ["", f"## {info['label'].capitalize()}", "",
               f"Matches closely on seeds 3–5: {yes(r['matches_closely'])}. Single-stage seed-3 rerun "
               f"bit-identical: {yes(r['rerun_seed3_identical'])}.", "",
               "### Confirmation seeds 3–5 (primary)", "", seed_table(main, r["tolerances"]), "",
               "Mean [min, max] over seeds 3–5:", "", summary_table(main), "",
               "### Selection seeds 0–2 (secondary)", "", seed_table(rest, r["tolerances"]), "",
               summary_table(rest)]
    md += ["", f"**Adoption rule outcome: {res['outcome']}.**", "",
           "Runtimes: the two-stage times come from the stored runs (jobs listed above), the single-stage times from "
           "this check's job; both cover the residual graph and the fit, on 1 thread."]
    (OUT / "tables.md").write_text("\n".join(md) + "\n")


def scale_bar(ax, e):
    length = round_length(np.ptp(e[:, 0]))
    ax.add_artist(AnchoredSizeBar(ax.transData, length, f"{length:g}", "lower right", pad=0.1, sep=3, borderpad=0.2,
                                  frameon=False, size_vertical=0, fontproperties={"size": 8}, label_top=False))


def embedding_figure(d, res):
    data = mg.load(mg.processed_path(RECHECK_SEED, ROOT, name="mnist_grass_0to9" if d == "0to9" else "mnist_grass"))
    y, brightness = data["y"], data["grass_T"].mean(axis=1)
    colors = np.asarray([BLUE, ORANGE] if d == "01" else plt.get_cmap("tab10").colors)
    order = np.random.default_rng(0).permutation(len(y))  # fixed draw order: no digit painted last on purpose
    embs = {"two_stage": np.load(stored(d, RECHECK_SEED)[1]), "single": new(d, RECHECK_SEED, "single")[1]}
    row = next(r for r in res[d]["seeds"] if r["seed"] == RECHECK_SEED)
    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm"}):
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for j, (v, name) in enumerate(VERSIONS):
            e = embs[v][order]
            axes[0, j].scatter(*e.T, c=colors[y[order]], s=1, edgecolor="none", rasterized=True)
            sc = axes[1, j].scatter(*e.T, c=brightness[order], s=1, cmap=BLUE_RAMP, edgecolor="none",
                                    rasterized=True)
            axes[0, j].set_title(f"{name}\ndigit kNN accuracy {row[v]['digit_acc']:.3f}", fontsize=10)
            axes[1, j].set_title(f"grass recall {row[v]['recall_grass']:.3f}", fontsize=10)
            for ax in axes[:, j]:
                ax.set_aspect("equal", adjustable="datalim")
                ax.axis("off")
                scale_bar(ax, e)
        handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=colors[k], label=str(k))
                   for k in range(len(colors))]
        fig.legend(handles=handles, title="digit", loc="upper right", bbox_to_anchor=(1.07, 0.9), frameon=False,
                   ncol=1 if d == "01" else 2)
        fig.colorbar(sc, ax=axes[1, :], fraction=0.03, label="mean grass-patch brightness")
        fig.suptitle(f"MNIST-on-grass {DATASETS[d]['label']}, seed {RECHECK_SEED}: top, colored by digit; bottom, by "
                     f"grass brightness\nkNN overlap {row['knn_overlap']:.3f}, Procrustes disparity "
                     f"{row['procrustes']:.3f}", fontsize=10)
        fig.savefig(OUT / f"embeddings_{d}_seed{RECHECK_SEED}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def lr_figure():
    two = new("0to9", RECHECK_SEED, "two_stage")[0]["learning_rates"]
    one = new("0to9", RECHECK_SEED, "single")[0]["learning_rates"]["single"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(two["stage1"] + two["stage2"], color=BLUE, lw=2, label="two-stage (250 + 500)")
    ax.plot(one, color=ORANGE, lw=2, label="single-stage (750)")
    ax.axvline(len(two["stage1"]), color="#d9d8d3", lw=1, zorder=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Learning rate per epoch, as set by the package (digits 0–9, seed 3)", fontsize=9)
    fig.savefig(OUT / "learning_rate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    res = json.loads((OUT / "results.json").read_text())
    noise_ref = json.loads((OUT / "noise_reference.json").read_text())
    tables(res, noise_ref)
    for d in DATASETS:
        embedding_figure(d, res)
    lr_figure()
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
