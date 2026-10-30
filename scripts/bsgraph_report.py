"""Background-suppressed graph, Phase 2 tables and figures (Phase 1's are in results/bsgraph/phase1.md).

New runs (results/bsgraph/runs_*.csv) next to the stored MNIST-on-grass confirmation runs (a), (c) m_B = 5, T1, T2,
T3 (results/mnist_grass/runs_confirmation.csv; reused under results/bsgraph/reuse.json). Tables: confirmation seeds
(mean [min, max] and per seed), the pre-registered comparisons on digit kNN accuracy (secondary metrics alongside, no
verdict), the descriptive (r_a) vs (a), and the selection-seed appendix. Figure: confirmation seed 3, (r_a), (r_c), (c)
and T3 colored by digit and by mean grass brightness. Needs data/processed/ and the stored embeddings (cluster).

    python scripts/bsgraph_report.py
Outputs: results/bsgraph/tables_phase2.md, comparisons.json, embeddings_seed3.png
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

OUT, STORED = ROOT / "results" / "bsgraph", ROOT / "results" / "mnist_grass"
FIG_SEED = 3
STORED_C = "c_m5"
METRICS = [("digit_acc", "digit kNN accuracy"), ("recall_digits", "recall vs clean digits"),
           ("recall_grass", "recall vs grass patches (lower is better)")]
NAMES = {"r_a": "(r_a) residual graph, background term off", "a": "(a) raw graph, background term off",
         "c_m5": "(c) raw graph + ξ_B, m_B = 5", "T1": "T1 t-SNE", "T2": "T2 PCA-50 → t-SNE",
         "T3": "T3 cPCA → t-SNE"}


def label(run):
    if run.startswith("r_c_m"):
        return f"(r_c) residual graph + ξ_B, m_B = {run[5:]}"
    return NAMES[run]


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
        out += ["", "Per seed, digit kNN accuracy:", "", "| run | " + " | ".join(f"seed {s}" for s in seeds) + " |",
                "|---" * (len(seeds) + 1) + "|"]
        for run in runs:
            v = {int(r["seed"]): float(r["digit_acc"]) for r in rows if r["run"] == run}
            out.append(f"| {label(run)} | " + " | ".join(f"{v[s]:.3f}" for s in seeds) + " |")
    return "\n".join(out)


def compare(rows, pairs, verdicts=True):
    out = ["| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean "
           "difference | recall vs grass, mean difference |", "|---|---|---|---|---|---|"]
    res = []
    for x, y in pairs:
        xy, yx = paired(rows, x, y, "digit_acc"), paired(rows, y, x, "digit_acc")
        verdict = (f"{label(x)} better, consistent across 3 paired seeds" if xy["consistent"] else
                   f"{label(y)} better, consistent across 3 paired seeds" if yx["consistent"] else "not consistent")
        sec = {m: paired(rows, x, y, m)["mean_diff"] for m in ("recall_digits", "recall_grass")}
        res.append({"x": x, "y": y, "per_seed": xy["per_seed"], "mean": xy["mean_diff"],
                    "x_better_consistent": xy["consistent"], "y_better_consistent": yx["consistent"],
                    "secondary_mean_diff": sec})
        out.append(f"| {label(x)} vs {label(y)} | {', '.join(f'{d + 0.0:+.3f}' for d in xy['per_seed'])} "
                   f"| {xy['mean_diff'] + 0.0:+.3f} | {verdict if verdicts else '(descriptive) ' + verdict} "
                   f"| {sec['recall_digits'] + 0.0:+.3f} | {sec['recall_grass'] + 0.0:+.3f} |")
    return "\n".join(out), res


def figure(rc):
    d = mg.load(mg.processed_path(FIG_SEED, ROOT))
    y, brightness = d["y"], d["grass_T"].mean(axis=1)
    panels = [(rc, OUT / "runs" / f"confirmation_seed{FIG_SEED}_{rc}"),
              ("r_a", OUT / "runs" / f"confirmation_seed{FIG_SEED}_r_a"),
              (STORED_C, STORED / "runs" / f"confirmation_seed{FIG_SEED}_{STORED_C}"),
              ("T3", STORED / "runs" / f"confirmation_seed{FIG_SEED}_T3")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(6, 3 * len(panels)))
    for ax_row, (run, stem) in zip(axes, panels):
        e = np.load(f"{stem}.npy")
        rec = json.loads(Path(f"{stem}.json").read_text())
        ax_row[0].scatter(*e.T, c=y, s=0.6, cmap="coolwarm", vmin=0, vmax=1, edgecolor="none", rasterized=True)
        sc = ax_row[1].scatter(*e.T, c=brightness, s=0.6, cmap="viridis", edgecolor="none", rasterized=True)
        ax_row[0].set_title(f"{label(run)}\nby digit (blue 0, red 1): acc {rec['digit_acc']:.3f}", fontsize=7)
        ax_row[1].set_title(f"{label(run)}\nby mean grass brightness: grass recall {rec['recall_grass']:.3f}",
                            fontsize=7)
        fig.colorbar(sc, ax=ax_row[1], fraction=0.04)
        for ax in ax_row:
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle(f"Background-suppressed graph, MNIST-on-grass confirmation seed {FIG_SEED}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / f"embeddings_seed{FIG_SEED}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    reuse = json.loads((OUT / "reuse.json").read_text())
    recheck = json.loads((OUT / "recheck.json").read_text())
    rc = f"r_c_m{sel['best_m_B']}"
    new_conf = read(OUT / "runs_confirmation.csv")
    stored = [r for r in read(STORED / "runs_confirmation.csv") if r["run"] in (STORED_C, "a", "T1", "T2", "T3")]
    rows = new_conf + stored
    pre_md, pre = compare(rows, [(rc, STORED_C), (rc, "T3"), (rc, "T1"), (rc, "T2"), (rc, "r_a")])
    desc_md, desc = compare(rows, [("r_a", "a")], verdicts=False)
    sel_rows = read(OUT / "runs_selection.csv")
    sel_runs = ["r_a"] + [r for r in dict.fromkeys(x["run"] for x in sel_rows) if r.startswith("r_c")]
    sel_runs = ["r_a"] + sorted(sel_runs[1:], key=lambda r: int(r[5:]))
    parts = ["# Background-suppressed graph, Phase 2 tables", "",
             "## Confirmation seeds (3, 4, 5): mean [min, max]", "",
             "(r_a) and (r_c) are new; (a), (c), T1, T2 and T3 are the stored MNIST-on-grass confirmation runs "
             f"(reuse check: code (a)/(c) depend on unchanged since `{reuse['stored_commit']}`: "
             f"{'yes' if reuse['code_unchanged'] else 'NO'}; dataset hashes checked by every run).", "",
             run_table(rows, [rc, "r_a", STORED_C, "a", "T1", "T2", "T3"], per_seed=True), "",
             "## Pre-registered comparisons (confirmation seeds; verdicts on digit kNN accuracy only)", "",
             "Differences are first run − second run.", "", pre_md, "",
             "## Descriptive comparison", "", desc_md, "",
             f"Determinism: {label(rc)} on seed {recheck['seed']} rerun: "
             f"{'bit-identical' if recheck['identical'] else 'NOT identical'}.", "",
             "## Appendix: selection seeds 0–2", "",
             f"Selection rule: {sel['rule']}. Chosen: m_B = {sel['best_m_B']}.", "", run_table(sel_rows, sel_runs)]
    (OUT / "tables_phase2.md").write_text("\n".join(parts) + "\n")
    (OUT / "comparisons.json").write_text(json.dumps({"r_c": rc, "preregistered": pre, "descriptive": desc}, indent=2))
    figure(rc)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
