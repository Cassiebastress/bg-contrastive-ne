"""Digit opacity on MNIST-on-grass 0/1: tables and figures, generated from results/mnist_grass_opacity/ (no number
typed by hand).

Tables: digit kNN accuracy per w and run (selection seeds, confirmation seeds, all six), the window and w*, the
pre-registered C1-C3 at w*, the crossing points, the secondary metrics, local cPCA's clusters, the checks and the
runtimes; once visual_judgment.md exists, the project owner's visual judgment next to the 0.60 rule.
Figures: accuracy vs opacity (all six seeds, mean with a min-max band); one 0 and one 1 at several w; the embeddings
at w* (seed 3); and the inspection figures for the visual judgment (T1 and r_a at every w, seed 3, gray and colored by
digit, titled with w only). Run colors: the reference categorical order (as in scripts/phase3_smoke.py), fixed per run;
digits 0/1 in the reference blue and orange. Needs data/processed/ and the embeddings (cluster).

    python scripts/opacity_report.py            # everything
    python scripts/opacity_report.py --tables   # tables only (no data or embeddings needed)
Outputs: results/mnist_grass_opacity/tables.md, tables.json and the figures below.
"""
import csv
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

from scripts.mnist_grass_0to9_report import round_length  # noqa: E402
from scripts.opacity_run import (OUT, R_A_MIN, RUNS, RUNS_DIR, SEEDS, SELECTION, T1_MAX, W_STORED,  # noqa: E402
                                 WS, load, name)

LABELS = {"r_a": "BC-NE (r_a)", "T1": "t-SNE (T1)", "a": "plain Neg-t-SNE (a)", "T3": "cPCA → t-SNE (T3)",
          "cPCA2D": "cPCA-2D", "local": "local cPCA"}
COLORS = dict(zip(RUNS, ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#008300", "#6250d6"]))
DIGIT_COLORS = np.array(["#2a78d6", "#eb6834"])  # 0 blue, 1 orange (reference slots 1-2, validator: all checks pass)
GRAY, INK, MUTED, RULE = "#8c8b85", "#1f1f1e", "#6b6a66", "#d9d8d3"
JUDGMENT = OUT / "visual_judgment.md"
FIG_SEED = 3


def read(stage):
    with open(OUT / f"runs_{stage}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"], r["w"], r["applicable"] = int(r["seed"]), float(r["w"]), r["applicable"] == "True"
        for m in ("digit_acc", "recall_digits", "recall_grass", "ari", "runtime_s"):
            r[m] = float(r[m]) if r[m] else None
    return rows


def vals(rows, run, w, m="digit_acc"):
    return [r[m] for r in rows if r["run"] == run and r["w"] == w and r["applicable"]]


def fmt(v):
    if not v:
        return "n/a"
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def acc_table(rows, m="digit_acc", means_only=False):
    out = ["| w | " + " | ".join(LABELS[r] for r in RUNS) + " |", "|---" * (len(RUNS) + 1) + "|"]
    for w in WS:
        cells = [(f"{np.mean(v):.3f}" if v else "n/a") if means_only else fmt(v)
                 for v in (vals(rows, run, w, m) for run in RUNS)]
        out.append(f"| {w:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def mean_curve(rows, run):
    return {w: float(np.mean(vals(rows, run, w))) for w in WS if vals(rows, run, w)}


def crossing(curve, level, below_strict):
    """Going down from the largest w: the first interval where the curve leaves the region (>= level for r_a, > level
    for T1), linearly interpolated. None if it never does on the grid."""
    ws = sorted(curve, reverse=True)
    for hi, lo in zip(ws, ws[1:]):
        inside_hi = curve[hi] >= level if below_strict else curve[hi] > level
        inside_lo = curve[lo] >= level if below_strict else curve[lo] > level
        if inside_hi and not inside_lo:
            return lo + (level - curve[lo]) * (hi - lo) / (curve[hi] - curve[lo])
    return None


def verdicts(conf, w_star):
    """C1-C3 at w* on the confirmation seeds."""
    seeds = SEEDS["confirmation"]
    acc = {run: {r["seed"]: r["digit_acc"] for r in conf if r["run"] == run and r["w"] == w_star} for run in RUNS}
    diff = [acc["r_a"][s] - acc["T1"][s] for s in seeds]
    c3 = ("r_a better, consistent across 3 paired seeds" if np.mean(diff) > 0 and all(d > 0 for d in diff) else
          "T1 better, consistent across 3 paired seeds" if np.mean(diff) < 0 and all(d < 0 for d in diff) else
          "not consistent")
    res = {"w_star": w_star, "r_a": [acc["r_a"][s] for s in seeds], "T1": [acc["T1"][s] for s in seeds],
           "C1": all(acc["r_a"][s] >= R_A_MIN for s in seeds), "C2": all(acc["T1"][s] <= T1_MAX for s in seeds),
           "C3_per_seed": diff, "C3_mean": float(np.mean(diff)), "C3": c3}
    yes = {True: "**holds**", False: "**fails**"}
    out = [f"At w\\* = {w_star:.2f}, confirmation seeds {', '.join(map(str, seeds))}:", "",
           "| check | per seed | verdict |", "|---|---|---|",
           f"| C1: r_a ≥ {R_A_MIN:.2f} on every seed | {', '.join(f'{v:.3f}' for v in res['r_a'])} "
           f"| {yes[res['C1']]} |",
           f"| C2: T1 ≤ {T1_MAX:.2f} on every seed | {', '.join(f'{v:.3f}' for v in res['T1'])} "
           f"| {yes[res['C2']]} |",
           f"| C3: r_a − T1 | {', '.join(f'{d:+.3f}' for d in diff)} (mean {res['C3_mean']:+.3f}) | {c3} |"]
    return "\n".join(out), res


def local_table(stage):
    out = ["| seed | w | K (silhouette) | clusters: target / background points | applicable | α per cluster |",
           "|---|---|---|---|---|---|"]
    for s in SEEDS[stage]:
        for w in WS:
            r = json.loads((RUNS_DIR / f"{name(stage, s, w, 'local')}.json").read_text())
            sizes = ", ".join(f"{t}/{b}" for t, b in r["cluster_sizes_T_B"].values())
            alphas = ", ".join(f"{a:.3g}" for a in r.get("alphas", {}).values()) or "—"
            out.append(f"| {s} | {w:.2f} | {r['K']} | {sizes} | {'yes' if r['applicable'] else '**no**'} | {alphas} |")
    return "\n".join(out)


def alpha_table(rows_by_stage):
    out = ["| w | label-free α of T3 / cPCA-2D, seeds 0–5 |", "|---|---|"]
    for w in WS:
        a = [json.loads((RUNS_DIR / f"{name(st, s, w, 'T3')}.json").read_text())["alpha"]
             for st in SEEDS for s in SEEDS[st]]
        out.append(f"| {w:.2f} | {', '.join(f'{v:.3g}' for v in a)} |")
    return "\n".join(out)


def checks():
    out = ["| stage | rerun (seed, w) | bit-identical, per run |", "|---|---|---|"]
    for st in SEEDS:
        c = json.loads((OUT / f"recheck_{st}.json").read_text())
        if "seed" in c:
            out.append(f"| {st} | ({c['seed']}, {c['w']:.2f}) | "
                       + ", ".join(f"{LABELS[r]} {'yes' if v else '**no**'}" for r, v in c["identical"].items()) + " |")
    out += ["", f"Every run checked its dataset's SHA-256 and that X_T({W_STORED}) reproduces the stored target bit "
                "for bit (a failure stops the run; none stopped)."]
    return "\n".join(out)


def runtime_table(rows):
    out = ["| process | median runtime per process (s) | range |", "|---|---|---|"]
    for run in ("r_a", "T1", "a", "T3", "local"):
        t = [r["runtime_s"] for r in rows if r["run"] == run]
        label = "cPCA fit + T3 + cPCA-2D" if run == "T3" else LABELS[run]
        out.append(f"| {label} | {np.median(t):.0f} | {min(t):.0f}–{max(t):.0f} |")
    return "\n".join(out)


def judgment_table(conf):
    """The project owner's visual judgment of T1 (seed 3), next to T1's seed-3 accuracy and the 0.60 rule."""
    if not JUDGMENT.exists():
        return None
    lines = [ln for ln in JUDGMENT.read_text().splitlines() if ln.startswith("| ") and not ln.startswith("| w ")]
    judged = {}
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells[0].replace(".", "").isdigit():
            judged[float(cells[0])] = cells[1:]
    acc = {r["w"]: r["digit_acc"] for r in conf if r["run"] == "T1" and r["seed"] == FIG_SEED}
    out = [f"| w | T1 accuracy, seed {FIG_SEED} | 0.60 rule: near chance | T1 separates the digits (project owner, "
           "visual) | r_a separates the digits (project owner, visual) |", "|---|---|---|---|---|"]
    for w in WS:
        j = judged.get(w, ["—", "—"])
        out.append(f"| {w:.2f} | {acc[w]:.3f} | {'yes' if acc[w] <= T1_MAX else 'no'} | {j[0]} | "
                   f"{j[1] if len(j) > 1 else '—'} |")
    return "\n".join(out)


def tables():
    sel, conf = read("selection"), read("confirmation")
    both = sel + conf
    s = json.loads(SELECTION.read_text())
    curves = {run: mean_curve(both, run) for run in RUNS}
    cross = {"r_a_below_0.80": crossing(curves["r_a"], R_A_MIN, True),
             "T1_to_0.60": crossing(curves["T1"], T1_MAX, False)}
    parts = {"selection": acc_table(sel), "confirmation": acc_table(conf), "all": acc_table(both)}
    md = ["## Digit kNN accuracy, selection seeds 0–2: mean [min, max]", "", parts["selection"], "",
          "## Window and w\\* (selection seeds)", "",
          f"Window W (mean r_a ≥ {R_A_MIN:.2f} and mean T1 ≤ {T1_MAX:.2f}): "
          + (", ".join(f"{w:.2f}" for w in s["window"]) if s["window"] else "empty")
          + f". w\\* (largest w in W): {s['w_star'] if s['w_star'] is None else format(s['w_star'], '.2f')}. "
          f"Largest mean r_a − T1 difference at w = {s['largest_gap_w']:.2f}.", "",
          "## Digit kNN accuracy, confirmation seeds 3–5: mean [min, max]", "", parts["confirmation"], ""]
    res = {}
    if s["w_star"] is not None:
        v_md, res = verdicts(conf, s["w_star"])
        md += ["## Pre-registered checks at w\\*", "", v_md, ""]
    md += ["## Digit kNN accuracy, all six seeds: mean [min, max]", "", parts["all"], "",
           "## Crossing points (descriptive, all six seeds, linear interpolation)", "",
           "| curve | crosses at w |", "|---|---|"]
    for k, v in cross.items():
        label = "mean r_a falls below 0.80" if k.startswith("r_a") else "mean T1 falls to 0.60"
        md.append(f"| {label} | {'not within the grid' if v is None else f'{v:.3f}'} |")
    md += ["", "## ARI (k-means, k = 2), all six seeds: mean", "", acc_table(both, "ari", True), "",
           "## Recall vs grass (lower is better), all six seeds: mean", "", acc_table(both, "recall_grass", True), "",
           "## Recall vs clean digits, all six seeds: mean", "", acc_table(both, "recall_digits", True), "",
           "## cPCA's label-free α per w", "", alpha_table(both), "",
           "## Local cPCA: clusters (selection and confirmation seeds)", "", local_table("selection"), "",
           local_table("confirmation"), "", "## Checks", "", checks(), "", "## Runtimes (one thread)", "",
           runtime_table(both)]
    jt = judgment_table(conf)
    if jt:
        md += ["", "## Visual judgment (project owner)", "", jt]
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "tables.json").write_text(json.dumps({"curves": {r: {str(w): v for w, v in c.items()}
                                                            for r, c in curves.items()},
                                                 "crossing": cross, "verdicts": res}, indent=2))
    return both, curves, s


def accuracy_figure(rows, s):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ws = np.array(sorted(WS))
    if s["window"]:
        lo, hi = min(s["window"]), max(s["window"])
        ax.axvspan(lo - 0.004, hi + 0.004, color=RULE, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(ws[-1], 0.505, "chance", color=MUTED, fontsize=8, ha="right", va="bottom")
    for run in RUNS:
        pts = [(w, vals(rows, run, w)) for w in ws if vals(rows, run, w)]
        x = np.array([w for w, _ in pts])
        m = np.array([np.mean(v) for _, v in pts])
        main = run in ("r_a", "T1")
        ax.fill_between(x, [min(v) for _, v in pts], [max(v) for _, v in pts], color=COLORS[run],
                        alpha=0.18 if main else 0.10, lw=0, zorder=2)
        ax.plot(x, m, color=COLORS[run], lw=2.2 if main else 1.2, marker="o", ms=4 if main else 3, label=LABELS[run],
                zorder=4 if main else 3)
    if s["w_star"] is not None:
        ax.axvline(s["w_star"], color=INK, lw=0.8, ls="--", zorder=1)
        ax.text(s["w_star"], 0.455, " w*", color=INK, fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("digit opacity w (target = w · digit + grass)")
    ax.set_ylabel("digit kNN accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w:.2f}" for w in ws], fontsize=7.5, rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Mean over seeds 0–5 (band: min–max); shaded: the window chosen on seeds 0–2", fontsize=9,
                 color=MUTED, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_opacity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def digits_figure():
    shown = (0.25, 0.10, 0.05, 0.03, 0.01)
    _, _, t = load(FIG_SEED, W_STORED)
    picks = [int(np.flatnonzero(t["y"] == d)[0]) for d in (0, 1)]
    fig, axes = plt.subplots(2, len(shown), figsize=(1.6 * len(shown), 3.5))
    for j, w in enumerate(shown):
        for i, p in enumerate(picks):
            img = (w * t["digits"][p] + t["grass_T"][p]).reshape(28, 28)
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=1.25, interpolation="nearest")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[0, j].set_title(f"w = {w:.2f}", fontsize=9)
    fig.suptitle(f"The first 0 and the first 1 of seed {FIG_SEED}, same grass patch at every w (gray scale fixed)",
                 fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / f"digits_by_opacity_seed{FIG_SEED}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter(ax, e, y=None, order=None):
    e = e[order]
    ax.scatter(*e.T, c=GRAY if y is None else DIGIT_COLORS[y[order]], s=1, edgecolor="none", rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def embeddings_figure(w_star):
    runs = ("r_a", "T1", "cPCA2D", "local")
    _, _, t = load(FIG_SEED, w_star)
    order = np.random.default_rng(0).permutation(len(t["y"]))
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.6))
    for ax, run in zip(axes, runs):
        rec = json.loads((RUNS_DIR / f"{name('confirmation', FIG_SEED, w_star, run)}.json").read_text())
        if not rec["applicable"]:
            ax.axis("off")
            ax.set_title(f"{LABELS[run]}\nnot applicable", fontsize=9.5)
            continue
        e = np.load(RUNS_DIR / f"{name('confirmation', FIG_SEED, w_star, run)}.npy")
        scatter(ax, e, t["y"], order)
        length = round_length(np.ptp(e[:, 0]))
        ax.add_artist(AnchoredSizeBar(ax.transData, length, f"{length:g}", "lower right", pad=0.1, sep=3,
                                      borderpad=0.2, frameon=False, size_vertical=0, fontproperties={"size": 8},
                                      label_top=False))
        ax.set_title(f"{LABELS[run]}\ndigit kNN acc {rec['digit_acc']:.3f} · ARI {rec['ari']:.3f}", fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c, label=str(d)) for d, c in enumerate(DIGIT_COLORS)]
    fig.legend(handles=handles, title="digit", frameon=False, loc="center right", fontsize=9)
    fig.suptitle(f"w = {w_star:.2f}, confirmation seed {FIG_SEED}", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(OUT / f"embeddings_w{round(w_star * 100):03d}_seed{FIG_SEED}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def inspection_figures():
    """Per run (T1, r_a): every w, gray on top and colored by digit below, titled with w only."""
    _, _, t = load(FIG_SEED, W_STORED)
    order = np.random.default_rng(0).permutation(len(t["y"]))
    ws = sorted(WS, reverse=True)
    for run in ("T1", "r_a"):
        fig, axes = plt.subplots(2, len(ws), figsize=(1.9 * len(ws), 4.4))
        for j, w in enumerate(ws):
            e = np.load(RUNS_DIR / f"{name('confirmation', FIG_SEED, w, run)}.npy")
            scatter(axes[0, j], e, None, order)
            scatter(axes[1, j], e, t["y"], order)
            axes[0, j].set_title(f"w = {w:.2f}", fontsize=9)
        axes[0, 0].text(-0.08, 0.5, "no labels", transform=axes[0, 0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=9, color=MUTED)
        axes[1, 0].text(-0.08, 0.5, "colored by digit", transform=axes[1, 0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=9, color=MUTED)
        fig.suptitle(f"{LABELS[run]}, confirmation seed {FIG_SEED} (0 blue, 1 orange)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"inspection_{run}_seed{FIG_SEED}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    rows, _, s = tables()
    if "--tables" in sys.argv:
        return
    accuracy_figure(rows, s)
    digits_figure()
    inspection_figures()
    if s["w_star"] is not None:
        embeddings_figure(s["w_star"])


if __name__ == "__main__":
    main()
