"""Phase 4 aggregation: metrics table, pre-registered comparisons, figures, and (with --stress-settings) the frozen
stress-test settings.

Reads results/phase4/runs_{level}_{stage}.csv and selection_{level}.json for whatever levels exist. Applies the rules in
common/experiment.py (Plan §5): the verdict uses the confirmation seeds 3-5; the same comparisons on the selection seeds
0-2 are reported as selection data only.

Run with the cne-bg env:  python scripts/phase4_report.py [--stress-settings]
"""
import argparse
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

from common.experiment import M_BS, PRIMARY, ROW_FIELDS, verdict
from common.repro import environment_info
from data.synthetic import LEVELS, MAIN_RHO, make_dataset

OUT = ROOT / "results" / "phase4"
STAGES = ("selection", "confirmation")
METRICS = [("acc_s", "signal kNN accuracy (higher is better)"), ("recall_s", "signal kNN recall vs S_T (higher)"),
           ("acc_b", "background kNN accuracy (lower)"), ("spearman_s", "signal distance Spearman (higher)")]
C_B, C_C, C_A, C_D = "#eb6834", "#1baf7a", "#2a78d6", "#6250d6"  # (b) orange, (c) aqua, (a) blue, (d) violet
COMP_FIELDS = ["level", "stage", "x", "y", "metric", "higher_is_better", "seeds", "per_seed", "mean_diff", "consistent"]


def read_runs(level, stage):
    path = OUT / f"runs_{level}_{stage}.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def values(rows, run, metric):
    return np.array([float(r[metric]) for r in rows if r["run"] == run])


def scatter(ax, embd, y, title):
    ax.scatter(*embd.T, c=y, s=0.6, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7)


def embedding(level, stage, seed, run):
    return np.load(OUT / "embeddings" / f"{level}_{stage}_seed{seed}_{run}.npy")


def labels(level, seed):
    t = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level]).truth
    return t["y_s"], t["y_b"]


def embedding_grid(level, rows):
    """Seed 0: rows = Stage 1 and all 12 Stage 2 runs; columns = colored by y_s / by y_b."""
    runs = ["stage1"] + [r["run"] for r in rows if int(r["seed"]) == 0 and r["group"] != "ref"]
    y_s, y_b = labels(level, 0)
    fig, axes = plt.subplots(len(runs), 2, figsize=(5, 2.4 * len(runs)))
    for ax_row, run in zip(axes, runs):
        r = next(x for x in rows if int(x["seed"]) == 0 and x["run"] == run)
        e = embedding(level, "selection", 0, run)
        flags = " [flag: collapse]" if r["flag_collapse"] == "True" else ""
        flags += " [flag: fragment]" if r["flag_fragment"] == "True" else ""
        scatter(ax_row[0], e, y_s, f"{run}: by y_s (acc {float(r['acc_s']):.2f}){flags}")
        scatter(ax_row[1], e, y_b, f"{run}: by y_b (acc {float(r['acc_b']):.2f})")
    fig.suptitle(f"{level}, seed 0 (selection)", fontsize=9)
    fig.savefig(OUT / f"embeddings_{level}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def allseeds_grid(level, sel, rows_by_stage):
    """Supplementary: seeds 0-5 for (a), the best (b) by signal accuracy, and (c) at m*_c."""
    runs = ["a", f"b_m{sel['best_b']['acc_s']}", f"c_m{sel['m_star_c']}"]
    seeds = [(s, "selection") for s in (0, 1, 2)] + [(s, "confirmation") for s in (3, 4, 5)]
    seeds = [(s, st) for s, st in seeds if rows_by_stage[st]]
    fig, axes = plt.subplots(len(seeds), 2 * len(runs), figsize=(2.2 * 2 * len(runs), 2.3 * len(seeds)), squeeze=False)
    for ax_row, (seed, stage) in zip(axes, seeds):
        y_s, y_b = labels(level, seed)
        for k, run in enumerate(runs):
            r = next(x for x in rows_by_stage[stage] if int(x["seed"]) == seed and x["run"] == run)
            e = embedding(level, stage, seed, run)
            scatter(ax_row[2 * k], e, y_s, f"seed {seed} {run}: y_s {float(r['acc_s']):.2f}")
            scatter(ax_row[2 * k + 1], e, y_b, f"seed {seed} {run}: y_b {float(r['acc_b']):.2f}")
    fig.suptitle(f"{level}: seeds 0-2 selection, 3-5 confirmation", fontsize=9)
    fig.savefig(OUT / f"embeddings_allseeds_{level}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def band(ax, rows, run, metric, color, label):
    v = values(rows, run, metric)
    if len(v):
        ax.axhspan(v.min(), v.max(), color=color, alpha=0.15, lw=0)
        ax.axhline(v.mean(), color=color, lw=1.2, ls="--", label=label)


def curves(levels, data, key_metrics, fname, logy=()):
    """Rows = levels, columns = metrics: (b) vs (c) over m_B (mean, min-max over seeds 0-2), (a) and (d) as bands,
    confirmation means (seeds 3-5) as open markers at the selected m_B."""
    fig, axes = plt.subplots(len(levels), len(key_metrics), figsize=(4.2 * len(key_metrics), 3.2 * len(levels)),
                             squeeze=False)
    for ax_row, level in zip(axes, levels):
        sel_rows, conf_rows = data[level]["selection"], data[level]["confirmation"]
        for ax, (metric, title) in zip(ax_row, key_metrics):
            band(ax, sel_rows, "a", metric, C_A, "(a) plain, seeds 0-2")
            band(ax, sel_rows, "d", metric, C_D, "(d) s=0, seeds 0-2")
            for group, color, name in (("b", C_B, "(b) uniform_extra"), ("c", C_C, "(c) xi_b")):
                vals = [values(sel_rows, f"{group}_m{m}", metric) for m in M_BS]
                mean = np.array([v.mean() for v in vals])
                ax.fill_between(M_BS, [v.min() for v in vals], [v.max() for v in vals], color=color, alpha=0.2, lw=0)
                ax.plot(M_BS, mean, "o-", color=color, lw=1.5, ms=4, label=f"{name}, seeds 0-2")
                for m in M_BS:
                    v = values(conf_rows, f"{group}_m{m}", metric)
                    if len(v):
                        ax.errorbar(m, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], fmt="s",
                                    mfc="white", mec=color, ecolor=color, ms=6, capsize=2)
            ax.set_xscale("log")
            ax.set_xticks(M_BS, [str(m) for m in M_BS])
            if metric in logy:
                ax.set_yscale("log")
            ax.set_title(f"{level}: {title}", fontsize=8)
            ax.set_xlabel("m_B (extra negatives per edge)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle("lines and bands: selection seeds 0-2 (mean, min-max); "
                 "open squares: confirmation seeds 3-5 (mean, min-max) at the selected m_B", fontsize=9, y=1.02)
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


TABLE_METRICS = [("acc_s", "signal acc"), ("recall_s", "signal recall"), ("acc_b", "background acc"),
                 ("spearman_s", "signal Spearman"), ("mean_phi", "mean phi")]
RUN_NAMES = {"highD_X_T": "high-D X_T (reference)", "stage1": "Stage 1 only (reference)", "a": "(a) plain",
             "d": "(d) s=0 reference"}


def run_name(run):
    if run in RUN_NAMES:
        return RUN_NAMES[run]
    group, m = run.split("_m")
    return f"({group}) {'uniform_extra' if group == 'b' else 'xi_b'}, m_B={m}"


def fmt(v, metric):
    digits = 4 if metric in ("recall_s", "spearman_s", "mean_phi") else 3
    return f"{v.mean():.{digits}f} [{v.min():.{digits}f}, {v.max():.{digits}f}]"


def run_table(rows):
    """Markdown table: one line per run, mean [min, max] over the seeds present, plus review flags."""
    runs = list(dict.fromkeys(r["run"] for r in rows))
    out = ["| run | " + " | ".join(name for _, name in TABLE_METRICS) + " | flags |",
           "|---" * (len(TABLE_METRICS) + 2) + "|"]
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        cells = []
        for metric, _ in TABLE_METRICS:
            v = np.array([float(r[metric]) for r in rr if r[metric] != ""])
            cells.append(fmt(v, metric) if len(v) else "")
        flags = sorted({k for r in rr for k in ("collapse", "fragment") if r[f"flag_{k}"] == "True"})
        out.append(f"| {run_name(run)} | " + " | ".join(cells) + f" | {', '.join(flags) or 'none'} |")
    return "\n".join(out)


def comparison_table(v):
    out = ["| comparison | metric | mean difference | per-seed differences | consistent across 3 paired seeds |",
           "|---|---|---|---|---|"]
    for c in v["comparisons"]:
        direction = "" if c["higher_is_better"] else " (drop)"
        per_seed = ", ".join(f"{d + 0.0:+.4f}" for d in c["per_seed"])  # + 0.0 turns -0.0 into 0.0
        out.append(f"| {run_name(c['x'])} vs {run_name(c['y'])} | {c['metric']}{direction} "
                   f"| {c['mean_diff'] + 0.0:+.4f} | {per_seed} | {'yes' if c['consistent'] else 'no'} |")
    return "\n".join(out)


def write_tables(levels, data, verdicts, selections):
    """results/phase4/tables.md: the tables REPORT.md quotes, generated from the run CSVs (never typed by hand)."""
    parts = ["# Phase 4 tables (generated by scripts/phase4_report.py)", "",
             "Values are mean [min, max] over seeds. Selection seeds 0-2 chose the settings; the verdict uses the "
             "fresh confirmation seeds 3-5.", ""]
    for level in levels:
        sel = selections[level]
        parts += [f"## {level}", "",
                  f"Selected on seeds 0-2: m*_c = {sel['m_star_c']}; best (b) by signal accuracy m_B = "
                  f"{sel['best_b']['acc_s']}, by signal recall m_B = {sel['best_b']['recall_s']}.", "",
                  "### Selection seeds 0-2", "", run_table(data[level]["selection"]), ""]
        if data[level]["confirmation"]:
            v = verdicts[level]["confirmation"]
            parts += ["### Confirmation seeds 3-5", "", run_table(data[level]["confirmation"]), "",
                      "### Pre-registered comparisons (confirmation seeds 3-5)", "", comparison_table(v), "",
                      f"Success criterion met: **{'yes' if v['success'] else 'no'}**"
                      f" ((c) runs flagged: {'yes' if v['c_flagged'] else 'no'}).", ""]
    (OUT / "tables.md").write_text("\n".join(parts))


def stress_outputs():
    """Stress test (graph-level hiding), not used for any choice: tables and a seed-0 figure, if results exist."""
    path = OUT / "stress" / "runs_stress.csv"
    if not path.exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    summary = json.loads((OUT / "stress" / "summary_stress.json").read_text())
    settings = summary["settings"]
    parts = ["# Stress test (graph-level hiding), not used for any choice", "",
             "Stress datasets: rho = 0.1 at every level, seeds 0-2. Settings frozen in stress_settings.json from the "
             "main selection seeds before any stress fit. Values are mean [min, max] over seeds.", ""]
    for level in LEVELS:
        lr = [r for r in rows if r["level"] == level]
        if not lr:
            continue
        diag = [summary["checks"][f"{level}/seed{s}"] for s in (0, 1, 2)]
        highd = ", ".join(f"{d['highd_acc_s_phase1']:.3f}" for d in diag)
        excess = ", ".join(f"{d['xi_b_diagnostics']['same_s_excess']:+.3f}" for d in diag)
        same_b = ", ".join(f"{d['xi_b_diagnostics']['same_b_xi']:.3f}" for d in diag)
        unhealthy = sorted({u for d in diag for u in d["unhealthy_runs"]})
        sel = {"m_star_c": settings[level]["m_star_c"], "best_b": settings[level]["best_b"]}
        v = verdict([r for r in lr if r["group"] != "ref"], sel)
        parts += [f"## {level}", "",
                  f"High-D X_T signal kNN accuracy (seeds 0-2, Phase 1 record): {highd}. "
                  f"xi_B same-y_b precision: {same_b}; same-y_s excess: {excess}. "
                  f"Unhealthy runs: {', '.join(unhealthy) or 'none'}.", "",
                  run_table(lr), "", comparison_table(v), "",
                  f"Pre-registered criterion applied descriptively: **{'met' if v['success'] else 'not met'}**.", ""]
    (OUT / "stress" / "tables_stress.md").write_text("\n".join(parts))

    levels = [lv for lv in LEVELS if any(r["level"] == lv for r in rows)]
    runs = {lv: settings[lv]["runs"] for lv in levels}
    ncol = max(len(r) for r in runs.values())
    fig, axes = plt.subplots(2 * len(levels), ncol, figsize=(2.4 * ncol, 2.4 * 2 * len(levels)), squeeze=False)
    for i, level in enumerate(levels):
        t = make_dataset(level, seed=0, signal_to_background=0.1).truth
        for j in range(ncol):
            for k, (y, name) in enumerate(((t["y_s"], "y_s"), (t["y_b"], "y_b"))):
                ax = axes[2 * i + k, j]
                if j >= len(runs[level]):
                    ax.axis("off")
                    continue
                run = runs[level][j]
                r = next(x for x in rows if x["level"] == level and int(x["seed"]) == 0 and x["run"] == run)
                e = np.load(OUT / "stress" / "embeddings" / f"{level}_stress_seed0_{run}.npy")
                metric = "acc_s" if name == "y_s" else "acc_b"
                scatter(ax, e, y, f"{level} {run}: by {name} ({float(r[metric]):.2f})")
    fig.suptitle("Stress test (graph-level hiding), seed 0, not used for any choice", fontsize=9)
    fig.savefig(OUT / "stress" / "stress_embeddings.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-settings", action="store_true")
    args = parser.parse_args()

    data = {level: {stage: read_runs(level, stage) for stage in STAGES} for level in LEVELS}
    levels = [lv for lv in LEVELS if data[lv]["selection"] and (OUT / f"selection_{lv}.json").exists()]
    all_rows = [r for lv in levels for st in STAGES for r in data[lv][st]]
    with open(OUT / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    comp_rows, verdicts, selections = [], {}, {}
    for level in levels:
        sel = json.loads((OUT / f"selection_{level}.json").read_text())
        selections[level] = sel
        verdicts[level] = {}
        for stage in STAGES:
            rows = [r for r in data[level][stage] if r["group"] != "ref"]
            if not rows:
                continue
            v = verdict(rows, sel)
            verdicts[level][stage] = v
            for c in v["comparisons"]:
                comp_rows.append({"level": level, "stage": stage, **{k: c[k] for k in COMP_FIELDS[2:]}})
    with open(OUT / "comparisons.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMP_FIELDS)
        writer.writeheader()
        writer.writerows(comp_rows)
    (OUT / "verdicts.json").write_text(json.dumps(
        {"environment": environment_info(), "primary_metrics": PRIMARY, "verdict_stage": "confirmation (seeds 3-5)",
         "selections": selections, "verdicts": verdicts}, indent=2))

    write_tables(levels, data, verdicts, selections)
    for level in levels:
        embedding_grid(level, data[level]["selection"])
        allseeds_grid(level, selections[level], data[level])
    curves(levels, data, METRICS, "metrics_vs_mB.png")
    curves(levels, data, [("mean_phi", "mean phi over fixed pairs (lower = more expanded)"),
                          ("embd_std", "embedding std")], "scale_vs_mB.png", logy=("mean_phi",))

    stress_outputs()

    for level in levels:
        for stage, v in verdicts[level].items():
            print(f"{level} {stage}: {v['c_run']} success={v['success']} flagged={v['c_flagged']}")
            for c in v["comparisons"]:
                print(f"    {c['x']} vs {c['y']:6s} {c['metric']:8s} mean {c['mean_diff'] + 0.0:+.4f} "
                      f"per-seed {[round(d, 4) for d in c['per_seed']]} consistent={c['consistent']}")

    if args.stress_settings:
        missing = [lv for lv in LEVELS if lv not in selections]
        if missing:
            raise SystemExit(f"selections missing for {missing}")
        settings = {lv: {"runs": ["a", "d", f"c_m{selections[lv]['m_star_c']}"]
                         + list(dict.fromkeys(f"b_m{selections[lv]['best_b'][k]}" for k in PRIMARY)),
                         "m_star_c": selections[lv]["m_star_c"], "best_b": selections[lv]["best_b"]}
                    for lv in LEVELS}
        (OUT / "stress_settings.json").write_text(json.dumps(
            {"rule": "frozen from selection_{level}.json before any stress fit; no tuning on stress data",
             "settings": settings}, indent=2))
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
