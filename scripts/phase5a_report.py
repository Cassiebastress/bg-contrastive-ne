"""Phase 5a tables and figures, generated from results/phase5a/ (no number typed by hand).

Per design: the confirmation table (mean [min, max] over the confirmation seeds), the pre-registered paired comparisons
((c) vs (e2) and (c) vs the oracle same-cluster control (o)), descriptive comparisons, the per-seed G1-G3 thresholds
(J), the xi_B diagnostics, excluded seeds, and the selection-seed appendix. Figures: seed-0 embeddings per design and
the Design J grid plot.

Run with the cne-bg env (after the main runs):  python scripts/phase5a_report.py
Outputs: results/phase5a/tables_{J,K}.md, embeddings_{J,K}.png, grid_J.png, comparisons.json, k_calibration.md,
predictions_status.md.
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

from common.benchmark import V3_J, make
from common.experiment import M_BS, paired
from data.synthetic_j import THETAS

OUT = ROOT / "results" / "phase5a"
COLS = [("acc_s", "signal acc", True), ("WC", "WC", True), ("G1", "G1", True), ("G2", "G2 (background acc)", False),
        ("G3_shares_signal", "G3 shares signal", True)]
PRIMARY = {"J": ["acc_s", "G1", "G2", "G3_shares_signal"], "K": ["WC"]}
SECONDARY = {"J": [], "K": ["acc_s"]}
LOWER_BETTER = {"G2"}
NAMES = {"a": "(a) plain Neg-t-SNE", "b": "(b) uniform extra negatives", "c": "(c) ξ_B negatives",
         "o": "(o) oracle same-cluster negatives", "e2": "(e2) cPCA → Neg-t-SNE", "e3": "(e3) cPCA → UMAP",
         "e2_k20": "(e2-k20) cPCA top 20 → Neg-t-SNE", "e2_k40": "(e2-k40) cPCA top 40 → Neg-t-SNE"}
CPCA_VS = {"J": ["e2"], "K": ["e2", "e2_k20", "e2_k40"]}


def read(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def vals(rows, run, metric):
    return np.array([float(r[metric]) for r in rows if r["run"] == run])


def fmt(v):
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]" if len(v) else ""


def label(run):
    group = run.split("_m")[0]
    return NAMES[group] + (f", m_B = {run.split('_m')[1]}" if "_m" in run else "")


def run_table(rows, runs, design):
    head = [name + (" (ill-posed for K)" if design == "K" and m.startswith("G") else "") for m, name, _ in COLS]
    out = ["| run | " + " | ".join(head) + " | flagged seeds |", "|---" * (len(COLS) + 2) + "|"]
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        flagged = sum(r["flag_collapse"] == "True" or r["flag_fragment"] == "True" for r in rr)
        out.append(f"| {label(run)} | " + " | ".join(fmt(vals(rows, run, m)) for m, _, _ in COLS)
                   + f" | {flagged}/{len(rr)} |")
    return "\n".join(out)


def thresholds_table(rows, runs):
    out = [f"Frozen thresholds, per seed: G1 ≥ {V3_J['G1']:.2f}, G2 ≤ {V3_J['G2']:.2f}, G3 shares signal ≥ "
           f"{V3_J['G3_shares_signal']:.2f}.", "",
           "| run | seeds meeting G1 | seeds meeting G2 | seeds meeting G3 | seeds meeting all three |",
           "|---|---|---|---|---|"]
    for run in runs:
        g1, g2, g3 = (vals(rows, run, m) for m in ("G1", "G2", "G3_shares_signal"))
        ok = (g1 >= V3_J["G1"]) & (g2 <= V3_J["G2"]) & (g3 >= V3_J["G3_shares_signal"])
        n = len(g1)
        out.append(f"| {label(run)} | {(g1 >= V3_J['G1']).sum()}/{n} | {(g2 <= V3_J['G2']).sum()}/{n} "
                   f"| {(g3 >= V3_J['G3_shares_signal']).sum()}/{n} | {ok.sum()}/{n} |")
    return "\n".join(out)


def comparisons(rows, x, ys, metrics):
    """Paired x - y per seed (raw differences shown; 'consistent' uses the sign for lower-is-better metrics)."""
    out = ["| comparison | metric | per-seed difference (x − y) | mean | verdict |", "|---|---|---|---|---|"]
    res = []
    for y in ys:
        for m in metrics:
            hib = m not in LOWER_BETTER
            xb, yb = paired(rows, x, y, m, hib), paired(rows, y, x, m, hib)
            raw = [float(a) - float(b) for a, b in zip(vals(rows, x, m), vals(rows, y, m))]
            verdict = (f"{x.split('_m')[0]} better, consistent across {len(raw)} paired seeds" if xb["consistent"]
                       else f"{y.split('_m')[0]} better, consistent across {len(raw)} paired seeds"
                       if yb["consistent"] else "not consistent")
            res.append({"x": x, "y": y, "metric": m, "per_seed_x_minus_y": raw, "mean": float(np.mean(raw)),
                        "x_better_consistent": xb["consistent"], "y_better_consistent": yb["consistent"]})
            out.append(f"| ({x.split('_m')[0]}) vs ({y.split('_m')[0]}) | {m}{' (lower is better)' if not hib else ''} "
                       f"| {', '.join(f'{d + 0.0:+.3f}' for d in raw)} | {np.mean(raw) + 0.0:+.3f} | {verdict} |")
    return "\n".join(out), res


def xi_table(summary):
    out = ["| seed | same-y_b precision of ξ_B | same-y_s excess | gates | coherence | detectability |",
           "|---|---|---|---|---|---|"]
    for seed, c in summary["checks"].items():
        d = c["xi_b_diagnostics"]
        out.append(f"| {seed} | {d['same_b_xi']:.3f} | {d['same_s_excess']:+.3f} "
                   f"| {'pass' if d['gate_same_b'] and d['gate_same_s'] else 'FAIL'} | {c['coherence']:.3f} "
                   f"| {c['detectability']:.3f} |")
    return "\n".join(out)


def excluded_note(summary):
    ex = summary["excluded_seeds"]
    if not ex:
        return "No confirmation seed was excluded."
    lines = ["Excluded confirmation seeds (validity judged from the data and (a) only; no comparison uses them):", ""]
    for e in ex:
        v = e["validity"]
        lines.append(f"- seed {e['seed']}: " + ", ".join(f"{k} {v[k]:.3f}" if isinstance(v[k], float) else
                                                         f"{k} {v[k]}" for k in v))
    return "\n".join(lines)


def selection_appendix(rows, selection):
    runs = (["a"] + [f"{g}_m{m}" for g in ("b", "c", "o") for m in M_BS] + ["e2", "e3"]
            + CPCA_VS[selection["design"]][1:])
    best = selection["best_m_B"]
    out = [f"Selection rule: {selection['rule']}. Chosen: (b) m_B = {best['b']}, (c) m_B = {best['c']}, "
           f"(o) m_B = {best['o']}.", "", run_table(rows, runs, selection["design"])]
    return "\n".join(out)


def embedding_figure(design, selection, cal):
    best = selection["best_m_B"]
    runs = ["a", f"b_m{best['b']}", f"c_m{best['c']}", f"o_m{best['o']}", "e2", "e3"]
    config = None
    if design == "J":
        config = {k: cal["chosen_J"][k] for k in ("theta", "projected", "gain")}
    t = make(design, 0, cal["rho"][design]["rho"], config).truth
    rows = [r for r in read(f"runs_{design}_selection.csv") if r["seed"] == "0"]
    fig, axes = plt.subplots(len(runs), 2, figsize=(5, 2.5 * len(runs)))
    for ax_row, run in zip(axes, runs):
        e = np.load(OUT / "embeddings" / f"{design}_selection_seed0_{run}.npy")
        r = next(x for x in rows if x["run"] == run)
        for ax, (y, lab) in zip(ax_row, ((t["y_s"], "y_s"), (t["y_b"], "y_b"))):
            ax.scatter(*e.T, c=y, s=0.6, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{run} by {lab}: signal acc {float(r['acc_s']):.2f}, WC {float(r['WC']):.2f}, "
                         f"G1 {float(r['G1']):.2f}", fontsize=6.5)
    fig.suptitle(f"Design {design}, selection seed 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"embeddings_{design}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def grid_figure():
    rows = read("grid_J.csv")
    panels = [("coherence", "coherence (high-D)"), ("detectability", "detectability (high-D WC)"),
              ("a_acc_s", "(a) signal acc"), ("e2_acc_s", "(e2) signal acc"), ("e2_WC", "(e2) WC"),
              ("e2_G1", "(e2) G1")]
    series = [(True, 2.0, "#1f77b4", "-"), (True, 6.0, "#1f77b4", "--"), (False, 2.0, "#d62728", "-"),
              (False, 6.0, "#d62728", "--")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for proj, gain, color, ls in series:
            rs = [r for r in rows if (r["projected"] == "True") == proj and float(r["gain"]) == gain]
            means = [np.mean([float(r[key]) for r in rs if int(r["theta"]) == t]) for t in THETAS]
            lo = [min(float(r[key]) for r in rs if int(r["theta"]) == t) for t in THETAS]
            hi = [max(float(r[key]) for r in rs if int(r["theta"]) == t) for t in THETAS]
            ax.plot(THETAS, means, color=color, ls=ls, lw=2, marker="o", ms=4,
                    label=f"{'projected' if proj else 'unprojected'}, g = {gain:g}")
            ax.vlines(THETAS, lo, hi, color=color, lw=1, alpha=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(THETAS)
        ax.set_xlabel("θ (degrees): 0 = additive, 90 = interaction only", fontsize=7)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].axhline(0.5, color="gray", lw=1, ls=":")
    axes[0, 1].axhline(0.85, color="gray", lw=1, ls=":")
    axes[1, 0].axhline(0.80, color="gray", lw=1, ls=":")
    axes[1, 2].axhline(0.50, color="gray", lw=1, ls=":")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Design J grid, selection seeds 0–2: mean (marker) and range (bar); dotted lines are the V2, V3, V4 "
                 "bars", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "grid_J.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def design_tables(design, cal):
    selection = json.loads((OUT / f"selection_{design}.json").read_text())
    conf = read(f"runs_{design}_confirmation.csv")
    sel_rows = read(f"runs_{design}_selection.csv")
    summary = json.loads((OUT / f"summary_{design}_confirmation.json").read_text())
    sel_summary = json.loads((OUT / f"summary_{design}_selection.json").read_text())
    best = selection["best_m_B"]
    c = f"c_m{best['c']}"
    runs = ["a", f"b_m{best['b']}", c, f"o_m{best['o']}", "e2", "e3"] + CPCA_VS[design][1:]
    pre, pre_res = comparisons(conf, c, CPCA_VS[design] + [f"o_m{best['o']}"], PRIMARY[design] + SECONDARY[design])
    primary = PRIMARY[design][0] if design == "K" else None
    beats = [r for r in pre_res if r["y"] in CPCA_VS[design] and r["metric"] == primary]
    beats_line = ("(c) beats cPCA on WC (consistent across 3 paired seeds against each of (e2), (e2-k20) and "
                  f"(e2-k40)): **{'yes' if beats and all(r['x_better_consistent'] for r in beats) else 'no'}**."
                  if design == "K" else "")
    desc, desc_res = comparisons(conf, c, ["a", f"b_m{best['b']}", "e3"], PRIMARY[design])
    seeds = ", ".join(str(s) for s in summary["seeds"])
    parts = [f"# Design {design} tables", "",
             f"## Confirmation seeds ({seeds}): mean [min, max]", "", run_table(conf, runs, design), ""]
    if design == "J":
        parts += ["## Frozen G1–G3 thresholds per confirmation seed", "", thresholds_table(conf, runs), ""]
    parts += ["## Pre-registered comparisons (confirmation seeds)", "",
              "Primary metrics: " + ", ".join(PRIMARY[design])
              + (f"; secondary: {', '.join(SECONDARY[design])}" if SECONDARY[design] else "") + ".", "", pre, "",
              beats_line, "",
              "## Descriptive comparisons (no verdict)", "", desc, "",
              "## ξ_B diagnostics and data checks (confirmation seeds)", "", xi_table(summary), "",
              "## Excluded seeds", "", excluded_note(summary), "",
              "## Appendix: selection seeds 0–2, every run", "", selection_appendix(sel_rows, selection), "",
              "### ξ_B diagnostics and data checks (selection seeds)", "", xi_table(sel_summary)]
    (OUT / f"tables_{design}.md").write_text("\n".join(parts) + "\n")
    embedding_figure(design, selection, cal)
    return {"preregistered": pre_res, "descriptive": desc_res, "selection": selection}


def k_calibration_figure(cal):
    """Design K, selection seed 0, from the calibration embeddings."""
    t = make("K", 0, cal["rho"]["K"]["rho"]).truth
    rows = {r["seed"]: r for r in read("k_validity.csv")}
    runs = ["a", "e2", "e3", "e2_k20", "e2_k40"]
    fig, axes = plt.subplots(len(runs), 2, figsize=(5, 2.5 * len(runs)))
    for ax_row, run in zip(axes, runs):
        e = np.load(OUT / "embeddings" / f"K_seed0_{run}.npy")
        for ax, (y, lab) in zip(ax_row, ((t["y_s"], "y_s"), (t["y_b"], "y_b"))):
            ax.scatter(*e.T, c=y, s=0.6, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{run} by {lab}: WC {float(rows['0'][f'{run}_WC']):.2f}", fontsize=7)
    fig.suptitle("Design K, selection seed 0 (calibration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "embeddings_K_calibration.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def k_calibration_table():
    """Design K calibration runs per seed (K stopped at V3-K, so these are its only embeddings)."""
    rows = read("k_validity.csv")
    runs = ["a", "e2", "e3", "e2_k20", "e2_k40"]
    out = ["# Design K calibration runs (selection seeds 0–2)", "",
           "(e2-k20) and (e2-k40) use the same label-free α as (e2), with the top 20 and top 40 cPCA components. "
           "V3-K is defined with the 10-component (e2) only.", "",
           "| run | WC | signal acc | G1 (ill-posed for K) | G2 (ill-posed for K) |", "|---|---|---|---|---|"]
    for run in runs:
        cells = [fmt(np.array([float(r[f"{run}_{m}"]) for r in rows])) for m in ("WC", "acc_s", "G1", "G2")]
        out.append(f"| {NAMES.get(run, run)} | " + " | ".join(cells) + " |")
    out += ["", f"α selected per seed: {', '.join(f'{float(r['selected_alpha']):.3g}' for r in rows)}. "
            f"10-D WC at the selected α: {', '.join(f'{float(r['selected_10d']):.3f}' for r in rows)}; "
            f"at the oracle α: {', '.join(f'{float(r['oracle_10d']):.3f}' for r in rows)}."]
    (OUT / "k_calibration.md").write_text("\n".join(out) + "\n")


def predictions_status(cal, res):
    """P1-P3 from results/phase5a/predictions.md, judged from the generated results (never revised)."""
    grid = {}
    for r in read("grid_J.csv"):
        grid.setdefault(r["config"], []).append(r)
    true = (lambda rs, keys: all(r[k] == "True" for r in rs for k in keys))
    proj_both = [n for n, rs in grid.items() if rs[0]["projected"] == "True" and rs[0]["theta"] != "0"
                 and true(rs, ("V3", "V4"))]
    valid = [n for n, rs in grid.items() if true(rs, ("V1", "V2", "V3", "V4"))]
    valid_unproj = [n for n in valid if grid[n][0]["projected"] == "False"]
    p1 = "not confirmed" if proj_both else "confirmed"
    p2 = "not applicable" if not valid else ("confirmed" if len(valid_unproj) == len(valid) else "not confirmed")
    chosen = cal["chosen_J"]
    p3_j = "not applicable (the chosen J configuration is projected)" if chosen["projected"] else "pending"
    if not chosen["projected"] and "J" in res:
        cmp = next(r for r in res["J"]["preregistered"] if r["y"] == "e2" and r["metric"] == "WC")
        conf = read("runs_J_confirmation.csv")
        c_run = res["J"]["preregistered"][0]["x"]
        g1_pass = {run: all(float(r["G1"]) >= V3_J["G1"] for r in conf if r["run"] == run) for run in (c_run, "e2")}
        ok = cmp["x_better_consistent"] and not any(g1_pass.values())
        p3_j = "confirmed" if ok else "not confirmed"
    k_rows = read("k_validity.csv")
    p3_k = ("not applicable (Design K stopped at calibration: V3-K failed on every seed, so (c) was never run)"
            if not any(r["V3"] == "True" for r in k_rows) else "pending")
    out = ["| prediction | status when committed | outcome | evidence |", "|---|---|---|---|",
           f"| P1: J projected configurations fail V3 only where coherence (V4) is low | pre-registered | **{p1}** "
           f"| projected configurations meeting V3 and V4 on every selection seed: "
           f"{', '.join(proj_both) if proj_both else 'none'} |",
           f"| P2: any valid J configuration is unprojected | descriptive, added after results | **{p2}** "
           f"| valid configurations: {', '.join(valid) if valid else 'none'} |",
           f"| P3 (J): (c) beats (e2) on WC; neither passes G1 | pre-registered | **{p3_j}** | chosen configuration "
           f"{chosen['config']} |",
           f"| P3 (K): (c) beats (e2) on WC; neither passes G1 | pre-registered | **{p3_k}** | (e2) WC on selection "
           f"seeds: {', '.join(f'{float(r['e2_WC']):.3f}' for r in k_rows)} |"]
    (OUT / "predictions_status.md").write_text("\n".join(out) + "\n")


def main():
    cal = json.loads((OUT / "calibration.json").read_text())
    grid_figure()
    k_calibration_table()
    k_calibration_figure(cal)
    res = {}
    for design in ("J", "K"):
        if (OUT / f"runs_{design}_confirmation.csv").exists():
            res[design] = design_tables(design, cal)
    (OUT / "comparisons.json").write_text(json.dumps(res, indent=2))
    predictions_status(cal, res)


if __name__ == "__main__":
    main()
