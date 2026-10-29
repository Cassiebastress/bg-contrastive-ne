"""Phase 5b tables and figures, generated from results/phase5b/ (no number typed by hand).

Per design: the confirmation table (mean [min, max] over the confirmation seeds, with "best cPCA" and the descriptive
"best linear projection" as per-seed maxima chosen with labels), the four pre-registered comparisons on WC, the
descriptive comparison with the linear ceiling, per-cluster WC (which clusters each method reveals), xi_B
diagnostics, excluded seeds, and the selection-seed appendix. Also the status of predictions P1-P5 and seed-0
embedding figures.

Run with the cne-bg env (after the main runs):  python scripts/phase5b_report.py
Outputs: results/phase5b/tables_{cyclic,between}.md, embeddings_{cyclic,between}.png, predictions_status.md,
comparisons.json.
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

from common.benchmark_l import K_LIST, WC_BAR
from common.experiment import paired
from data.synthetic_l import make_l

OUT = ROOT / "results" / "phase5b"
COLS = [("WC", "WC"), ("acc_s", "signal acc"), ("G1", "G1"), ("G1_k200", "G1 at k = 200"), ("G2", "G2"),
        ("G3_shares_signal", "G3 shares signal")]
NAMES = {"a": "(a) plain Neg-t-SNE", "b": "(b) uniform extra negatives", "c": "(c) ξ_B negatives",
         "o": "(o) oracle same-cluster negatives", "e2": "(e2) label-free cPCA → Neg-t-SNE",
         "e3": "(e3) label-free cPCA → UMAP", "best_cpca": "best cPCA variant (labels; per-seed max)",
         "best_linear": "best labeled linear projection (descriptive; per-seed max)"}


def read(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]" if len(v) else ""


def label(run):
    g = run.split("_m")[0] if "_m" in run else run
    return NAMES.get(g, run) + (f", m_B = {run.split('_m')[1]}" if "_m" in run else "")


def with_best(rows):
    """Add per-seed 'best_cpca' and 'best_linear' rows (copies of the max-WC row in each group)."""
    out = [r for r in rows if r["group"] not in ("cpca", "linear")]
    for group, name in (("cpca", "best_cpca"), ("linear", "best_linear")):
        for seed in sorted({r["seed"] for r in rows}, key=int):
            g = [r for r in rows if r["group"] == group and r["seed"] == seed]
            if g:
                best = max(g, key=lambda r: float(r["WC"]))
                out.append({**best, "run": name, "source_run": best["run"]})
    return out


def run_table(rows, runs):
    out = ["| run | " + " | ".join(n for _, n in COLS) + " | flagged seeds |", "|---" * (len(COLS) + 2) + "|"]
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        if not rr:
            continue
        flagged = sum(r.get("flag_collapse") == "True" or r.get("flag_fragment") == "True" for r in rr)
        out.append(f"| {label(run)} | " + " | ".join(fmt([float(r[m]) for r in rr]) for m, _ in COLS)
                   + f" | {flagged}/{len(rr)} |")
    return "\n".join(out)


def compare(rows, x, ys):
    out = ["| comparison | per-seed WC difference (x − y) | mean | verdict |", "|---|---|---|---|"]
    res = []
    for y in ys:
        xb, yb = paired(rows, x, y, "WC"), paired(rows, y, x, "WC")
        verdict = ("(c) better, consistent across 3 paired seeds" if xb["consistent"]
                   else f"{label(y)} better, consistent across 3 paired seeds" if yb["consistent"]
                   else "not consistent")
        res.append({"x": x, "y": y, "per_seed": xb["per_seed"], "mean": xb["mean_diff"],
                    "x_better_consistent": xb["consistent"], "y_better_consistent": yb["consistent"]})
        out.append(f"| (c) vs {label(y)} | {', '.join(f'{d + 0.0:+.3f}' for d in xb['per_seed'])} "
                   f"| {xb['mean_diff'] + 0.0:+.3f} | {verdict} |")
    return "\n".join(out), res


def revealed(row):
    return ",".join(str(i) for i in range(6) if float(row[f"WC_c{i}"]) >= WC_BAR) or "none"


def per_cluster_table(rows, runs):
    out = ["Per-cluster WC; a cluster counts as revealed if its WC ≥ 0.80.", "",
           "| seed | run | per-cluster WC (clusters 0–5) | revealed |", "|---|---|---|---|"]
    for seed in sorted({r["seed"] for r in rows}, key=int):
        for run in runs:
            r = next((x for x in rows if x["seed"] == seed and x["run"] == run), None)
            if r is None:
                continue
            name = label(run) + (f" ({r['source_run']})" if "source_run" in r else "")
            out.append(f"| {seed} | {name} | {', '.join(f'{float(r[f'WC_c{i}']):.2f}' for i in range(6))} "
                       f"| {revealed(r)} |")
    return "\n".join(out)


def cpca_grid(rows, summary):
    """Every cPCA variant grouped by alpha range and k: per-seed best WC in each cell, mean [min, max] over seeds."""
    cp = [r for r in rows if r["group"] == "cpca"]
    chosen = {seed: c["alpha"]["selected_alpha"] for seed, c in summary["checks"].items()}
    buckets = [("α < 1 (best)", lambda r: float(r["alpha"]) < 1),
               ("α > 1 (best)", lambda r: float(r["alpha"]) > 1),
               ("label-free α", lambda r: np.isclose(float(r["alpha"]), chosen[r["seed"]]))]
    out = ["Each cell: the per-seed best WC among the variants in that α range at that k, then mean [min, max] over "
           "seeds. The label-free row is the selected α at each k ((e2) is its k = 10 entry, embedded separately).", "",
           "| α | " + " | ".join(f"k = {k}" for k in K_LIST) + " |", "|---" * (len(K_LIST) + 1) + "|"]
    for name, keep in buckets:
        cells = []
        for k in K_LIST:
            best = [max((float(r["WC"]) for r in cp if r["seed"] == s and int(r["k"]) == k and keep(r)), default=None)
                    for s in chosen]
            cells.append(fmt([b for b in best if b is not None]))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def xi_table(summary):
    out = ["| seed | ξ_B same-y_b precision | same-y_s excess | detectability | graph signal |",
           "|---|---|---|---|---|"]
    for seed, c in summary["checks"].items():
        d = c["xi_b_diagnostics"]
        out.append(f"| {seed} | {d['same_b_xi']:.3f} | {d['same_s_excess']:+.3f} | {c['detectability']:.3f} "
                   f"| {c['graph_signal']:.3f} |")
    return "\n".join(out)


def excluded_note(summary):
    ex = summary["excluded_seeds"]
    if not ex:
        return "No confirmation seed was excluded."
    return "\n".join(["Excluded confirmation seeds (judged from the data and (a) only; in no comparison):", ""]
                     + [f"- seed {e['seed']}: " + ", ".join(f"{k} {v}" for k, v in e["validity"].items())
                        for e in ex])


def figure(variant, selection, cal):
    best = selection["best_m_B"]
    runs = ["a", f"b_m{best['b']}", f"c_m{best['c']}", f"o_m{best['o']}", "e2", "e3"]
    paths = {r: OUT / "embeddings" / f"{variant}_selection_seed0_{r}.npy" for r in runs}
    rec = next(r for r in read(f"validate_{variant}.csv") if r["seed"] == "0")
    for key, name in (("best cPCA (labels)", rec["cpca_ceiling_variant"]),
                      ("best linear (labels)", rec["linear_ceiling_projection"])):
        paths[f"{key}: {name}"] = OUT / "embeddings" / f"{variant}_cal_seed0_{name}.npy"
    t = make_l(variant, 0, cal["rho"][variant]["rho"]).truth
    rows = {r["run"]: r for r in read(f"runs_{variant}_selection.csv") if r["seed"] == "0"}
    ceil = {r["name"]: r for r in read(f"ceilings_{variant}.csv") if r["seed"] == "0"}
    fig, axes = plt.subplots(len(paths), 2, figsize=(5, 2.5 * len(paths)))
    for ax_row, (run, path) in zip(axes, paths.items()):
        e = np.load(path)
        wc = float(rows[run]["WC"]) if run in rows else float(ceil[run.split(": ")[1]]["emb_WC"])
        for ax, (y, lab) in zip(ax_row, ((t["y_s"], "y_s"), (t["y_b"], "y_b"))):
            ax.scatter(*e.T, c=y, s=0.6, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{run}\nby {lab}: WC {wc:.2f}", fontsize=6.5)
    fig.suptitle(f"L-{variant}, selection seed 0", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / f"embeddings_{variant}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def design_tables(variant, cal):
    selection = json.loads((OUT / f"selection_{variant}.json").read_text())
    best = selection["best_m_B"]
    raw = read(f"runs_{variant}_confirmation.csv")
    conf = with_best(raw)
    sel = read(f"runs_{variant}_selection.csv")
    summary = json.loads((OUT / f"summary_{variant}_confirmation.json").read_text())
    sel_summary = json.loads((OUT / f"summary_{variant}_selection.json").read_text())
    c, o, b = (f"{g}_m{best[g]}" for g in ("c", "o", "b"))
    runs = ["a", b, c, o, "e2", "e3", "best_cpca", "best_linear"]
    pre, pre_res = compare(conf, c, ["e2", "best_cpca", o, b])
    desc, desc_res = compare(conf, c, ["best_linear", "a", "e3"])
    sel_runs = ["a"] + [r for r in dict.fromkeys(x["run"] for x in sel) if "_m" in r] + ["e2", "e3"]
    parts = [f"# L-{variant} tables", "",
             f"## Confirmation seeds ({', '.join(map(str, summary['seeds']))}): mean [min, max]", "",
             "G1, G1 at k = 200, G2 and G3 are descriptive only for Design L.", "", run_table(conf, runs), "",
             "## Pre-registered comparisons on WC (confirmation seeds)", "",
             "(c) beats best cPCA only if it exceeds the per-seed maximum over every cPCA variant on every seed.", "",
             pre, "", "## Descriptive comparisons on WC (no verdict)", "", desc, "",
             "## Per-cluster WC (confirmation seeds)", "",
             per_cluster_table(conf, [c, "e2", "best_cpca", "best_linear"]), "",
             "## cPCA variants by α and k (confirmation seeds)", "", cpca_grid(raw, summary), "",
             "## ξ_B diagnostics and data checks (confirmation seeds)", "", xi_table(summary), "",
             "## Excluded seeds", "", excluded_note(summary), "",
             "## Appendix: selection seeds 0–2, every run", "",
             f"Selection rule: {selection['rule']}. Chosen: (b) m_B = {best['b']}, (c) m_B = {best['c']}, "
             f"(o) m_B = {best['o']}.", "", run_table(sel, sel_runs), "",
             "### ξ_B diagnostics and data checks (selection seeds)", "", xi_table(sel_summary)]
    (OUT / f"tables_{variant}.md").write_text("\n".join(parts) + "\n")
    figure(variant, selection, cal)
    return {"preregistered": pre_res, "descriptive": desc_res, "selection": selection}


def predictions_status(res):
    def consistent(variant, y_name):
        r = next(x for x in res[variant]["preregistered"] if x["y"] == y_name)
        return r["x_better_consistent"]

    rows = []
    cyc = read("validate_cyclic.csv") if (OUT / "validate_cyclic.csv").exists() else []
    if cyc:
        lin = [float(r["linear_ceiling"]) for r in cyc]
        ok = 0.50 <= float(np.mean(lin)) <= 0.70
        rows.append(("P1 (L-cyclic): embedding-level linear ceiling ≈ 0.5–0.7", "confirmed" if ok else "not confirmed",
                     f"mean over selection seeds {np.mean(lin):.3f} (per seed {', '.join(f'{x:.3f}' for x in lin)})"))
    if "cyclic" in res:
        e2, bc = consistent("cyclic", "e2"), consistent("cyclic", "best_cpca")
        rows.append(("P2 (L-cyclic): (c) beats (e2) and best cPCA on WC", "confirmed" if e2 and bc else
                     "not confirmed", f"vs (e2): {'consistent' if e2 else 'not consistent'}; vs best cPCA: "
                     f"{'consistent' if bc else 'not consistent'}"))
    btw = read("validate_between.csv") if (OUT / "validate_between.csv").exists() else []
    if btw:
        e2_wc = [float(r["e2_WC"]) for r in btw]
        ceil = [float(r["cpca_ceiling"]) for r in btw]
        alphas = [float(r["selected_alpha"]) for r in btw]
        ok = all(w <= WC_BAR for w in e2_wc) and all(c >= 0.90 for c in ceil)
        rows.append(("P3 (L-between): label-free (e2) WC ≤ 0.80; cPCA oracle ceiling ≥ 0.90",
                     "confirmed" if ok else "not confirmed",
                     f"(e2) WC {', '.join(f'{x:.3f}' for x in e2_wc)} at label-free α "
                     f"{', '.join(f'{a:.3g}' for a in alphas)}; cPCA ceiling {', '.join(f'{x:.3f}' for x in ceil)} "
                     "(selection seeds, embedding level)"))
    if "between" in res:
        e2, bc = consistent("between", "e2"), consistent("between", "best_cpca")
        rows.append(("P4 (L-between): (c) beats (e2) but not best cPCA", "confirmed" if e2 and not bc else
                     "not confirmed", f"vs (e2): {'consistent' if e2 else 'not consistent'}; vs best cPCA: "
                     f"{'consistent' if bc else 'not consistent'}"))
    for variant in ("cyclic", "between"):
        if variant in res:
            sel = res[variant]["selection"]["best_m_B"]
            o_name, b_name = f"o_m{sel['o']}", f"b_m{sel['b']}"
            o_ok, b_ok = consistent(variant, o_name), consistent(variant, b_name)
            rows.append((f"P5 (L-{variant}): (c) beats (o) and (b) on WC", "confirmed" if o_ok and b_ok else
                         "not confirmed", f"vs (o): {'consistent' if o_ok else 'not consistent'}; vs (b): "
                         f"{'consistent' if b_ok else 'not consistent'}"))
    out = ["| prediction | outcome | evidence |", "|---|---|---|"]
    out += [f"| {p} | **{s}** | {e} |" for p, s, e in rows]
    (OUT / "predictions_status.md").write_text("\n".join(out) + "\n")


def main():
    cal = json.loads((OUT / "calibration.json").read_text())
    res = {}
    for variant in ("cyclic", "between"):
        if (OUT / f"runs_{variant}_confirmation.csv").exists():
            res[variant] = design_tables(variant, cal)
    (OUT / "comparisons.json").write_text(json.dumps(res, indent=2))
    predictions_status(res)


if __name__ == "__main__":
    main()
