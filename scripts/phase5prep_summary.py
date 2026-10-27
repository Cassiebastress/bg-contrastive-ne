"""Phase 5 prep summary: generated tables and figures for the cPCA baseline (Phase_5_Plan.md §2 B3).

Reads results/phase5prep/cpca_runs.csv, alphas.csv and cpca_check.json and writes:
- cpca_summary.md: per level, (a), (c), (e1), (e2), (e3), (e2-oracle) on the confirmation seeds (mean [min, max]);
  the pre-registered criteria 1-4 applied descriptively (results/phase5prep/metrics_preregistration.md); per-seed
  paired differences (c) - (e2) with the Phase 4 "consistent across 3 paired seeds" rule; alphas per dataset;
  appendices with the selection seeds and every alpha representative.
- cpca_criteria.json, and figures cpca_embeddings_{level}.png (seed 0).

Run with the cne-bg env:  python scripts/phase5prep_summary.py
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
from data.synthetic import LEVELS, MAIN_RHO, make_dataset

OUT = ROOT / "results" / "phase5prep"
P4 = ROOT / "results" / "phase4"
MAIN_RUNS = ["a", "c", "e1", "e2", "e3", "e2_oracle"]
NAMES = {"a": "(a) plain Neg-t-SNE (Phase 4)", "c": "(c) xi_b at m*_c (Phase 4)", "e1": "(e1) cPCA-2D",
         "e2": "(e2) cPCA -> Neg-t-SNE", "e3": "(e3) cPCA -> UMAP",
         "e2_oracle": "(e2-oracle) cPCA -> Neg-t-SNE, oracle alpha chosen by 10-D cPCA signal accuracy"}
COLS = [("acc_s", "signal acc", True), ("recall_s", "signal recall", True), ("G1", "G1", True),
        ("G2", "G2 (background acc)", False), ("G3_shares_signal", "G3 shares signal", True),
        ("G3_shares_background", "G3 shares background", None)]
THRESH = {"G1": 0.50, "G2": 0.50, "G3_shares_signal": 0.50, "acc_tolerance": 0.05}


def read(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def main_rows(rows, level, stage):
    """(a), (c), (e2-oracle), and (e1)-(e3) at the selected alpha."""
    return [r for r in rows if r["level"] == level and r["stage"] == stage
            and (r["run"] in ("a", "c", "e2_oracle") or r["selected"] == "True")]


def vals(rows, run, metric):
    return np.array([float(r[metric]) for r in rows if r["run"] == run])


def fmt(v, d=3):
    return f"{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]" if len(v) else ""


def table(rows):
    out = ["| run | " + " | ".join(name for _, name, _ in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for run in MAIN_RUNS:
        cells = [fmt(vals(rows, run, m), 4 if m == "recall_s" else 3) for m, _, _ in COLS]
        out.append(f"| {NAMES[run]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def criteria(rows, c_mean):
    """Pre-registered criteria 1-4 (descriptive): 1-3 on every confirmation seed, 4 on the confirmation mean."""
    res = {}
    for run in MAIN_RUNS:
        g1, g2, g3, acc = (vals(rows, run, m) for m in ("G1", "G2", "G3_shares_signal", "acc_s"))
        res[run] = {"1_G1_all_seeds": bool((g1 >= THRESH["G1"]).all()), "G1_min": float(g1.min()),
                    "2_G2_all_seeds": bool((g2 <= THRESH["G2"]).all()), "G2_max": float(g2.max()),
                    "3_G3_all_seeds": bool((g3 >= THRESH["G3_shares_signal"]).all()), "G3_min": float(g3.min()),
                    "4_acc_mean": bool(acc.mean() >= c_mean - THRESH["acc_tolerance"]),
                    "acc_mean": float(acc.mean())}
        keys = ("1_G1_all_seeds", "2_G2_all_seeds", "3_G3_all_seeds", "4_acc_mean")
        res[run]["all_four"] = all(res[run][k] for k in keys)
    return res


def criteria_table(res, c_mean):
    yes = {True: "yes", False: "no"}
    out = [f"Thresholds: G1 >= 0.50, G2 <= 0.50 and G3 shares signal >= 0.50 on every confirmation seed; signal "
           f"accuracy mean >= (c)'s confirmation mean - 0.05 = {c_mean - THRESH['acc_tolerance']:.3f}.", "",
           "| run | 1: G1 (min) | 2: G2 (max) | 3: G3 shares signal (min) | 4: signal acc (mean) | all four |",
           "|---|---|---|---|---|---|"]
    for run in MAIN_RUNS:
        r = res[run]
        out.append(f"| {NAMES[run]} | {yes[r['1_G1_all_seeds']]} ({r['G1_min']:.3f}) "
                   f"| {yes[r['2_G2_all_seeds']]} ({r['G2_max']:.3f}) | {yes[r['3_G3_all_seeds']]} ({r['G3_min']:.3f}) "
                   f"| {yes[r['4_acc_mean']]} ({r['acc_mean']:.3f}) | {yes[r['all_four']]} |")
    return "\n".join(out)


def paired_table(rows):
    out = ["| metric | (c) - (e2) per seed | mean | consistent across 3 paired seeds |", "|---|---|---|---|"]
    res = {}
    for m, name, hib in COLS[:5]:
        if m == "recall_s":
            continue
        c_better, e2_better = paired(rows, "c", "e2", m, hib), paired(rows, "e2", "c", m, hib)
        raw = [float(a) - float(b) for a, b in zip(vals(rows, "c", m), vals(rows, "e2", m))]
        verdict = "(c) better" if c_better["consistent"] else "(e2) better" if e2_better["consistent"] else "neither"
        res[m] = {"per_seed_c_minus_e2": raw, "verdict": verdict}
        out.append(f"| {name}{'' if hib else ' (lower is better)'} | {', '.join(f'{d + 0.0:+.3f}' for d in raw)} "
                   f"| {np.mean(raw) + 0.0:+.3f} | {verdict} |")
    return "\n".join(out), res


def alpha_table(alpha_rows, level):
    out = ["| seed | stage | selected alpha | representatives (variance ratio) | oracle alpha | 10-D signal acc: "
           "selected / oracle |", "|---|---|---|---|---|---|"]
    for r in alpha_rows:
        if r["level"] != level:
            continue
        pairs = zip(r["representatives"].split(";"), r["variance_ratios"].split(";"))
        reps = ", ".join(f"{float(a):.3g} ({float(v):.2f})" for a, v in pairs)
        out.append(f"| {r['seed']} | {r['stage']} | {float(r['selected_alpha']):.3g} | {reps} "
                   f"| {float(r['oracle_alpha']):.3g} | {float(r['selected_acc_10d']):.3f} / "
                   f"{float(r['oracle_acc_10d']):.3f} |")
    return "\n".join(out)


def reps_table(rows, level):
    out = ["| seed | alpha | selected | e1 signal acc | e2 signal acc | e3 signal acc | e2 G1 | e3 G1 | e2 G2 "
           "| e3 G2 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    lr = [r for r in rows if r["level"] == level and r["run"] in ("e1", "e2", "e3")]
    for seed in dict.fromkeys(r["seed"] for r in lr):
        for alpha in dict.fromkeys(r["alpha"] for r in lr if r["seed"] == seed):
            g = {r["run"]: r for r in lr if r["seed"] == seed and r["alpha"] == alpha}
            out.append(f"| {seed} | {float(alpha):.3g} | {'yes' if g['e1']['selected'] == 'True' else ''} "
                       + " | ".join(f"{float(g[run][m]):.3f}" for m, run in
                                    (("acc_s", "e1"), ("acc_s", "e2"), ("acc_s", "e3"), ("G1", "e2"), ("G1", "e3"),
                                     ("G2", "e2"), ("G2", "e3"))) + " |")
    return "\n".join(out)


def figure(rows, alpha_rows, level):
    a_row = next(r for r in alpha_rows if r["level"] == level and r["seed"] == "0")
    alpha = float(a_row["selected_alpha"])
    sel = json.loads((P4 / f"selection_{level}.json").read_text())
    paths = {"a": P4 / "embeddings" / f"{level}_selection_seed0_a.npy",
             "c": P4 / "embeddings" / f"{level}_selection_seed0_c_m{sel['m_star_c']}.npy",
             **{run: OUT / "embeddings" / f"{level}_seed0_{run}_alpha{alpha:.6g}.npy" for run in ("e1", "e2", "e3")}}
    t = make_dataset(level, seed=0, signal_to_background=MAIN_RHO[level]).truth
    fig, axes = plt.subplots(len(paths), 2, figsize=(5, 2.5 * len(paths)))
    for ax_row, (run, path) in zip(axes, paths.items()):
        e = np.load(path)
        r = next(x for x in rows if x["level"] == level and x["seed"] == "0" and x["run"] == run
                 and (run in ("a", "c") or x["selected"] == "True"))
        for ax, (y, lab, m) in zip(ax_row, ((t["y_s"], "y_s", "acc_s"), (t["y_b"], "y_b", "G2"))):
            ax.scatter(*e.T, c=y, s=0.6, alpha=0.5, cmap="tab10", vmin=0, vmax=9, edgecolor="none", rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{run}: by {lab} (acc {float(r[m]):.2f}, G1 {float(r['G1']):.2f})", fontsize=7)
    fig.suptitle(f"{level}, seed 0; cPCA at alpha = {alpha:.3g}", fontsize=9)
    fig.savefig(OUT / f"cpca_embeddings_{level}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


NOISE_VAR = 0.01  # the generator's isotropic noise variance (std 0.1)


def signal_outside_background(level, seeds):
    """Post-hoc data diagnostic: the fraction of signal variance (S_T) lying in directions where the clean background
    (Bg_T) has variance below the noise variance, and how many directions carry background variance above it."""
    out = []
    for seed in seeds:
        t = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level]).truth
        w, V = np.linalg.eigh(np.cov(t["Bg_T"], rowvar=False))
        free = V[:, w < NOISE_VAR]
        S = t["S_T"] - t["S_T"].mean(axis=0)
        out.append((int((w >= NOISE_VAR).sum()), float(np.sum((S @ free) ** 2) / np.sum(S ** 2))))
    return out


def outside_table(rows):
    md = ["| level | seeds | background directions with variance >= noise variance (of 50) | fraction of signal "
          "variance in the remaining, background-free directions |", "|---|---|---|---|"]
    res = {}
    for level in LEVELS:
        seeds = sorted({int(r["seed"]) for r in rows if r["level"] == level})
        vals_ = signal_outside_background(level, seeds)
        res[level] = vals_
        dims = [v[0] for v in vals_]
        frac = [v[1] for v in vals_]
        md.append(f"| {level} | {', '.join(map(str, seeds))} | {min(dims)}–{max(dims)} | "
                  f"{min(frac):.2f}–{max(frac):.2f} |")
    return "\n".join(md) + "\n", res


def main():
    rows, alpha_rows = read("cpca_runs.csv"), read("alphas.csv")
    check = json.loads((OUT / "cpca_check.json").read_text())
    md = ["# cPCA baseline summary (generated by scripts/phase5prep_summary.py)", "",
          "Main tables: confirmation seeds, mean [min, max]. (e1)-(e3) use the label-free selected alpha. Global "
          "metrics G1-G3 are defined in results/phase5prep/metrics_preregistration.md.", "",
          f"Reference-package check (easy seed 0): max principal angle "
          f"{check['package_check']['max_principal_angle_rad']:.2e} rad (tolerance {ANGLE_TOL:.0e}); representative "
          f"alphas identical: {check['package_check']['agrees']}. Determinism: (e2) rerun bit-identical "
          f"{check['determinism']['e2_bit_identical']}, (e3) {check['determinism']['e3_bit_identical']}.", ""]
    all_crit, all_paired = {}, {}
    for level in LEVELS:
        conf = main_rows(rows, level, "confirmation")
        c_mean = float(vals(conf, "c", "acc_s").mean())
        crit = criteria(conf, c_mean)
        ptab, pres = paired_table(conf)
        all_crit[level], all_paired[level] = crit, pres
        seeds = ", ".join(sorted({r["seed"] for r in conf}, key=int))
        md += [f"## {level} (confirmation seeds {seeds})", "", table(conf), "",
               "### Pre-registered criteria 1-4, applied descriptively", "", criteria_table(crit, c_mean), "",
               "### (c) vs (e2), paired by dataset seed", "", ptab, "",
               "### alpha per dataset", "", alpha_table(alpha_rows, level), ""]
        figure(rows, alpha_rows, level)
    md += ["## Appendix: selection seeds 0-2", ""]
    for level in LEVELS:
        md += [f"### {level}", "", table(main_rows(rows, level, "selection")), ""]
    md += ["## Appendix: every alpha representative (all seeds)", ""]
    for level in LEVELS:
        md += [f"### {level}", "", reps_table(rows, level), ""]
    (OUT / "cpca_summary.md").write_text("\n".join(md))
    outside_md, outside = outside_table(rows)
    (OUT / "signal_outside_background.md").write_text(outside_md)
    (OUT / "cpca_criteria.json").write_text(json.dumps({"thresholds": THRESH, "criteria": all_crit,
                                                        "paired_c_vs_e2": all_paired,
                                                        "signal_outside_background": outside}, indent=2))
    for level in LEVELS:
        print(level, {run: c["all_four"] for run, c in all_crit[level].items()},
              {m: p["verdict"] for m, p in all_paired[level].items()})


ANGLE_TOL = 1e-6

if __name__ == "__main__":
    main()
