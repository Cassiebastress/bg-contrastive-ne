"""Tuned T3 tables and heatmap, generated from results/mnist_grass_0to9_t3grid/ and the stored runs it reuses (no
number typed by hand). Uses only the recorded results (CSV/JSON), not the embeddings.

Tables: T3* and PCA* with their selection means, the two pre-registered comparisons (digit kNN accuracy, secondary
metrics alongside), the descriptive comparisons, every run on the confirmation seeds, the oracle T3 per seed against
(r_c), the 10 best settings on the selection seeds and the checks. Heatmap: mean digit kNN accuracy over (k, alpha),
selection and confirmation seeds, one-hue blue ramp (dataviz reference palette), T3* and the label-free setting marked.

    python scripts/t3grid_report.py
Outputs: results/mnist_grass_0to9_t3grid/tables.md, comparisons.json, heatmap_digit_acc.png
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
from scripts.mnist_grass_0to9_report import BLUE_RAMP  # noqa: E402
from scripts.t3grid_run import ALPHAS, KS, N_CPCA, OLD, OUT, SEEDS, SINGLE, best  # noqa: E402

METRICS = [("digit_acc", "digit kNN accuracy"), ("recall_digits", "recall vs clean digits"),
           ("recall_grass", "recall vs grass (lower is better)"), ("ari", "ARI (k-means, k = 10)")]
INK, MUTED = "#1f1f1e", "#6b6a66"


def grid(stage):
    with open(OUT / f"runs_{stage}.csv") as f:
        return [{**r, "seed": int(r["seed"]), "alpha_index": int(r["alpha_index"]), "k": int(r["k"]),
                 **{m: float(r[m]) for m, _ in METRICS}} for r in csv.DictReader(f)]


def setting(s):
    return f"k = {s['k']}, α = {s['alpha']:.4g}"


def confirmation_rows(sel, conf):
    """Per confirmation seed, every run the comparisons use, as rows {seed, run, metrics}."""
    with open(OLD / "runs_confirmation.csv") as f:
        old = [r for r in csv.DictReader(f) if r["rep"] in ("", "0")]
    m_b = json.loads((OLD / "selection.json").read_text())["r_c"]["best_m_B"]
    names = {f"r_c_m{m_b}": "r_c", "r_a": "r_a_two_stage", "T3": "T3", "T2": "T2"}
    out = [{"seed": int(r["seed"]), "run": names[r["run"]], **{m: float(r[m]) for m, _ in METRICS}}
           for r in old if r["run"] in names]
    for s in SEEDS["confirmation"]:
        rec = json.loads((SINGLE / f"0to9_seed{s}_single.json").read_text())
        out.append({"seed": s, "run": "r_a_single", **{m: rec[m] for m, _ in METRICS}})
    for run, key in (("T3*", "T3_star"), ("PCA*", "PCA_star")):
        out += [{**r, "run": run} for r in conf if (r["k"], r["alpha_index"]) ==
                (sel[key]["k"], sel[key]["alpha_index"])]
    for s in SEEDS["confirmation"]:
        o = best(conf, lambda r: r["seed"] == s)
        r = next(r for r in conf if r["seed"] == s and (r["k"], r["alpha_index"]) == (o["k"], o["alpha_index"]))
        out.append({**r, "run": "oracle"})
    return out


LABELS = {"r_c": "(r_c) BC-NE+BN, m_B = 5", "r_a_single": "(r_a) BC-NE, single-stage",
          "r_a_two_stage": "(r_a) BC-NE, two-stage", "T3*": "T3\\* tuned cPCA → t-SNE",
          "T3": "T3 label-free cPCA-10 → t-SNE", "PCA*": "PCA\\* tuned PCA → t-SNE", "T2": "T2 PCA-50 → t-SNE",
          "oracle": "oracle T3 (per seed, own labels)"}
SHORT = {"r_c": "(r_c)", "r_a_single": "single-stage (r_a)", "r_a_two_stage": "two-stage (r_a)", "T3*": "T3\\*",
         "T3": "label-free T3", "PCA*": "PCA\\*", "T2": "T2"}


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def run_table(rows, runs):
    out = ["| run | " + " | ".join(n for _, n in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for run in runs:
        out.append(f"| {LABELS[run]} | " + " | ".join(fmt([r[m] for r in rows if r["run"] == run]) for m, _ in METRICS)
                   + " |")
    seeds = SEEDS["confirmation"]
    out += ["", "Per seed, digit kNN accuracy:", "", "| run | " + " | ".join(f"seed {s}" for s in seeds) + " |",
            "|---" * (len(seeds) + 1) + "|"]
    for run in runs:
        v = {r["seed"]: r["digit_acc"] for r in rows if r["run"] == run}
        out.append(f"| {LABELS[run]} | " + " | ".join(f"{v[s]:.3f}" for s in seeds) + " |")
    return "\n".join(out)


def compare(rows, pairs, verdicts):
    out = ["| comparison | per-seed difference in digit kNN accuracy | mean | verdict | recall vs digits, mean "
           "difference | recall vs grass, mean difference | ARI, mean difference |", "|---|---|---|---|---|---|---|"]
    res = []
    for x, y in pairs:
        xy, yx = paired(rows, x, y, "digit_acc"), paired(rows, y, x, "digit_acc")
        verdict = (f"{SHORT[x]} better, consistent across 3 paired seeds" if xy["consistent"] else
                   f"{SHORT[y]} better, consistent across 3 paired seeds" if yx["consistent"] else "not consistent")
        sec = {m: paired(rows, x, y, m)["mean_diff"] for m in ("recall_digits", "recall_grass", "ari")}
        res.append({"x": x, "y": y, "per_seed": xy["per_seed"], "mean": xy["mean_diff"],
                    "x_better_consistent": xy["consistent"], "y_better_consistent": yx["consistent"],
                    "secondary_mean_diff": sec, "preregistered": verdicts})
        out.append(f"| {LABELS[x]} vs {LABELS[y]} | {', '.join(f'{d + 0.0:+.3f}' for d in xy['per_seed'])} "
                   f"| {xy['mean_diff'] + 0.0:+.3f} | {verdict if verdicts else '(descriptive) ' + verdict} "
                   f"| {sec['recall_digits'] + 0.0:+.3f} | {sec['recall_grass'] + 0.0:+.3f} "
                   f"| {sec['ari'] + 0.0:+.3f} |")
    return "\n".join(out), res


def means(rows):
    m = {}
    for r in rows:
        m.setdefault((r["k"], r["alpha_index"]), []).append(r["digit_acc"])
    return {s: float(np.mean(v)) for s, v in m.items()}


def selection_table(sel, sel_means, conf_means):
    lf = int(np.flatnonzero(ALPHAS == json.loads((OLD / "runs" / "selection_seed0_T3.json").read_text())["alpha"])[0])
    out = ["| setting | role | mean digit kNN accuracy, seeds 0–2 | seeds 3–5 |", "|---|---|---|---|"]
    for key, role in (("T3_star", "T3\\*"), ("PCA_star", "PCA\\*")):
        s = sel[key]
        out.append(f"| {setting(s)} | {role} | {sel_means[(s['k'], s['alpha_index'])]:.3f} "
                   f"| {conf_means[(s['k'], s['alpha_index'])]:.3f} |")
    out.append(f"| k = {N_CPCA}, α = {ALPHAS[lf]:.4g} | label-free T3 setting (seeds 0–2, 4, 5) "
               f"| {sel_means[(N_CPCA, lf)]:.3f} | {conf_means[(N_CPCA, lf)]:.3f} |")
    top = sorted(sel_means, key=lambda s: (-sel_means[s], s[0], ALPHAS[s[1]]))[:10]
    conf_rank = {s: i + 1 for i, s in enumerate(sorted(conf_means, key=lambda s: -conf_means[s]))}
    out += ["", "The 10 best settings on the selection seeds:", "",
            "| rank | setting | mean, seeds 0–2 | mean, seeds 3–5 | rank on seeds 3–5 |", "|---|---|---|---|---|"]
    out += [f"| {i + 1} | k = {k}, α = {ALPHAS[a]:.4g} | {sel_means[(k, a)]:.3f} | {conf_means[(k, a)]:.3f} "
            f"| {conf_rank[(k, a)]} |" for i, (k, a) in enumerate(top)]
    return "\n".join(out), lf


def by_k_table(sel_means, conf_means):
    """Per k: the best alpha on the selection seeds, and alpha = 1 and alpha = 0, with their seed-set means."""
    one, zero = int(np.flatnonzero(ALPHAS == 1.0)[0]), 0
    out = ["| k | best α on seeds 0–2 | its mean, seeds 0–2 | its mean, seeds 3–5 | α = 1, seeds 3–5 | α = 0 (PCA), "
           "seeds 3–5 |", "|---|---|---|---|---|---|"]
    for k in KS:
        a = max(range(len(ALPHAS)), key=lambda i: (sel_means[(k, i)], -ALPHAS[i]))
        out.append(f"| {k} | {ALPHAS[a]:.4g} | {sel_means[(k, a)]:.3f} | {conf_means[(k, a)]:.3f} "
                   f"| {conf_means[(k, one)]:.3f} | {conf_means[(k, zero)]:.3f} |")
    return "\n".join(out)


def oracle_table(rows):
    out = ["| seed | oracle setting | oracle T3 | T3\\* | (r_c) | (r_c) − oracle |", "|---|---|---|---|---|---|"]
    for s in SEEDS["confirmation"]:
        o = next(r for r in rows if r["seed"] == s and r["run"] == "oracle")
        star = next(r for r in rows if r["seed"] == s and r["run"] == "T3*")["digit_acc"]
        rc = next(r for r in rows if r["seed"] == s and r["run"] == "r_c")["digit_acc"]
        out.append(f"| {s} | k = {o['k']}, α = {float(o['alpha']):.4g} | {o['digit_acc']:.3f} | {star:.3f} | {rc:.3f} "
                   f"| {rc - o['digit_acc'] + 0.0:+.3f} |")
    return "\n".join(out)


def rerun_cell(rr):
    if "rerun_identical" not in rr[0]:
        return "—"
    return "yes" if all(r["rerun_identical"] for r in rr) else "NO"


def checks_table():
    out = ["| stage | seed | label-free α | grid point reproduces stored T3 (SHA-256) |", "|---|---|---|---|"]
    for stage in SEEDS:
        for c in json.loads((OUT / f"sha_check_{stage}.json").read_text())["checks"]:
            out.append(f"| {stage} | {c['seed']} | {c['label_free_alpha']:.4g} | {'yes' if c['identical'] else 'NO'} |")
    reuse = json.loads((OUT / "reuse.json").read_text())
    out += ["", "Reused runs (code unchanged since the run's commit; stored `.npy` matches its SHA-256; for (r_c) and "
            "two-stage (r_a), rerun on the cluster and compared bit for bit):", "",
            "| run | seeds | commit | Slurm job | code unchanged | .npy SHA-256 matches | rerun bit-identical |",
            "|---|---|---|---|---|---|---|"]
    groups = {}
    for r in reuse["runs"]:
        groups.setdefault((r["run"], r["commit"], r["slurm_job"]), []).append(r)
    for (run, commit, job), rr in groups.items():
        out.append(f"| {run} | {', '.join(str(r['seed']) for r in rr)} | `{commit}` | {job} "
                   f"| {'yes' if all(r['code_unchanged'] for r in rr) else 'NO'} "
                   f"| {'yes' if all(r['npy_sha_matches'] for r in rr) else 'NO'} "
                   f"| {rerun_cell(rr)} |")
    return "\n".join(out)


def heatmap(sel, sel_means, conf_means, lf):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    mats = [np.array([[m[(k, a)] for a in range(len(ALPHAS))] for k in KS]) for m in (sel_means, conf_means)]
    lo, hi = min(m.min() for m in mats), max(m.max() for m in mats)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), constrained_layout=True, sharex=True)
    ticks = [0] + [int(np.argmin(np.abs(np.log(ALPHAS[1:]) - np.log(v)))) + 1 for v in (1, 10, 100, 1000)]
    star = (KS.index(sel["T3_star"]["k"]), sel["T3_star"]["alpha_index"])
    for ax, mat, title in zip(axes, mats, ("Selection seeds 0–2 (T3* chosen here)", "Confirmation seeds 3–5")):
        im = ax.imshow(mat, cmap=BLUE_RAMP, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
        ax.scatter([star[1]], [star[0]], marker="s", s=90, facecolors="none", edgecolors=INK, linewidths=1.6,
                   label="T3* (selected)")
        ax.scatter([lf], [KS.index(N_CPCA)], marker="o", s=70, facecolors="none", edgecolors="#eb6834",
                   linewidths=1.6, label=f"label-free T3 (k = {N_CPCA}, α = {ALPHAS[lf]:.4g})")
        ax.set_yticks(range(len(KS)), [str(k) for k in KS])
        ax.set_ylabel("components k")
        ax.set_title(title, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
    axes[1].set_xticks(ticks, [f"{ALPHAS[i]:.4g}" for i in ticks])
    axes[1].set_xlabel("α, one column per grid value (0 = plain PCA, then 0.1 to 1000 log-spaced, plus 1)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower left", ncol=2, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean digit kNN accuracy")
    fig.savefig(OUT / "heatmap_digit_acc.png", dpi=200)
    plt.close(fig)


def main():
    sel = json.loads((OUT / "selection.json").read_text())
    conf = grid("confirmation")
    rows = confirmation_rows(sel, conf)
    sel_means, conf_means = means(grid("selection")), means(conf)
    sel_tab, lf = selection_table(sel, sel_means, conf_means)
    pre_tab, pre = compare(rows, [("r_c", "T3*"), ("r_a_single", "T3*")], verdicts=True)
    desc_tab, desc = compare(rows, [("r_a_two_stage", "T3*"), ("T3*", "T3"), ("PCA*", "T2")], verdicts=False)
    runs = ["r_c", "r_a_single", "r_a_two_stage", "T3*", "oracle", "T3", "PCA*", "T2"]
    text = f"""# Tuned T3 on MNIST-on-grass 0–9: tables (generated)

Generated by `scripts/t3grid_report.py` from `results/mnist_grass_0to9_t3grid/` and the stored runs it reuses.

## T3\\* and PCA\\* (chosen on seeds 0–2)

Rule: {sel['rule']}.
Grid: k ∈ {{{', '.join(map(str, KS))}}} × {len(ALPHAS)} values of α, {len(KS) * len(ALPHAS)} settings.

{sel_tab}

Mean digit kNN accuracy by number of components:

{by_k_table(sel_means, conf_means)}

## Pre-registered comparisons (confirmation seeds 3–5)

{pre_tab}

## Descriptive comparisons (no verdicts)

{desc_tab}

## All runs, confirmation seeds 3–5 (mean [min, max])

{run_table(rows, runs)}

## Oracle T3 per seed (upper bound, chosen with each seed's own labels)

{oracle_table(rows)}

## Checks

{checks_table()}
"""
    (OUT / "tables.md").write_text(text)
    (OUT / "comparisons.json").write_text(json.dumps({"preregistered": pre, "descriptive": desc}, indent=2))
    heatmap(sel, sel_means, conf_means, lf)
    print(text)


if __name__ == "__main__":
    main()
