"""MNIST-on-grass 0-9: descriptive pre-checks and graph quality on all seeds (no embedding is scored; nothing here
selects or gates anything).

  --seed S --k-loc K    residual graphs for every r in {0, 1, 2, 5, 10} at this k_loc -> prechecks/seed{S}_k{K}.json
  --seed S --reference  graphs on raw X_T, PCA-50 of X_T and cPCA-10 (label-free alpha); digit kNN accuracy of the clean
                        digits; the xi_B checks (scripts/mnist_grass_prechecks.xi_checks) -> prechecks/seed{S}_ref.json
  --make-jobs           cluster/jobs/mnist_grass_0to9_prechecks.txt (6 seeds x (3 k_loc + reference) = 24 processes)
  --summarize           prechecks.json and the generated prechecks.md, seeds 0-2 and 3-5 reported separately

Graph metrics (scripts/bsgraph_phase1.graph_metrics, in the space the graph is built on): digit kNN accuracy (k = 15,
stratified 5-fold; chance 0.10), edge purity, kNN recall at 15 vs the grass patches (lower is better) and vs the clean
digits. The residual default (k_loc = 30, r = 2) is fixed in advance and marked in the tables.
"""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402
from scripts.bsgraph_phase1 import K_LOCS, RS, fmt, graph_metrics  # noqa: E402
from scripts.mnist_grass_0to9_run import NAME, PREREG, SEEDS as STAGE_SEEDS, load  # noqa: E402

OUT = ROOT / "results" / NAME
PRE = OUT / "prechecks"
SEEDS = range(6)
DEFAULT = "k30_r2"
METRICS = [("graph_digit_acc", "graph digit accuracy"), ("edge_purity", "edge purity"),
           ("graph_grass_recall", "grass recall (lower is better)"), ("graph_digit_recall", "clean-digit recall")]


def write(name, rec):
    PRE.mkdir(parents=True, exist_ok=True)
    (PRE / f"{name}.json").write_text(json.dumps({**rec, "environment": environment_info()}, indent=2))
    print(json.dumps(rec), flush=True)


def residual_part(seed, k_loc):
    from method.residual_graph import local_background, residuals
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    local = local_background(X_T, X_B, k_loc)
    rows = [{"variant": f"k{k_loc}_r{r}", "k_loc": k_loc, "r": r, **graph_metrics(residuals(X_T, local, r), t)}
            for r in RS]
    write(f"seed{seed}_k{k_loc}", {"seed": seed, "rows": rows})


def reference_part(seed):
    from sklearn.decomposition import PCA

    from baselines import cpca
    from eval.metrics import knn_accuracy
    from scripts.mnist_grass_prechecks import xi_checks
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    f = cpca.fit(X_T, X_B)
    alpha = cpca.select_alpha(f, seed)[0]
    spaces = {"raw": X_T, "pca50": PCA(50, svd_solver="full").fit_transform(X_T), "cpca10": cpca.project(f, alpha, 10)}
    rows = [{"variant": name, **graph_metrics(Z, t)} for name, Z in spaces.items()]
    write(f"seed{seed}_ref", {"seed": seed, "rows": rows, "cpca_alpha": alpha,
                              "digit_acc_clean": knn_accuracy(t["digits"], t["y"]), "xi": xi_checks(X_T, X_B, t, seed)})


def make_jobs():
    lines = [f"$PY scripts/mnist_grass_0to9_prechecks.py --seed {s} --k-loc {k}" for k in sorted(K_LOCS, reverse=True)
             for s in SEEDS]
    lines += [f"$PY scripts/mnist_grass_0to9_prechecks.py --seed {s} --reference" for s in SEEDS]
    path = ROOT / "cluster" / "jobs" / f"{NAME}_prechecks.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def table(by, names, seeds):
    out = ["| graph | " + " | ".join(n for _, n in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name in names:
        shown = f"**{name} (default)**" if name == DEFAULT else name
        out.append(f"| {shown} | " + " | ".join(fmt([by[s][name][m] for s in seeds]) for m, _ in METRICS) + " |")
    return "\n".join(out)


def xi_table(refs, seeds):
    out = ["| seed | clean-digit accuracy | same digit, ξ_B | chance | excess | grass dist, ξ_B pairs "
           "| grass dist, random pairs | grass dist, grass-kNN | list length min / median / p90 / max | empty rows "
           "| tie-split rows | capped rows |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        x = refs[s]["xi"]
        out.append(f"| {s} | {refs[s]['digit_acc_clean']:.3f} | {x['xi_same_digit']:.3f} | "
                   f"{x['same_digit_chance']:.3f} | {x['xi_same_digit_excess']:+.3f} | {x['grass_dist_xi_pairs']:.2f} "
                   f"| {x['grass_dist_random_pairs']:.2f} | {x['grass_dist_grass_knn']:.2f} | {x['len_min']} / "
                   f"{x['len_median']:.0f} / {x['len_p90']:.0f} / {x['len_max']} | {x['n_empty_rows']} | "
                   f"{x['frac_tie_split_rows']:.3f} | {x['frac_capped_rows']:.3f} |")
    return "\n".join(out)


def summarize():
    by, refs = {s: {} for s in SEEDS}, {}
    for p in sorted(PRE.glob("seed*.json")):
        rec = json.loads(p.read_text())
        for row in rec["rows"]:
            by[rec["seed"]][row["variant"]] = row
        if "xi" in rec:
            refs[rec["seed"]] = rec
    variants = [f"k{k}_r{r}" for k in K_LOCS for r in RS]
    md = ["# MNIST-on-grass 0–9 pre-checks and graph quality (descriptive)", "",
          "Graph metrics in the space each graph is built on; mean [min, max] over seeds. Chance for digit accuracy "
          f"is 0.10. The residual default ({DEFAULT}: k_loc = 30, r = 2) was fixed in advance; the grid is a "
          "robustness check and selects nothing."]
    for stage, seeds in STAGE_SEEDS.items():
        md += ["", f"## Seeds {seeds[0]}–{seeds[-1]} ({stage})", "",
               "### Reference graphs and the residual default", "",
               table(by, ["raw", "pca50", "cpca10", DEFAULT], seeds), "",
               "### Residual grid", "", table(by, variants, seeds), "",
               "### ξ_B on raw pixels (k_B = 15, L = 30)", "", xi_table(refs, seeds)]
    md += ["", "cpca10: the top 10 cPCA components at the label-free α (per seed: "
           + ", ".join(f"{s}: {refs[s]['cpca_alpha']:.4g}" for s in SEEDS) + ")."]
    (OUT / "prechecks.md").write_text("\n".join(md) + "\n")
    (OUT / "prechecks.json").write_text(json.dumps(
        {"graphs": {str(s): by[s] for s in SEEDS}, "reference": {str(s): refs[s] for s in SEEDS}}, indent=2))
    print("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k-loc", type=int)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--make-jobs", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs()
    if args.summarize:
        return summarize()
    assert_committed(PREREG)
    reference_part(args.seed) if args.reference else residual_part(args.seed, args.k_loc)


if __name__ == "__main__":
    main()
