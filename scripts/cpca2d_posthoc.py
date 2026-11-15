"""Post hoc, descriptive: score the 2-D cPCA projections the MNIST-on-grass T3 runs saved, as an embedding (cPCA-2D).

cPCA-2D was not part of the 0/1 or 0-9 pre-registrations; it is scored here after their results were known and carries
no verdict. For each T3 run (results/mnist_grass/runs and results/mnist_grass_0to9/runs, seeds 0-5, reruns skipped),
the saved projection at the label-free alpha that T3 used (its run JSON) is the embedding. Before scoring, the
cPCA fit is recomputed from the dataset (SHA-256 checked against data/MANIFEST.md): the label-free alpha and the
projection must equal the saved ones bit for bit, or the script stops.

Metrics as in the runs themselves (eval.metrics): digit kNN accuracy (k = 15, stratified 5-fold), recall vs clean
digits and vs grass (k = 15), ARI (k-means with k = the number of digits, n_init = 10, random_state = the dataset seed),
and the fragmentation check. Run on the cluster (THREADS=1), where the projections and data/processed/ live.

    python scripts/cpca2d_posthoc.py
Outputs: results/cpca2d_posthoc/runs.csv, posthoc.json, tables.md
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import cpca  # noqa: E402
from common.experiment import FRAGMENT_FRAC, FRAGMENT_SMALL  # noqa: E402
from common.repro import environment_info, setup_determinism  # noqa: E402
from eval.metrics import component_stats, kmeans_ari, knn_accuracy, knn_recall  # noqa: E402

OUT = ROOT / "results" / "cpca2d_posthoc"
K = 15
STAGES = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
FIELDS = ["data", "stage", "seed", "alpha", "digit_acc", "recall_digits", "recall_grass", "ari", "n_components",
          "frac_in_small_components", "flag_fragment", "t3_digit_acc"]


def loaders():
    from scripts.bsgraph_phase1 import load as load_01
    from scripts.mnist_grass_0to9_run import load as load_09
    return {"0/1": (ROOT / "results" / "mnist_grass" / "runs", load_01, 2),
            "0-9": (ROOT / "results" / "mnist_grass_0to9" / "runs", load_09, 10)}


def score(data, stage, seed, runs_dir, load, n_classes):
    t3 = json.loads((runs_dir / f"{stage}_seed{seed}_T3.json").read_text())
    with np.load(runs_dir / f"{stage}_seed{seed}_T3_cpca2d.npz") as f:
        E = f[f"{t3['alpha']:.6g}"]
    X_T, X_B, t = load(seed)
    fit = cpca.fit(X_T, X_B)
    alpha = cpca.select_alpha(fit, seed)[0]
    if alpha != t3["alpha"] or not np.array_equal(cpca.project(fit, alpha, 2), E):
        print(f"STOP (cPCA-2D post hoc): {data} {stage} seed {seed}: recomputed alpha or projection differs",
              flush=True)
        sys.exit(2)
    comp = component_stats(E, k=K, small=FRAGMENT_SMALL)
    return {"data": data, "stage": stage, "seed": seed, "alpha": alpha,
            "digit_acc": knn_accuracy(E, t["y"], k=K), "recall_digits": knn_recall(E, t["digits"], k=K),
            "recall_grass": knn_recall(E, t["grass_T"], k=K), "ari": kmeans_ari(E, t["y"], n_classes, seed), **comp,
            "flag_fragment": comp["frac_in_small_components"] > FRAGMENT_FRAC, "t3_digit_acc": t3["digit_acc"]}


def fmt(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]"


def tables(rows):
    out = ["# cPCA-2D on MNIST-on-grass, post hoc (descriptive)", "",
           "The 2-D cPCA projection at T3's label-free alpha, used as the embedding. Mean [min, max] over seeds; "
           "every projection and alpha was recomputed from the data and matched the saved one bit for bit.", ""]
    for stage, seeds in STAGES.items():
        out += [f"## {stage.capitalize()} seeds {', '.join(map(str, seeds))}", "",
                "| data | digit kNN accuracy | ARI | recall vs grass | recall vs clean digits | T3 digit kNN accuracy "
                "(same seeds) | fragmented |", "|---|---|---|---|---|---|---|"]
        for data in ("0/1", "0-9"):
            rr = [r for r in rows if r["data"] == data and r["stage"] == stage]
            out.append(f"| {data} | " + " | ".join(fmt([r[m] for r in rr]) for m in (
                "digit_acc", "ari", "recall_grass", "recall_digits", "t3_digit_acc"))
                + f" | {sum(r['flag_fragment'] for r in rr)} of {len(rr)} |")
        out.append("")
    out += ["## Per seed", "", "| data | stage | seed | α | digit kNN accuracy | ARI | recall vs grass |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {r['data']} | {r['stage']} | {r['seed']} | {r['alpha']:.3g} | {r['digit_acc']:.3f} | {r['ari']:.3f} "
            f"| {r['recall_grass']:.3f} |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    setup_determinism(0)
    rows = []
    for data, (runs_dir, load, n_classes) in loaders().items():
        for stage, seeds in STAGES.items():
            for seed in seeds:
                rows.append(score(data, stage, seed, runs_dir, load, n_classes))
                print(json.dumps(rows[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "posthoc.json").write_text(json.dumps(
        {"label": "descriptive, computed post hoc (cPCA-2D was not pre-registered for the MNIST-on-grass runs)",
         "embedding": "2-D cPCA projection at T3's label-free alpha (saved *_T3_cpca2d.npz)",
         "check": "label-free alpha and projection recomputed from the dataset, bit-identical on every seed",
         "kmeans": {"n_clusters": "number of digits", "n_init": 10, "random_state": "dataset seed"},
         "rows": rows, "environment": environment_info()}, indent=2))
    (OUT / "tables.md").write_text(tables(rows))
    print(tables(rows))


if __name__ == "__main__":
    main()
