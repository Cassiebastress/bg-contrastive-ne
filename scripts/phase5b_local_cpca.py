"""Phase 5b extension: the local (cluster-then-cPCA) baseline on Design L (predictions P6-P9 in
results/phase5b/predictions.md, committed before this script's first result).

  --design cyclic|between   confirmation seeds 3-5 (the datasets of the main runs, checked by hash): local cPCA,
                            label-free (k-means on X_B, K by silhouette) and oracle (true clusters), at k in
                            {3, 5, 10}; each embedded once with (a)'s two-stage Neg-t-SNE and scored on WC.
  --recheck                 rerun seed 3 and compare rows and embeddings bit for bit

Stop and report: a cluster with fewer than 50 target or background points; a kNN edge crossing clusters in the joint
representation; a non-finite embedding; the recheck not bit-identical; a dataset hash differing from the main runs.

Run with the cne-bg env:  caffeinate -s python scripts/phase5b_local_cpca.py --design cyclic
Outputs go to results/phase5b/ (embeddings are gitignored).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import local_cpca as lc
from common.benchmark_l import embed, score
from common.repro import assert_committed, environment_info, setup_determinism
from data.synthetic_l import VARIANTS, make_l

OUT = ROOT / "results" / "phase5b"
PREDICTIONS = OUT / "predictions.md"
SEEDS = (3, 4, 5)
K_LOCAL = (3, 5, 10)
VERSIONS = ("labelfree", "oracle")
METRICS = ["WC", "acc_s", "acc_b", "G1", "G1_k200", "G2", "G3_shares_signal"] + [f"WC_c{i}" for i in range(6)]
FIELDS = ["variant", "seed", "run", "group", "version", "k", "K", "s", "cross_cluster_knn_frac", *METRICS,
          "runtime_s"]


def stop(msg):
    print(f"STOP (local cPCA extension): {msg}", flush=True)
    sys.exit(2)


def dataset(variant, seed, cal, main):
    """The main runs' dataset for this seed, verified by its X_T and X_B hashes."""
    setup_determinism(seed)
    ds = make_l(variant, seed, cal["rho"][variant]["rho"])
    ref = main["checks"][str(seed)]
    if (hashlib.sha256(ds.X_T.tobytes()).hexdigest() != ref["sha256_X_T"]
            or hashlib.sha256(ds.X_B.tobytes()).hexdigest() != ref["sha256_X_B"]):
        stop(f"L-{variant} seed {seed}: dataset differs from the main runs")
    return ds


def run_seed(variant, seed, cal, main, save=True):
    ds = dataset(variant, seed, cal, main)
    rows, embds, info = [], {}, {}
    for version in VERSIONS:
        lab_T, lab_B, cl = lc.clusters_for(ds, version, seed)
        small = lc.check_sizes(lab_T, lab_B)
        if small:
            stop(f"L-{variant} seed {seed} {version}: clusters too small for cPCA {small}")
        per = lc.local_alphas(ds.X_T, ds.X_B, lab_T, lab_B, seed)
        info[version] = {**cl, "alphas": {str(j): p["alpha"] for j, p in per.items()},
                         "representatives": {str(j): p["representatives"] for j, p in per.items()},
                         "cluster_sizes_T": {str(j): int((lab_T == j).sum()) for j in per}, "representations": []}
        for k in K_LOCAL:
            Z, rep = lc.representation(ds.X_T, lab_T, per, k)
            info[version]["representations"].append(rep)
            if rep["cross_cluster_knn_frac"] > 0:
                stop(f"L-{variant} seed {seed} {version} k={k}: {rep['cross_cluster_knn_frac']:.4f} of kNN edges "
                     "cross clusters")
            t0 = time.perf_counter()
            E = embed(Z, seed)
            runtime = time.perf_counter() - t0
            if not np.isfinite(E).all():
                stop(f"L-{variant} seed {seed} {version} k={k}: non-finite embedding")
            run = f"local_{version}_k{k}"
            rows.append({"variant": variant, "seed": seed, "run": run, "group": "local", "version": version, "k": k,
                         "K": cl["K"], "s": rep["s"], "cross_cluster_knn_frac": rep["cross_cluster_knn_frac"],
                         **score(E, ds.truth), "runtime_s": runtime})
            embds[run] = E
            print(f"L-{variant} seed={seed} {run:20s} K {cl['K']} WC {rows[-1]['WC']:.3f} ({runtime:.0f}s)",
                  flush=True)
    if save:
        for run, e in embds.items():
            np.save(OUT / "embeddings" / f"{variant}_local_seed{seed}_{run}.npy", e)
    return rows, embds, info


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def recheck(variant, cal, main):
    with open(OUT / f"runs_{variant}_local_cpca.csv") as f:
        saved = [r for r in csv.DictReader(f) if int(r["seed"]) == SEEDS[0]]
    rows, embds, _ = run_seed(variant, SEEDS[0], cal, main, save=False)
    fields = [f for f in FIELDS if f != "runtime_s"]
    rows_ok = len(rows) == len(saved) and all(str(new[f]) == old[f] for new, old in zip(rows, saved) for f in fields)
    embd_ok = {run: bool(np.array_equal(e, np.load(OUT / "embeddings" / f"{variant}_local_seed{SEEDS[0]}_{run}.npy")))
               for run, e in embds.items()}
    result = {"environment": environment_info(), "variant": variant, "seed": SEEDS[0], "rows_identical": rows_ok,
              "embeddings_identical": embd_ok, "all_identical": rows_ok and all(embd_ok.values())}
    (OUT / f"recheck_{variant}_local_cpca.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    if not result["all_identical"]:
        stop(f"L-{variant}: rerun of seed {SEEDS[0]} is not bit-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=VARIANTS, required=True)
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    assert_committed(PREDICTIONS)
    cal = json.loads((OUT / "calibration.json").read_text())
    main_summary = json.loads((OUT / f"summary_{args.design}_confirmation.json").read_text())
    (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
    if args.recheck:
        return recheck(args.design, cal, main_summary)
    rows, info = [], {}
    for seed in SEEDS:
        seed_rows, _, info[seed] = run_seed(args.design, seed, cal, main_summary)
        rows += seed_rows
        write_csv(OUT / f"runs_{args.design}_local_cpca.csv", rows)
    summary = {"environment": environment_info(), "variant": args.design, "seeds": list(SEEDS),
               "versions": list(VERSIONS), "k": list(K_LOCAL), "clusters": info}
    (OUT / f"summary_{args.design}_local_cpca.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
