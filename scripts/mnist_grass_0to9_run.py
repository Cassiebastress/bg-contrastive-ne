"""MNIST-on-grass with digits 0-9: one run on one dataset seed per process (after the committed pre-registration,
results/mnist_grass_0to9/preregistration.md).

  --stage selection|confirmation --seed S --run R [--rep N]
        R is a, c_m<m>, T1, T2, T3 (built by scripts/mnist_grass_run.embed) or r_a, r_c_m<m> (scripts/bsgraph_run.embed:
        the residual graph at k_loc = 30, r = 2). Writes results/mnist_grass_0to9/runs/{stage}_seed{S}_{R}[_rep{N}].json
        and the embedding .npy (gitignored).
  --make-jobs STAGE   cluster/jobs/mnist_grass_0to9_{STAGE}.txt, longest runs first
  --merge STAGE       runs_{STAGE}.csv (+ collapse flags: (c) against (a), (r_c) against (r_a), same seed); for
                      selection also selection.json (m_B for (c) and for (r_c): highest mean digit accuracy over seeds
                      0-2, ties -> smaller m_B) and the T1 seed-0 rerun check; for confirmation the (r_c) seed-3 rerun
                      check. Reruns must be bit-identical (recheck_{STAGE}.json).

Metrics (evaluation only): digit kNN accuracy (primary; k = 15, stratified 5-fold, 10 classes), kNN recall at 15
against the clean digits and against the grass patches, ARI of k-means (k = 10, n_init = 10, random_state = seed) on
the embedding against the digits (secondary), per-digit accuracy and the confusion matrix (out-of-fold predictions of
the same classifier), and the Phase 4 review flags.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402

NAME = "mnist_grass_0to9"
OUT = ROOT / "results" / NAME
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration.md"
SELECTION = OUT / "selection.json"
SEEDS = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
M_GRID = (1, 2, 5, 10, 20, 40)
CONTROLS = ("a", "r_a", "T1", "T2", "T3")
RECHECK = {"selection": (0, "T1"), "confirmation": (3, "r_c")}  # (seed, run group) rerun and required bit-identical
N_CLASSES = 10
K = 15
FIELDS = ["stage", "seed", "run", "rep", "digit_acc", "recall_digits", "recall_grass", "ari", "embd_std",
          "std_ratio_vs_ref", "flag_collapse", "n_components", "frac_in_small_components", "flag_fragment",
          "nonfinite_batches", "max_std_growth", "alpha", "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (MNIST-on-grass 0-9): {msg}", flush=True)
    sys.exit(2)


def load(seed):
    """The 0-9 dataset, after checking its SHA-256 against data/MANIFEST.md."""
    from data import mnist_grass as mg
    path = mg.processed_path(seed, ROOT, name=NAME)
    want = re.search(rf"`{path.name}` \| `([0-9a-f]{{64}})`", (ROOT / "data" / "MANIFEST.md").read_text())
    if want is None or hashlib.sha256(path.read_bytes()).hexdigest() != want.group(1):
        stop(f"{path.name} does not match its SHA-256 in data/MANIFEST.md")
    return mg.for_embedding(mg.load(path))


def embed(run, seed, X_T, X_B):
    if run == "r_a" or run.startswith("r_c_m"):
        from scripts.bsgraph_run import embed as residual_embed
        return residual_embed(run, seed, X_T, X_B)
    from scripts.mnist_grass_run import embed as raw_embed
    return raw_embed(run, seed, X_T, X_B)


def run_one(stage, seed, run, rep):
    from common.experiment import FRAGMENT_FRAC, FRAGMENT_SMALL, MAX_STD_GROWTH
    from eval.metrics import component_stats, kmeans_ari, knn_accuracy, knn_predictions, knn_recall
    assert_committed(PREREG)
    if stage == "confirmation":
        assert_committed(SELECTION)
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    t0 = time.perf_counter()
    E, extra = embed(run, seed, X_T, X_B)
    runtime = time.perf_counter() - t0
    proj = extra.pop("projections_2d", None)
    y = t["y"]
    pred = knn_predictions(E, y, k=K)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    comp = component_stats(E, k=K, small=FRAGMENT_SMALL)
    rec = {"stage": stage, "seed": seed, "run": run, "rep": rep, "digit_acc": knn_accuracy(E, y, k=K),
           "recall_digits": knn_recall(E, t["digits"], k=K), "recall_grass": knn_recall(E, t["grass_T"], k=K),
           "ari": kmeans_ari(E, y, N_CLASSES, seed),
           "per_digit_acc": (np.diag(confusion) / confusion.sum(axis=1)).tolist(), "confusion": confusion.tolist(),
           **comp, "flag_fragment": comp["frac_in_small_components"] > FRAGMENT_FRAC, "runtime_s": runtime,
           "sha256": hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest(), **extra,
           "environment": environment_info()}
    rec.setdefault("nonfinite_batches", int(not np.isfinite(E).all()))
    name = f"{stage}_seed{seed}_{run}" + (f"_rep{rep}" if rep else "")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    np.save(RUNS_DIR / f"{name}.npy", E)
    if proj is not None:
        np.savez(RUNS_DIR / f"{name}_cpca2d.npz", **proj)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("environment", "confusion")}, default=str), flush=True)
    if rec["nonfinite_batches"] or rec.get("max_std_growth", 0) > MAX_STD_GROWTH:
        stop(f"{name}: non-finite or exploding (std growth {rec.get('max_std_growth')})")


def runs_for(stage):
    """Longest first: the background-negative runs by decreasing m_B, then the references and t-SNE controls."""
    if stage == "selection":
        return [f"{g}_m{m}" for m in sorted(M_GRID, reverse=True) for g in ("c", "r_c")] + list(CONTROLS)
    sel = json.loads(SELECTION.read_text())
    return [f"c_m{sel['c']['best_m_B']}", f"r_c_m{sel['r_c']['best_m_B']}"] + list(CONTROLS)


def recheck_run(stage):
    seed, group = RECHECK[stage]
    return seed, next(r for r in runs_for(stage) if r == group or r.startswith(f"{group}_m"))


def make_jobs(stage):
    lines = [f"$PY scripts/mnist_grass_0to9_run.py --stage {stage} --seed {s} --run {r}"
             for r in runs_for(stage) for s in SEEDS[stage]]
    seed, run = recheck_run(stage)
    lines.append(f"$PY scripts/mnist_grass_0to9_run.py --stage {stage} --seed {seed} --run {run} --rep 2")
    path = ROOT / "cluster" / "jobs" / f"{NAME}_{stage}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def select(main, group):
    means = {m: float(np.mean([r["digit_acc"] for r in main if r["run"] == f"{group}_m{m}"])) for m in M_GRID}
    return {"means": {str(m): v for m, v in means.items()}, "best_m_B": max(M_GRID, key=lambda m: (means[m], -m))}


def merge(stage):
    from common.experiment import COLLAPSE_STD_RATIO
    recs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob(f"{stage}_seed*.json"))]
    main = [r for r in recs if not r["rep"]]
    for r in main:
        ref = "a" if r["run"].startswith("c_m") else "r_a" if r["run"].startswith("r_c_m") else None
        a = next((x for x in main if x["seed"] == r["seed"] and x["run"] == ref), None)
        if a is not None:
            r["std_ratio_vs_ref"] = r["embd_std"] / a["embd_std"]
            r["flag_collapse"] = r["std_ratio_vs_ref"] < COLLAPSE_STD_RATIO
    with open(OUT / f"runs_{stage}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: (r["seed"], r["run"], r["rep"] or 0)))
    if stage == "selection":
        sel = {"metric": "digit kNN accuracy", "c": select(main, "c"), "r_c": select(main, "r_c"),
               "rule": "highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B",
               "environment": environment_info()}
        SELECTION.write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: v for k, v in sel.items() if k != "environment"}, indent=2))
    seed, run = recheck_run(stage)
    first = np.load(RUNS_DIR / f"{stage}_seed{seed}_{run}.npy")
    again = np.load(RUNS_DIR / f"{stage}_seed{seed}_{run}_rep2.npy")
    result = {"stage": stage, "run": run, "seed": seed, "identical": bool(np.array_equal(first, again)),
              "environment": environment_info()}
    (OUT / f"recheck_{stage}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}))
    print(f"merged {len(recs)} runs -> runs_{stage}.csv")
    if not result["identical"]:
        stop(f"rerun of {run} on seed {seed} is not bit-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(SEEDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--make-jobs", choices=list(SEEDS))
    parser.add_argument("--merge", choices=list(SEEDS))
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs(args.make_jobs)
    if args.merge:
        return merge(args.merge)
    run_one(args.stage, args.seed, args.run, args.rep)


if __name__ == "__main__":
    main()
