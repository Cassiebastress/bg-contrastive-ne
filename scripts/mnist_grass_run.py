"""MNIST-on-grass Phases F and G: one run on one dataset seed per process (after the committed pre-registration).

  --stage selection|confirmation --seed S --run R [--rep N]
        R is a (plain two-stage Neg-t-SNE), c_m<m> (ξ_B negatives, m_B = m), T1 (openTSNE on X_T), T2 (PCA-50 fit on
        X_T -> openTSNE) or T3 (cPCA at the label-free alpha, top 10 components -> openTSNE). (a)/(c): exact kNN graph
        (k = 15) and ξ_B (k_B = 15, L = 30) on raw 784-D pixels, PCA init, the two-stage schedule; each process
        computes its own (deterministic) Stage 1. openTSNE: library defaults, random_state = seed. Writes
        results/mnist_grass/runs/{stage}_seed{S}_{R}[_rep{N}].json and the embedding .npy (gitignored).
  --make-jobs STAGE   cluster/jobs/mnist_grass_{STAGE}.txt, one command per line (cluster/parallel.sbatch)
  --merge STAGE       runs_{STAGE}.csv (+ the collapse flag of (c) against (a) on the same seed); for selection also
                      selection.json (m_B with the highest mean digit accuracy, ties -> smaller m_B); for confirmation
                      also recheck.json ((c) on seed 3 rerun must be bit-identical)

Metrics (evaluation only): digit kNN accuracy (primary; k = 15, stratified 5-fold), kNN recall at 15 against the clean
digits and against the target grass patches (lower is better), and the Phase 4 review flags.
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

from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402

OUT = ROOT / "results" / "mnist_grass"
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration.md"
SELECTION = OUT / "selection.json"
SEEDS = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
M_GRID = (1, 2, 5, 10, 20, 40)
CONTROLS = ("a", "T1", "T2", "T3")
RECHECK_SEED = 3
K, K_B, L, N_PCA, N_CPCA = 15, 15, 30, 50, 10
FIELDS = ["stage", "seed", "run", "rep", "digit_acc", "recall_digits", "recall_grass", "embd_std", "std_ratio_vs_a",
          "flag_collapse", "n_components", "frac_in_small_components", "flag_fragment", "nonfinite_batches",
          "max_std_growth", "alpha", "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (MNIST-on-grass): {msg}", flush=True)
    sys.exit(2)


def load(seed):
    from data import mnist_grass as mg
    return mg.for_embedding(mg.load(mg.processed_path(seed, ROOT)))


def tsne(Z, seed):
    from openTSNE import TSNE
    return np.asarray(TSNE(random_state=seed).fit(np.ascontiguousarray(Z, dtype=np.float32)))


def embed(run, seed, X_T, X_B):
    """The embedding and run-specific fields (fit health for (a)/(c), the chosen alpha and projections for T3)."""
    extra = {}
    if run == "a" or run.startswith("c_m"):
        from common.embed import fit_neg, pca_init, stage2
        from common.experiment import run_spec
        from common.knn import knn_graph
        from method.bg_term import BackgroundTerm
        from method.xi_b import build_xi_b
        G = knn_graph(X_T, k=K)
        xi = build_xi_b(X_T, X_B, k_B=K_B, L=L, seed=seed) if run != "a" else None
        s1 = fit_neg(pca_init(X_T), G, seed, batch_hook=BackgroundTerm("none"), n_epochs=250)
        fit = stage2(s1["embd"], G, seed, xi=xi, **run_spec(run))
        std = np.array(fit["scale"]["embd_std"])
        log = fit.get("log", {})
        nonfinite = int(sum(log.get("n_nonfinite_batches", [0]))) + int(not all(fit["scale"]["finite"]))
        extra = {"embd_std": float(std[-1]), "max_std_growth": float(std.max() / std[0]),
                 "nonfinite_batches": nonfinite, "stage1_sha256": hashlib.sha256(s1["embd"].tobytes()).hexdigest()}
        return fit["embd"], extra
    if run == "T1":
        return tsne(X_T, seed), extra
    if run == "T2":
        from sklearn.decomposition import PCA
        return tsne(PCA(N_PCA, svd_solver="full").fit_transform(X_T), seed), extra
    if run == "T3":
        from baselines import cpca
        f = cpca.fit(X_T, X_B)
        alpha, reps, ratios, _ = cpca.select_alpha(f, seed)
        extra = {"alpha": alpha, "representatives": reps, "variance_ratios": ratios,
                 "projections_2d": {f"{a:.6g}": cpca.project(f, a, 2) for a in reps}}
        return tsne(cpca.project(f, alpha, N_CPCA), seed), extra
    raise ValueError(f"unknown run {run!r}")


def run_one(stage, seed, run, rep):
    from common.experiment import FRAGMENT_FRAC, FRAGMENT_SMALL, MAX_STD_GROWTH
    from eval.metrics import component_stats, knn_accuracy, knn_recall
    assert_committed(PREREG)
    if stage == "confirmation":
        assert_committed(SELECTION)
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    t0 = time.perf_counter()
    E, extra = embed(run, seed, X_T, X_B)
    runtime = time.perf_counter() - t0
    proj = extra.pop("projections_2d", None)
    comp = component_stats(E, k=K, small=FRAGMENT_SMALL)
    rec = {"stage": stage, "seed": seed, "run": run, "rep": rep, "digit_acc": knn_accuracy(E, t["y"]),
           "recall_digits": knn_recall(E, t["digits"], k=K), "recall_grass": knn_recall(E, t["grass_T"], k=K),
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
    print(json.dumps({k: v for k, v in rec.items() if k not in ("environment",)}, default=str), flush=True)
    if rec["nonfinite_batches"] or rec.get("max_std_growth", 0) > MAX_STD_GROWTH:
        stop(f"{name}: non-finite or exploding (std growth {rec.get('max_std_growth')})")


def runs_for(stage):
    if stage == "selection":
        return [f"c_m{m}" for m in M_GRID] + list(CONTROLS)
    sel = json.loads(SELECTION.read_text())
    return [f"c_m{sel['best_m_B']}"] + list(CONTROLS)


def make_jobs(stage):
    lines = [f"$PY scripts/mnist_grass_run.py --stage {stage} --seed {s} --run {r}"
             for s in SEEDS[stage] for r in runs_for(stage)]
    if stage == "confirmation":
        lines.append(f"$PY scripts/mnist_grass_run.py --stage confirmation --seed {RECHECK_SEED} "
                     f"--run {runs_for(stage)[0]} --rep 2")
    path = ROOT / "cluster" / "jobs" / f"mnist_grass_{stage}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def merge(stage):
    from common.experiment import COLLAPSE_STD_RATIO
    recs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob(f"{stage}_seed*.json"))]
    main = [r for r in recs if not r["rep"]]
    for r in main:
        a = next((x for x in main if x["seed"] == r["seed"] and x["run"] == "a"), None)
        if r["run"].startswith("c_m") and a is not None:
            r["std_ratio_vs_a"] = r["embd_std"] / a["embd_std"]
            r["flag_collapse"] = r["std_ratio_vs_a"] < COLLAPSE_STD_RATIO
    with open(OUT / f"runs_{stage}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: (r["seed"], r["run"], r["rep"] or 0)))
    if stage == "selection":
        means = {m: float(np.mean([r["digit_acc"] for r in main if r["run"] == f"c_m{m}"])) for m in M_GRID}
        best = max(M_GRID, key=lambda m: (means[m], -m))
        sel = {"metric": "digit kNN accuracy", "means": {str(m): v for m, v in means.items()}, "best_m_B": best,
               "rule": "highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B",
               "environment": environment_info()}
        SELECTION.write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: v for k, v in sel.items() if k != "environment"}, indent=2))
    else:
        run = runs_for(stage)[0]
        first = np.load(RUNS_DIR / f"confirmation_seed{RECHECK_SEED}_{run}.npy")
        again = np.load(RUNS_DIR / f"confirmation_seed{RECHECK_SEED}_{run}_rep2.npy")
        result = {"run": run, "seed": RECHECK_SEED, "identical": bool(np.array_equal(first, again)),
                  "environment": environment_info()}
        (OUT / "recheck.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "environment"}))
        if not result["identical"]:
            stop(f"rerun of {run} on seed {RECHECK_SEED} is not bit-identical")
    print(f"merged {len(recs)} runs -> runs_{stage}.csv")


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
