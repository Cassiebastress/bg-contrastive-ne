"""Background-suppressed graph, Phase 2: embeddings on the selected residual graph (results/bsgraph/
preregistration_phase2.md). One run on one dataset seed per process, as in scripts/mnist_grass_run.py.

  --stage selection|confirmation --seed S --run R [--rep N]
        R is r_a (two-stage Neg-t-SNE on the residual graph, background term off) or r_c_m<m> (the same + raw-pixel
        xi_B negatives, m_B = m). Initialization: PCA of the raw X_T, as for (a) and (c). Writes
        results/bsgraph/runs/{stage}_seed{S}_{R}[_rep{N}].json and the embedding .npy (gitignored).
  --make-jobs STAGE   cluster/jobs/bsgraph_{STAGE}.txt
  --merge STAGE       runs_{STAGE}.csv; selection.json (selection) or recheck.json + reuse.json (confirmation): the
                      reuse check of the stored MNIST-on-grass (a), (c), T1-T3 results (dataset hashes and the code
                      (a)/(c) depend on unchanged since 5556465)
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402

OUT = ROOT / "results" / "bsgraph"
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration_phase2.md"
SELECTION = OUT / "selection.json"
SEEDS = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
M_GRID = (1, 2, 5, 10, 20, 40)
K_LOC, R = 30, 2  # selected in Phase 1 (results/bsgraph/phase1.json)
K, K_B, L = 15, 15, 30
RECHECK_SEED = 3
STORED_COMMIT = "5556465"  # the commit the MNIST-on-grass confirmation runs ran at
AC_CODE = ["common/embed.py", "common/knn.py", "common/experiment.py", "common/repro.py", "method/bg_term.py",
           "method/xi_b.py", "data/mnist_grass.py", "eval/metrics.py", "scripts/mnist_grass_run.py"]
FIELDS = ["stage", "seed", "run", "rep", "digit_acc", "recall_digits", "recall_grass", "embd_std", "std_ratio_vs_a",
          "flag_collapse", "n_components", "frac_in_small_components", "flag_fragment", "nonfinite_batches",
          "max_std_growth", "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (background-suppressed graph, Phase 2): {msg}", flush=True)
    sys.exit(2)


def embed(run, seed, X_T, X_B):
    from common.embed import fit_neg, pca_init, stage2
    from common.experiment import run_spec
    from method.bg_term import BackgroundTerm
    from method.residual_graph import residual_graph
    from method.xi_b import build_xi_b
    G, _ = residual_graph(X_T, X_B, K_LOC, R, k=K)
    label = "a" if run == "r_a" else run[2:]  # r_c_m5 -> c_m5
    xi = build_xi_b(X_T, X_B, k_B=K_B, L=L, seed=seed) if label != "a" else None
    s1 = fit_neg(pca_init(X_T), G, seed, batch_hook=BackgroundTerm("none"), n_epochs=250)
    fit = stage2(s1["embd"], G, seed, xi=xi, **run_spec(label))
    std = np.array(fit["scale"]["embd_std"])
    log = fit.get("log", {})
    nonfinite = int(sum(log.get("n_nonfinite_batches", [0]))) + int(not all(fit["scale"]["finite"]))
    return fit["embd"], {"embd_std": float(std[-1]), "max_std_growth": float(std.max() / std[0]),
                         "nonfinite_batches": nonfinite,
                         "stage1_sha256": hashlib.sha256(s1["embd"].tobytes()).hexdigest()}


def run_one(stage, seed, run, rep):
    from common.experiment import FRAGMENT_FRAC, FRAGMENT_SMALL, MAX_STD_GROWTH
    from eval.metrics import component_stats, knn_accuracy, knn_recall
    from scripts.bsgraph_phase1 import load
    assert_committed(PREREG)
    if stage == "confirmation":
        assert_committed(SELECTION)
    setup_determinism(seed)
    X_T, X_B, t = load(seed)  # checks the dataset hash against data/MANIFEST.md
    t0 = time.perf_counter()
    E, extra = embed(run, seed, X_T, X_B)
    comp = component_stats(E, k=K, small=FRAGMENT_SMALL)
    rec = {"stage": stage, "seed": seed, "run": run, "rep": rep, "digit_acc": knn_accuracy(E, t["y"]),
           "recall_digits": knn_recall(E, t["digits"], k=K), "recall_grass": knn_recall(E, t["grass_T"], k=K),
           **comp, "flag_fragment": comp["frac_in_small_components"] > FRAGMENT_FRAC,
           "runtime_s": time.perf_counter() - t0,
           "sha256": hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest(),
           "residual_graph": {"k_loc": K_LOC, "r": R}, **extra, "environment": environment_info()}
    name = f"{stage}_seed{seed}_{run}" + (f"_rep{rep}" if rep else "")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    np.save(RUNS_DIR / f"{name}.npy", E)
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}, default=str), flush=True)
    if rec["nonfinite_batches"] or rec["max_std_growth"] > MAX_STD_GROWTH:
        stop(f"{name}: non-finite or exploding (std growth {rec['max_std_growth']:.2f})")


def runs_for(stage):
    if stage == "selection":
        return ["r_a"] + [f"r_c_m{m}" for m in M_GRID]
    return ["r_a", f"r_c_m{json.loads(SELECTION.read_text())['best_m_B']}"]


def make_jobs(stage):
    lines = [f"$PY scripts/bsgraph_run.py --stage {stage} --seed {s} --run {r}" for s in SEEDS[stage]
             for r in runs_for(stage)]
    if stage == "confirmation":
        lines.append(f"$PY scripts/bsgraph_run.py --stage confirmation --seed {RECHECK_SEED} "
                     f"--run {runs_for(stage)[1]} --rep 2")
    path = ROOT / "cluster" / "jobs" / f"bsgraph_{stage}.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def reuse_check():
    """The stored MNIST-on-grass results can be reused if the code (a)/(c) depend on is unchanged since they ran (the
    dataset hashes are checked by every run of this phase, against the same manifest)."""
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", STORED_COMMIT, "HEAD", "--", *AC_CODE])
    return {"stored_commit": STORED_COMMIT, "files": AC_CODE, "code_unchanged": diff.returncode == 0}


def merge(stage):
    from common.experiment import COLLAPSE_STD_RATIO
    recs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob(f"{stage}_seed*.json"))]
    main = [r for r in recs if not r["rep"]]
    for r in main:
        a = next((x for x in main if x["seed"] == r["seed"] and x["run"] == "r_a"), None)
        if r["run"].startswith("r_c") and a is not None:
            r["std_ratio_vs_a"] = r["embd_std"] / a["embd_std"]
            r["flag_collapse"] = r["std_ratio_vs_a"] < COLLAPSE_STD_RATIO
    with open(OUT / f"runs_{stage}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: (r["seed"], r["run"], r["rep"] or 0)))
    if stage == "selection":
        means = {m: float(np.mean([r["digit_acc"] for r in main if r["run"] == f"r_c_m{m}"])) for m in M_GRID}
        best = max(M_GRID, key=lambda m: (means[m], -m))
        sel = {"metric": "digit kNN accuracy", "means": {str(m): v for m, v in means.items()}, "best_m_B": best,
               "rule": "highest mean digit kNN accuracy over selection seeds 0-2; ties -> smaller m_B",
               "environment": environment_info()}
        SELECTION.write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: v for k, v in sel.items() if k != "environment"}, indent=2))
    else:
        run = runs_for(stage)[1]
        first = np.load(RUNS_DIR / f"confirmation_seed{RECHECK_SEED}_{run}.npy")
        again = np.load(RUNS_DIR / f"confirmation_seed{RECHECK_SEED}_{run}_rep2.npy")
        recheck = {"run": run, "seed": RECHECK_SEED, "identical": bool(np.array_equal(first, again)),
                   "environment": environment_info()}
        (OUT / "recheck.json").write_text(json.dumps(recheck, indent=2))
        reuse = reuse_check()
        (OUT / "reuse.json").write_text(json.dumps(reuse, indent=2))
        print(json.dumps({"recheck_identical": recheck["identical"], **reuse}))
        if not recheck["identical"]:
            stop(f"rerun of {run} on seed {RECHECK_SEED} is not bit-identical")
        if not reuse["code_unchanged"]:
            stop("the code (a)/(c) depend on changed since the stored runs: rerun (c) on seed 3 and require "
                 "bit-identity before reusing the stored results")
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
