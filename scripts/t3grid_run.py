"""Tuned cPCA -> t-SNE baseline (T3) on MNIST-on-grass 0-9 (results/mnist_grass_0to9_t3grid/preregistration.md).

Grid: k cPCA components in KS x alpha in cpca.ALPHAS plus 1 (ALPHAS here: 42 values, sorted; the alpha index is the
position in it). Each setting is T3's pipeline: cpca.fit on raw pixels, cpca.project(f, alpha, k), openTSNE with
library defaults and random_state = the dataset seed (scripts/mnist_grass_run.tsne).

  --stage selection|confirmation --seed S --alpha-index I
        one process, all k: runs/{stage}_seed{S}_a{I:02d}.json (per-k metrics and SHA-256) and the embeddings
        runs/{stage}_seed{S}_a{I:02d}_k{k}.npy (gitignored)
  --make-jobs STAGE   cluster/jobs/mnist_grass_0to9_t3grid_{STAGE}.txt, 126 commands
  --merge STAGE       runs_{STAGE}.csv and the SHA check (sha_check_{STAGE}.json: the grid point (label-free alpha,
                      k = 10) must reproduce the stored T3 embedding on every seed); for selection also selection.json
                      (T3* and PCA*: highest mean digit accuracy over seeds 0-2, ties -> fewer components, then smaller
                      alpha); for confirmation also reuse.json (the stored runs the comparisons reuse, with the
                      reruns below)
  --rerun RUN --seed S
                      RUN in RERUN: rerun a stored 0-9 confirmation run exactly as scripts/mnist_grass_0to9_run.py does
                      and record whether it is bit-identical (reruns/{RUN}_seed{S}.json). The pre-registered remedy
                      after the first reuse check failed, which came from a bug in that check (CODE listed
                      scripts/single_stage_run.py, a file only the single-stage runs depend on, for every run)
  --make-jobs rerun   cluster/jobs/mnist_grass_0to9_t3grid_rerun.txt
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

from baselines import cpca  # noqa: E402
from common.repro import assert_committed, environment_info, setup_determinism  # noqa: E402

NAME = "mnist_grass_0to9_t3grid"
OUT = ROOT / "results" / NAME
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration.md"
SELECTION = OUT / "selection.json"
OLD = ROOT / "results" / "mnist_grass_0to9"  # the 0-9 phase: stored (r_c), (r_a), T2, T3
SINGLE = ROOT / "results" / "single_stage" / "runs"  # stored single-stage (r_a), 0-9
SEEDS = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
KS = (2, 5, 10, 20, 30, 50, 100)
ALPHAS = np.sort(np.append(cpca.ALPHAS, 1.0))
N_CPCA = 10  # T3's number of components
N_CLASSES = 10
K = 15
METRICS = ("digit_acc", "recall_digits", "recall_grass", "ari")
FIELDS = ["stage", "seed", "alpha_index", "alpha", "k", *METRICS, "runtime_s", "sha256"]
# everything a stored run depends on, from the dataset files to the embedding; the single-stage runs also on SINGLE_CODE
CODE = ["common", "method", "baselines/cpca.py", "eval/metrics.py", "data/mnist_grass.py", "scripts/bsgraph_run.py",
        "scripts/mnist_grass_run.py", "scripts/mnist_grass_0to9_run.py", "third_party/contrastive-ne"]
SINGLE_CODE = ["scripts/single_stage_run.py"]
RERUN = {"r_c": "r_c_m5", "r_a_two_stage": "r_a"}  # reuse-check group -> stored 0-9 run, rerun on seeds 3-5
RERUN_DIR = OUT / "reruns"


def stop(msg):
    print(f"STOP (T3 grid): {msg}", flush=True)
    sys.exit(2)


def load(seed):
    from scripts.mnist_grass_0to9_run import load as load09
    return load09(seed)


def sha(E):
    return hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()


def npy(stage, seed, i, k):
    return RUNS_DIR / f"{stage}_seed{seed}_a{i:02d}_k{k}.npy"


def run_one(stage, seed, i):
    from eval.metrics import kmeans_ari, knn_accuracy, knn_recall
    from scripts.mnist_grass_run import tsne
    assert_committed(PREREG)
    if stage == "confirmation":
        assert_committed(SELECTION)
    setup_determinism(seed)
    X_T, X_B, t = load(seed)
    alpha = float(ALPHAS[i])
    f = cpca.fit(X_T, X_B)
    per_k = []
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for k in KS:
        t0 = time.perf_counter()
        E = tsne(cpca.project(f, alpha, k), seed)
        runtime = time.perf_counter() - t0
        if not np.isfinite(E).all():
            stop(f"{stage} seed {seed} alpha {alpha:.6g} k {k}: non-finite embedding")
        np.save(npy(stage, seed, i, k), E)
        per_k.append({"k": k, "digit_acc": knn_accuracy(E, t["y"], k=K),
                      "recall_digits": knn_recall(E, t["digits"], k=K),
                      "recall_grass": knn_recall(E, t["grass_T"], k=K), "ari": kmeans_ari(E, t["y"], N_CLASSES, seed),
                      "runtime_s": runtime, "sha256": sha(E)})
        print(json.dumps({"stage": stage, "seed": seed, "alpha": alpha, **per_k[-1]}), flush=True)
    rec = {"stage": stage, "seed": seed, "alpha_index": i, "alpha": alpha, "per_k": per_k,
           "environment": environment_info()}
    (RUNS_DIR / f"{stage}_seed{seed}_a{i:02d}.json").write_text(json.dumps(rec, indent=2))


def make_jobs(stage):
    if stage == "rerun":
        lines = [f"$PY scripts/t3grid_run.py --rerun {r} --seed {s}" for r in RERUN.values()
                 for s in SEEDS["confirmation"]]
    else:
        lines = [f"$PY scripts/t3grid_run.py --stage {stage} --seed {s} --alpha-index {i}"
                 for i in range(len(ALPHAS)) for s in SEEDS[stage]]
    path = ROOT / "cluster" / "jobs" / f"{NAME}_{stage}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def rows(stage):
    """One row per (seed, alpha, k) of a stage, after checking that the grid is complete."""
    recs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob(f"{stage}_seed*_a*.json"))]
    if len(recs) != len(SEEDS[stage]) * len(ALPHAS):
        stop(f"{stage}: {len(recs)} of {len(SEEDS[stage]) * len(ALPHAS)} runs")
    return [{"stage": stage, "seed": r["seed"], "alpha_index": r["alpha_index"], "alpha": r["alpha"], **pk}
            for r in recs for pk in r["per_k"]]


def best(rows_, keep=lambda r: True):
    """The setting with the highest mean digit accuracy; ties -> fewer components, then smaller alpha."""
    means = {}
    for r in rows_:
        if keep(r):
            means.setdefault((r["k"], r["alpha_index"]), []).append(r["digit_acc"])
    k, i = max(means, key=lambda s: (np.mean(means[s]), -s[0], -ALPHAS[s[1]]))
    return {"k": k, "alpha_index": i, "alpha": float(ALPHAS[i]), "mean_digit_acc": float(np.mean(means[(k, i)])),
            "n_seeds": len(means[(k, i)])}


def sha_check(stage, rows_):
    """The grid point (label-free alpha, k = 10) against the stored T3 run of the 0-9 phase, per seed."""
    out = []
    for seed in SEEDS[stage]:
        old = json.loads((OLD / "runs" / f"{stage}_seed{seed}_T3.json").read_text())
        i = int(np.flatnonzero(ALPHAS == old["alpha"])[0]) if old["alpha"] in ALPHAS else None
        new = next((r for r in rows_ if r["seed"] == seed and r["alpha_index"] == i and r["k"] == N_CPCA), None)
        out.append({"seed": seed, "label_free_alpha": old["alpha"], "alpha_index": i, "stored_sha256": old["sha256"],
                    "grid_sha256": new and new["sha256"], "identical": bool(new) and new["sha256"] == old["sha256"]})
    return out


def code_unchanged(commit, extra=()):
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "HEAD", "--", *CODE, *extra]
                          ).returncode == 0


def rerun_one(run, seed):
    """A stored 0-9 confirmation run, recomputed as scripts/mnist_grass_0to9_run.run_one does, against its SHA-256."""
    from scripts.mnist_grass_0to9_run import embed
    assert_committed(PREREG)
    setup_determinism(seed)
    X_T, X_B, _ = load(seed)
    E, _ = embed(run, seed, X_T, X_B)
    stored = json.loads((OLD / "runs" / f"confirmation_seed{seed}_{run}.json").read_text())["sha256"]
    rec = {"run": run, "seed": seed, "sha256": sha(E), "stored_sha256": stored, "identical": sha(E) == stored,
           "environment": environment_info()}
    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    (RERUN_DIR / f"{run}_seed{seed}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}), flush=True)


def reuse_check():
    """Each stored run the comparisons reuse: code unchanged since its commit, .npy SHA-256 as recorded (load checks
    the dataset hash) and, for (r_c) and two-stage (r_a), the rerun bit-identical."""
    m_b = json.loads((OLD / "selection.json").read_text())["r_c"]["best_m_B"]
    stored = {"r_c": [OLD / "runs" / f"confirmation_seed{s}_r_c_m{m_b}" for s in SEEDS["confirmation"]],
              "r_a_two_stage": [OLD / "runs" / f"confirmation_seed{s}_r_a" for s in SEEDS["confirmation"]],
              "r_a_single": [SINGLE / f"0to9_seed{s}_single" for s in SEEDS["confirmation"]],
              "T2": [OLD / "runs" / f"{st}_seed{s}_T2" for st, ss in SEEDS.items() for s in ss],
              "T3": [OLD / "runs" / f"{st}_seed{s}_T3" for st, ss in SEEDS.items() for s in ss]}
    for s in (*SEEDS["selection"], *SEEDS["confirmation"]):
        load(s)
    out = []
    for run, paths in stored.items():
        for p in paths:
            rec = json.loads(p.with_suffix(".json").read_text())
            commit = rec["environment"]["project_git"]["head"]
            out.append({"run": run, "file": p.name, "seed": rec["seed"], "commit": commit[:7],
                        "slurm_job": rec["environment"]["slurm_job_id"],
                        "code_unchanged": code_unchanged(commit, SINGLE_CODE if run == "r_a_single" else ()),
                        "npy_sha_matches": sha(np.load(p.with_suffix(".npy"))) == rec["sha256"]})
            if run in RERUN:
                rerun = RERUN_DIR / f"{RERUN[run]}_seed{rec['seed']}.json"
                out[-1]["rerun_identical"] = rerun.exists() and json.loads(rerun.read_text())["identical"]
    ok = all(r["code_unchanged"] and r["npy_sha_matches"] and r.get("rerun_identical", True) for r in out)
    return {"m_B": m_b, "runs": out, "reusable": ok}


def merge(stage):
    rows_ = rows(stage)
    with open(OUT / f"runs_{stage}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows_, key=lambda r: (r["seed"], r["alpha_index"], r["k"])))
    check = sha_check(stage, rows_)
    (OUT / f"sha_check_{stage}.json").write_text(json.dumps({"checks": check, "environment": environment_info()},
                                                            indent=2))
    print(json.dumps([{k: c[k] for k in ("seed", "label_free_alpha", "identical")} for c in check]))
    if not all(c["identical"] for c in check):
        stop(f"{stage}: the label-free grid point does not reproduce the stored T3 embedding")
    if stage == "selection":
        sel = {"metric": "digit kNN accuracy", "T3_star": best(rows_),
               "PCA_star": best(rows_, lambda r: r["alpha"] == 0),
               "rule": "highest mean digit kNN accuracy over selection seeds 0-2; ties -> fewer components, then "
                       "smaller alpha", "grid": {"k": list(KS), "alpha": ALPHAS.tolist()},
               "environment": environment_info()}
        SELECTION.write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: sel[k] for k in ("T3_star", "PCA_star")}, indent=2))
    else:
        reuse = reuse_check()
        (OUT / "reuse.json").write_text(json.dumps({**reuse, "environment": environment_info()}, indent=2))
        print(json.dumps({"reusable": reuse["reusable"]}))
        if not reuse["reusable"]:
            stop("stored runs not reusable: rerun (r_c) and (r_a) on seeds 3-5 before comparing")
    print(f"merged {len(rows_)} settings x seeds -> runs_{stage}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(SEEDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha-index", type=int)
    parser.add_argument("--make-jobs", choices=[*SEEDS, "rerun"])
    parser.add_argument("--rerun", choices=list(RERUN.values()))
    parser.add_argument("--merge", choices=list(SEEDS))
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs(args.make_jobs)
    if args.merge:
        return merge(args.merge)
    if args.rerun:
        return rerun_one(args.rerun, args.seed)
    run_one(args.stage, args.seed, args.alpha_index)


if __name__ == "__main__":
    main()
