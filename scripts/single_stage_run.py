"""Single-stage vs two-stage (r_a), descriptive (docs/plans/single_stage_plan.md), on MNIST-on-grass 0/1 and 0-9.

(r_a) is two-stage Neg-t-SNE on the residual graph (k_loc = 30, r = 2), background term off: 250 epochs, then 500 more
from that embedding with a fresh learning-rate schedule (scripts/bsgraph_run.embed). r_a-single is one fit of 750
epochs on the same graph from the same initialization (PCA of raw X_T, rescaled) with package defaults (neg_spec = 1.0,
early exaggeration on, a no-op at that setting because the package only exaggerates when `s` is set,
third_party/contrastive-ne/src/cne/cne.py:462). The dataset seed picks the data; the training seed (default: the
dataset seed, as in the stored runs) drives the optimizer. No shared code is changed.

  --dataset 01|0to9 --seed S --variant two_stage|single [--train-seed T] [--rep N]
        one run -> results/single_stage/runs/{dataset}_seed{S}_{variant}[_t{T}][_rep{N}].json and .npy (gitignored)
  --make-jobs 1|2   job 1: the seed-3 two-stage rerun and the noise reference (training seeds 1003, 2003) per dataset;
                    job 2: r_a-single on seeds 0-5 per dataset and its seed-3 rerun
  --check           after job 1: the reuse check of the stored two-stage runs (dataset hashes, code unchanged since
                    each run's commit, .npy SHA-256 and digit accuracy reproduced, seed-3 rerun bit-identical), the
                    noise ranges (noise_reference.json/.md) and the generated criterion.md, committed before job 2
  --merge           after job 2: the single-stage rerun check, all metrics for both versions from the embeddings, the
                    agreement between them, "matches closely" per seed and dataset and the adoption outcome
                    (results.json, runs.csv)
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
from scripts.bsgraph_run import K, K_LOC, R  # noqa: E402

OUT = ROOT / "results" / "single_stage"
RUNS_DIR = OUT / "runs"
PLAN = ROOT / "docs" / "plans" / "single_stage_plan.md"
CRITERION = OUT / "criterion.md"
SEEDS, PRIMARY = range(6), (3, 4, 5)
RECHECK_SEED = 3
NOISE_SEEDS = (1003, 2003)
N1, N2 = 250, 500
FLOOR = 0.005  # the tolerance never goes below this
DATASETS = {"01": {"label": "digits 0/1", "stored": ROOT / "results" / "bsgraph" / "runs", "n_classes": 2},
            "0to9": {"label": "digits 0–9", "stored": ROOT / "results" / "mnist_grass_0to9" / "runs",
                     "n_classes": 10}}
# everything a stored (r_a) run depends on, from the dataset files to the embedding
CODE = ["common/embed.py", "common/knn.py", "common/repro.py", "common/experiment.py", "method/bg_term.py",
        "method/residual_graph.py", "data/mnist_grass.py", "scripts/bsgraph_run.py", "scripts/bsgraph_phase1.py",
        "scripts/mnist_grass_0to9_run.py", "third_party/contrastive-ne"]
METRICS = ("digit_acc", "recall_digits", "recall_grass", "ari")
FIELDS = ["dataset", "seed", "variant", "train_seed", "rep", *METRICS, "embd_std", "nonfinite", "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (single-stage check): {msg}", flush=True)
    sys.exit(2)


def load(dataset, seed):
    """X_T, X_B and the evaluation truth, after the dataset's SHA-256 check against data/MANIFEST.md."""
    if dataset == "01":
        from scripts.bsgraph_phase1 import load as load01
        return load01(seed)
    from scripts.mnist_grass_0to9_run import load as load09
    return load09(seed)


def stored(dataset, seed):
    """The stored two-stage (r_a) run of the bsgraph (0/1) or 0-9 phase: its JSON record and .npy path."""
    stage = "selection" if seed < 3 else "confirmation"
    path = DATASETS[dataset]["stored"] / f"{stage}_seed{seed}_r_a"
    return json.loads(path.with_suffix(".json").read_text()), path.with_suffix(".npy")


def run_name(dataset, seed, variant, train_seed, rep):
    return f"{dataset}_seed{seed}_{variant}" + (f"_t{train_seed}" if train_seed != seed else "") + \
        (f"_rep{rep}" if rep else "")


def metrics(E, t, n_classes, seed):
    from eval.metrics import kmeans_ari, knn_accuracy, knn_recall
    return {"digit_acc": knn_accuracy(E, t["y"], k=K), "recall_digits": knn_recall(E, t["digits"], k=K),
            "recall_grass": knn_recall(E, t["grass_T"], k=K), "ari": kmeans_ari(E, t["y"], n_classes, seed)}


def sha(E):
    return hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()


def fit(variant, X_T, X_B, train_seed):
    """The embedding, the per-epoch learning rates, the resolved package parameters and the final embedding std."""
    from common.embed import fit_neg, pca_init, stage2
    from common.experiment import run_spec
    from method.bg_term import BackgroundTerm
    from method.residual_graph import residual_graph
    G, _ = residual_graph(X_T, X_B, K_LOC, R, k=K)
    if variant == "two_stage":  # exactly scripts/bsgraph_run.embed("r_a"), with the training seed separated out
        s1 = fit_neg(pca_init(X_T), G, train_seed, batch_hook=BackgroundTerm("none"), n_epochs=N1)
        f = stage2(s1["embd"], G, train_seed, n2=N2, **run_spec("a"))
        params, lrs = {"stage1": s1["params"], "stage2": f["params"]}, {"stage1": s1["lrs"], "stage2": f["lrs"]}
        finite = all(s1["scale"]["finite"]) and all(f["scale"]["finite"])
    else:
        f = fit_neg(pca_init(X_T), G, train_seed, batch_hook=BackgroundTerm("none"), n_epochs=N1 + N2)
        p = f["params"]
        if not (p["s"] is None and p["neg_spec"] == 1.0 and p["early_exaggeration"]):
            stop(f"unexpected package settings {p}: early exaggeration may change the objective")
        params, lrs, finite = {"single": p}, {"single": f["lrs"]}, all(f["scale"]["finite"])
    return f["embd"], lrs, params, float(f["scale"]["embd_std"][-1]), not finite


def run_one(dataset, seed, variant, train_seed, rep):
    assert_committed(PLAN)
    if variant == "single":
        assert_committed(CRITERION)
    setup_determinism(seed)
    X_T, X_B, t = load(dataset, seed)
    t0 = time.perf_counter()
    E, lrs, params, std, nonfinite = fit(variant, X_T, X_B, train_seed)
    runtime = time.perf_counter() - t0
    nonfinite = nonfinite or not np.isfinite(E).all()
    rec = {"dataset": dataset, "seed": seed, "variant": variant, "train_seed": train_seed, "rep": rep,
           **metrics(E, t, DATASETS[dataset]["n_classes"], seed), "embd_std": std, "nonfinite": nonfinite,
           "runtime_s": runtime, "sha256": sha(E), "residual_graph": {"k_loc": K_LOC, "r": R}, "params": params,
           "learning_rates": lrs, "environment": environment_info()}
    name = run_name(dataset, seed, variant, train_seed, rep)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    np.save(RUNS_DIR / f"{name}.npy", E)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("environment", "params", "learning_rates")}),
          flush=True)
    if nonfinite:
        stop(f"{name}: non-finite embedding")


def make_jobs(job):
    cmd = "$PY scripts/single_stage_run.py --dataset {} --seed {} --variant {}"
    if job == "1":
        lines = [cmd.format(d, RECHECK_SEED, "two_stage") + extra for d in DATASETS
                 for extra in ["", *(f" --train-seed {s}" for s in NOISE_SEEDS)]]
    else:
        lines = [cmd.format(d, s, "single") for d in DATASETS for s in SEEDS]
        lines += [cmd.format(d, RECHECK_SEED, "single") + " --rep 2" for d in DATASETS]
    path = ROOT / "cluster" / "jobs" / f"single_stage_{job}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def new(dataset, seed, variant, train_seed=None, rep=0):
    name = run_name(dataset, seed, variant, seed if train_seed is None else train_seed, rep)
    return json.loads((RUNS_DIR / f"{name}.json").read_text()), np.load(RUNS_DIR / f"{name}.npy")


def code_unchanged(commit):
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "HEAD", "--", *CODE]).returncode == 0


def reuse_check(dataset):
    """Per stored run: dataset hash (load stops on a mismatch), code unchanged since its commit, .npy SHA-256 and digit
    accuracy reproduced; then the seed-3 two-stage rerun against the stored embedding."""
    n_classes = DATASETS[dataset]["n_classes"]
    rows = []
    for seed in SEEDS:
        rec, npy = stored(dataset, seed)
        E = np.load(npy)
        _, _, t = load(dataset, seed)
        commit = rec["environment"]["project_git"]["head"]
        rows.append({"seed": seed, "commit": commit[:7], "slurm_job": rec["environment"]["slurm_job_id"],
                     "code_unchanged": code_unchanged(commit), "npy_sha_matches": sha(E) == rec["sha256"],
                     "digit_acc_reproduced": metrics(E, t, n_classes, seed)["digit_acc"] == rec["digit_acc"]})
    _, E_rerun = new(dataset, RECHECK_SEED, "two_stage")
    rerun_identical = bool(np.array_equal(E_rerun, np.load(stored(dataset, RECHECK_SEED)[1])))
    ok = rerun_identical and all(r["npy_sha_matches"] and r["digit_acc_reproduced"] for r in rows)
    return {"stored_runs": rows, "rerun_seed3_identical": rerun_identical, "reusable": ok}


def noise(dataset):
    """The range of digit accuracy and grass recall over the stored seed-3 run and the two training-seed reruns."""
    rec, _ = stored(dataset, RECHECK_SEED)
    runs = [{"train_seed": RECHECK_SEED, "digit_acc": rec["digit_acc"], "recall_grass": rec["recall_grass"]}]
    for s in NOISE_SEEDS:
        r, _ = new(dataset, RECHECK_SEED, "two_stage", train_seed=s)
        runs.append({"train_seed": s, "digit_acc": r["digit_acc"], "recall_grass": r["recall_grass"]})
    rng = {m: float(np.ptp([r[m] for r in runs])) for m in ("digit_acc", "recall_grass")}
    return {"runs": runs, "noise_acc": rng["digit_acc"], "noise_grass": rng["recall_grass"],
            "tol_acc": max(FLOOR, rng["digit_acc"]), "tol_grass": max(FLOOR, rng["recall_grass"])}


def criterion_md(noise_by):
    rows = "\n".join(f"| {DATASETS[d]['label']} | {n['noise_acc']:.4f} | {n['tol_acc']:.4f} | {n['noise_grass']:.4f} "
                     f"| {n['tol_grass']:.4f} |" for d, n in noise_by.items())
    return f"""# Single-stage vs two-stage (r_a): criterion (generated; committed before any single-stage run)

Generated by `scripts/single_stage_run.py --check` from `noise_reference.json`. The plan is
`docs/plans/single_stage_plan.md`.

## Tolerances

The noise ranges come from seed 3 of each dataset: two-stage (r_a), stored run (training seed 3) plus training seeds
{NOISE_SEEDS[0]} and {NOISE_SEEDS[1]}. Each tolerance is max({FLOOR}, noise range).

| dataset | noise_acc | tolerance, digit kNN accuracy | noise_grass | tolerance, grass recall |
|---|---|---|---|---|
{rows}

## "Matches closely" (descriptive, not a verdict)

Applied separately to each dataset, on confirmation seeds 3, 4 and 5. It holds for a dataset if, on every one of those
seeds, r_a-single minus the stored two-stage (r_a) satisfies
- |Δ digit kNN accuracy| ≤ the dataset's accuracy tolerance, and
- |Δ grass recall| ≤ the dataset's grass tolerance.

## Adoption rule

- If "matches closely" holds on both datasets, future experiments use single-stage (r_a) as the method's core form.
- If it fails on either dataset, they keep two-stage (r_a).
- Either way, (r_c) keeps its two-stage schedule.

## Package behavior (read from the code)

- **Early exaggeration** (`third_party/contrastive-ne/src/cne/cne.py:462`) uses the exaggerated spectrum parameter only
  when `self.s is not None`. r_a-single sets only neg_spec = 1.0, so `s` stays unset and the objective is the same in
  every epoch. Each single-stage run asserts that `s` is unset.
- **Learning rate** (`new_lr`, `cne.py:702`): linear annealing over each fit's own `n_epochs`. Two-stage anneals over
  250 epochs, then restarts and anneals over 500; single-stage anneals once over 750. Every run records its per-epoch
  rates.
"""


def check():
    result = {d: {"reuse": reuse_check(d), "noise": noise(d)} for d in DATASETS}
    (OUT / "noise_reference.json").write_text(json.dumps({**result, "environment": environment_info()}, indent=2))
    print(json.dumps(result, indent=2))
    bad = [d for d in DATASETS if not result[d]["reuse"]["reusable"]]
    if bad:
        stop(f"stored two-stage (r_a) runs not reusable for {bad}")
    CRITERION.write_text(criterion_md({d: result[d]["noise"] for d in DATASETS}))
    print(f"wrote {CRITERION}: commit it before job 2")


def merge():
    from scipy.spatial import procrustes

    from eval.metrics import knn_recall
    tol = json.loads((OUT / "noise_reference.json").read_text())
    recs, out = [], {}
    for d, info in DATASETS.items():
        rerun_rec, rerun = new(d, RECHECK_SEED, "single", rep=2)
        identical = bool(np.array_equal(rerun, new(d, RECHECK_SEED, "single")[1]))
        if not identical:
            stop(f"{d}: the single-stage rerun on seed {RECHECK_SEED} is not bit-identical")
        n = tol[d]["noise"]
        seeds = []
        for seed in SEEDS:
            old_rec, npy = stored(d, seed)
            two = np.load(npy)
            one_rec, one = new(d, seed, "single")
            _, _, t = load(d, seed)
            m_two = metrics(two, t, info["n_classes"], seed)
            delta = {m: one_rec[m] - m_two[m] for m in METRICS}
            row = {"seed": seed, "two_stage": m_two, "single": {m: one_rec[m] for m in METRICS}, "delta": delta,
                   "knn_overlap": knn_recall(one, two, k=K), "procrustes": float(procrustes(two, one)[2]),
                   "runtime_two_stage": old_rec["runtime_s"], "runtime_single": one_rec["runtime_s"],
                   "within_acc": abs(delta["digit_acc"]) <= n["tol_acc"],
                   "within_grass": abs(delta["recall_grass"]) <= n["tol_grass"]}
            row["matches"] = row["within_acc"] and row["within_grass"]
            seeds.append(row)
            recs.append({"dataset": d, "seed": seed, "variant": "two_stage", "train_seed": seed, "rep": 0, **m_two,
                         "runtime_s": old_rec["runtime_s"], "sha256": old_rec["sha256"]})
            recs.append({k: one_rec.get(k) for k in FIELDS})
        recs.append({k: rerun_rec.get(k) for k in FIELDS})
        out[d] = {"seeds": seeds, "tolerances": {"acc": n["tol_acc"], "grass": n["tol_grass"]},
                  "rerun_seed3_identical": identical,
                  "matches_closely": all(r["matches"] for r in seeds if r["seed"] in PRIMARY)}
    adopt = all(out[d]["matches_closely"] for d in DATASETS)
    result = {**out, "adopt_single_stage": adopt,
              "outcome": "single-stage (r_a) becomes the core form" if adopt else "two-stage (r_a) is kept",
              "environment": environment_info()}
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(json.dumps({d: {"matches_closely": out[d]["matches_closely"]} for d in DATASETS}), result["outcome"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=["two_stage", "single"])
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--make-jobs", choices=["1", "2"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs(args.make_jobs)
    if args.check:
        return check()
    if args.merge:
        return merge()
    run_one(args.dataset, args.seed, args.variant, args.seed if args.train_seed is None else args.train_seed, args.rep)


if __name__ == "__main__":
    main()
