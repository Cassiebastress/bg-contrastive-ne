"""Digit opacity on MNIST-on-grass 0/1 (results/mnist_grass_opacity/preregistration.md).

The stored 0/1 datasets, with the target rebuilt at opacity w: X_T(w) = w * digits + grass_T (float64, cast to
float32 as the stored targets are); X_B unchanged. Every process checks that X_T(0.25) equals the stored target.

  --stage selection|confirmation --seed S --w W --proc P [--rep N]
        P is r_a (single-stage BC-NE, scripts/single_stage_run.fit), T1 (openTSNE), a (single-stage Neg-t-SNE on the
        raw kNN graph), cpca (one cPCA fit at the label-free alpha: T3 = top 10 -> openTSNE, and cPCA2D = the 2-D
        projection) or local (label-free local cPCA, k = 10 -> as a). Writes runs/{stage}_seed{S}_w{100w:03d}_{run}
        [_rep{N}].json per run and the embedding .npy (gitignored)
  --make-jobs selection|confirmation
        cluster/jobs/mnist_grass_opacity_{stage}.txt: every (seed, w, proc), plus the reruns at (seed 0, w = 0.25)
        for selection and at (seed 3, w*) for confirmation
  --make-jobs smoke   the selection job's (seed 0, w = 0.25) processes only
  --merge selection|confirmation
        runs_{stage}.csv after the completeness and rerun checks; for selection also selection.json (the window W and
        w*: the largest w with mean r_a >= 0.80 and mean T1 <= 0.60 over seeds 0-2)
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

NAME = "mnist_grass_opacity"
OUT = ROOT / "results" / NAME
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration.md"
SELECTION = OUT / "selection.json"
SEEDS = {"selection": (0, 1, 2), "confirmation": (3, 4, 5)}
WS = (0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)
W_STORED = 0.25
PROCS = {"r_a": ("r_a",), "T1": ("T1",), "a": ("a",), "cpca": ("T3", "cPCA2D"), "local": ("local",)}
RUNS = tuple(r for runs in PROCS.values() for r in runs)
RECHECK = {"selection": 0, "confirmation": 3}  # seed of the reruns (selection at w = 0.25, confirmation at w*)
R_A_MIN, T1_MAX = 0.80, 0.60
N_EPOCHS = 750  # single-stage: 250 + 500
N_CPCA = 10  # T3's components and local cPCA's k
N_CLASSES = 2
K = 15
METRICS = ("digit_acc", "recall_digits", "recall_grass", "ari")
FIELDS = ["stage", "seed", "w", "run", "rep", "applicable", *METRICS, "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (opacity): {msg}", flush=True)
    sys.exit(2)


def wtag(w):
    return f"w{round(w * 100):03d}"


def name(stage, seed, w, run, rep=0):
    return f"{stage}_seed{seed}_{wtag(w)}_{run}" + (f"_rep{rep}" if rep else "")


def load(seed, w):
    """X_T(w), X_B and the evaluation truth; stops unless X_T(0.25) reproduces the stored target bit for bit."""
    from scripts.bsgraph_phase1 import load as load01
    X_T, X_B, t = load01(seed)

    def rebuild(v):
        return (v * t["digits"] + t["grass_T"]).astype(np.float32)

    if not np.array_equal(rebuild(W_STORED), X_T):
        stop(f"seed {seed}: X_T({W_STORED}) does not reproduce the stored target")
    return rebuild(w), X_B, t


def sha(E):
    return hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()


def metrics(E, t, seed):
    from eval.metrics import kmeans_ari, knn_accuracy, knn_recall
    return {"digit_acc": knn_accuracy(E, t["y"], k=K), "recall_digits": knn_recall(E, t["digits"], k=K),
            "recall_grass": knn_recall(E, t["grass_T"], k=K), "ari": kmeans_ari(E, t["y"], N_CLASSES, seed)}


def neg_tsne(X, seed, graph=None):
    """Single-stage Neg-t-SNE as single_stage_run.fit("single"), on the raw kNN graph of X unless a graph is given."""
    from common.embed import fit_neg, pca_init
    from common.knn import knn_graph
    from method.bg_term import BackgroundTerm
    f = fit_neg(pca_init(X), knn_graph(X, k=K) if graph is None else graph, seed, batch_hook=BackgroundTerm("none"),
                n_epochs=N_EPOCHS)
    return f["embd"], {"embd_std": float(f["scale"]["embd_std"][-1]), "params": f["params"]}


def local_cpca(X_T, X_B, seed):
    """Label-free local cPCA (k = 10) embedded as run a, or None with the reason when a cluster is too small."""
    from baselines import local_cpca as lc
    lab_B, centers, info = lc.select_clusters(X_B, seed)
    lab_T = lc.assign(X_T, centers)
    small = lc.check_sizes(lab_T, lab_B)
    sizes = {str(j): [int((lab_T == j).sum()), int((lab_B == j).sum())] for j in np.unique(lab_B)}
    info = {**info, "cluster_sizes_T_B": sizes}
    if small:
        return None, {**info, "too_small": small}
    per = lc.local_alphas(X_T, X_B, lab_T, lab_B, seed)
    Z, rep = lc.representation(X_T, lab_T, per, N_CPCA)
    E, extra = neg_tsne(Z.astype(np.float32), seed)
    return E, {**info, **rep, **extra, "alphas": {str(j): p["alpha"] for j, p in per.items()}}


def embed(proc, seed, X_T, X_B):
    """{run: (embedding or None, run-specific fields)} for one process."""
    if proc == "r_a":
        from scripts.single_stage_run import fit
        E, _, params, std, _ = fit("single", X_T, X_B, seed)
        return {"r_a": (E, {"embd_std": std, "params": params})}
    if proc == "a":
        return {"a": neg_tsne(X_T, seed)}
    if proc == "T1":
        from scripts.mnist_grass_run import tsne
        return {"T1": (tsne(X_T, seed), {})}
    if proc == "cpca":
        from baselines import cpca
        from scripts.mnist_grass_run import tsne
        f = cpca.fit(X_T, X_B)
        alpha, reps, ratios, _ = cpca.select_alpha(f, seed)
        info = {"alpha": alpha, "representatives": reps, "variance_ratios": ratios}
        return {"cPCA2D": (cpca.project(f, alpha, 2), info), "T3": (tsne(cpca.project(f, alpha, N_CPCA), seed), info)}
    if proc == "local":
        return {"local": local_cpca(X_T, X_B, seed)}
    raise ValueError(f"unknown process {proc!r}")


def run_one(stage, seed, w, proc, rep):
    assert_committed(PREREG)
    if stage == "confirmation":
        assert_committed(SELECTION)
    setup_determinism(seed)
    X_T, X_B, t = load(seed, w)
    t0 = time.perf_counter()
    out = embed(proc, seed, X_T, X_B)
    runtime = time.perf_counter() - t0  # the whole process: shared by the runs of one process
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for run, (E, extra) in out.items():
        rec = {"stage": stage, "seed": seed, "w": w, "run": run, "rep": rep, "applicable": E is not None}
        if E is not None:
            if not np.isfinite(E).all():
                stop(f"{name(stage, seed, w, run, rep)}: non-finite embedding")
            rec.update({**metrics(E, t, seed), "sha256": sha(E)})
            np.save(RUNS_DIR / f"{name(stage, seed, w, run, rep)}.npy", E)
        rec.update({"runtime_s": runtime, **extra, "environment": environment_info()})
        (RUNS_DIR / f"{name(stage, seed, w, run, rep)}.json").write_text(json.dumps(rec, indent=2, default=str))
        print(json.dumps({k: v for k, v in rec.items() if k in FIELDS}), flush=True)


def make_jobs(stage):
    cmd = "$PY scripts/opacity_run.py --stage {} --seed {} --w {} --proc {}"
    if stage == "smoke":
        lines = [cmd.format("selection", RECHECK["selection"], W_STORED, p) for p in PROCS]
    else:
        lines = [cmd.format(stage, s, w, p) for w in WS for s in SEEDS[stage] for p in PROCS]
        w_re = W_STORED if stage == "selection" else json.loads(SELECTION.read_text())["w_star"]
        if w_re is not None:
            lines += [cmd.format(stage, RECHECK[stage], w_re, p) + " --rep 1" for p in PROCS]
    path = ROOT / "cluster" / "jobs" / f"{NAME}_{stage}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def read(stage, seed, w, run, rep=0):
    p = RUNS_DIR / f"{name(stage, seed, w, run, rep)}.json"
    if not p.exists():
        stop(f"missing {p.name}")
    return json.loads(p.read_text())


def rerun_check(stage, w):
    """Each run of the rerun (seed, w): applicability and SHA-256 equal to the main run's."""
    seed = RECHECK[stage]
    out = {}
    for run in RUNS:
        a, b = read(stage, seed, w, run), read(stage, seed, w, run, rep=1)
        out[run] = a["applicable"] == b["applicable"] and a.get("sha256") == b.get("sha256")
    return {"seed": seed, "w": w, "identical": out, "all_identical": all(out.values())}


def means(rows_, run):
    return {w: float(np.mean([r["digit_acc"] for r in rows_ if r["run"] == run and r["w"] == w])) for w in WS}


def merge(stage):
    rows_ = [read(stage, s, w, run) for w in WS for s in SEEDS[stage] for run in RUNS]
    with open(OUT / f"runs_{stage}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_)
    w_re = W_STORED if stage == "selection" else json.loads(SELECTION.read_text())["w_star"]
    check = rerun_check(stage, w_re) if w_re is not None else None
    (OUT / f"recheck_{stage}.json").write_text(json.dumps({**(check or {}), "environment": environment_info()},
                                                          indent=2))
    print(json.dumps(check))
    if check and not check["all_identical"]:
        stop(f"{stage}: a rerun is not bit-identical")
    if stage == "selection":
        r_a, t1 = means(rows_, "r_a"), means(rows_, "T1")
        window = [w for w in WS if r_a[w] >= R_A_MIN and t1[w] <= T1_MAX]
        gap = {w: r_a[w] - t1[w] for w in WS}
        sel = {"metric": "digit kNN accuracy", "window": window, "w_star": max(window) if window else None,
               "rule": f"largest w with mean r_a >= {R_A_MIN} and mean T1 <= {T1_MAX} over seeds 0-2",
               "mean_r_a": {str(w): v for w, v in r_a.items()}, "mean_T1": {str(w): v for w, v in t1.items()},
               "largest_gap_w": max(WS, key=lambda w: (gap[w], w)), "environment": environment_info()}
        SELECTION.write_text(json.dumps(sel, indent=2))
        print(json.dumps({k: sel[k] for k in ("window", "w_star", "largest_gap_w")}))
    print(f"merged {len(rows_)} runs -> runs_{stage}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(SEEDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--w", type=float, choices=WS)
    parser.add_argument("--proc", choices=list(PROCS))
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--make-jobs", choices=[*SEEDS, "smoke"])
    parser.add_argument("--merge", choices=list(SEEDS))
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs(args.make_jobs)
    if args.merge:
        return merge(args.merge)
    run_one(args.stage, args.seed, args.w, args.proc, args.rep)


if __name__ == "__main__":
    main()
