"""Single-cell RNA-seq, Zheng et al. 2017 AML027 (results/scrna/preregistration.md).

  --precheck
        the pre-checks: label kNN accuracy in high dimension (500 genes, PCA-50, cPCA-10, residuals), the label-free
        alpha on every seed, the background-retention reference values, and the replication of the paper's figure
        (its preprocessing; cPCA-2D at its alphas and at ours). Writes prechecks.json and prechecks.md, and the
        replication projections to runs/replication.npz (gitignored)
  --seed S --proc P [--rep N]
        P is r_a (BC-NE, scripts/single_stage_run.fit("single")), a (Neg-t-SNE on the raw kNN graph), T1 (openTSNE),
        T2 (PCA-50 -> openTSNE), U2 (PCA-50 -> UMAP), cpca (one cPCA fit at the label-free alpha: T3 = top 10 ->
        openTSNE, cPCA2D and PCA2D = 2-D projections), bg12_r_a / bg12_cpca (as r_a / cpca with background donors 1 + 2;
        runs r_a_bg12, T3_bg12, cPCA2D_bg12), sens_k{K}_r{R} (BC-NE at k_loc = K, r = R) or oracle (cPCA-2D and T3 at
        every alpha of cpca.ALPHAS; metrics only). Writes runs/seed{S}_{run}[_rep{N}].json and the embedding .npy
  --make-jobs precheck|smoke|main
        cluster/jobs/scrna_{precheck,smoke,main}.txt (precheck: the pre-check alone, run before any embedding; smoke:
        seed 0 of each main process and one sensitivity process; main: every (seed, process) and the seed-0 reruns)
  --merge
        runs.csv and oracle.csv after the completeness and rerun checks; recheck.json
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
from scripts.bsgraph_run import K, K_LOC, R  # noqa: E402

NAME = "scrna"
OUT = ROOT / "results" / NAME
RUNS_DIR = OUT / "runs"
PREREG = OUT / "preregistration.md"
SEEDS = tuple(range(10))
N_EPOCHS = 750  # single-stage: 250 + 500
N_CPCA, N_PCA = 10, 50
N_CLASSES = 2
K_B = 15  # background neighbors of the retention metric
SENS = [(k, r) for k in (10, 30, 60) for r in (0, 2, 5) if (k, r) != (K_LOC, R)]
PAPER_ALPHAS = (0.0, 3.45510729459, 28.9426612472, 388.815518031)  # the paper notebook's 2-group output
MAIN = ("r_a", "a", "T1", "T2", "U2", "cpca", "bg12_r_a", "bg12_cpca")
PROCS = MAIN + tuple(f"sens_k{k}_r{r}" for k, r in SENS) + ("oracle",)
RUNS_OF = {"cpca": ("T3", "cPCA2D", "PCA2D"), "bg12_r_a": ("r_a_bg12",), "bg12_cpca": ("T3_bg12", "cPCA2D_bg12")}
METRICS = ("label_acc", "ari", "bg_retention")
FIELDS = ["seed", "run", "rep", *METRICS, "n_components", "frac_in_small_components", "flag_fragment", "alpha",
          "runtime_s", "sha256"]


def stop(msg):
    print(f"STOP (scrna): {msg}", flush=True)
    sys.exit(2)


def runs_of(proc):
    return RUNS_OF.get(proc, (proc,))


def name(seed, run, rep=0):
    return f"seed{seed}_{run}" + (f"_rep{rep}" if rep else "")


def load_data():
    """The processed dataset after checking its SHA-256 against data/MANIFEST_scrna.md."""
    from data import scrna
    path = scrna.processed_path(ROOT)
    want = re.search(rf"`{path.name}` \| `([0-9a-f]{{64}})`", scrna.manifest_path(ROOT).read_text())
    if want is None or scrna.sha256(path) != want.group(1):
        stop(f"{path.name} does not match its SHA-256 in data/MANIFEST_scrna.md")
    return scrna.load(path)


def sha(E):
    return hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()


def overlaps_of(d):
    """|N_B(i) ∩ N_B(j)| over target pairs, with N_B the K_B nearest donor-1 background cells (500-gene space)."""
    from data.scrna import for_embedding
    from method.xi_b import background_neighbor_overlaps
    X_T, X_B, _ = for_embedding(d)
    return background_neighbor_overlaps(X_T, X_B, K_B)


def metrics(E, y, overlaps, seed):
    from common.experiment import FRAGMENT_FRAC, FRAGMENT_SMALL
    from eval.metrics import background_retention, component_stats, kmeans_ari, knn_accuracy
    comp = component_stats(E, k=K, small=FRAGMENT_SMALL)
    return {"label_acc": knn_accuracy(E, y, k=K), "ari": kmeans_ari(E, y, N_CLASSES, seed),
            "bg_retention": background_retention(E, overlaps, K_B, k=K), **comp,
            "flag_fragment": comp["frac_in_small_components"] > FRAGMENT_FRAC}


def pca(X, n):
    from sklearn.decomposition import PCA
    return PCA(n_components=n, svd_solver="full").fit_transform(np.asarray(X, dtype=np.float64))


def neg_tsne(X, seed, graph=None):
    """Single-stage Neg-t-SNE as single_stage_run.fit("single"), on the raw kNN graph of X unless a graph is given."""
    from common.embed import fit_neg, pca_init
    from common.knn import knn_graph
    from method.bg_term import BackgroundTerm
    f = fit_neg(pca_init(X), knn_graph(X, k=K) if graph is None else graph, seed, batch_hook=BackgroundTerm("none"),
                n_epochs=N_EPOCHS)
    return f["embd"], {"embd_std": float(f["scale"]["embd_std"][-1]), "params": f["params"]}


def cpca_runs(X_T, X_B, seed, suffix="", pca2d=True):
    from baselines import cpca
    from scripts.mnist_grass_run import tsne
    f = cpca.fit(X_T, X_B)
    alpha, reps, ratios, _ = cpca.select_alpha(f, seed)
    info = {"alpha": alpha, "representatives": reps, "variance_ratios": ratios}
    out = {f"cPCA2D{suffix}": (cpca.project(f, alpha, 2), info),
           f"T3{suffix}": (tsne(cpca.project(f, alpha, N_CPCA), seed), info)}
    if pca2d:
        out["PCA2D"] = (cpca.project(f, 0.0, 2), {})
    return out


def embed(proc, seed, d):
    """{run: (embedding, run-specific fields)} for one process."""
    from data.scrna import for_embedding
    X_T, X_B, _ = for_embedding(d)
    if proc in ("r_a", "bg12_r_a"):
        from scripts.single_stage_run import fit
        if proc == "bg12_r_a":
            X_B = for_embedding(d, background="donors12")[1]
        E, _, params, std, _ = fit("single", X_T, X_B, seed)
        return {runs_of(proc)[0]: (E, {"embd_std": std, "params": params, "n_B": len(X_B)})}
    if proc == "a":
        return {"a": neg_tsne(X_T, seed)}
    if proc in ("T1", "T2"):
        from scripts.mnist_grass_run import tsne
        return {proc: (tsne(X_T if proc == "T1" else pca(X_T, N_PCA), seed), {})}
    if proc == "U2":
        import umap
        return {"U2": (umap.UMAP(random_state=seed).fit_transform(pca(X_T, N_PCA).astype(np.float32)), {})}
    if proc == "cpca":
        return cpca_runs(X_T, X_B, seed)
    if proc == "bg12_cpca":
        return cpca_runs(X_T, for_embedding(d, background="donors12")[1], seed, suffix="_bg12", pca2d=False)
    m = re.fullmatch(r"sens_k(\d+)_r(\d+)", proc)
    if m:
        from method.residual_graph import residual_graph
        k_loc, r = int(m.group(1)), int(m.group(2))
        G, _ = residual_graph(X_T, X_B, k_loc, r, k=K)
        E, extra = neg_tsne(X_T, seed, graph=G)
        return {proc: (E, {**extra, "k_loc": k_loc, "r": r})}
    raise ValueError(f"unknown process {proc!r}")


def oracle(seed, d, overlaps):
    """cPCA-2D and T3 at every alpha of cpca.ALPHAS: metrics only (labels pick the best alpha later, descriptively)."""
    from baselines import cpca
    from data.scrna import for_embedding
    from scripts.mnist_grass_run import tsne
    X_T, X_B, t = for_embedding(d)
    f = cpca.fit(X_T, X_B)
    rows = []
    for alpha in cpca.ALPHAS:
        for run, E in (("cPCA2D", cpca.project(f, alpha, 2)), ("T3", tsne(cpca.project(f, alpha, N_CPCA), seed))):
            m = metrics(E, t["y"], overlaps, seed)
            rows.append({"seed": seed, "run": run, "alpha": float(alpha), **{k: m[k] for k in METRICS}})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def run_one(seed, proc, rep):
    assert_committed(PREREG)
    setup_determinism(seed)
    d = load_data()
    overlaps = overlaps_of(d)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if proc == "oracle":
        rows = oracle(seed, d, overlaps)
        (RUNS_DIR / f"{name(seed, 'oracle', rep)}.json").write_text(json.dumps(
            {"seed": seed, "rows": rows, "runtime_s": time.perf_counter() - t0, "environment": environment_info()},
            indent=2))
        return
    out = embed(proc, seed, d)
    runtime = time.perf_counter() - t0  # the whole process: shared by the runs of one process
    for run, (E, extra) in out.items():
        if not np.isfinite(E).all():
            stop(f"{name(seed, run, rep)}: non-finite embedding")
        rec = {"seed": seed, "run": run, "rep": rep, **metrics(E, d["y"], overlaps, seed), "runtime_s": runtime,
               "sha256": sha(E), **extra, "environment": environment_info()}
        np.save(RUNS_DIR / f"{name(seed, run, rep)}.npy", E)
        (RUNS_DIR / f"{name(seed, run, rep)}.json").write_text(json.dumps(rec, indent=2, default=str))
        print(json.dumps({k: v for k, v in rec.items() if k in FIELDS}), flush=True)


def precheck():
    """Descriptive pre-checks, before any embedding (results/scrna/preregistration.md, Pre-checks)."""
    from baselines import cpca
    from data.scrna import for_embedding
    from eval.metrics import background_retention, knn_accuracy, mean_pair_background_jaccard
    from method.residual_graph import residual_graph
    assert_committed(PREREG)
    setup_determinism(0)
    d = load_data()
    X_T, X_B, t = for_embedding(d)
    y = t["y"]
    overlaps = overlaps_of(d)
    f = cpca.fit(X_T, X_B)
    alphas = {s: cpca.select_alpha(f, s) for s in SEEDS}
    alpha0 = alphas[0][0]
    _, Rsd = residual_graph(X_T, X_B, K_LOC, R, k=K)
    spaces = {"500 genes": X_T, "PCA-50": pca(X_T, N_PCA),
              f"cPCA-10 (alpha {alpha0:.3g})": cpca.project(f, alpha0, N_CPCA),
              f"residuals (k_loc {K_LOC}, r {R})": Rsd}
    acc = {k: knn_accuracy(np.asarray(v, dtype=np.float64), y, k=K) for k, v in spaces.items()}
    retention = {"all pairs": mean_pair_background_jaccard(overlaps, K_B),
                 "500-gene kNN graph": background_retention(X_T, overlaps, K_B, k=K),
                 "residual kNN graph": background_retention(Rsd, overlaps, K_B, k=K)}
    # replication: the paper's preprocessing, cPCA-2D at its alphas and at the label-free alpha on that input
    fp = cpca.fit(d["X_T_paper"], d["X_B_paper"])
    alpha_p = cpca.select_alpha(fp, 0)[0]
    proj = {f"{a:.6g}": cpca.project(fp, a, 2) for a in (*PAPER_ALPHAS, alpha_p)}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS_DIR / "replication.npz", **proj)
    replication = {"label_free_alpha": alpha_p, "label_acc": {a: knn_accuracy(P, y, k=K) for a, P in proj.items()}}
    counts = {lab: float(np.median(d["log10_total_T"][y == i])) for i, lab in enumerate(("pre", "post"))}
    res = {"cells": {"pre": int((y == 0).sum()), "post": int((y == 1).sum()), "background": len(X_B)},
           "median_log10_total_counts": {**counts, "background": float(np.median(d["log10_total_B"]))},
           "label_acc_high_dim": acc, "bg_retention_reference": retention,
           "label_free_alpha": {str(s): {"alpha": a[0], "representatives": a[1]} for s, a in alphas.items()},
           "replication": replication, "environment": environment_info()}
    (OUT / "prechecks.json").write_text(json.dumps(res, indent=2, default=str))
    md = ["# Pre-checks (descriptive, before any embedding)", "",
          f"Cells: pre {res['cells']['pre']}, post {res['cells']['post']}, background {res['cells']['background']}. "
          "Median log10 total counts: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                    res["median_log10_total_counts"].items()) + ".", "",
          "## Label kNN accuracy in high dimension (k = 15, stratified 5-fold)", "", "| space | accuracy |",
          "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in acc.items()] + [
          "", "## Background-structure retention: reference values", "", "| graph | mean Jaccard |", "|---|---|"] + [
          f"| {k} | {v:.3f} |" for k, v in retention.items()] + [
          "", "## Label-free α per seed", "", "| seed | α | representatives |", "|---|---|---|"] + [
          f"| {s} | {a[0]:.3g} | {', '.join(f'{r:.3g}' for r in a[1])} |" for s, a in alphas.items()] + [
          "", "## Replication: the paper's preprocessing, cPCA-2D", "",
          f"Label-free α on the paper's preprocessing: {alpha_p:.3g}.", "",
          "| α | label kNN accuracy |", "|---|---|"] + [
          f"| {a} | {v:.3f} |" for a, v in replication["label_acc"].items()]
    (OUT / "prechecks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def make_jobs(kind):
    cmd = "$PY scripts/scrna_run.py --seed {} --proc {}"
    if kind == "precheck":
        lines = ["$PY scripts/scrna_run.py --precheck"]
    elif kind == "smoke":
        lines = [cmd.format(0, p) for p in (*MAIN, f"sens_k{SENS[0][0]}_r{SENS[0][1]}")]
    else:
        lines = [cmd.format(s, p) for s in SEEDS for p in PROCS]
        lines += [cmd.format(0, p) + " --rep 1" for p in PROCS if p != "oracle"]
    path = ROOT / "cluster" / "jobs" / f"{NAME}_{kind}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} commands -> {path.relative_to(ROOT)}")


def read(seed, run, rep=0):
    p = RUNS_DIR / f"{name(seed, run, rep)}.json"
    if not p.exists():
        stop(f"missing {p.name}")
    return json.loads(p.read_text())


def merge():
    runs = [run for p in PROCS if p != "oracle" for run in runs_of(p)]
    rows = [read(s, run) for s in SEEDS for run in runs]
    with open(OUT / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    oracle_rows = [r for s in SEEDS for r in read(s, "oracle")["rows"]]
    with open(OUT / "oracle.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "run", "alpha", *METRICS])
        writer.writeheader()
        writer.writerows(oracle_rows)
    identical = {run: read(0, run)["sha256"] == read(0, run, rep=1)["sha256"] for run in runs}
    (OUT / "recheck.json").write_text(json.dumps({"seed": 0, "identical": identical,
                                                  "all_identical": all(identical.values()),
                                                  "environment": environment_info()}, indent=2))
    print(json.dumps(identical))
    if not all(identical.values()):
        stop("a rerun is not bit-identical")
    print(f"merged {len(rows)} runs and {len(oracle_rows)} alpha-grid rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--proc", choices=PROCS)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--precheck", action="store_true")
    parser.add_argument("--make-jobs", choices=["precheck", "smoke", "main"])
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.make_jobs:
        return make_jobs(args.make_jobs)
    if args.merge:
        return merge()
    if args.precheck:
        return precheck()
    run_one(args.seed, args.proc, args.rep)


if __name__ == "__main__":
    main()
