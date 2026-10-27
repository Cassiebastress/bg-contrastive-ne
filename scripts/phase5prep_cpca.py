"""Phase 5 prep, Part B (Phase_5_Plan.md §2): the cPCA baseline on every Phase 4 dataset.

Per dataset (easy and medium seeds 0-5; hard seeds 0-2 and 4-6): cPCA fit, the package's automatic alpha
representatives, the label-free selected alpha (largest target-to-background variance ratio), and the oracle alpha
chosen by 10-D cPCA signal accuracy (labeled oracle everywhere). Runs at every representative alpha:
  (e1) the 2-D cPCA projection; (e2) top-10 cPCA -> the two-stage Neg-t-SNE of run (a); (e3) top-10 cPCA -> UMAP.
Plus (e2-oracle) at the oracle alpha, and Phase 4's (a) and (c) at m*_c reloaded from stored embeddings.
Checks: our cPCA vs the reference `contrastive` package on easy seed 0 (subspaces and representative alphas), and
bit-identical reruns of (e2) and (e3) on easy seed 0.

Run with the cne-bg env:  caffeinate -i python scripts/phase5prep_cpca.py
Outputs: results/phase5prep/cpca_runs.csv, alphas.csv, cpca_check.json (embeddings are gitignored .npy files).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.linalg import subspace_angles

from baselines import cpca
from baselines.cpca_embed import embed_e2, embed_e3
from common.repro import environment_info, setup_determinism
from data.synthetic import LEVELS, MAIN_RHO, make_dataset
from eval.global_metrics import global_scores
from eval.metrics import knn_accuracy, knn_recall

P4 = ROOT / "results" / "phase4"
OUT = ROOT / "results" / "phase5prep"
EMB = OUT / "embeddings"
ANGLE_TOL = 1e-6
FIELDS = ["level", "seed", "stage", "run", "alpha", "selected", "acc_s", "recall_s", "G1", "G2", "G3_shares_signal",
          "G3_shares_background", "G3_neither", "runtime_s"]
ALPHA_FIELDS = ["level", "seed", "stage", "selected_alpha", "representatives", "variance_ratios", "oracle_alpha",
                "oracle_acc_10d", "selected_acc_10d"]


def seed_sets(level):
    conf = json.loads((P4 / f"summary_{level}_confirmation.json").read_text())
    return [("selection", s) for s in (0, 1, 2)] + [("confirmation", s) for s in conf["seeds"]]


def score(X, t):
    g = global_scores(X, t["y_s"], t["y_b"])
    return {"acc_s": knn_accuracy(X, t["y_s"]), "recall_s": knn_recall(X, t["S_T"]), "G1": g["G1"], "G2": g["G2"],
            **{k: g[k] for k in ("G3_shares_signal", "G3_shares_background", "G3_neither")}}


def emb_path(level, seed, run, alpha):
    return EMB / f"{level}_seed{seed}_{run}_alpha{alpha:.6g}.npy"


def verify_package():
    """Our cPCA vs the reference package (standardize=False) on easy seed 0."""
    import contrastive
    ds = make_dataset("easy", seed=0, signal_to_background=MAIN_RHO["easy"])
    f = cpca.fit(ds.X_T, ds.X_B)
    pkg = contrastive.CPCA(n_components=2, standardize=False)
    pkg.fit(ds.X_T.astype(np.float64), ds.X_B.astype(np.float64))
    angles = [float(subspace_angles(cpca.project(f, a, 2), np.real(pkg.cpca_alpha(dataset=pkg.fg, alpha=a))).max())
              for a in cpca.ALPHAS]
    np.random.seed(0)  # the package's spectral clustering draws from numpy's global generator
    best, _, _, _ = pkg.find_spectral_alphas(40, 3, cpca.N_CLUSTERS)
    reps_pkg = np.concatenate(([0.0], best))
    reps_ours, _ = cpca.representative_alphas(f, seed=0)
    out = {"package": "contrastive", "version": contrastive.__dict__.get("__version__", "1.2.0"),
           "max_principal_angle_rad": max(angles), "tolerance_rad": ANGLE_TOL,
           "representatives_package": reps_pkg.tolist(), "representatives_ours": reps_ours.tolist(),
           "affinity_max_abs_diff": float(np.abs(pkg.affinity_matrix - cpca.affinity_matrix(f)).max())}
    out["agrees"] = max(angles) <= ANGLE_TOL and np.allclose(reps_pkg, reps_ours)
    return out


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    check = {"environment": environment_info(), "package_check": verify_package()}
    print(json.dumps(check["package_check"], indent=2), flush=True)
    if not check["package_check"]["agrees"]:
        sys.exit("STOP: the reference package disagrees with our cPCA beyond tolerance")

    rows, alpha_rows = [], []
    for level in LEVELS:
        sel = json.loads((P4 / f"selection_{level}.json").read_text())
        c_run = f"c_m{sel['m_star_c']}"
        for stage, seed in seed_sets(level):
            setup_determinism(seed)
            ds = make_dataset(level, seed=seed, signal_to_background=MAIN_RHO[level])
            t = ds.truth
            f = cpca.fit(ds.X_T, ds.X_B)
            chosen, reps, ratios, _ = cpca.select_alpha(f, seed)
            if len(reps) < 2:
                sys.exit(f"STOP: {level} seed {seed}: no usable alpha representative besides 0")
            acc10 = [knn_accuracy(cpca.project(f, a, 10), t["y_s"]) for a in cpca.ALPHAS]
            oracle = float(cpca.ALPHAS[int(np.argmax(acc10))])  # first maximum: ties go to the smaller alpha
            alpha_rows.append({"level": level, "seed": seed, "stage": stage, "selected_alpha": chosen,
                               "representatives": ";".join(f"{a:.6g}" for a in reps),
                               "variance_ratios": ";".join(f"{r:.6g}" for r in ratios), "oracle_alpha": oracle,
                               "oracle_acc_10d": max(acc10),
                               "selected_acc_10d": acc10[int(np.flatnonzero(cpca.ALPHAS == chosen)[0])]})

            def add(run, alpha, selected, X, t0):
                rows.append({"level": level, "seed": seed, "stage": stage, "run": run, "alpha": alpha,
                             "selected": selected, **score(X, t), "runtime_s": time.perf_counter() - t0})

            e2_by_alpha = {}
            for a in reps:
                is_sel = a == chosen
                t0 = time.perf_counter()
                Z2 = cpca.project(f, a, 2)
                add("e1", a, is_sel, Z2, t0)
                np.save(emb_path(level, seed, "e1", a), Z2.astype(np.float32))
                Z10 = cpca.project(f, a, 10)
                t0 = time.perf_counter()
                e2 = embed_e2(Z10, seed)
                add("e2", a, is_sel, e2, t0)
                np.save(emb_path(level, seed, "e2", a), e2)
                e2_by_alpha[a] = e2
                t0 = time.perf_counter()
                e3 = embed_e3(Z10, seed)
                add("e3", a, is_sel, e3, t0)
                np.save(emb_path(level, seed, "e3", a), e3)
            t0 = time.perf_counter()
            e2o = e2_by_alpha[oracle] if oracle in e2_by_alpha else embed_e2(cpca.project(f, oracle, 10), seed)
            add("e2_oracle", oracle, False, e2o, t0)
            np.save(emb_path(level, seed, "e2_oracle", oracle), e2o)
            for run, label in (("a", "a"), ("c", c_run)):
                e = np.load(P4 / "embeddings" / f"{level}_{stage}_seed{seed}_{label}.npy")
                add(run, "", False, e, time.perf_counter())
            r = [x for x in rows if x["level"] == level and x["seed"] == seed and x["selected"] is True]
            summary = " ".join(f"{x['run']} acc_s {x['acc_s']:.3f} G1 {x['G1']:.3f}" for x in r)
            print(f"{level} {stage} seed={seed} alpha={chosen:.4g} reps={[round(a, 3) for a in reps]} "
                  f"oracle={oracle:.4g} | {summary}", flush=True)
            with open(OUT / "cpca_runs.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with open(OUT / "alphas.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ALPHA_FIELDS)
                writer.writeheader()
                writer.writerows(alpha_rows)

    # determinism: rerun (e2) and (e3) on easy seed 0 at the selected alpha
    setup_determinism(0)
    ds = make_dataset("easy", seed=0, signal_to_background=MAIN_RHO["easy"])
    f = cpca.fit(ds.X_T, ds.X_B)
    chosen = cpca.select_alpha(f, 0)[0]
    Z10 = cpca.project(f, chosen, 10)
    check["determinism"] = {
        "e2_bit_identical": bool(np.array_equal(embed_e2(Z10, 0), np.load(emb_path("easy", 0, "e2", chosen)))),
        "e3_bit_identical": bool(np.array_equal(embed_e3(Z10, 0), np.load(emb_path("easy", 0, "e3", chosen))))}
    (OUT / "cpca_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check["determinism"], indent=2))
    if not all(check["determinism"].values()):
        sys.exit("STOP: cPCA baseline runs are not bit-reproducible")


if __name__ == "__main__":
    main()
