"""Phase B of the MNIST-on-grass plan: validate the cluster on a recorded synthetic result (no MNIST involved).

  --part ref --rep {1,2}    L-cyclic confirmation seed 3 exactly as in scripts/phase5b_run.py: Stage 1, then (a)
                            and (c) m_B = 20. Run twice (two processes); the two must be bit-identical (the gate).
  --part opt --opt-seed S   (c) m_B = 20 on the same seed-3 dataset with optimization seed S (embedding, Stage 1/2 and
                            xi_B tie-splitting), for the cluster's optimization-seed spread.
  --summarize               gate verdict, and (descriptive) the Mac-vs-cluster difference next to the cluster's
                            optimization-seed spread, the Mac's seed-to-seed spread (seeds 3-5) and the smallest
                            per-seed margin among the Phase 5b verdicts (results/phase5b/comparisons.json).

Outputs: results/mnist_grass/validation/ (embeddings gitignored). Run on the cluster with 4 threads, as on the Mac:
sbatch cluster/mnist_grass_validate.sbatch
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "mnist_grass" / "validation"
P5B = ROOT / "results" / "phase5b"
SEED = 3
RUNS = ("a", "c_m20")
OPT_SEEDS = (1000, 1001, 1002)


def embed_and_score(ctx, labels):
    from common.benchmark_l import score
    from common.experiment import run_stage1, run_stage2
    s1 = run_stage1(ctx)
    out, embds = {"stage1_sha256": hashlib.sha256(s1["embd"].tobytes()).hexdigest()}, {"stage1": s1["embd"]}
    for label in labels:
        E = run_stage2(ctx, s1, label)["embd"]
        sc = score(E, ctx.ds.truth)
        out[label] = {"WC": sc["WC"], "acc_s": sc["acc_s"], "sha256": hashlib.sha256(E.tobytes()).hexdigest()}
        embds[label] = E
    return out, embds


def run_part(name, ctx, labels):
    from common.repro import environment_info
    rec, embds = embed_and_score(ctx, labels)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"environment": environment_info(), **rec}, indent=2))
    for label, e in embds.items():
        np.save(OUT / f"{name}_{label}.npy", e)
    print(json.dumps(rec), flush=True)


def ref(rep):
    import scripts.phase5b_run as r
    cal = json.loads(r.CAL.read_text())
    run_part(f"ref_rep{rep}", r.dataset("cyclic", SEED, cal), RUNS)


def opt(opt_seed):
    import scripts.phase5b_run as r
    from common.experiment import prepare_dataset
    from common.repro import setup_determinism
    from data.synthetic_l import make_l
    from method.bg_term import same_cluster_xi
    cal = json.loads(r.CAL.read_text())
    setup_determinism(opt_seed)
    ds = make_l("cyclic", SEED, cal["rho"]["cyclic"]["rho"])  # the seed-3 dataset (its own seed streams)
    ctx = prepare_dataset(ds, "L-cyclic", opt_seed, oracle_xi=same_cluster_xi(ds.truth["y_b"]))
    run_part(f"opt_seed{opt_seed}", ctx, ("c_m20",))


def smallest_5b_margin():
    """Smallest |per-seed WC difference| over every Phase 5b paired comparison (pre-registered and local extension)."""
    res = json.loads((P5B / "comparisons.json").read_text())
    margins = []
    for variant, r in res.items():
        comps = r["preregistered"] + ((r.get("local") or {}).get("comparisons") or [])
        margins += [(abs(d), variant, c["y"]) for c in comps for d in c["per_seed"]]
    return min(margins)


def summarize():
    from common.repro import environment_info
    rec = {p.stem: json.loads(p.read_text()) for p in OUT.glob("*.json") if p.stem != "validation"}
    r1, r2 = rec["ref_rep1"], rec["ref_rep2"]
    labels = ("stage1",) + RUNS
    identical = {k: bool(np.array_equal(np.load(OUT / f"ref_rep1_{k}.npy"), np.load(OUT / f"ref_rep2_{k}.npy")))
                 for k in labels}
    gate = all(identical.values()) and all(r1[k] == r2[k] for k in RUNS)
    with open(P5B / "runs_cyclic_confirmation.csv") as f:
        mac = [r for r in csv.DictReader(f) if r["run"] in RUNS]
    rows = []
    for run in RUNS:
        for m in ("WC", "acc_s"):
            mac3 = float(next(r[m] for r in mac if r["run"] == run and r["seed"] == str(SEED)))
            mac_seeds = [float(r[m]) for r in mac if r["run"] == run]
            opt_vals = [rec[f"opt_seed{s}"][run][m] for s in OPT_SEEDS] if run == "c_m20" else []
            rows.append({"run": run, "metric": m, "cluster": r1[run][m], "mac": mac3, "diff": r1[run][m] - mac3,
                         "cluster_opt_seed_range": [min(opt_vals), max(opt_vals)] if opt_vals else None,
                         "mac_seed_range_3_5": [min(mac_seeds), max(mac_seeds)]})
    margin, m_variant, m_y = smallest_5b_margin()
    max_diff = max(abs(r["diff"]) for r in rows if r["metric"] == "WC")
    result = {"environment": environment_info(), "gate": "cluster run twice is bit-identical (rows and embeddings)",
              "gate_passed": gate, "embeddings_identical": identical,
              "why": "x86 cluster vs ARM Mac (different BLAS and SIMD): Mac bit-identity is impossible, so the "
                     "Mac-vs-cluster difference is descriptive",
              "comparison": rows, "max_abs_WC_diff_mac_vs_cluster": max_diff,
              "smallest_phase5b_per_seed_margin": {"abs_WC_diff": margin, "variant": m_variant, "vs": m_y},
              "diff_below_smallest_margin": max_diff < margin}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    md = ["| run | metric | cluster | Mac | cluster − Mac | cluster, optimization seeds 1000–1002 "
          "| Mac, seeds 3–5 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rng = (lambda x: f"{x[0]:.3f}–{x[1]:.3f}" if x else "—")
        md.append(f"| {r['run']} | {r['metric']} | {r['cluster']:.3f} | {r['mac']:.3f} | {r['diff']:+.3f} "
                  f"| {rng(r['cluster_opt_seed_range'])} | {rng(r['mac_seed_range_3_5'])} |")
    md += ["", f"Gate (cluster run twice, bit-identical): **{'passed' if gate else 'FAILED'}**. "
           f"Largest |Mac − cluster| WC difference {max_diff:.3f}; smallest per-seed margin among the Phase 5b "
           f"verdicts {margin:.3f} (L-{m_variant}, (c) vs {m_y})."]
    (OUT / "validation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if not gate:
        print("STOP (MNIST-on-grass Phase B): the two cluster runs are not bit-identical", flush=True)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["ref", "opt"])
    parser.add_argument("--rep", type=int)
    parser.add_argument("--opt-seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        return summarize()
    ref(args.rep) if args.part == "ref" else opt(args.opt_seed)


if __name__ == "__main__":
    main()
