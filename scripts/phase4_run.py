"""Phase 4 main runs for one level (Plan §5).

  --stage selection     seeds 0-2: Stage 1 + all 12 Stage 2 runs [(a), (b) and (c) at m_B in {1,2,5,10,20}, (d)];
                        then the pre-registered selection rule -> results/phase4/selection_{level}.json
  --stage confirmation  fresh seeds from 3 (requires the committed selection): (a), (d), (c) at m*_c, (b) at m*_c,
                        and (b) at each per-metric best m_B. A seed whose dataset fails the validity check (judged
                        from (a) and the oracle only) is excluded and the next seed is drawn, until 3 valid seeds;
                        more than 2 exclusions stops the level.
  --recheck             determinism check: rerun selection seed 0 and compare with the saved rows and embeddings

Checks (Plan §6, stop and report on failure): dataset and xi_B hashes against Phases 1-2 (seeds 0-2); (a) reproduces
the Phase 1 accuracies exactly (seeds 0-2); validity criterion and xi_B gates on the fresh seeds 3-5; every run
finite with std growth <= 10x.

Run with the cne-bg env, keeping the machine awake:
  caffeinate -i python scripts/phase4_run.py --level easy --stage selection
Outputs go to results/phase4/ (embeddings and per-epoch logs are gitignored).
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

from common.experiment import (ROW_FIELDS, SELECTION_RUNS, VALIDITY, confirmation_runs, epoch_arrays, fit_row,
                               healthy, prepare, reference_row, run_stage1, run_stage2, select_settings)
from common.repro import assert_committed, environment_info
from data.synthetic import LEVELS, MAIN_RHO
from eval.metrics import knn_accuracy
from eval.xi_b_diag import diagnose

OUT = ROOT / "results" / "phase4"
SEEDS = {"selection": [0, 1, 2], "confirmation": [3, 4, 5]}
N_CONFIRM, FIRST_CONFIRM_SEED = 3, 3
MAX_EXCLUDED = 2  # Plan §5: more excluded confirmation seeds than this means rho is miscalibrated -> stop


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stop(msg):
    print(f"STOP (Plan §6): {msg}", flush=True)
    sys.exit(2)


def dataset_checks(ctx, stage):
    """Seeds 0-2: hashes must match Phases 1-2. Seeds 3-5: record hashes and high-D numbers, check the xi_B gates."""
    ds, level, seed = ctx.ds, ctx.level, ctx.seed
    hashes = {"sha256_X_T": hashlib.sha256(ds.X_T.tobytes()).hexdigest(),
              "sha256_X_B": hashlib.sha256(ds.X_B.tobytes()).hexdigest(), "xi_b": ctx.xi.sha256()}
    out = {"hashes": hashes}
    if stage == "selection":
        rec = next(r for r in read_csv(ROOT / "results/phase1/datasets.csv")
                   if r["set"] == "main" and r["level"] == level and int(r["seed"]) == seed)
        xi_rec = next(r for r in read_csv(ROOT / "results/phase2/xi_b_diagnostics.csv")
                      if r["level"] == level and int(r["seed"]) == seed)
        ok = (rec["sha256_X_T"] == hashes["sha256_X_T"] and rec["sha256_X_B"] == hashes["sha256_X_B"]
              and xi_rec["sha256"] == hashes["xi_b"])
        if not ok:
            stop(f"{level} seed {seed}: dataset or xi_B hash differs from Phases 1-2")
        out["hashes_match_phases_1_2"] = True
    else:
        t = ds.truth
        diag, _ = diagnose(ds, ctx.xi, ctx.G)
        out.update({"highd_acc_s": knn_accuracy(ds.X_T, t["y_s"]), "highd_acc_b": knn_accuracy(ds.X_T, t["y_b"]),
                    "oracle_acc_s": knn_accuracy(t["S_T"], t["y_s"]), "xi_b_diagnostics": diag})
        if not (diag["gate_same_b"] and diag["gate_same_s"]):
            stop(f"{level} seed {seed}: xi_B gates fail ({diag['same_b_xi']:.3f}, excess {diag['same_s_excess']:+.3f})")
    return out


def phase1_sanity(row):
    """(a) must reproduce Phase 1's embedding accuracies exactly (same dataset, schedule and seed)."""
    rec = next(r for r in read_csv(ROOT / "results/phase1/validity.csv")
               if r["level"] == row["level"] and float(r["rho"]) == MAIN_RHO[row["level"]]
               and int(r["seed"]) == row["seed"])
    ok = float(rec["emb_acc_s"]) == row["acc_s"] and float(rec["emb_acc_b"]) == row["acc_b"]
    return {"phase1_acc_s": float(rec["emb_acc_s"]), "phase1_acc_b": float(rec["emb_acc_b"]), "match": ok}


def validity(row, oracle):
    return (row["acc_b"] >= VALIDITY["min_emb_acc_b"] and row["acc_s"] <= VALIDITY["max_emb_acc_s"]
            and oracle >= VALIDITY["min_oracle_acc_s"])


def run_seed(level, stage, seed, labels, save=True):
    """Stage 1 + the given Stage 2 runs on one dataset. Returns rows, checks, and the embeddings."""
    ctx = prepare(level, seed, MAIN_RHO[level])
    checks = dataset_checks(ctx, stage)
    rows = [reference_row(ctx, stage)]
    embds = {}
    s1 = run_stage1(ctx)
    rows.append(fit_row(ctx, stage, "stage1", "ref", s1))
    embds["stage1"] = s1["embd"]
    a_std = None
    for label in labels:
        fit = run_stage2(ctx, s1, label)
        row = fit_row(ctx, stage, label, label[0], fit, a_std=a_std)
        if label == "a":
            a_std = row["embd_std"]
            row["std_ratio_vs_a"], row["flag_collapse"] = 1.0, False
        rows.append(row)
        embds[label] = fit["embd"]
        print(f"{level} {stage} seed={seed} {label:6s} acc_s {row['acc_s']:.3f} acc_b {row['acc_b']:.3f} "
              f"recall_s {row['recall_s']:.3f} ({row['runtime_s']:.0f}s)", flush=True)
        if save:
            np.save(OUT / "embeddings" / f"{level}_{stage}_seed{seed}_{label}.npy", fit["embd"])
            np.savez(OUT / "logs" / f"{level}_{stage}_seed{seed}_{label}.npz", **epoch_arrays(fit))
        if not healthy(row):
            stop(f"{level} seed {seed} {label}: non-finite or exploding (std growth {row['max_std_growth']:.2f})")
        if stage == "confirmation" and label == "a":
            # exclusion rule (Plan §5): validity is judged from (a) and the oracle only, before any other run
            checks["validity"] = {"emb_acc_b": row["acc_b"], "emb_acc_s": row["acc_s"],
                                  "oracle_acc_s": checks["oracle_acc_s"],
                                  "valid": validity(row, checks["oracle_acc_s"])}
            if not checks["validity"]["valid"]:
                print(f"{level} confirmation seed={seed}: EXCLUDED, not a valid test {checks['validity']}", flush=True)
                return rows, checks, embds
    a_row = next(r for r in rows if r["run"] == "a")
    if stage == "selection":
        checks["phase1_sanity"] = phase1_sanity(a_row)
        if not checks["phase1_sanity"]["match"]:
            stop(f"{level} seed {seed}: (a) does not reproduce Phase 1 accuracies {checks['phase1_sanity']}")
    if save:
        np.save(OUT / "embeddings" / f"{level}_{stage}_seed{seed}_stage1.npy", s1["embd"])
    return rows, checks, embds


def recheck(level):
    """Rerun selection seed 0 and compare every saved row field (except runtime) and every embedding bit-for-bit."""
    saved = [r for r in read_csv(OUT / f"runs_{level}_selection.csv") if int(r["seed"]) == 0]
    rows, _, embds = run_seed(level, "selection", 0, SELECTION_RUNS, save=False)
    fields = [f for f in ROW_FIELDS if f != "runtime_s"]
    rows_ok = all(str(new[f]) == old[f] for new, old in zip(rows, saved) for f in fields) and len(rows) == len(saved)
    embd_ok = {label: bool(np.array_equal(e, np.load(OUT / "embeddings" / f"{level}_selection_seed0_{label}.npy")))
               for label, e in embds.items()}
    result = {"environment": environment_info(), "level": level, "seed": 0, "rows_identical": rows_ok,
              "embeddings_identical": embd_ok, "all_identical": rows_ok and all(embd_ok.values())}
    (OUT / f"recheck_{level}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    if not result["all_identical"]:
        stop(f"{level}: rerun of selection seed 0 is not bit-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--stage", choices=list(SEEDS))
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    for sub in ("embeddings", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if args.recheck:
        return recheck(args.level)
    level, stage = args.level, args.stage

    selection = None
    if stage == "confirmation":
        sel_path = OUT / f"selection_{level}.json"
        assert_committed(sel_path)
        selection = json.loads(sel_path.read_text())
        labels = confirmation_runs(selection)
    else:
        labels = SELECTION_RUNS

    rows, checks, used, excluded = [], {}, [], []
    seed = 0 if stage == "selection" else FIRST_CONFIRM_SEED
    while len(used) < (len(SEEDS["selection"]) if stage == "selection" else N_CONFIRM):
        seed_rows, checks[seed], _ = run_seed(level, stage, seed, labels)
        if stage == "confirmation" and not checks[seed]["validity"]["valid"]:
            # excluded seeds never enter the run table or any comparison (Plan §5)
            excluded.append({"seed": seed, "validity": checks[seed]["validity"],
                             "rows": [r for r in seed_rows if r["group"] != "ref"]})
            if len(excluded) > MAX_EXCLUDED:
                stop(f"{level}: {len(excluded)} confirmation seeds excluded before {N_CONFIRM} valid ones; "
                     f"rho is likely miscalibrated for this level")
        else:
            rows += seed_rows
            used.append(seed)
            write_csv(OUT / f"runs_{level}_{stage}.csv", rows)  # after every seed, so partial progress is kept
        seed += 1

    summary = {"environment": environment_info(), "level": level, "stage": stage, "rho": MAIN_RHO[level],
               "seeds": used, "excluded_seeds": excluded, "runs": labels, "checks": checks}
    if stage == "selection":
        selected = select_settings([r for r in rows if r["group"] != "ref"])
        (OUT / f"selection_{level}.json").write_text(json.dumps({"level": level, **selected}, indent=2))
        summary["selection"] = selected
        print(json.dumps({k: v for k, v in selected.items() if k != "means"}, indent=2))
    else:
        summary["selection"] = selection
    (OUT / f"summary_{level}_{stage}.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
