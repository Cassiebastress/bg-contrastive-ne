"""Phase 4 stress test (graph-level hiding): run once, at the end, with settings frozen before any stress fit.

Reads the committed results/phase4/stress_settings.json (per level: (a), (d), (c) at m*_c, and the best (b), all
chosen on the main selection seeds) and runs them on the stress datasets (STRESS_RHO, seeds 0-2). Nothing here is tuned
or rerun after the fact. xi_B diagnostics for the stress datasets are recorded (reported, not gated). Health problems
are recorded and the run continues, so the single stress run is complete; they are reported, never fixed by rerunning.

Run with the cne-bg env, keeping the machine awake:  caffeinate -i python scripts/phase4_stress.py
Outputs go to results/phase4/stress/ (embeddings and per-epoch logs are gitignored).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.experiment import (ROW_FIELDS, epoch_arrays, fit_row, healthy, prepare, reference_row, run_stage1,
                               run_stage2)
from common.repro import assert_committed, environment_info
from data.synthetic import LEVELS, STRESS_RHO
from eval.xi_b_diag import diagnose

OUT = ROOT / "results" / "phase4" / "stress"
SETTINGS = ROOT / "results" / "phase4" / "stress_settings.json"
SEEDS = [0, 1, 2]


def main():
    assert_committed(SETTINGS)
    settings = json.loads(SETTINGS.read_text())["settings"]
    for sub in ("embeddings", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results/phase1/datasets.csv") as f:
        recorded = [r for r in csv.DictReader(f) if r["set"] == "stress"]

    rows, checks = [], {}
    for level in LEVELS:
        for seed in SEEDS:
            ctx = prepare(level, seed, STRESS_RHO[level])
            rec = next(r for r in recorded if r["level"] == level and int(r["seed"]) == seed)
            hashes_ok = (rec["sha256_X_T"] == hashlib.sha256(ctx.ds.X_T.tobytes()).hexdigest()
                         and rec["sha256_X_B"] == hashlib.sha256(ctx.ds.X_B.tobytes()).hexdigest())
            if not hashes_ok:
                sys.exit(f"STOP: stress dataset {level} seed {seed} differs from the Phase 1 record")
            diag, _ = diagnose(ctx.ds, ctx.xi, ctx.G)
            checks[f"{level}/seed{seed}"] = {"hashes_match_phase1": True,
                                             "highd_acc_s_phase1": float(rec["highd_acc_s"]),
                                             "xi_b_diagnostics": diag, "unhealthy_runs": []}
            rows.append(reference_row(ctx, "stress"))
            s1 = run_stage1(ctx)
            rows.append(fit_row(ctx, "stress", "stage1", "ref", s1))
            np.save(OUT / "embeddings" / f"{level}_stress_seed{seed}_stage1.npy", s1["embd"])
            a_std = None
            for label in settings[level]["runs"]:
                fit = run_stage2(ctx, s1, label)
                row = fit_row(ctx, "stress", label, label[0], fit, a_std=a_std)
                if label == "a":
                    a_std = row["embd_std"]
                    row["std_ratio_vs_a"], row["flag_collapse"] = 1.0, False
                rows.append(row)
                np.save(OUT / "embeddings" / f"{level}_stress_seed{seed}_{label}.npy", fit["embd"])
                np.savez(OUT / "logs" / f"{level}_stress_seed{seed}_{label}.npz", **epoch_arrays(fit))
                if not healthy(row):
                    checks[f"{level}/seed{seed}"]["unhealthy_runs"].append(label)
                print(f"stress {level} seed={seed} {label:6s} acc_s {row['acc_s']:.3f} acc_b {row['acc_b']:.3f} "
                      f"recall_s {row['recall_s']:.3f} ({row['runtime_s']:.0f}s)", flush=True)
            with open(OUT / "runs_stress.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    (OUT / "summary_stress.json").write_text(json.dumps(
        {"environment": environment_info(), "label": "Stress test (graph-level hiding), not used for any choice",
         "stress_rho": STRESS_RHO, "seeds": SEEDS, "settings": settings, "checks": checks}, indent=2, default=str))


if __name__ == "__main__":
    main()
