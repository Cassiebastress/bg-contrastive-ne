"""Freeze the Phase 4 datasets: main (MAIN_RHO) and graph-level-hiding stress test (STRESS_RHO), seeds 0-2.

Records, per dataset, SHA-256 hashes of X_T and X_B (so any later generator drift is detectable) and kNN accuracies
of the labels in the high-D target X_T and of y_s in the clean signal S_T. No embeddings are computed: the stress
datasets are run only once, at the end of Phase 4.

Run with the cne-bg env:  python scripts/phase1_datasets.py
Outputs: results/phase1/datasets.csv, results/phase1/datasets.json
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import environment_info, setup_determinism
from data.synthetic import DEFAULTS, LEVELS, MAIN_RHO, STRESS_RHO, make_dataset
from eval.metrics import knn_accuracy

OUT = ROOT / "results" / "phase1"
SEEDS = [0, 1, 2]
FIELDS = ["set", "level", "rho", "seed", "highd_acc_b", "highd_acc_s", "oracle_acc_s", "sha256_X_T", "sha256_X_B"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rhos in [("main", MAIN_RHO), ("stress", STRESS_RHO)]:
        for level in LEVELS:
            for seed in SEEDS:
                setup_determinism(seed)
                ds = make_dataset(level, seed=seed, signal_to_background=rhos[level])
                t = ds.truth
                rows.append({
                    "set": name, "level": level, "rho": rhos[level], "seed": seed,
                    "highd_acc_b": knn_accuracy(ds.X_T, t["y_b"]), "highd_acc_s": knn_accuracy(ds.X_T, t["y_s"]),
                    "oracle_acc_s": knn_accuracy(t["S_T"], t["y_s"]),
                    "sha256_X_T": hashlib.sha256(ds.X_T.tobytes()).hexdigest(),
                    "sha256_X_B": hashlib.sha256(ds.X_B.tobytes()).hexdigest(),
                })
                r = rows[-1]
                print(f"{name:6s} {level:6s} rho={r['rho']} seed={seed}  high-D y_b {r['highd_acc_b']:.3f}  "
                      f"high-D y_s {r['highd_acc_s']:.3f}  oracle {r['oracle_acc_s']:.3f}")

    with open(OUT / "datasets.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "environment": environment_info(), "main_rho": MAIN_RHO, "stress_rho": STRESS_RHO, "seeds": SEEDS,
        "generator_defaults": DEFAULTS, "n_T": 5000, "n_B": 5000, "D": 50,
        "stress_test_rule": "run the method once, at the end of Phase 4, with the best main-run settings; no tuning",
        "datasets": rows,
    }
    (OUT / "datasets.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
