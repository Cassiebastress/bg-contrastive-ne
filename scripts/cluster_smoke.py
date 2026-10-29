"""Cluster timing and reproducibility smoke test (no experiment result; writes only to ~/cluster_smoke/).

L-cyclic selection seed 0, run twice in one process through scripts/phase5b_run.run_seed (save=False): Stage 1,
(a), (c) m_B = 20, (o) m_B = 80, (e2) and (e3). Checks that the two passes are bit-identical (rows and embeddings),
times each, and compares WC with the Mac's committed values (results/phase5b/runs_cyclic_selection.csv; descriptive:
another CPU architecture is not expected to be bit-identical to the Mac).

Run with the cne-bg env (on the cluster: sbatch cluster/smoke.sbatch):  python scripts/cluster_smoke.py
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.phase5b_run as run  # noqa: E402
from common.repro import environment_info  # noqa: E402

LABELS = ["a", "c_m20", "o_m80", "e2"]  # e2 also runs e3
OUT = Path.home() / "cluster_smoke"


def one_pass(cal):
    t0 = time.perf_counter()
    rows, _, embds = run.run_seed("cyclic", "selection", 0, LABELS, cal, save=False)
    return rows, embds, time.perf_counter() - t0


def main():
    cal = json.loads(run.CAL.read_text())
    passes = [one_pass(cal) for _ in range(2)]
    (r1, e1, t1), (r2, e2, t2) = passes
    fields = [f for f in run.ROW_FIELDS if f != "runtime_s"]
    rows_identical = all(str(a[f]) == str(b[f]) for a, b in zip(r1, r2) for f in fields)
    embd_identical = {k: bool(np.array_equal(e1[k], e2[k])) for k in e1}
    with open(run.OUT / "runs_cyclic_selection.csv") as f:
        mac = {r["run"]: float(r["WC"]) for r in csv.DictReader(f) if r["seed"] == "0"}
    runs = [{"run": r["run"], "WC": float(r["WC"]), "WC_mac": mac.get(r["run"]),
             "WC_minus_mac": float(r["WC"]) - mac[r["run"]] if r["run"] in mac else None,
             "runtime_s_pass1": r["runtime_s"] or None} for r in r1]
    result = {"environment": environment_info(), "pass_seconds": [t1, t2], "rows_identical": rows_identical,
              "embeddings_identical": embd_identical,
              "all_identical": rows_identical and all(embd_identical.values()), "runs": runs}
    OUT.mkdir(exist_ok=True)
    path = OUT / f"smoke_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2, default=str))
    print(f"written to {path}")


if __name__ == "__main__":
    main()
