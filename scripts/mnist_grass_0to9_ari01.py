"""Post-hoc, descriptive: ARI on the saved 0/1 MNIST-on-grass embeddings, so the 0-9 ARI has a 0/1 counterpart.

ARI was not pre-registered for the 0/1 runs; it is computed here after their results were known and carries no
verdict. For every saved 0/1 embedding (results/mnist_grass/runs: (a), (c) at every m_B, T1, T2, T3;
results/bsgraph/runs: (r_a), (r_c) at every m_B; selection and confirmation seeds; reruns skipped): k-means with k = 2,
n_init = 10, random_state = the dataset seed, and the ARI against the 0/1 digit labels. Each embedding's SHA-256 is
checked against the one its run recorded. Run on the cluster, where the embeddings and data/processed/ live.

    python scripts/mnist_grass_0to9_ari01.py
Outputs: results/mnist_grass_0to9/ari_01_posthoc.json and .csv (the 0/1 results themselves are not touched).
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import environment_info, setup_determinism  # noqa: E402
from data import mnist_grass as mg  # noqa: E402
from eval.metrics import kmeans_ari  # noqa: E402

OUT = ROOT / "results" / "mnist_grass_0to9"
SOURCES = {"mnist_grass": ROOT / "results" / "mnist_grass" / "runs", "bsgraph": ROOT / "results" / "bsgraph" / "runs"}
PATTERN = re.compile(r"(selection|confirmation)_seed(\d+)_(a|c_m\d+|T[123]|r_a|r_c_m\d+)\.npy")


def main():
    setup_determinism(0)
    labels, rows = {}, []
    for source, runs_dir in SOURCES.items():
        for p in sorted(runs_dir.glob("*.npy")):
            m = PATTERN.fullmatch(p.name)
            if m is None:
                continue
            stage, seed, run = m.group(1), int(m.group(2)), m.group(3)
            if seed not in labels:
                labels[seed] = mg.load(mg.processed_path(seed, ROOT))["y"]
            E = np.load(p)
            recorded = json.loads(p.with_suffix(".json").read_text())["sha256"]
            if hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest() != recorded:
                print(f"STOP (ARI 0/1 post hoc): {p.name} does not match the SHA-256 its run recorded", flush=True)
                sys.exit(2)
            rows.append({"source": source, "stage": stage, "seed": seed, "run": run,
                         "ari": kmeans_ari(E, labels[seed], 2, seed)})
            print(json.dumps(rows[-1]), flush=True)
    rows.sort(key=lambda r: (r["stage"], r["seed"], r["source"], r["run"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ari_01_posthoc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "stage", "seed", "run", "ari"])
        w.writeheader()
        w.writerows(rows)
    (OUT / "ari_01_posthoc.json").write_text(json.dumps(
        {"label": "descriptive, computed post hoc (ARI was not pre-registered for the 0/1 runs)",
         "kmeans": {"n_clusters": 2, "n_init": 10, "random_state": "dataset seed"}, "rows": rows,
         "environment": environment_info()}, indent=2))
    print(f"{len(rows)} embeddings -> ari_01_posthoc.csv")


if __name__ == "__main__":
    main()
