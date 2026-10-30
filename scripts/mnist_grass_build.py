"""Build the MNIST-on-grass datasets for seeds 0-5 (on the cluster, where data/raw/ lives) and check them.

For each seed: build, run data.mnist_grass.check on the real data (shapes, ranges, exact reconstruction, disjoint
pools), rebuild in memory and require bit-identity, save data/processed/mnist_grass_seed{s}.npz, and record the digit
class counts. Writes results/mnist_grass/build.json. A failed check stops the build (spec section 9).
Afterwards (last step of the job, so no run records an edited tracked file) scripts/mnist_grass_manifest.py
adds the processed checksums to data/MANIFEST.md.

    python scripts/mnist_grass_build.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.repro import environment_info
from data import mnist_grass as mg

SEEDS = range(6)
RAW, PROCESSED, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results" / "mnist_grass"


def main():
    grass = mg.load_grass(mg.grass_files(RAW / "grass"))
    records = []
    for seed in SEEDS:
        d = mg.build(seed, RAW / "grass", RAW / "mnist.npz")
        try:
            mg.check(d, grass)
            again = mg.build(seed, RAW / "grass", RAW / "mnist.npz")
            assert all(np.array_equal(d[k], again[k]) for k in d), "rebuild is not bit-identical"
        except AssertionError as e:
            print(f"STOP (MNIST-on-grass build): seed {seed} fails a build check: {e}", flush=True)
            sys.exit(2)
        mg.save(d, PROCESSED / f"mnist_grass_seed{seed}.npz")
        counts = {str(k): int(v) for k, v in zip(*np.unique(d["y"], return_counts=True))}
        records.append({"seed": seed, "digit_counts": counts, "n_target_pool": len(d["target_pool"]),
                        "n_background_pool": len(d["background_pool"]),
                        "distinct_target_sources": int(len(np.unique(d["target_src"]))),
                        "distinct_background_sources": int(len(np.unique(d["background_src"]))),
                        "filenames_sha256": str(d["filenames_sha256"]), "checks": "passed"})
        print(json.dumps(records[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "build.json").write_text(json.dumps({"environment": environment_info(), "n_grass_images": len(grass),
                                                "seeds": records}, indent=2))


if __name__ == "__main__":
    main()
