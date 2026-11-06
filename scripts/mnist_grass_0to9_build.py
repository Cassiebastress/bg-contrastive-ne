"""Build the MNIST-on-grass 0-9 datasets for seeds 0-5 (on the cluster, where data/raw/ lives) and check them.

First the regressions, before anything is written:
- 0/1 rebuild: for each seed, the stored data/processed/mnist_grass_seed{s}.npz still matches its SHA-256 in
  data/MANIFEST.md, and a fresh default (0/1) build has the same keys and bit-identical arrays (file bytes are not
  compared: np.savez stamps the time into the zip);
- the xi_B check refactor: scripts/mnist_grass_prechecks.xi_checks on 0/1 seed 0 reproduces every xi_B field of the
  stored results/mnist_grass/prechecks/seed0.json exactly.
Then for each seed: build with digits 0-9 (balanced, 500 per digit), run data.mnist_grass.check on the real data,
rebuild in memory and require bit-identity, save data/processed/mnist_grass_0to9_seed{s}.npz, and record the digit
counts. Writes results/mnist_grass_0to9/build.json. Any failure stops the build. Afterwards (last step of the job)
scripts/mnist_grass_manifest.py adds the new checksums to data/MANIFEST.md.

    python scripts/mnist_grass_0to9_build.py
"""
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

SEEDS = range(6)
DIGITS = tuple(range(10))
NAME = "mnist_grass_0to9"
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / NAME


def stop(msg):
    print(f"STOP (MNIST-on-grass 0-9 build): {msg}", flush=True)
    sys.exit(2)


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def regression_0_1():
    manifest = (ROOT / "data" / "MANIFEST.md").read_text()
    out = []
    for seed in SEEDS:
        path = mg.processed_path(seed, ROOT)
        want = re.search(rf"`{path.name}` \| `([0-9a-f]{{64}})`", manifest)
        file_ok = want is not None and hashlib.sha256(path.read_bytes()).hexdigest() == want.group(1)
        rebuild_ok = same(mg.load(path), mg.build(seed, RAW / "grass", RAW / "mnist.npz"))
        out.append({"seed": seed, "stored_file_matches_manifest": file_ok, "rebuild_identical": rebuild_ok})
        print(json.dumps(out[-1]), flush=True)
        if not (file_ok and rebuild_ok):
            stop(f"0/1 regression fails on seed {seed}")
    return out


def refactor_check():
    from scripts.mnist_grass_prechecks import xi_checks
    stored = json.loads((ROOT / "results" / "mnist_grass" / "prechecks" / "seed0.json").read_text())
    setup_determinism(0)
    X_T, X_B, t = mg.for_embedding(mg.load(mg.processed_path(0, ROOT)))
    fresh = xi_checks(X_T, X_B, t, 0)
    mismatched = [k for k, v in fresh.items() if stored[k] != v]
    rec = {"seed": 0, "fields": len(fresh), "identical": not mismatched, "mismatched": mismatched}
    print(json.dumps(rec), flush=True)
    if mismatched:
        stop(f"xi_checks no longer reproduces the stored 0/1 pre-checks: {mismatched}")
    return rec


def main():
    regression = regression_0_1()
    refactor = refactor_check()
    grass = mg.load_grass(mg.grass_files(RAW / "grass"))
    records = []
    for seed in SEEDS:
        d = mg.build(seed, RAW / "grass", RAW / "mnist.npz", digits=DIGITS)
        try:
            mg.check(d, grass)
            assert same(d, mg.build(seed, RAW / "grass", RAW / "mnist.npz", digits=DIGITS)), "rebuild not identical"
        except AssertionError as e:
            stop(f"seed {seed} fails a build check: {e}")
        mg.save(d, mg.processed_path(seed, ROOT, name=NAME))
        counts = {str(k): int(v) for k, v in zip(*np.unique(d["y"], return_counts=True))}
        records.append({"seed": seed, "digit_counts": counts, "n_target_pool": len(d["target_pool"]),
                        "n_background_pool": len(d["background_pool"]),
                        "distinct_target_sources": int(len(np.unique(d["target_src"]))),
                        "distinct_background_sources": int(len(np.unique(d["background_src"]))),
                        "filenames_sha256": str(d["filenames_sha256"]), "checks": "passed"})
        print(json.dumps(records[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "build.json").write_text(json.dumps({"environment": environment_info(), "n_grass_images": len(grass),
                                                "digits": list(DIGITS), "regression_0_1": regression,
                                                "xi_checks_refactor": refactor, "seeds": records}, indent=2))


if __name__ == "__main__":
    main()
