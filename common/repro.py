"""Project-wide reproducibility setup and environment recording (see Initial_Project_Plan.md, Reproducibility rules)."""
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "d5f0b530f35be4d6247300a601fb5a860bc0a1c8"

# Fixed torch thread count (Apple M3: 4 performance + 4 efficiency cores; 4 is also torch's default here).
# If exact-equality checks fail, retry with 1 before investigating further. CNE_NUM_THREADS overrides it (cluster).
NUM_THREADS = int(os.environ.get("CNE_NUM_THREADS", 4))


def setup_determinism(seed, num_threads=NUM_THREADS):
    """Call at startup: deterministic torch algorithms, fixed thread count, torch/numpy/random seeded from `seed`."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _cpu_model():
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    if sys.platform.startswith("linux"):
        with open("/proc/cpuinfo") as f:
            names = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
        if names:
            return names[0]
    return platform.processor() or platform.machine()


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    # results/ is excluded: a run's own outputs must not mark the code it ran from as dirty
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)results"],
                           capture_output=True, text=True)
    return {"head": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}


def environment_info():
    """Environment details to store with every results file."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "packages": {d: _version(d) for d in
                     ["numpy", "scipy", "scikit-learn", "torchvision", "annoy", "matplotlib", "contrastive-ne",
                      "contrastive", "umap-learn", "numba", "llvmlite", "pynndescent", "openTSNE"]},
        "contrastive_ne_upstream_commit": UPSTREAM_COMMIT,
        "project_git": _git_head(),
        # node name only inside a Slurm job (cluster node names are not personal; a laptop's can be)
        "hostname": platform.node() if os.environ.get("SLURM_JOB_ID") else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cne_device": os.environ.get("CNE_DEVICE", "cpu"),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def assert_committed(path):
    """Raise unless `path` is tracked by git and has no uncommitted changes (used to freeze pre-registered choices)."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True)
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel],
                            capture_output=True, text=True)
    if tracked.returncode != 0 or status.stdout.strip():
        raise RuntimeError(f"{rel} must be committed (and unmodified) before this step")
