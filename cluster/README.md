# Running on the Math Slurm cluster

Docs: https://docs.sci.mcgill.ca/MATH/slurm/. Account `yang-2026`; queues (QOS) `gpu_h100_pro`, `gpu_debug_nvidia`.

## Log in

    ssh cbastr2@stat.math.mcgill.ca     # on campus or on the McGill VPN
    ssh cbastr2@jump.math.mcgill.ca     # otherwise
    module load slurm                   # sbatch, squeue, scancel

## One-time setup

    git clone https://github.com/Cassiebastress/bg-contrastive-ne.git ~/bg-contrastive-ne
    cd ~/bg-contrastive-ne
    sbatch cluster/setup.sbatch         # builds ~/miniforge3/envs/cne-bg and runs the tests (≤ 1 h)

## Run a job

Submit from the repo root, after `git pull`, so the code a job records (`project_git` in every results JSON) is a
pushed commit. Logs go to `cluster/logs/` (gitignored).

    sbatch cluster/smoke.sbatch         # timing and reproducibility test; writes ~/cluster_smoke/smoke_<job>.json

## Run seeds in parallel

Per run, the cluster is about 2.4–3× slower than the M3 Mac (many tiny steps; single-core speed), so the gain
comes from running many seeds at once. Write one command per line (one seed of one run script) and submit:

    sbatch --export=ALL,THREADS=1 cluster/parallel.sbatch cluster/jobs/<name>.txt

`cluster/parallel_lib.sh` runs at most (24 CPUs) / THREADS commands at a time, each with THREADS threads, and logs
each to `cluster/logs/<job>/<n>.out`. Run scripts for new phases take `--seed` (one seed per process, its own
output file) and a separate merge step. Timing job 6019 (`cluster/threads.sbatch`, one (c) run per process):

| processes × threads | one (c) run (s) | runs per hour |
|---|---|---|
| 1 × 4 | 131 | 14 |
| 6 × 4 | 123 | 97 |
| 12 × 2 | 165 | 161 |
| 24 × 1 | 233 | 275 |

so THREADS = 1 (24 at a time) is the default. Keep one THREADS value within a phase: it can change results.

## Watch and cancel

    squeue -u cbastr2                   # your jobs
    squeue -p gpu_h100_pro              # everyone's, before submitting
    tail -f cluster/logs/<name>-<job>.out
    scancel <job>

## Conventions

- **Device:** `CNE_DEVICE=cpu` (the default). The ξ_B sampling code (`method/bg_term.py`) is CPU-only and untested on
  CUDA.
- **Threads:** `CNE_NUM_THREADS` (torch; default 4, as on the Mac); BLAS/OpenMP limited to the job's CPUs.
- **CPU-only jobs:** `gpu_h100_pro` accepts jobs without a GPU (checked with `sbatch --test-only`; the QOS caps a
  job at 24 CPUs and 256 GB), so no job holds a GPU it doesn't use. Always set a time limit (`-t`); cancel jobs as
  soon as they are done.
- **Platforms:** no comparison mixes Mac-recorded and cluster results.
- **Reproducibility:** results are deterministic within the cluster but not bit-identical to the Mac's.
