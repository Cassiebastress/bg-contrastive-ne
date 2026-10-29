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

## Watch and cancel

    squeue -u cbastr2                   # your jobs
    squeue -p gpu_h100_pro              # everyone's, before submitting
    tail -f cluster/logs/<name>-<job>.out
    scancel <job>

## Conventions

- **Device:** `CNE_DEVICE=cpu` (the default). The ξ_B sampling code (`method/bg_term.py`) is CPU-only and untested on
  CUDA.
- **Threads:** `CNE_NUM_THREADS` (torch; default 4, as on the Mac); BLAS/OpenMP limited to the job's CPUs.
- **GPU etiquette:** every queue requires a GPU, but our jobs use CPUs. Request an RTX 6000
  (`--gres=gpu:rtx6000:1`), not an H100; always set a time limit (`-t`); cancel jobs as soon as they are done.
- **Reproducibility:** results are deterministic within the cluster but not bit-identical to the Mac's.
