#!/bin/bash -l
# Timing and reproducibility test (scripts/cluster_smoke.py). Submit from the repo root:  sbatch cluster/smoke.sbatch
#SBATCH -J cne-smoke
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
# Fixed thread counts: torch as on the Mac; BLAS/OpenMP limited to the job's CPUs (the node has 184).
export CNE_DEVICE=cpu CNE_NUM_THREADS=4
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
       MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$HOME/miniforge3/envs/cne-bg/bin/python" scripts/cluster_smoke.py
