#!/bin/bash -l
# Run the commands in a file in parallel (cluster/parallel_lib.sh). Submit from the repo root, e.g.
#   sbatch --export=ALL,THREADS=2 cluster/parallel.sbatch cluster/jobs/phaseX_selection.txt
# JOBS defaults to (CPUs of the job) / THREADS. Adjust -t on the command line for long runs (sbatch -t 3:00:00 ...).
#SBATCH -J cne-parallel
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
source cluster/parallel_lib.sh
THREADS="${THREADS:-2}"
JOBS="${JOBS:-$((SLURM_CPUS_PER_TASK / THREADS))}"
echo "job $SLURM_JOB_ID on $(hostname): $1, $JOBS at a time, $THREADS threads each, code $(git rev-parse --short HEAD)"
run_parallel "$1" "$JOBS" "$THREADS" "cluster/logs/$SLURM_JOB_ID"
