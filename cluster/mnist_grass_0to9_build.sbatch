#!/bin/bash -l
# MNIST-on-grass 0-9, job 1: the 0/1 regressions and the 0-9 build for seeds 0-5 (scripts/mnist_grass_0to9_build.py,
# 1 thread), then the processed checksums into data/MANIFEST.md (last, so no run records an edited tracked file).
# Submit from the repo root:  sbatch cluster/mnist_grass_0to9_build.sbatch
#SBATCH -J cne-mnist10-build
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
source cluster/parallel_lib.sh
echo "job $SLURM_JOB_ID on $(hostname), code $(git rev-parse --short HEAD)"
build="$(mktemp)"
echo "\$PY scripts/mnist_grass_0to9_build.py" > "$build"
run_parallel "$build" 1 1 "cluster/logs/$SLURM_JOB_ID/build"
cat "cluster/logs/$SLURM_JOB_ID/build/1.out"
"$PY" scripts/mnist_grass_manifest.py
