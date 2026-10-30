#!/bin/bash -l
# MNIST-on-grass job 2: build seeds 0-5 (scripts/mnist_grass_build.py), then the Phase D pre-checks
# (scripts/mnist_grass_prechecks.py: seeds 0-2 and the openTSNE determinism check in parallel, 1 thread each), then
# the processed checksums into data/MANIFEST.md (last, so no run records an edited tracked file). Submit from the
# repo root:  sbatch cluster/mnist_grass_build.sbatch
#SBATCH -J cne-mnist-build
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
source cluster/parallel_lib.sh
echo "job $SLURM_JOB_ID on $(hostname), code $(git rev-parse --short HEAD)"
build="$(mktemp)" pre="$(mktemp)"
echo "\$PY scripts/mnist_grass_build.py" > "$build"
run_parallel "$build" 1 1 "cluster/logs/$SLURM_JOB_ID/build"
cat "cluster/logs/$SLURM_JOB_ID/build/1.out"
printf '%s\n' "\$PY scripts/mnist_grass_prechecks.py --seed 0" "\$PY scripts/mnist_grass_prechecks.py --seed 1" \
              "\$PY scripts/mnist_grass_prechecks.py --seed 2" "\$PY scripts/mnist_grass_prechecks.py --tsne-check" > "$pre"
run_parallel "$pre" 4 1 "cluster/logs/$SLURM_JOB_ID/prechecks"
"$PY" scripts/mnist_grass_prechecks.py --summarize
"$PY" scripts/mnist_grass_manifest.py
