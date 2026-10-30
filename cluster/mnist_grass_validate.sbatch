#!/bin/bash -l
# MNIST-on-grass Phase B: validate the cluster on L-cyclic confirmation seed 3 (scripts/mnist_grass_validate.py).
# Five processes with 4 threads each (the Mac's torch setting), then the summary. Submit from the repo root.
#SBATCH -J cne-validate
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
source cluster/parallel_lib.sh
cmds="$(mktemp)"
printf '%s\n' "\$PY scripts/mnist_grass_validate.py --part ref --rep 1" \
                "\$PY scripts/mnist_grass_validate.py --part ref --rep 2" \
                "\$PY scripts/mnist_grass_validate.py --part opt --opt-seed 1000" \
                "\$PY scripts/mnist_grass_validate.py --part opt --opt-seed 1001" \
                "\$PY scripts/mnist_grass_validate.py --part opt --opt-seed 1002" > "$cmds"
echo "job $SLURM_JOB_ID on $(hostname), code $(git rev-parse --short HEAD)"
run_parallel "$cmds" 5 4 "cluster/logs/$SLURM_JOB_ID"
CNE_NUM_THREADS=4 "$PY" scripts/mnist_grass_validate.py --summarize
