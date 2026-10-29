#!/bin/bash -l
# One-time environment build (cluster/setup_env.sh). Submit from the repo root:  sbatch cluster/setup.sbatch
# An RTX 6000 rather than an H100 keeps the scarce H100s free; the job needs CPUs, not the GPU.
#SBATCH -J cne-setup
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
bash cluster/setup_env.sh
