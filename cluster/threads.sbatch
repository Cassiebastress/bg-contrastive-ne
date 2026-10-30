#!/bin/bash -l
# Thread-layout timing (scripts/cluster_threads.py): one (c) m_B = 20 run per process on L-cyclic, in four layouts
# that fill the job's 24 CPUs differently. Writes only to ~/cluster_threads/<job>/.  sbatch cluster/threads.sbatch
#SBATCH -J cne-threads
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH -A yang-2026
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 1:00:00
#SBATCH -o cluster/logs/%x-%j.out
set -euo pipefail
source cluster/parallel_lib.sh
out="$HOME/cluster_threads/$SLURM_JOB_ID"
for layout in 1x4 6x4 12x2 24x1; do
    procs="${layout%x*}" threads="${layout#*x}"
    cmds="$(mktemp)"
    for seed in $(seq 0 $((procs - 1))); do
        echo "\$PY scripts/cluster_threads.py --seed $seed --layout $layout --out $out" >> "$cmds"
    done
    echo "== layout $layout"
    run_parallel "$cmds" "$procs" "$threads" "cluster/logs/$SLURM_JOB_ID/$layout"
done
"$PY" scripts/cluster_threads.py --summarize "$out"
