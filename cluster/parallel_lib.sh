# Run independent commands in parallel inside one Slurm job (sourced by cluster/*.sbatch).
#
#   run_parallel COMMANDS_FILE JOBS THREADS LOGDIR
#
# Each non-empty, non-# line of COMMANDS_FILE is one command (typically one seed of one run script), run with
# `bash -c`, at most JOBS at a time, each with THREADS threads for torch and the math libraries. Output goes to
# LOGDIR/<n>.out and the exit code to LOGDIR/<n>.rc. Returns 1 if any command failed. $PY is the env's Python.

export PY="$HOME/miniforge3/envs/cne-bg/bin/python"

run_parallel() {
    local file="$1" jobs="$2" threads="$3" logdir="$4" n=0 line
    mkdir -p "$logdir"
    export CNE_DEVICE=cpu CNE_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
           MKL_NUM_THREADS="$threads" NUMBA_NUM_THREADS="$threads"
    while IFS= read -r line || [ -n "$line" ]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do sleep 1; done
        n=$((n + 1))
        echo "[$(date +%T)] start $n: $line"
        (bash -c "$line" > "$logdir/$n.out" 2>&1; echo $? > "$logdir/$n.rc") &
    done < "$file"
    wait
    local failed=0 rc
    for rc in "$logdir"/*.rc; do
        if [ "$(cat "$rc")" != 0 ]; then echo "FAILED: ${rc%.rc}.out (exit $(cat "$rc"))"; failed=1; fi
    done
    echo "[$(date +%T)] done: $n commands, $([ $failed = 0 ] && echo 'all succeeded' || echo 'some failed')"
    return $failed
}
