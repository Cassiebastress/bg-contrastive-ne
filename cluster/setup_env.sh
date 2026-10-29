#!/bin/bash -l
# Build the cne-bg environment on the Math Slurm cluster (run once, inside a job: see cluster/setup.sbatch).
#
# Miniforge in ~/miniforge3 (a fixed path the job scripts can rely on; installed if absent), a conda env with the
# Mac's Python, the PyTorch CPU wheels of the Mac's torch version, the pins of env/requirements-linux.txt, contrastive
# 1.2.0 from its sha256-checked sdist, the vendored contrastive-ne (editable), then the test suite.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME="$HOME/miniforge3"
ENV="$CONDA_HOME/envs/cne-bg"
PY="$ENV/bin/python"
CONTRASTIVE_SHA256=7d6c2ba9f3ef44bedaad8476a127e908b352a90abff14a799287372475f1f7b2

echo "node $(hostname), job ${SLURM_JOB_ID:-none}, repo $REPO at $(git -C "$REPO" rev-parse --short HEAD)"

if ! curl -sfI --max-time 15 https://pypi.org/simple/ > /dev/null; then
    echo "STOP: no internet access from this node (pypi.org unreachable); not working around it"
    exit 2
fi

if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    installer="$(mktemp -d)/miniforge.sh"
    curl -sfL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$CONDA_HOME"
fi
"$CONDA_HOME/bin/conda" --version

if [ ! -x "$PY" ]; then
    "$CONDA_HOME/bin/conda" create -y -p "$ENV" -c conda-forge python=3.12.15
fi

"$PY" -m pip install --no-input torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install --no-input -r "$REPO/env/requirements-linux.txt"

sdist_dir="$(mktemp -d)"
"$PY" -m pip download --no-input --no-deps --no-binary :all: contrastive==1.2.0 -d "$sdist_dir"
sdist="$(ls "$sdist_dir"/contrastive-1.2.0.*)"
echo "$CONTRASTIVE_SHA256  $sdist" | sha256sum -c -
"$PY" -m pip install --no-input --no-deps "$sdist"

"$PY" -m pip install --no-input --no-deps -e "$REPO/third_party/contrastive-ne"

cd "$REPO"
"$PY" -m pytest -q -p no:cacheprovider
"$PY" -c "import json; from common.repro import environment_info; print(json.dumps(environment_info(), indent=1))"
echo "OK: environment ready at $ENV"
