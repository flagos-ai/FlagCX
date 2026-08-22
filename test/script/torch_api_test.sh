#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=INIT

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO_ROOT/plugin/torch${PYTHONPATH:+:$PYTHONPATH}"

hash -r

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import torch' >/dev/null 2>&1; then
            PYTHON_BIN=$candidate
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x /root/miniconda3/envs/flagscale-train/bin/python ]]; then
        PYTHON_BIN=/root/miniconda3/envs/flagscale-train/bin/python
    else
        echo "[ERROR] Could not find a Python interpreter with torch installed"
        exit 1
    fi
fi

CMD_BASE="$PYTHON_BIN -m torch.distributed.run --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr=\"localhost\""
PY_SCRIPT='../../plugin/torch/example/example.py'

echo "[INFO] Launching PyTorch API tests in homogeneous mode"
while true; do
    PORT=$(shuf -i 20000-65535 -n 1)
    (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
done
CMD="$CMD_BASE --master_port=$PORT $PY_SCRIPT"
echo "$CMD"
eval "$CMD"
echo "[INFO] Completed PyTorch API tests in homogeneous mode"
echo "--------------------------------------------------------"

if [[ "${FLAGCX_SKIP_HETERO:-0}" == "1" ]]; then
    echo "[INFO] Skipping heterogeneous PyTorch API tests for this backend"
    exit 0
fi

# Wait for previous torchrun processes to fully release sockets
sleep 5

echo "[INFO] Launching PyTorch API tests in heterogeneous mode"
export FLAGCX_CLUSTER_SPLIT_LIST=2
export FLAGCX_MEM_ENABLE=1
while true; do
    PORT=$(shuf -i 20000-65535 -n 1)
    # Ensure port is not in use or in TIME_WAIT
    if ss -tlna | grep -q ":${PORT} "; then
        continue
    fi
    (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
done
CMD="$CMD_BASE --master_port=$PORT $PY_SCRIPT"
echo "$CMD"
eval "$CMD"
echo "[INFO] Completed PyTorch API tests in heterogeneous mode"
echo "--------------------------------------------------------"
