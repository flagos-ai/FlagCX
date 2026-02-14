#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=INIT

CMD_BASE='torchrun --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr="localhost"'
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

echo "[INFO] Launching PyTorch API tests in heterogeneous mode"
export FLAGCX_CLUSTER_SPLIT_LIST=2
while true; do
    PORT=$(shuf -i 20000-65535 -n 1)
    (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
done
CMD="$CMD_BASE --master_port=$PORT $PY_SCRIPT"
echo "$CMD"
eval "$CMD"
echo "[INFO] Completed PyTorch API tests in heterogeneous mode"
echo "--------------------------------------------------------"
