#!/bin/bash
set -euo pipefail
source /root/miniconda3/bin/activate flagscale-train

export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_addr 127.0.0.1 --master_port 8281 /share/project/gzy/FlagCX/plugin/torch/example/example.py
