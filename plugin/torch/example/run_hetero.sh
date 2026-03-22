#!/bin/bash

# Check if the debug flag is provided as an argument
if [ "$1" == "debug" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=all
    echo "NCCL debug information enabled."
else
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
    echo "NCCL debug information disabled."
fi

export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=ALL
export FLAGCX_USEDEV=1
#export GLOO_SOCKET_IFNAME=ibs4
#export FLAGCX_SOCKET_IFNAME=ibs4
export GLOO_SOCKET_IFNAME=eth0
export FLAGCX_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH=/usr/local/corex/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export LD_LIBRARY_PATH=./:/usr/local/corex/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Need to preload customized gloo library specified for FlagCX linkage
# export LD_PRELOAD=/usr/local/lib/libgloo.so
# export LD_PRELOAD=/usr/local/nccl/build/lib/libnccl.so
CMD='torchrun --nproc_per_node 8 --nnodes=2 --node_rank=1 --master_addr="10.31.30.232" --master_port=8122 example.py'

echo $CMD
eval $CMD
