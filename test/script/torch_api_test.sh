#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FLAGCX_DEBUG="${FLAGCX_DEBUG:-INFO}"
export FLAGCX_DEBUG_SUBSYS="${FLAGCX_DEBUG_SUBSYS:-INIT}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
TEST_RUNNER=${FLAGCX_TORCH_TEST_RUNNER:-$REPO_ROOT/.github/scripts/ci/run_with_timeout.sh}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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

readonly HETERO_USE_HETERO_COMM=${FLAGCX_USE_HETERO_COMM-}
readonly HETERO_CLUSTER_SPLIT_LIST=${FLAGCX_CLUSTER_SPLIT_LIST:-2}
readonly HETERO_MEM_ENABLE=${FLAGCX_MEM_ENABLE:-1}
readonly HETERO_VMM_ENABLE=${FLAGCX_VMM_ENABLE-}
readonly HETERO_P2P_TRANSPORT=${FLAGCX_P2P_TRANSPORT-}
readonly HETERO_P2P_DISABLE=${FLAGCX_P2P_DISABLE-}

# Always remove mode-selecting variables before constructing either child
# environment. This prevents a platform setup script (PPU in particular) from
# turning the first, homogeneous invocation into a heterogeneous one.
MODE_ENV=(
    env
    -u FLAGCX_USE_HOST_COMM
    -u FLAGCX_USE_HETERO_COMM
    -u FLAGCX_CLUSTER_SPLIT_LIST
    -u FLAGCX_MEM_ENABLE
    -u FLAGCX_VMM_ENABLE
    -u FLAGCX_P2P_TRANSPORT
    -u FLAGCX_P2P_DISABLE
)

HETERO_ENV=(
    "FLAGCX_CLUSTER_SPLIT_LIST=$HETERO_CLUSTER_SPLIT_LIST"
    "FLAGCX_MEM_ENABLE=$HETERO_MEM_ENABLE"
)
[[ -z "$HETERO_USE_HETERO_COMM" ]] || \
    HETERO_ENV+=("FLAGCX_USE_HETERO_COMM=$HETERO_USE_HETERO_COMM")
[[ -z "$HETERO_VMM_ENABLE" ]] || \
    HETERO_ENV+=("FLAGCX_VMM_ENABLE=$HETERO_VMM_ENABLE")
[[ -z "$HETERO_P2P_TRANSPORT" ]] || \
    HETERO_ENV+=("FLAGCX_P2P_TRANSPORT=$HETERO_P2P_TRANSPORT")
[[ -z "$HETERO_P2P_DISABLE" ]] || \
    HETERO_ENV+=("FLAGCX_P2P_DISABLE=$HETERO_P2P_DISABLE")

find_free_port() {
    "$PYTHON_BIN" -c \
        'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

run_torch_api_test() {
    local mode=$1
    shift
    local port
    port=$(find_free_port)
    local -a command=(
        "$PYTHON_BIN" -m torch.distributed.run
        --nproc_per_node 8
        --nnodes=1
        --node_rank=0
        --master_addr=localhost
        "--master_port=$port"
        "$REPO_ROOT/plugin/torch/example/example.py"
    )

    echo "[INFO] Launching PyTorch API tests in $mode mode"
    printf '[INFO] Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    FLAGCX_CI_TEST_LABEL="PyTorch API tests ($mode)" \
        "$TEST_RUNNER" "${MODE_ENV[@]}" "$@" "${command[@]}"
    echo "[INFO] Completed PyTorch API tests in $mode mode"
    echo "--------------------------------------------------------"
}

run_torch_api_test homogeneous

if [[ "${FLAGCX_SKIP_HETERO:-0}" == "1" ]]; then
    echo "[INFO] Skipping heterogeneous PyTorch API tests for this backend"
    exit 0
fi

# Give the first torchrun a short grace period to release process-group state.
sleep "${FLAGCX_TORCH_TEST_INTER_MODE_DELAY:-5}"
run_torch_api_test heterogeneous "${HETERO_ENV[@]}"
