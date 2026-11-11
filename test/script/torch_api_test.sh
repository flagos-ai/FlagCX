#!/usr/bin/env bash
set -euo pipefail

export FLAGCX_DEBUG="${FLAGCX_DEBUG:-INFO}"
export FLAGCX_DEBUG_SUBSYS="${FLAGCX_DEBUG_SUBSYS:-INIT}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
TEST_RUNNER=${FLAGCX_TORCH_TEST_RUNNER:-$REPO_ROOT/.github/scripts/ci/run_with_timeout.sh}
if [[ "${FLAGCX_TORCH_USE_INSTALLED_PACKAGE:-0}" != "1" ]]; then
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ "${CI:-false}" == "true" ]]; then
    # All platform jobs build this library before running the shared Torch test.
    export FLAGCX_CI_EXPECT_FLAGCX_LIBRARY="${FLAGCX_CI_EXPECT_FLAGCX_LIBRARY:-$REPO_ROOT/build/lib/libflagcx.so}"
fi

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
readonly HETERO_SKIP_OPS=${FLAGCX_TORCH_HETERO_SKIP_OPS-}

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
    -u FLAGCX_TORCH_SKIP_OPS
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
[[ -z "$HETERO_SKIP_OPS" ]] || \
    HETERO_ENV+=("FLAGCX_TORCH_SKIP_OPS=$HETERO_SKIP_OPS")

find_free_port() {
    "$PYTHON_BIN" -c \
        'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

run_torch_api_test() {
    local mode=$1
    shift
    local port
    local diagnostic_dir=${FLAGCX_CI_DIAGNOSTIC_DIR:-}
    if [[ -n "${FLAGCX_CI_TORCH_DIAGNOSTIC_ROOT:-}" ]]; then
        diagnostic_dir="$FLAGCX_CI_TORCH_DIAGNOSTIC_ROOT/$mode"
        mkdir -p "$diagnostic_dir"
    fi
    port=$(find_free_port)
    local -a command=(
        "$PYTHON_BIN" -m torch.distributed.run
        --nproc_per_node 4
        --nnodes=1
        --node_rank=0
        --master_addr=localhost
        "--master_port=$port"
    )
    if [[ "${FLAGCX_TORCH_RANK_BACKTRACE:-0}" == "1" ]]; then
        command+=(
            "$REPO_ROOT/.github/scripts/ci/torch_rank_backtrace.py"
            "$REPO_ROOT/plugin/torch/example/example.py"
        )
    else
        command+=("$REPO_ROOT/plugin/torch/example/example.py")
    fi

    echo "[INFO] Launching PyTorch API tests in $mode mode"
    printf '[INFO] Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    local status=0
    FLAGCX_CI_DIAGNOSTIC_DIR="$diagnostic_dir" \
    FLAGCX_CI_TEST_LABEL="PyTorch API tests ($mode)" \
        "$TEST_RUNNER" "${MODE_ENV[@]}" "$@" "${command[@]}" || status=$?
    if ((status != 0)); then
        echo "[ERROR] PyTorch API tests in $mode mode failed with exit=$status" >&2
        return "$status"
    fi
    echo "[INFO] Completed PyTorch API tests in $mode mode"
    echo "--------------------------------------------------------"
}

homogeneous_status=0
if [[ "${FLAGCX_TORCH_HETERO_DIAGNOSTIC_ONLY:-0}" != "1" ]]; then
    run_torch_api_test homogeneous || homogeneous_status=$?
    if ((homogeneous_status != 0)) &&
        [[ "${FLAGCX_TORCH_CONTINUE_ON_FAILURE:-0}" != "1" ]]; then
        exit "$homogeneous_status"
    fi
fi

if [[ "${FLAGCX_SKIP_HETERO:-0}" == "1" ]]; then
    echo "[INFO] Skipping heterogeneous PyTorch API tests for this backend"
    exit "$homogeneous_status"
fi

# Give the first torchrun a short grace period to release process-group state.
if [[ "${FLAGCX_TORCH_HETERO_DIAGNOSTIC_ONLY:-0}" != "1" ]]; then
    sleep "${FLAGCX_TORCH_TEST_INTER_MODE_DELAY:-5}"
fi
heterogeneous_status=0
run_torch_api_test heterogeneous "${HETERO_ENV[@]}" || heterogeneous_status=$?
echo "[INFO] PyTorch API mode results: homogeneous=$homogeneous_status heterogeneous=$heterogeneous_status"
if ((homogeneous_status != 0)); then
    exit "$homogeneous_status"
fi
exit "$heterogeneous_status"
