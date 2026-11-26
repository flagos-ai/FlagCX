#!/usr/bin/env bash

# Mthreads MUSA unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

export FLAGCX_ADAPTOR=musa
export USE_MUSA=1
export DEVICE_HOME="${DEVICE_HOME:-/usr/local/musa}"
export CCL_HOME="${CCL_HOME:-$DEVICE_HOME}"

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}
if [[ -x /usr/local/openmpi/bin/mpirun ]]; then
  # The local Mthreads CI container ships OpenMPI here, while older images
  # used /usr/local/mpi. Prefer the installed compiler wrapper so MPI link
  # flags and runtime launcher resolve to the same installation.
  export MPI_HOME=/usr/local/openmpi
elif [[ -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" ]]; then
  FLAGCX_CI_MPI_HOME=$(mktemp -d)
  mkdir -p "$FLAGCX_CI_MPI_HOME/bin"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" \
    "$FLAGCX_CI_MPI_HOME/bin/mpirun"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/include" "$FLAGCX_CI_MPI_HOME/include"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/lib" "$FLAGCX_CI_MPI_HOME/lib"
  export MPI_HOME=$FLAGCX_CI_MPI_HOME
elif [[ ! -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun" && -x /usr/bin/mpirun ]]; then
  export MPI_HOME=/usr
else
  export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
fi

export PATH="$DEVICE_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CCL_HOME/lib:$DEVICE_HOME/lib:$DEVICE_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_SOCKET_IFNAME="${FLAGCX_SOCKET_IFNAME:-^lo,docker,br-,veth}"
# Capture native MCCL initialization/transport failures in every MUSA suite.
# FLAGCX_DEBUG configures FlagCX only, not the underlying native collective.
export MCCL_DEBUG="${MCCL_DEBUG:-TRACE}"
export MCCL_DEBUG_SUBSYS="${MCCL_DEBUG_SUBSYS:-ALL}"

FLAGCX_CI_MUSA_DEVICES=${FLAGCX_CI_MUSA_DEVICES:-0,1,2,3}
IFS=, read -r -a FLAGCX_CI_MUSA_DEVICE_IDS <<< "$FLAGCX_CI_MUSA_DEVICES"
if [[ ${#FLAGCX_CI_MUSA_DEVICE_IDS[@]} -ne 4 ]]; then
  echo "FLAGCX_CI_MUSA_DEVICES must list exactly four device IDs" >&2
  return 1 2>/dev/null || exit 1
fi
export MUSA_VISIBLE_DEVICES=$FLAGCX_CI_MUSA_DEVICES
# Require GPU MR now that RDMA support is being revalidated. Unsupported
# registration must fail the check rather than pass as a capability skip.
export FLAGCX_CI_GPU_MR_POLICY=gpu-mr-required
# Every wrapped unit/MPI invocation retains a delayed process-stack snapshot;
# the suite transcript also captures direct runner invocations.
export FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=120
export FLAGCX_CI_MPI_DIAGNOSTICS_AFTER=120
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1 DEVICE_HOME="$DEVICE_HOME" CCL_HOME="$CCL_HOME" MPI_HOME="$MPI_HOME")
FLAGCX_CI_TEST_MAKE_ARGS=("${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}")
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
FLAGCX_CI_ENABLE_PXN=0
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,2,4,7
FLAGCX_CI_MUSA_NODE1=$(IFS=,; echo "${FLAGCX_CI_MUSA_DEVICE_IDS[*]:0:FLAGCX_CI_NODE_NP}")
FLAGCX_CI_MUSA_NODE2=$(IFS=,; echo "${FLAGCX_CI_MUSA_DEVICE_IDS[*]:FLAGCX_CI_NODE_NP}")
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=$FLAGCX_CI_MUSA_NODE1
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=$FLAGCX_CI_MUSA_NODE2
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)
export NP=4

flagcx_ci_configure_suite() {
  case "$1" in
    device_api|device_api_unified_ir|perf-test-experimental)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
  esac
  case "$1" in
    device_api|device_api_unified_ir)
      # Try the installed SDK, including native musa::atomic_ref, rather than
      # inferring support from its version or silently skipping these suites.
      export FLAGCX_CI_DEVICE_API_MPI_RUNNER="$FLAGCX_CI_ENV_DIR/../ci/run_device_api_checked.py"
      export FLAGCX_CI_DEVICE_API_ROUNDS=${FLAGCX_CI_DEVICE_API_ROUNDS:-1}
      export FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=0
      FLAGCX_CI_DEVICE_API_FLAGS=(-b 1K -e 16M -f 2 -R 2)
      FLAGCX_CI_DEVICE_API_P2P_DISABLE=0
      FLAGCX_CI_DEVICE_API_MPI_ARGS=(
        -x FLAGCX_IB_HCA
        -x FLAGCX_IB_DISABLE=0 -x FLAGCX_DMABUF_ENABLE=1
        -x FLAGCX_SIGNAL_HOST_ENABLE=1
        -x MCCL_IB_DISABLE=1 -x MCCL_TOPO_ENHANCE=0
      )
      FLAGCX_CI_UNIFIED_IR_MPI_ARGS=(
        "${FLAGCX_CI_DEVICE_API_MPI_ARGS[@]}"
        -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_P2P_DISABLE=0
      )
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Mthreads MUSA environment for CI workload: $suite"
  command -v mpirun
  command -v g++
  if [[ "$suite" == device_api || "$suite" == device_api_unified_ir ]]; then
    echo "Building native Device API with the current MUSA SDK: $DEVICE_HOME"
    "$DEVICE_HOME/bin/mcc" --version || return
  fi
  case "$suite" in
    runner|device_api_host|device_api|device_api_unified_ir|rma|symmem|perf|torch-api|perf-test-experimental)
      python3 "$FLAGCX_CI_ENV_DIR/../ci/check_musa_runtime.py" || return
      ;;
  esac
  command -v mthreads-gmi || command -v musa-smi || command -v mthreads-smi || true
  if command -v mthreads-gmi >/dev/null 2>&1; then
    mthreads-gmi
  elif command -v musa-smi >/dev/null 2>&1; then
    musa-smi
  elif command -v mthreads-smi >/dev/null 2>&1; then
    mthreads-smi
  fi
  ls -l /dev/mtgpu* /dev/musa* /dev/mthreads* 2>/dev/null || true
}

flagcx_ci_select_rdma() {
  local suite=$1
  local -a hca_paths=() hca_names=()
  local hca_path

  shopt -s nullglob
  hca_paths=(/sys/class/infiniband/*)
  shopt -u nullglob
  if [[ ${#hca_paths[@]} -eq 0 ]]; then
    echo "Mthreads $suite tests require an active RDMA device." >&2
    return 1
  fi

  if [[ -z "${FLAGCX_IB_HCA:-}" ]]; then
    for hca_path in "${hca_paths[@]}"; do
      hca_names+=("${hca_path##*/}")
    done
    local IFS=,
    export FLAGCX_IB_HCA="${hca_names[*]}"
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_select_rdma "$suite" || return
  flagcx_ci_validate_rdma_static Mthreads "$suite" \
    "/sys/class/infiniband/*" || return
  echo "Mthreads RDMA selection: FLAGCX_IB_HCA=$FLAGCX_IB_HCA"
}
