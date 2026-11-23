#!/usr/bin/env bash

# Mthreads MUSA unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

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

export PATH="/usr/local/musa/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/musa/lib:/usr/local/musa/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_SOCKET_IFNAME="${FLAGCX_SOCKET_IFNAME:-^lo,docker,br-,veth}"
FLAGCX_CI_IGNORE_TEST=$(cat <<'JSON'
{
  "gtest": [
    {
      "suite": "adaptor",
      "tests": ["DeviceAdaptorTest.StreamWaitEvent"],
      "reason": "StreamWaitEvent currently fails on MUSA"
    }
  ],
  "phases": [
    {
      "suite": "symmem",
      "name": "vmm-local-and-remote-without-net",
      "reason": "MUSA does not implement SymMem VMM allocation and mapping; ordinary IPC is tested"
    },
    {
      "suite": "symmem",
      "name": "vmm-net-data",
      "reason": "MUSA advertises neither DMA-BUF nor VA registration for VMM allocations; ordinary NET and the VMM rejection contract are tested"
    }
  ]
}
JSON
)
export FLAGCX_CI_IGNORE_TEST

FLAGCX_CI_MUSA_DEVICES=${FLAGCX_CI_MUSA_DEVICES:-0,1,2,3}
IFS=, read -r -a FLAGCX_CI_MUSA_DEVICE_IDS <<< "$FLAGCX_CI_MUSA_DEVICES"
if [[ ${#FLAGCX_CI_MUSA_DEVICE_IDS[@]} -ne 4 ]]; then
  echo "FLAGCX_CI_MUSA_DEVICES must list exactly four device IDs" >&2
  return 1 2>/dev/null || exit 1
fi
export MUSA_VISIBLE_DEVICES=$FLAGCX_CI_MUSA_DEVICES
# The current MUSA backend has no Device API kernel target or unified IR platform
# implementation. Keep both suites visible in CI and report this gap.
FLAGCX_CI_UNIFIED_IR_SUPPORTED=0
FLAGCX_CI_DEVICE_API_SUPPORTED=0
# Require GPU MR now that RDMA support is being revalidated. Unsupported
# registration must fail the check rather than pass as a capability skip.
export FLAGCX_CI_GPU_MR_POLICY=gpu-mr-required
# Every wrapped unit/MPI invocation retains a delayed process-stack snapshot;
# the suite transcript also captures direct runner invocations.
export FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=120
export FLAGCX_CI_MPI_DIAGNOSTICS_AFTER=120
# StreamWaitEvent currently fails on MUSA; keep the remaining device adaptor
# checks active while the backend issue is tracked.
export FLAGCX_CI_DEVICE_GTEST_FILTER='-*.*StreamWaitEvent'

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_MUSA=1)
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=0
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
FLAGCX_CI_ENABLE_PXN=0
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,2,4,7
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=${FLAGCX_CI_MUSA_DEVICE_IDS[0]},${FLAGCX_CI_MUSA_DEVICE_IDS[1]}
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=${FLAGCX_CI_MUSA_DEVICE_IDS[2]},${FLAGCX_CI_MUSA_DEVICE_IDS[3]}
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)
export NP=4

flagcx_ci_skip_suite() {
  local suite=$1
  if [[ "$suite" == device_api &&
        "$FLAGCX_CI_DEVICE_API_SUPPORTED" == 0 ]]; then
    echo "Skipping device_api: MUSA Device API test kernel target is unavailable"
    return 0
  fi
  if [[ "$suite" == device_api_unified_ir &&
        "$FLAGCX_CI_UNIFIED_IR_SUPPORTED" == 0 ]]; then
    echo "Skipping device_api_unified_ir: MUSA Device API test kernel target and unified IR backend are unavailable"
    return 0
  fi
  return 1
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Mthreads MUSA environment for CI workload: $suite"
  command -v mpirun
  command -v g++
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
