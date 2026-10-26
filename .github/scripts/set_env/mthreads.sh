#!/usr/bin/env bash

# Mthreads MUSA unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}
if [[ -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" ]]; then
  FLAGCX_CI_MPI_HOME=$(mktemp -d)
  mkdir -p "$FLAGCX_CI_MPI_HOME/bin"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" \
    "$FLAGCX_CI_MPI_HOME/bin/mpirun"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/include" "$FLAGCX_CI_MPI_HOME/include"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/lib" "$FLAGCX_CI_MPI_HOME/lib"
  export MPI_HOME=$FLAGCX_CI_MPI_HOME
else
  export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
fi

export PATH="/usr/local/musa/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/musa/lib:/usr/local/musa/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_MUSA=1)
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=0
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,2,4,7
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=0,1
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x MUSA_VISIBLE_DEVICES=2,3
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)
export NP=4

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Mthreads MUSA environment for CI workload: $suite"
  command -v mpirun
  command -v g++
  command -v musa-smi || command -v mthreads-smi || true
  if command -v musa-smi >/dev/null 2>&1; then
    musa-smi
  elif command -v mthreads-smi >/dev/null 2>&1; then
    mthreads-smi
  fi
  ls -l /dev/musa* /dev/mthreads* 2>/dev/null || true
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
