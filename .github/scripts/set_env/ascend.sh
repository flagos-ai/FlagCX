#!/usr/bin/env bash
FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export DEVICE_HOME=${DEVICE_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
export CCL_HOME=${CCL_HOME:-$DEVICE_HOME}
export MPI_HOME=${MPI_HOME:-/usr/local/mpi}
export PATH="$DEVICE_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$DEVICE_HOME/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/dcmi:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=0
FLAGCX_CI_ENABLE_PXN=0
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
export NP=4

# One Ascend host provides enough devices to run two logical host contexts
# through MPI MPMD for inter-mode transport coverage.
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)

flagcx_ci_configure_suite() {
  case "$1" in
    core|adaptor|device_api_host|topo|p2p|rma|runner|service|symmem|perf|torch-api) ;;
    *) echo "Ascend CI does not enable suite: $1" >&2; return 1 ;;
  esac
}

flagcx_ci_run_suite_override() {
  local suite=$1
  local suite_dir=$2

  # CANN does not expose the symmetric-memory VMM entry points. Exercise
  # ordinary-allocation IPC and NET/MR independently; neither is evidence
  # of VMM or visibility parity. See .github/ASCEND_CI_ISSUES.md.
  if [[ "$suite" != "symmem" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
  local unit_status=0 ipc_status=0 fallback_status=0 network_status=0
  FLAGCX_CI_TEST_LABEL="Ascend symmem unit tests" \
    "$TEST_RUNNER" "$suite_dir/build/bin/symmem_unit_tests" || unit_status=$?

  FLAGCX_CI_MPI_LABEL="Ascend symmem IPC local" \
    "$MPI_RUNNER" -np "${FLAGCX_CI_SYMMEM_LOCAL_NP:-2}" --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_IB_DISABLE=1 \
      "$suite_dir/build/bin/symmem_mpi_tests" \
      --gtest_filter=-SymMemTest.CrossGpuReadViaPeerPtr:SymMemTest.CrossGpuWriteViaPeerPtr:SymMemTest.HybridLocalAndRemoteAccess || ipc_status=$?

  local net_filter="SymMemTest.HybridLocalAndRemoteAccess:SymMemTest.RankLocalStatusFailureConvergesDeterministically:SymMemTest.FullMeshRoundFailureConvergesAndRetries:SymMemTest.RankLocalMrFailureDoesNotPublishPartialWindow:SymMemTest.MrRollbackFailureRetainsWindowLeaseUntilRetry:SymMemTest.AsymmetricDeregisterUsesCollectivePublishSlot:SymMemTest.RepeatedRegisterDeregister:SymMemTest.DuplicateWindowsShareMrUntilLastDeregister:SymMemTest.CommDestroyReleasesLiveWindow"
  local -a net_env=(
    -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_CLUSTER_SPLIT_LIST=2
    -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0
    -x FLAGCX_IB_DISABLE=0 -x FLAGCX_P2P_DISABLE=1
    -x FLAGCX_CI_REQUIRE_NET_MR=1 -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB
    -x LD_LIBRARY_PATH
  )
  FLAGCX_CI_MPI_LABEL="Ascend symmem local NET fallback" \
    "$MPI_RUNNER" --allow-run-as-root \
      -np "$FLAGCX_CI_NODE_NP" "${net_env[@]}" \
      -x FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK=1 \
      "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
      "$suite_dir/build/bin/symmem_mpi_tests" \
      --gtest_filter=SymMemTest.RankLocalIpcFailureUsesNetworkMrFallback:SymMemTest.P2pDisabledLocalWindowAcquiresNetworkMrBeforePublish || fallback_status=$?
  FLAGCX_CI_MPI_LABEL="Ascend symmem NET MR and fallback" \
    "$MPI_RUNNER" --allow-run-as-root \
      -np "$FLAGCX_CI_NODE_NP" "${net_env[@]}" \
      "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
      "$suite_dir/build/bin/symmem_mpi_tests" --gtest_filter="$net_filter" \
      : -np "$FLAGCX_CI_NODE_NP" "${net_env[@]}" \
      "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
      "$suite_dir/build/bin/symmem_mpi_tests" --gtest_filter="$net_filter" || network_status=$?

  echo "Ascend symmem results: unit=$unit_status IPC=$ipc_status fallback=$fallback_status NET=$network_status"
  echo "Ascend coverage gap: VMM/MR routes and RMA visibility require backend support; see .github/ASCEND_CI_ISSUES.md"
  if ((unit_status != 0 || ipc_status != 0 || fallback_status != 0 || network_status != 0)); then
    return 1
  fi
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Ascend environment for CI workload: $suite"
  test -d "$DEVICE_HOME/include"
  test -e /dev/davinci_manager
  test -e /dev/devmm_svm
  test -e /dev/hisi_hdc
  test -x /usr/local/sbin/npu-smi
  test -e /dev/infiniband/rdma_cm
  echo "Ascend host and container device inventory:"
  npu-smi info
  ls -l /dev/davinci* /dev/infiniband
  command -v g++
  if [[ "$suite" == core ]]; then
    flagcx_ci_validate_rdma core
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static Ascend "$suite" "/sys/class/infiniband/*"
}
