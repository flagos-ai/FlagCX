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
export LD_LIBRARY_PATH="$DEVICE_HOME/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_ENABLE_IBUC=1
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=0
FLAGCX_CI_ENABLE_PXN=0
# NPU memory registration is part of the Ascend IBRC contract. Require the
# four one-sided GPU cases to transfer and verify data.
export FLAGCX_CI_GPU_MR_POLICY=gpu-mr-required
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
    core|adaptor|device_api_host|p2p|rma|runner|service|symmem|perf|torch-api) ;;
    *) echo "Ascend CI does not enable suite: $1" >&2; return 1 ;;
  esac
}

# PCI lookup is covered by the perf and Torch workloads. These suites still
# run only their single-process phases until their remaining capabilities and
# multi-rank lifecycle issues are addressed in later changes.
flagcx_ci_run_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  case "$suite" in
  rma | runner | symmem) ;;
  *)
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
    return
    ;;
  esac

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
  local unit_status=0

  case "$suite" in
  rma)
    FLAGCX_CI_TEST_LABEL="Ascend rma unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}" || unit_status=$?
    echo "Ascend coverage gap: rma IPC and network MPI are not run. PCI lookup is now wired, but the CANN net adaptor does not advertise GPU MR. RMA visibility also needs cannAdaptorStreamWaitValue64/StreamWriteValue64 and an Ascend device-consumer kernel; multi-rank RMA remains pending validation"
    ;;
  runner)
    FLAGCX_CI_TEST_LABEL="Ascend runner unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}" || unit_status=$?
    echo "Ascend coverage gap: runner collective MPI is not run. hcclAdaptorCommGetAsyncError() returns flagcxNotSupported, so synchronizeAndCheckAsyncError() yields 9 at coll_allgather.cpp:23 and every isolated collective fails on the query rather than on the collective itself"
    ;;
  symmem)
    FLAGCX_CI_TEST_LABEL="Ascend symmem unit tests" \
      "$TEST_RUNNER" "$suite_dir/build/bin/symmem_unit_tests" || unit_status=$?
    echo "Ascend coverage gap: symmem IPC, local NET fallback and NET MR are not run. A prior multi-rank diagnostic exited 139 during process teardown inside flagcxProxyService; that lifecycle failure still needs a hardware stack trace. SymMem VMM routes separately need cannAdaptorSymPhysAlloc/Free plus flat map/unmap, and CANN IPC needs the cannAdaptorIpcMemHandle* contract"
    ;;
  esac

  echo "Ascend $suite results: unit=$unit_status (multi-rank phases skipped)"
  return "$unit_status"
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
  command -v python3
  python3 "$FLAGCX_CI_ENV_DIR/../ci/check_ascend_pci_runtime.py"
  if [[ "$suite" == core ]]; then
    flagcx_ci_validate_rdma core
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static Ascend "$suite" "/sys/class/infiniband/*"
}
