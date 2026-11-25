#!/usr/bin/env bash

# Enflame TOPS-specific unit-test environment setup.

FLAGCX_CI_ENFLAME_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr}
export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
export PATH="/opt/tops/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/opt/tops/lib:/usr/local/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_ADAPTOR=enflame

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ENFLAME=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ENFLAME=1)
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
# The shared multi-node SymMem runner also uses this capability to gate its
# remote VMM data matrices when RDMA is present.
export FLAGCX_CI_SYMMEM_VMM_SUPPORTED=0
export FLAGCX_CI_RMA_GDR_VISIBILITY_CONSUMER_SUPPORTED=0
export NP=4


FLAGCX_CI_NODE1_MPI_ARGS=(
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)

flagcx_ci_configure_suite() {
  local suite=$1
  case "$suite" in
    adaptor)
      # TOPS currently reports SystemError for the invalid-argument probes in
      # HostGetDevicePointer; keep the rest of the adaptor contract enabled.
      export GTEST_FILTER="-DeviceAdaptorTest.HostGetDevicePointer"
      if ! flagcx_ci_has_rdma; then
        export GTEST_FILTER+=":NetAdaptorLoopback.*"
        export FLAGCX_CI_CHECK_ADAPTOR_GPU_TESTS=0
      else
        export FLAGCX_CI_CHECK_ADAPTOR_GPU_TESTS=1
      fi
      ;;
    p2p)
      if ! flagcx_ci_has_rdma; then
        # This assertion requires an actual IB P2P device; other protocol and
        # shared-engine unit tests remain enabled.
        export GTEST_FILTER="-P2pAdaptorTest.DevicesReturnsPositive"
      fi
      ;;
    rma)
      # TOPS streamWaitValue64 fails in this counter-ordering case; retain all
      # other IPC RMA cases and the complete RMA unit suite.
      export FLAGCX_CI_RMA_IPC_FILTER="RmaTest.IpcResolvedPeerPointerSupportsDirectCopy:RmaTest.IpcPutWithoutNetworkMr:RmaTest.GetAsyncOffsets:RmaTest.PutSignalSmall:RmaTest.PutSignalLarge:RmaTest.SignalOnlyNoData:RmaTest.MultipleSignals"
      ;;
    runner)
      # Gather/Scatter currently return incorrect rank data on S60, and the
      # heterogeneous ECCL HYBRID path hangs during runner initialization.
      export FLAGCX_CI_RUNNER_MPI_FILTER="--gtest_filter=-FlagCXCollTest.Gather:FlagCXCollTest.Scatter"
      export FLAGCX_CI_SKIP_RUNNER_HYBRID=1
      ;;
  esac
}

flagcx_ci_skip_suite() {
  local suite=$1
  case "$suite" in
    device_api|device_api_unified_ir)
      echo "Skipping $suite: ECCL device API SDK headers are not available in the CI image"
      return 0
      ;;
  esac
  return 1
}

flagcx_ci_has_rdma() {
  compgen -G "/sys/class/infiniband_verbs/uverbs*" >/dev/null || return 1
  compgen -G "/dev/infiniband/uverbs*" >/dev/null || return 1
  [[ -e /dev/infiniband/rdma_cm ]] || return 1

  local hca port state link_layer
  while IFS= read -r hca; do
    [[ -n "$hca" ]] || continue
    while IFS= read -r port; do
      [[ -r "$port/state" && -r "$port/link_layer" ]] || continue
      state=$(<"$port/state")
      link_layer=$(<"$port/link_layer")
      if [[ "$state" == 4:* &&
            ( "${link_layer,,}" == "ethernet" ||
              "${link_layer,,}" == "infiniband" ) ]]; then
        return 0
      fi
    done < <(compgen -G "$hca/ports/*" || true)
  done < <(compgen -G "/sys/class/infiniband/*" || true)
  return 1
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Enflame environment for CI workload: $suite"
  command -v topscc
  command -v mpirun
  mpirun --version
  topscc --version
  efsmi || true

  if flagcx_ci_has_rdma; then
    export FLAGCX_CI_HAS_RDMA=1
  else
    export FLAGCX_CI_HAS_RDMA=0
  fi
}
