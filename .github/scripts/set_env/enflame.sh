#!/usr/bin/env bash

# Enflame TOPS-specific unit-test environment setup.

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
      export GTEST_FILTER="-DeviceAdaptorTest.HostGetDevicePointer:NetAdaptorLoopback.*"
      ;;
    device_api|device_api_unified_ir)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Enflame environment for CI workload: $suite"
  command -v topscc
  command -v mpirun
  mpirun --version
  topscc --version
  efsmi || true
}
