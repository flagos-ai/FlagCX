#!/usr/bin/env bash

# T-Head PPU-specific unit-test environment setup.

export PATH="/usr/local/PPU_SDK/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

# Some images ship a wrapper around OpenMPI. Use the real launcher so the
# common test script can pass normal OpenMPI arguments.
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

# PPU uses the BAREX ACCL transport for P2P. IBRC/RMA is intentionally not
# selected by this platform configuration.
export FLAGCX_P2P_TRANSPORT=accl
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_PPU=1 USE_ACCL_BAREX=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_PPU=1 USE_ACCL_BAREX=1)
FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=8
export NP=8

# Keep the common runner/device test helpers satisfied if those suites are
# enabled later. PPU's current suite list does not run multi-node tests.
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
      # BAREX has no DMA-BUF callback. Keep the remaining adaptor contract
      # tests enabled, including GPU MR registration.
      export GTEST_FILTER="-NetAdaptorMemory.RegMrDmaBufRegistration"
      ;;
    p2p)
      # These suites call the IBRC vtable directly. The Engine tests use the
      # runtime transport selector and are retained for ACCL coverage.
      export GTEST_FILTER="-P2pAdaptorStruct.*:P2pAdaptorTest.*:P2pLoopbackTest.*:P2pBatchStruct.*:P2pBatchTest.*:P2pEngineRpcIbTest.ConnectAcceptIsLocalSameHost"
      export FLAGCX_P2P_TRANSPORT=accl
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing T-Head PPU environment for unit-test suite: $suite"
  command -v mpirun
  mpirun --version

  if command -v ppu-smi >/dev/null 2>&1; then
    ppu-smi || true
  fi

  echo "PPU devices:"
  ls -l /dev/alixpu* 2>/dev/null || true
  echo "ACCL/RDMA devices:"
  ls -l /dev/infiniband 2>/dev/null || true
  ibv_devices 2>/dev/null || true
}
