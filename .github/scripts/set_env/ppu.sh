#!/usr/bin/env bash

# T-Head PPU-specific unit-test environment setup.

export PATH="/usr/local/PPU_SDK/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
FLAGCX_CI_PPU_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FLAGCX_CI_PPU_DLOPEN_SHIM_SOURCE="$FLAGCX_CI_PPU_ENV_DIR/../ci/ppu_pccl_dlopen_shim.c"
FLAGCX_CI_PPU_DLOPEN_SHIM_DIR="${RUNNER_TEMP:-/tmp}/flagcx-ci-ppu"
FLAGCX_CI_PPU_DLOPEN_SHIM="$FLAGCX_CI_PPU_DLOPEN_SHIM_DIR/libflagcx-ppu-pccl-deepbind.so"

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

# PPU uses BAREX ACCL for both P2P and RMA. The RMA suite remains enabled so
# ACCL one-sided support is exercised as it is completed.
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
    rma)
      export FLAGCX_P2P_TRANSPORT=accl
      ;;
    runner)
      # PCCL on PPU currently hangs in the heterogeneous runner variants.
      # Keep the regular collective runner coverage enabled.
      unset FLAGCX_USE_HETERO_COMM FLAGCX_CLUSTER_SPLIT_LIST FLAGCX_MEM_ENABLE FLAGCX_VMM_ENABLE
      export NCCL_P2P_DISABLE=1
      export NCCL_SHM_DISABLE=1
      ;;
  esac
}

flagcx_ci_run_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  if [[ "$suite" == "runner" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    FLAGCX_CI_TEST_LABEL="runner unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
    cd "$suite_dir"
    FLAGCX_CI_MPI_LABEL="runner default" \
      "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      ./build/bin/runner_mpi_tests
    echo "Skipping PPU runner heterogeneous MPI variants: PCCL ACCL backend currently hangs in FLAGCX_CLUSTER_SPLIT_LIST mode."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}


flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing T-Head PPU environment for unit-test suite: $suite"

  # BAREX exposes libu2mm symbols globally. PCCL uses the same names for
  # function-pointer objects, so deep-bind PCCL to prevent a startup crash.
  mkdir -p "$FLAGCX_CI_PPU_DLOPEN_SHIM_DIR"
  "${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra \
    "$FLAGCX_CI_PPU_DLOPEN_SHIM_SOURCE" \
    -o "$FLAGCX_CI_PPU_DLOPEN_SHIM" -ldl
  case ":${LD_PRELOAD:-}:" in
    *":$FLAGCX_CI_PPU_DLOPEN_SHIM:"*) ;;
    *) export LD_PRELOAD="$FLAGCX_CI_PPU_DLOPEN_SHIM${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
  echo "PCCL deep-bind shim: $FLAGCX_CI_PPU_DLOPEN_SHIM"
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
