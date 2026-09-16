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
      # Fail rather than silently falling back to IBRC/socket: this suite is
      # the integration coverage for the BAREX one-sided contract.
      export FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX
      ;;
    p2p)
      # Keep ACCL engine coverage, but skip the same-host GPU READ tests until
      # ACCL supports this PPU self-loopback path; RMA remains enabled separately.
      export GTEST_FILTER="-P2pAdaptorStruct.*:P2pAdaptorTest.*:P2pLoopbackTest.*:P2pBatchStruct.*:P2pBatchTest.*:P2pEngineRpcIbTest.ConnectAcceptIsLocalSameHost:FlagcxP2pEngineReadTest.*"
      export FLAGCX_P2P_TRANSPORT=accl
      ;;
    rma)
      export FLAGCX_P2P_TRANSPORT=accl
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
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      -u FLAGCX_CLUSTER_SPLIT_LIST -u FLAGCX_MEM_ENABLE \
      -u FLAGCX_VMM_ENABLE -u FLAGCX_P2P_TRANSPORT \
      -u FLAGCX_P2P_DISABLE \
      "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      ./build/bin/runner_mpi_tests
    FLAGCX_CI_MPI_LABEL="runner BAREX heterogeneous SendRecv smoke" \
      "$MPI_RUNNER" -np 2 --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl \
      ./build/bin/runner_mpi_tests \
      --gtest_filter=FlagCXCollTest.SendRecv
    FLAGCX_CI_MPI_LABEL="runner BAREX heterogeneous" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl \
      ./build/bin/runner_mpi_tests
    FLAGCX_CI_MPI_LABEL="runner BAREX forced NET" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl \
      -x FLAGCX_P2P_DISABLE=1 \
      ./build/bin/runner_mpi_tests
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}


flagcx_ci_prepare() {
  local suite=$1
  local project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
  echo "Preparing T-Head PPU environment for unit-test suite: $suite"

  # Self-hosted PPU runners may reuse a workspace previously mounted at a
  # different container path. Remove only generated build trees so stale CMake
  # absolute paths cannot leak into this job.
  rm -rf "$project_root/build" \
    "$project_root/third-party/googletest/build"

  # Deep-bind the vendor collective library to avoid symbol interposition.
  mkdir -p "$FLAGCX_CI_PPU_DLOPEN_SHIM_DIR"
  "${CC:-cc}" -shared -fPIC -O2 -Wall -Wextra \
    "$FLAGCX_CI_PPU_DLOPEN_SHIM_SOURCE" \
    -o "$FLAGCX_CI_PPU_DLOPEN_SHIM" -ldl
  case ":${LD_PRELOAD:-}:" in
    *":$FLAGCX_CI_PPU_DLOPEN_SHIM:"*) ;;
    *) export LD_PRELOAD="$FLAGCX_CI_PPU_DLOPEN_SHIM${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
  echo "PPU deep-bind shim: $FLAGCX_CI_PPU_DLOPEN_SHIM"
  command -v mpirun
  mpirun --version

  if command -v ppu-smi >/dev/null 2>&1; then
    ppu-smi || true
  fi

  echo "PPU devices:"
  ls -l /dev/alixpu* 2>/dev/null || true
  echo "ACCL/RDMA devices:"
  ls -l /sys/class/infiniband 2>/dev/null || true
  ls -l /sys/class/infiniband_verbs 2>/dev/null || true
  ls -l /dev/infiniband 2>/dev/null || true
  ibv_devices 2>/dev/null || true
}

flagcx_ci_validate_rdma() {
  local suite=$1

  # PPU ACCL/BAREX uses the vsolar HCA together with the standard uverbs and
  # vendor command nodes. Check only kernel/sysfs visibility here; provider
  # diagnostics remain informational and do not depend on ibv_devices or
  # ibv_devinfo being installed in the image.
  if ! compgen -G "/sys/class/infiniband/vsolar_*" >/dev/null; then
    echo "PPU $suite tests require a vsolar_* RDMA HCA in /sys/class/infiniband." >&2
    return 1
  fi
  if ! compgen -G "/sys/class/infiniband_verbs/uverbs*" >/dev/null; then
    echo "PPU $suite tests require uverbs entries in /sys/class/infiniband_verbs." >&2
    return 1
  fi
  if ! compgen -G "/dev/infiniband/uverbs*" >/dev/null; then
    echo "PPU $suite tests require /dev/infiniband/uverbs* device nodes." >&2
    return 1
  fi
  if [[ ! -e /dev/infiniband/rdma_cm ]]; then
    echo "PPU $suite tests require /dev/infiniband/rdma_cm." >&2
    return 1
  fi
  if ! compgen -G "/dev/infiniband/fic2_soe_ucmd*" >/dev/null; then
    echo "PPU $suite tests require /dev/infiniband/fic2_soe_ucmd* control nodes." >&2
    return 1
  fi
}
