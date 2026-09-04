#!/usr/bin/env bash

# Hygon DCU-specific unit-test environment setup.

export CUDA_PATH="${CUDA_PATH:-/opt/dtk/cuda/cuda-12}"
export CUDA_HOME="${CUDA_HOME:-$CUDA_PATH}"

if [[ -f /opt/dtk/env.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/dtk/env.sh
fi

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/opt/mpi}

# Use the real OpenMPI launcher if the image provides a wrapper.
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

export FLAGCX_ADAPTOR=du
export USE_DU=1

FLAGCX_CI_COMMON_MAKE_ARGS=(
  USE_DU=1
  DEVICE_HOME="$CUDA_PATH"
  CCL_HOME="$CUDA_PATH"
)

# The DU makefile does not currently pull the default device-api backend into
# libflagcx.so by itself, so CI passes it through as an extra source.
FLAGCX_CI_PROJECT_MAKE_ARGS=(
  "${FLAGCX_CI_COMMON_MAKE_ARGS[@]}"
  PLATFORM_EXTRA_SRCS=flagcx/adaptor/device_api/default_dev_api_backend.cc
)
FLAGCX_CI_TEST_MAKE_ARGS=("${FLAGCX_CI_COMMON_MAKE_ARGS[@]}")

FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=8
export NP=8

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    adaptor|p2p)
      export FLAGCX_DEBUG=TRACE
      export FLAGCX_DEBUG_SUBSYS=ALL
      ;;
    device_api)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Hygon DCU environment for unit-test suite: $suite"
  echo "FLAGCX_IB_PCI_RELAXED_ORDERING=${FLAGCX_IB_PCI_RELAXED_ORDERING:-<unset>}"
  command -v mpirun
  command -v nvcc
  mpirun --version
  nvcc --version
  hy-smi --showproductname || true

  echo "Network interfaces visible inside the CI container:"
  ls -la /sys/class/net 2>/dev/null || true
  if command -v ip >/dev/null 2>&1; then
    ip -o link show | grep -E 'bond[0-3](:|@)' || true
    ip -o addr show | grep -E 'bond[0-3](:|@)' || true
  else
    echo "ip command is unavailable in the CI image"
  fi

  echo "RDMA devices visible inside the CI container:"
  ls -la /sys/class/infiniband 2>/dev/null || true
  ls -la /sys/class/infiniband_verbs 2>/dev/null || true
  ls -la /dev/infiniband 2>/dev/null || true
  if command -v ibv_devices >/dev/null 2>&1; then
    IBV_SHOW_WARNINGS=1 ibv_devices || true
  fi
  if command -v ibv_devinfo >/dev/null 2>&1; then
    IBV_SHOW_WARNINGS=1 ibv_devinfo -v || true
  fi
}

flagcx_ci_build_suite_override() {
  local suite=$1

  if [[ "$suite" == "device_api" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping Hygon device_api build: DU test kernels do not provide all launchers required by the current device_api tests."
    return
  fi

  FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
}

flagcx_ci_run_suite_override() {
  local suite=$1

  if [[ "$suite" == "device_api" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping Hygon device_api tests: DU launcher coverage is incomplete in the current test kernels."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}
