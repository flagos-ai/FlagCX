#!/usr/bin/env bash

# Hygon DCU-specific unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

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
    device_api)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
    adaptor|p2p|rma|runner|symmem)
      # Temporary full diagnostics while the Hygon SHCA/heterogeneous paths
      # are being qualified across all communication suites.
      export FLAGCX_DEBUG=TRACE
      export FLAGCX_DEBUG_SUBSYS=ALL
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Hygon DCU environment for CI workload: $suite"
  command -v mpirun
  command -v nvcc
  mpirun --version
  nvcc --version
  hy-smi --showproductname || true

  if [[ "$suite" == "adaptor" || "$suite" == "p2p" ||
        "$suite" == "rma" || "$suite" == "runner" ||
        "$suite" == "symmem" ]]; then
    echo "Network interfaces visible inside the CI container:"
    ls /sys/class/net 2>/dev/null || true
    echo "RDMA devices visible inside the CI container:"
    ls /sys/class/infiniband 2>/dev/null || true
    ls /sys/class/infiniband_verbs 2>/dev/null || true
    ls /dev/infiniband 2>/dev/null || true
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static Hygon "$suite" \
    "/sys/class/infiniband/shca_*" || return
  if [[ ! -e /dev/shca-ioctl ]]; then
    echo "Hygon $suite tests require /dev/shca-ioctl." >&2
    return 1
  fi
  if [[ ! -d /var/log/shca ]]; then
    echo "Hygon $suite tests require /var/log/shca for the SHCA provider." >&2
    return 1
  fi
}

flagcx_ci_build_suite_override() {
  local suite=$1
  local suite_dir=${2:-}
  shift 2 || true
  local -a args=("$@")

  if [[ "$suite" == "device_api" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping Hygon device_api build: DU test kernels do not provide all launchers required by the current device_api tests."
    return
  fi

  if [[ "$suite" == "symmem" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    cmake -S "$PROJECT_ROOT/third-party/googletest" \
      -B "$PROJECT_ROOT/third-party/googletest/build"
    cmake --build "$PROJECT_ROOT/third-party/googletest/build" --parallel "$(nproc)"
    make -C "$suite_dir" --jobs="$(nproc)" "${args[@]}"
    return
  fi

  FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
}

flagcx_ci_run_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  if [[ "$suite" == "device_api" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping Hygon device_api tests: DU launcher coverage is incomplete in the current test kernels."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}
