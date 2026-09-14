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
    device_api)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
    adaptor|p2p|rma)
      # Surface the libibverbs device/port selection path when the kernel
      # sysfs preflight succeeds but IBRC still reports zero usable devices.
      export FLAGCX_DEBUG="${FLAGCX_DEBUG:-INFO}"
      export FLAGCX_DEBUG_SUBSYS="${FLAGCX_DEBUG_SUBSYS:-INIT,NET,ENV}"
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Hygon DCU environment for unit-test suite: $suite"
  command -v mpirun
  command -v nvcc
  mpirun --version
  nvcc --version
  hy-smi --showproductname || true

  if [[ "$suite" == "adaptor" || "$suite" == "p2p" ||
        "$suite" == "rma" ]]; then
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
  local link_layer_file link_layer state_file state
  local found_link_layer=0
  local found_supported_link_layer=0
  local found_active_port=0

  # FlagCX obtains link_layer and state from the verbs HCA ports (shca_*),
  # rather than from their ib0..ib3 network interfaces. Reject the runner
  # before building unless at least one port meets IBRC's selection criteria.
  while IFS= read -r link_layer_file; do
    [[ -n "$link_layer_file" ]] || continue
    found_link_layer=1
    link_layer=$(<"$link_layer_file")
    state_file=${link_layer_file%/link_layer}/state
    if [[ -r "$state_file" ]]; then
      state=$(<"$state_file")
    else
      state="<missing>"
    fi
    echo "Hygon RDMA port: $link_layer_file=$link_layer, $state_file=$state"
    case "${link_layer,,}" in
      ethernet|infiniband)
        found_supported_link_layer=1
        if [[ "$state" == 4:* ]]; then
          found_active_port=1
        fi
        ;;
    esac
  done < <(compgen -G "/sys/class/infiniband/shca_*/ports/*/link_layer" || true)

  if [[ "$found_link_layer" != 1 ]]; then
    echo "Hygon $suite tests require RDMA, but no SHCA port link_layer file is visible in sysfs." >&2
    return 1
  fi
  if [[ "$found_supported_link_layer" != 1 ]]; then
    echo "Hygon $suite tests require an SHCA port whose link_layer is Ethernet (RoCE) or InfiniBand; the runner reported only unsupported values such as Unspecified." >&2
    return 1
  fi
  if [[ "$found_active_port" != 1 ]]; then
    echo "Hygon $suite tests require an SHCA port whose link_layer is Ethernet (RoCE) or InfiniBand and whose state is 4: ACTIVE." >&2
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

  if [[ "$suite" == "runner" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    FLAGCX_CI_TEST_LABEL="runner unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
    cd "$suite_dir"
    FLAGCX_CI_MPI_LABEL="runner default" \
      "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      ./build/bin/runner_mpi_tests
    echo "Skipping Hygon runner hetero-mode MPI variants: current DU path stalls in the single-node hetero simulation."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}
