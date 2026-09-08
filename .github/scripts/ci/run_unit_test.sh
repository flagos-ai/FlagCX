#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <set-env-script> <suite>" >&2
  exit 2
fi

SET_ENV_SCRIPT=$1
SUITE=$2
PROJECT_ROOT=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
MPI_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_mpi_with_timeout.sh"
TEST_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_with_timeout.sh"

if [[ ! -f "$SET_ENV_SCRIPT" ]]; then
  echo "Platform environment script not found: $SET_ENV_SCRIPT" >&2
  exit 1
fi

# The platform script owns accelerator-specific compiler flags and device
# topology. It is sourced (rather than executed) so it can provide arrays and
# hook functions without unsafe string evaluation.
# shellcheck source=/dev/null
source "$SET_ENV_SCRIPT"

if declare -F flagcx_ci_configure_suite >/dev/null; then
  flagcx_ci_configure_suite "$SUITE"
fi

: "${MPI_HOME:?The platform set_env script must define MPI_HOME}"
declare -p FLAGCX_CI_PROJECT_MAKE_ARGS >/dev/null 2>&1 || {
  echo "The platform set_env script must define FLAGCX_CI_PROJECT_MAKE_ARGS" >&2
  exit 1
}
declare -p FLAGCX_CI_TEST_MAKE_ARGS >/dev/null 2>&1 || {
  echo "The platform set_env script must define FLAGCX_CI_TEST_MAKE_ARGS" >&2
  exit 1
}

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${LD_LIBRARY_PATH:-}"

flagcx_ci_require_rdma() {
  local suite=$1
  local platform_name

  platform_name=$(basename "$SET_ENV_SCRIPT" .sh)
  case "$platform_name" in
    cuda|metax|hygon) ;;
    *) return 0 ;;
  esac

  case "$suite" in
    adaptor|p2p) ;;
    *) return 0 ;;
  esac

  echo "Running $platform_name RDMA preflight for unit-test suite: $suite"

  if ! compgen -G "/sys/class/infiniband/*" >/dev/null ||
    ! compgen -G "/dev/infiniband/uverbs*" >/dev/null; then
    echo "$platform_name $suite tests require RDMA devices, but the runner did not expose /sys/class/infiniband and /dev/infiniband/uverbs* to the test container." >&2
    return 1
  fi

  if declare -F flagcx_ci_validate_rdma >/dev/null; then
    flagcx_ci_validate_rdma "$suite"
  fi

  echo "RDMA preflight passed"
}

if declare -F flagcx_ci_prepare >/dev/null; then
  flagcx_ci_prepare "$SUITE"
fi
flagcx_ci_require_rdma "$SUITE"

build_googletest() {
  cmake -S "$PROJECT_ROOT/third-party/googletest" \
    -B "$PROJECT_ROOT/third-party/googletest/build"
  cmake --build "$PROJECT_ROOT/third-party/googletest/build" --parallel "$(nproc)"
}

build_project() {
  local -a args=("${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}")
  make -C "$PROJECT_ROOT" --jobs="$(nproc)" "${args[@]}"
}

build_suite() {
  local suite_dir="$PROJECT_ROOT/test/unittest/$SUITE"
  if [[ "$SUITE" == device_api_host ||
        "$SUITE" == device_api_unified_ir ]]; then
    suite_dir="$PROJECT_ROOT/test/unittest/device_api"
  fi
  local -a args=("${FLAGCX_CI_TEST_MAKE_ARGS[@]}")

  if declare -F flagcx_ci_build_suite_override >/dev/null; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
    flagcx_ci_build_suite_override "$SUITE" "$suite_dir" "${args[@]}"
    if [[ "$FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED" == 1 ]]; then
      return
    fi
  fi

  if [[ "$SUITE" == device_api_host ]]; then
    make -C "$suite_dir" --jobs="$(nproc)" unit "${args[@]}"
  elif [[ "$SUITE" == device_api ||
          "$SUITE" == device_api_unified_ir ]]; then
    make -C "$suite_dir" --jobs="$(nproc)" mpi "${args[@]}"
  else
    make -C "$suite_dir" --jobs="$(nproc)" "${args[@]}"
  fi
}

run_device_api_host() {
  local suite_dir="$PROJECT_ROOT/test/unittest/device_api"
  local -a args=("${FLAGCX_CI_TEST_MAKE_ARGS[@]}")
  FLAGCX_CI_TEST_LABEL="device_api_host unit tests" \
    "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
}

run_device_api() {
  local suite_dir="$PROJECT_ROOT/test/unittest/device_api"
  local -a common_env=(
    -x FLAGCX_USE_HETERO_COMM=1
    -x FLAGCX_MEM_ENABLE=1
    -x FLAGCX_VMM_ENABLE=0
    -x FLAGCX_P2P_DISABLE=1
    -x LD_LIBRARY_PATH
  )
  local -a flags=(-b 1M -e 4M -f 2 -R 1)

  declare -p FLAGCX_CI_NODE1_MPI_ARGS >/dev/null 2>&1 || {
    echo "The platform set_env script must define FLAGCX_CI_NODE1_MPI_ARGS" >&2
    exit 1
  }
  declare -p FLAGCX_CI_NODE2_MPI_ARGS >/dev/null 2>&1 || {
    echo "The platform set_env script must define FLAGCX_CI_NODE2_MPI_ARGS" >&2
    exit 1
  }
  : "${FLAGCX_CI_INTRA_NP:?The platform set_env script must define FLAGCX_CI_INTRA_NP}"
  : "${FLAGCX_CI_NODE_NP:?The platform set_env script must define FLAGCX_CI_NODE_NP}"

  cd "$suite_dir"
  FLAGCX_CI_MPI_LABEL="device_api intra" \
    "$MPI_RUNNER" -np "$FLAGCX_CI_INTRA_NP" --allow-run-as-root "${common_env[@]}" \
    build/bin/test_device_api_intra "${flags[@]}"
  FLAGCX_CI_MPI_LABEL="device IR intra" \
    "$MPI_RUNNER" -np "$FLAGCX_CI_INTRA_NP" --allow-run-as-root "${common_env[@]}" \
    build/bin/test_device_ir_intra "${flags[@]}"

  FLAGCX_CI_MPI_LABEL="device_api inter" \
    "$MPI_RUNNER" --allow-run-as-root \
    -np "$FLAGCX_CI_NODE_NP" "${common_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
    build/bin/test_device_api_inter "${flags[@]}" \
    : -np "$FLAGCX_CI_NODE_NP" "${common_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
    build/bin/test_device_api_inter "${flags[@]}"
  FLAGCX_CI_MPI_LABEL="device IR inter" \
    "$MPI_RUNNER" --allow-run-as-root \
    -np "$FLAGCX_CI_NODE_NP" "${common_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
    build/bin/test_device_ir_inter "${flags[@]}" \
    : -np "$FLAGCX_CI_NODE_NP" "${common_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
    build/bin/test_device_ir_inter "${flags[@]}"
}

run_device_api_unified_ir() {
  local suite_dir="$PROJECT_ROOT/test/unittest/device_api"
  local -a common_env=(
    -x FLAGCX_USE_HETERO_COMM=1
    -x FLAGCX_VMM_ENABLE=0
    -x FLAGCX_IB_GID_INDEX=3
    -x NCCL_DEBUG=INFO
    -x NCCL_DEBUG_SUBSYS=INIT
    -x NCCL_NVLS_ENABLE=0
    -x NCCL_IB_GID_INDEX=3
    -x LD_LIBRARY_PATH
  )
  local -a intra_env=(
    "${common_env[@]}"
    -x FLAGCX_DEBUG=TRACE
    -x FLAGCX_DEBUG_SUBSYS=ALL
  )
  local -a inter_env=(
    "${common_env[@]}"
    -x FLAGCX_DEBUG=INFO
    -x FLAGCX_DEBUG_SUBSYS=PROXY
  )
  local -a intra_fallback_env=(
    "${intra_env[@]}"
    -x FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=1
  )
  local -a inter_fallback_env=(
    "${inter_env[@]}"
    -x FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=1
  )
  local -a intra_flags=(-b 1K -e 16M -f 2 -R 1)
  local -a inter_flags=(-b 1K -e 16M -f 2 -R 1)
  # Two sizes cover both initial and reused signal/shadow/counter state while
  # keeping the forced-fallback regression reasonably small.
  local -a fallback_flags=(-b 1K -e 2K -f 2 -R 1)

  declare -p FLAGCX_CI_NODE1_MPI_ARGS >/dev/null 2>&1 || {
    echo "The platform set_env script must define FLAGCX_CI_NODE1_MPI_ARGS" >&2
    exit 1
  }
  declare -p FLAGCX_CI_NODE2_MPI_ARGS >/dev/null 2>&1 || {
    echo "The platform set_env script must define FLAGCX_CI_NODE2_MPI_ARGS" >&2
    exit 1
  }
  : "${FLAGCX_CI_INTRA_NP:?The platform set_env script must define FLAGCX_CI_INTRA_NP}"
  : "${FLAGCX_CI_NODE_NP:?The platform set_env script must define FLAGCX_CI_NODE_NP}"

  cd "$suite_dir"

  # Keep P2P enabled so the intra test covers signal/counter buffers and their
  # shadows. The INTER team below crosses the two logical nodes through NET.
  FLAGCX_CI_MPI_LABEL="unified IR intra" \
    "$MPI_RUNNER" -np "$FLAGCX_CI_INTRA_NP" --allow-run-as-root "${intra_env[@]}" \
    build/bin/test_device_ir_unified_intra "${intra_flags[@]}"

  FLAGCX_CI_MPI_LABEL="unified IR inter" \
    "$MPI_RUNNER" --allow-run-as-root \
    -np "$FLAGCX_CI_NODE_NP" "${inter_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
    build/bin/test_device_ir_unified_inter "${inter_flags[@]}" \
    : -np "$FLAGCX_CI_NODE_NP" "${inter_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
    build/bin/test_device_ir_unified_inter "${inter_flags[@]}"

  # Fault injection: disable only one-sided data/signal IPC.  Barriers retain
  # their IPC transport so these runs specifically validate IPC-to-Net
  # fallback for S18-S25.
  FLAGCX_CI_MPI_LABEL="unified IR intra forced fallback" \
    "$MPI_RUNNER" -np "$FLAGCX_CI_INTRA_NP" --allow-run-as-root \
    "${intra_fallback_env[@]}" \
    build/bin/test_device_ir_unified_intra "${fallback_flags[@]}"

  FLAGCX_CI_MPI_LABEL="unified IR inter forced fallback" \
    "$MPI_RUNNER" --allow-run-as-root \
    -np "$FLAGCX_CI_NODE_NP" "${inter_fallback_env[@]}" \
    "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
    build/bin/test_device_ir_unified_inter "${fallback_flags[@]}" \
    : -np "$FLAGCX_CI_NODE_NP" "${inter_fallback_env[@]}" \
    "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
    build/bin/test_device_ir_unified_inter "${fallback_flags[@]}"
}

run_suite() {
  local suite_dir="$PROJECT_ROOT/test/unittest/$SUITE"
  if [[ "$SUITE" == device_api_host ||
        "$SUITE" == device_api_unified_ir ]]; then
    suite_dir="$PROJECT_ROOT/test/unittest/device_api"
  fi
  local -a args=("${FLAGCX_CI_TEST_MAKE_ARGS[@]}")

  if declare -F flagcx_ci_run_suite_override >/dev/null; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
    flagcx_ci_run_suite_override "$SUITE" "$suite_dir" "${args[@]}"
    if [[ "$FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED" == 1 ]]; then
      return
    fi
  fi

  case "$SUITE" in
    adaptor)
      FLAGCX_CI_TEST_LABEL="$SUITE unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      # The core P2P FIFO relies on cross-process, cross-GPU writable IPC
      # mappings. Exercise device allocations before broader runner/RMA suites
      # so an unsupported mapping fails at its source.
      FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_ADAPTOR_MPI_TIMEOUT:-5m}" \
        FLAGCX_CI_MPI_LABEL="adaptor cross-GPU IPC tests" \
        make -C "$suite_dir" run-mpi "${args[@]}" \
        MPI_NP=2 MPIRUN="$MPI_RUNNER" \
        MPI_ARGS="-x FLAGCX_VMM_ENABLE=0"
      ;;
    core|service)
      FLAGCX_CI_TEST_LABEL="$SUITE unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      ;;
    p2p)
      FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 \
        FLAGCX_CI_TEST_LABEL="p2p unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      ;;
    rma)
      FLAGCX_CI_MPI_LABEL="rma MPI tests" \
        make -C "$suite_dir" run-mpi "${args[@]}" MPIRUN="$MPI_RUNNER"
      ;;
    runner)
      : "${FLAGCX_CI_RUNNER_NP:?The platform set_env script must define FLAGCX_CI_RUNNER_NP}"
      FLAGCX_CI_TEST_LABEL="runner unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      if [[ "${FLAGCX_CI_RUNNER_MPI_SUPPORTED:-1}" != "1" ]]; then
        echo "Skipping runner MPI tests: ${FLAGCX_CI_RUNNER_MPI_SKIP_REASON:-unsupported by this platform}"
        return
      fi
      cd "$suite_dir"
      FLAGCX_CI_MPI_LABEL="runner default" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        ./build/bin/runner_mpi_tests
      FLAGCX_CI_MPI_LABEL="runner heterogeneous" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        ./build/bin/runner_mpi_tests
      # A platform may opt out when forcing NET cannot select a real RDMA
      # adaptor. Do not silently turn this coverage into a socket test.
      if [[ "${FLAGCX_CI_RUNNER_FORCE_NET_SUPPORTED:-1}" == "1" ]]; then
        FLAGCX_CI_MPI_LABEL="runner forced NET" \
          "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
          -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
          -x FLAGCX_P2P_DISABLE=1 \
          -x FLAGCX_VMM_ENABLE=0 \
          ./build/bin/runner_mpi_tests
      else
        echo "Skipping forced-NET runner test: ${FLAGCX_CI_RUNNER_FORCE_NET_SKIP_REASON:-unsupported by this platform}"
      fi
      FLAGCX_CI_MPI_LABEL="runner registered heterogeneous" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_TEST_REGISTER_BUFFERS=1 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_VMM_ENABLE=0 \
        ./build/bin/runner_mpi_tests
      if [[ "${FLAGCX_CI_RUNNER_FORCE_NET_SUPPORTED:-1}" == "1" ]]; then
        FLAGCX_CI_MPI_LABEL="runner registered forced NET" \
          "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
          -x FLAGCX_TEST_REGISTER_BUFFERS=1 \
          -x FLAGCX_MEM_ENABLE=1 \
          -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
          -x FLAGCX_P2P_DISABLE=1 \
          -x FLAGCX_VMM_ENABLE=0 \
          ./build/bin/runner_mpi_tests
      else
        echo "Skipping registered forced-NET runner test: ${FLAGCX_CI_RUNNER_FORCE_NET_SKIP_REASON:-unsupported by this platform}"
      fi
      ;;
    symmem)
      bash "$PROJECT_ROOT/test/script/symmem_test.sh"
      ;;
    device_api)
      run_device_api
      ;;
    device_api_host)
      run_device_api_host
      ;;
    device_api_unified_ir)
      run_device_api_unified_ir
      ;;
    *)
      echo "Unsupported unit test suite: $SUITE" >&2
      exit 2
      ;;
  esac
}

case "$SUITE" in
  adaptor|core|device_api_host|p2p|rma|runner|service|symmem)
    build_googletest
    ;;
  device_api|device_api_unified_ir)
    ;;
  *)
    echo "Unsupported unit test suite: $SUITE" >&2
    exit 2
    ;;
esac

build_project
build_suite
run_suite
