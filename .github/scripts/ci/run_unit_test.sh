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

# Keep every hardware backend on the same diagnostic and allocation baseline.
# Individual invocations may add transport selectors, but must not silently
# change the device-memory allocator or reduce the information available when a
# hardware-only failure needs to be diagnosed.
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=ALL
export FLAGCX_VMM_ENABLE=0

: "${MPI_HOME:?The platform set_env script must define MPI_HOME}"
declare -p FLAGCX_CI_PROJECT_MAKE_ARGS >/dev/null 2>&1 || {
  echo "The platform set_env script must define FLAGCX_CI_PROJECT_MAKE_ARGS" >&2
  exit 1
}
declare -p FLAGCX_CI_TEST_MAKE_ARGS >/dev/null 2>&1 || {
  echo "The platform set_env script must define FLAGCX_CI_TEST_MAKE_ARGS" >&2
  exit 1
}
if ! declare -p FLAGCX_CI_IBUC_ENV >/dev/null 2>&1; then
  FLAGCX_CI_IBUC_ENV=()
fi
: "${FLAGCX_CI_ENABLE_IBUC:=0}"
: "${FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE:=0}"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${LD_LIBRARY_PATH:-}"

flagcx_ci_require_rdma() {
  local suite=$1
  local platform_name

  platform_name=$(basename "$SET_ENV_SCRIPT" .sh)
  case "$platform_name" in
    cuda|metax|hygon|ppu) ;;
    *) return 0 ;;
  esac

  case "$suite" in
    adaptor|p2p|rma|runner|symmem) ;;
    *) return 0 ;;
  esac

  echo "Running $platform_name static RDMA preflight for unit-test suite: $suite"
  if ! declare -F flagcx_ci_validate_rdma >/dev/null; then
    echo "$platform_name does not provide the required static RDMA validator." >&2
    return 1
  fi
  flagcx_ci_validate_rdma "$suite"

  echo "Static RDMA preflight passed"
}

if declare -F flagcx_ci_prepare >/dev/null; then
  flagcx_ci_prepare "$SUITE"
fi
# RMA has separate IPC and network invocations. Its RDMA preflight runs only
# before the network invocation so missing RDMA cannot hide IPC regressions.
if [[ "$SUITE" != rma ]]; then
  flagcx_ci_require_rdma "$SUITE"
fi

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
    -x FLAGCX_KERNEL_PROXY_PARALLELISM=4
    -x FLAGCX_IB_QPS_PER_CONNECTION=2
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
    -x FLAGCX_DEBUG=INFO
    -x FLAGCX_DEBUG_SUBSYS=ALL
  )
  local -a inter_env=(
    "${common_env[@]}"
    -x FLAGCX_DEBUG=INFO
    -x FLAGCX_DEBUG_SUBSYS=ALL
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
      local unit_status=0
      local ibuc_status=0
      local ipc_status=0
      local geometry_status=0
      # Build more than one RC QP in every RDMA adaptor job. The current
      # one-sided API intentionally stays on one ordered QP until the transport
      # layer can express QP selection together with ordering boundaries.
      FLAGCX_IB_QPS_PER_CONNECTION=2 \
        FLAGCX_CI_TEST_LABEL="$SUITE unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}" || \
        unit_status=$?

      if ((FLAGCX_CI_ENABLE_IBUC != 0)); then
        # IBUC is another implementation of the net-adaptor contract, not an
        # independent test suite. Build it in isolated output directories so
        # the regular IBRC binary remains available, then run its contract.
        local ibuc_project_build="$PROJECT_ROOT/build-ibuc"
        local ibuc_test_build="$suite_dir/build-ibuc"
        local -a ibuc_project_args=(
          "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
          USE_IBUC=1
          BUILDDIR="$ibuc_project_build"
        )
        local -a ibuc_test_args=(
          "${args[@]}"
          USE_IBUC=1
          BUILDDIR="$ibuc_test_build"
          FLAGCX_LIB="$ibuc_project_build/lib"
        )
        make -C "$PROJECT_ROOT" --jobs="$(nproc)" \
          "${ibuc_project_args[@]}" || ibuc_status=$?
        if ((ibuc_status == 0)); then
          make -C "$suite_dir" --jobs="$(nproc)" \
            "${ibuc_test_args[@]}" || ibuc_status=$?
        fi
        if ((ibuc_status == 0)); then
          env "${FLAGCX_CI_IBUC_ENV[@]}" \
            LD_LIBRARY_PATH="$ibuc_project_build/lib:$LD_LIBRARY_PATH" \
            FLAGCX_CI_EXPECT_NET_ADAPTOR=IBUC \
            FLAGCX_IB_QPS_PER_CONNECTION=2 \
            FLAGCX_IBUC_SPLIT_DATA_ON_QPS=1 \
            GTEST_FILTER="NetAdaptorInterface.IbucAdvertisesTwoSidedContract:NetAdaptorLoopback.SendRecv:NetAdaptorLoopback.RegisterGpuMr:NetAdaptorLoopback.Ibuc*:IbucOwnershipTest.*:IbucRetransmissionTest.*" \
            FLAGCX_CI_TEST_LABEL="IBUC net adaptor tests" \
            "$TEST_RUNNER" make -C "$suite_dir" run-unit \
              "${ibuc_test_args[@]}" || ibuc_status=$?
        fi
      fi
      # Exercise device IPC handles across processes and physical devices. Use
      # exactly two ranks so GPU 0 and GPU 1 exercise both exporter/importer
      # directions and the full-mesh mapping setup used by RMA, with an
      # independent timeout from the adaptor unit tests. Always run this
      # invocation even when the RDMA loopback tests fail so an RDMA environment
      # problem cannot hide device IPC coverage. Disable VMM to exercise the
      # same IPC-exportable GDR allocation used by the RMA suite.
      FLAGCX_CI_MPI_LABEL="$SUITE IPC MPI tests" \
        make -C "$suite_dir" run-mpi "${args[@]}" \
        MPIRUN="$MPI_RUNNER" MPI_NP=2 \
        MPI_ENV="-x FLAGCX_VMM_ENABLE=0" || ipc_status=$?

      # Exercise the real IBRC connection handshake with asymmetric rank-local
      # settings. Both peers must report the same terminal error, cleanup must
      # not hang, and neither connector may be published as connected.
      if [[ "$(basename "$SET_ENV_SCRIPT" .sh)" == "cuda" ]]; then
        local geometry_filter="IbConnectionGeometryMpiTest.MismatchedPeersFailConsistentlyWithoutPublishingConnectors"
        local -a geometry_common_env=(
          -x FLAGCX_USE_HETERO_COMM=1
          -x FLAGCX_MEM_ENABLE=1
          -x FLAGCX_VMM_ENABLE=0
          -x FLAGCX_P2P_DISABLE=1
          -x FLAGCX_CI_EXPECT_IB_GEOMETRY_MISMATCH=1
          -x LD_LIBRARY_PATH
        )
        FLAGCX_CI_MPI_TIMEOUT=5m \
          FLAGCX_CI_MPI_LABEL="IBRC QP-count mismatch" \
          "$MPI_RUNNER" --allow-run-as-root \
          -np 1 "${geometry_common_env[@]}" \
          "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
          -x FLAGCX_IB_QPS_PER_CONNECTION=1 \
          -x FLAGCX_IB_SPLIT_DATA_ON_QPS=0 \
          "$suite_dir/build/bin/adaptor_mpi_tests" \
          --gtest_filter="$geometry_filter" \
          : -np 1 "${geometry_common_env[@]}" \
          "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
          -x FLAGCX_IB_QPS_PER_CONNECTION=2 \
          -x FLAGCX_IB_SPLIT_DATA_ON_QPS=0 \
          "$suite_dir/build/bin/adaptor_mpi_tests" \
          --gtest_filter="$geometry_filter" || geometry_status=$?

        FLAGCX_CI_MPI_TIMEOUT=5m \
          FLAGCX_CI_MPI_LABEL="IBRC split-data mismatch" \
          "$MPI_RUNNER" --allow-run-as-root \
          -np 1 "${geometry_common_env[@]}" \
          "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
          -x FLAGCX_IB_QPS_PER_CONNECTION=2 \
          -x FLAGCX_IB_SPLIT_DATA_ON_QPS=0 \
          "$suite_dir/build/bin/adaptor_mpi_tests" \
          --gtest_filter="$geometry_filter" \
          : -np 1 "${geometry_common_env[@]}" \
          "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
          -x FLAGCX_IB_QPS_PER_CONNECTION=2 \
          -x FLAGCX_IB_SPLIT_DATA_ON_QPS=1 \
          "$suite_dir/build/bin/adaptor_mpi_tests" \
          --gtest_filter="$geometry_filter" || geometry_status=$?
      fi
      if ((unit_status != 0 || ibuc_status != 0 || ipc_status != 0 ||
           geometry_status != 0)); then
        echo "Adaptor failures: IBRC=$unit_status IBUC=$ibuc_status IPC=$ipc_status geometry=$geometry_status" >&2
        return 1
      fi
      ;;
    core|service)
      FLAGCX_CI_TEST_LABEL="$SUITE unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      ;;
    p2p)
      local unit_status=0
      local mpi_status=0
      local shared_status=0
      local shared_mpi_status=0
      local p2p_mpi_env=""
      if [[ "${FLAGCX_P2P_TRANSPORT:-ib}" == "accl" ]]; then
        p2p_mpi_env="-x FLAGCX_P2P_TRANSPORT=accl"
      fi
      FLAGCX_P2P_QPS_PER_CONN=2 FLAGCX_IB_QPS_PER_CONNECTION=2 \
        FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0 \
        FLAGCX_CI_TEST_LABEL="p2p unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}" || \
        unit_status=$?
      FLAGCX_P2P_QPS_PER_CONN=2 FLAGCX_IB_QPS_PER_CONNECTION=2 \
        FLAGCX_CI_MPI_LABEL="p2p Engine WRITE MPI tests" \
        make -C "$suite_dir" run-mpi "${args[@]}" \
        MPIRUN="$MPI_RUNNER" \
        MPI_ENV="$p2p_mpi_env" || mpi_status=$?
      # Keep the production/default artifact on the legacy Engine. Platforms
      # build the shared implementation in isolated output trees and run the
      # same unit and WRITE MPI tests against it.
      if ((FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE != 0)); then
        local shared_project_build="$PROJECT_ROOT/build-p2p-shared"
        local shared_test_build="$suite_dir/build-p2p-shared"
        local -a shared_project_args=(
          "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
          USE_SHARED_P2P_ENGINE=1
          BUILDDIR="$shared_project_build"
        )
        local -a shared_test_args=(
          "${args[@]}"
          USE_SHARED_P2P_ENGINE=1
          BUILDDIR="$shared_test_build"
          FLAGCX_LIB="$shared_project_build/lib"
        )

        make -C "$PROJECT_ROOT" --jobs="$(nproc)" \
          "${shared_project_args[@]}" || shared_status=$?
        if ((shared_status == 0)); then
          make -C "$suite_dir" --jobs="$(nproc)" \
            "${shared_test_args[@]}" || shared_status=$?
        fi
        if ((shared_status == 0)); then
          LD_LIBRARY_PATH="$shared_project_build/lib:$LD_LIBRARY_PATH" \
            FLAGCX_P2P_QPS_PER_CONN=2 FLAGCX_IB_QPS_PER_CONNECTION=2 \
            FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 \
            FLAGCX_VMM_ENABLE=0 \
            FLAGCX_CI_TEST_LABEL="p2p shared-engine unit tests" \
            "$TEST_RUNNER" make -C "$suite_dir" run-unit \
              "${shared_test_args[@]}" || shared_status=$?
        fi
        if ((shared_status == 0)); then
          LD_LIBRARY_PATH="$shared_project_build/lib:$LD_LIBRARY_PATH" \
            FLAGCX_P2P_QPS_PER_CONN=2 FLAGCX_IB_QPS_PER_CONNECTION=2 \
            FLAGCX_CI_MPI_LABEL="p2p shared-engine WRITE MPI tests" \
            make -C "$suite_dir" run-mpi \
              "${shared_test_args[@]}" MPIRUN="$MPI_RUNNER" \
              MPI_ENV="$p2p_mpi_env" || shared_mpi_status=$?
        fi
      fi

      if ((unit_status != 0 || mpi_status != 0 || shared_status != 0 ||
           shared_mpi_status != 0)); then
        echo "P2P failures: unit=$unit_status MPI_WRITE=$mpi_status shared=$shared_status shared_MPI_WRITE=$shared_mpi_status" >&2
        return 1
      fi
      ;;
    rma)
      local ipc_status=0
      local network_status=0
      local visibility_status=0
      FLAGCX_CI_TEST_LABEL="rma unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      # Keep these as separate invocations so each transport has its own
      # timeout. Collect both statuses so a failure in one path cannot prevent
      # the other path from running and producing a useful result.
      FLAGCX_CI_MPI_LABEL="rma IPC MPI tests" \
        make -C "$suite_dir" run-mpi-ipc "${args[@]}" \
        MPIRUN="$MPI_RUNNER" || ipc_status=$?
      if flagcx_ci_require_rdma "$SUITE"; then
        FLAGCX_CI_MPI_LABEL="rma network MPI tests" \
          make -C "$suite_dir" run-mpi-net "${args[@]}" \
          MPIRUN="$MPI_RUNNER" || network_status=$?

        # Visibility conformance is deliberately a separate process from the
        # general RMA suite: requirement overrides and VMM route selection are
        # cached once, and each case must exercise one unambiguous policy.
        local platform_name visibility_filter base_rma_platform_env
        local expected_read=success expected_write=success
        local expected_read_required=1 expected_write_required=
        platform_name=$(basename "$SET_ENV_SCRIPT" .sh)
        visibility_filter="RmaTest.DirectConsumerReadVisibility:RmaTest.DirectConsumerWriteVisibility"
        base_rma_platform_env="${RMA_PLATFORM_ENV:-}"
        case "$platform_name" in
          metax)
            # MACA can complete GET visibility through the provider flush, but
            # its stream wait cannot acquire incoming remote WRITEs yet.
            expected_write=unsupported
            expected_write_required=1
            ;;
          hygon)
            # DU uses the existing RMA correctness tests. No separate device
            # consumer kernel is built for this platform.
            ;;
          ppu)
            # BAREX intentionally retains a temporary no-op/NONE policy. The
            # READ consumer runs; WRITE records a capability note if BAREX
            # cannot send a one-sided signal. Forced READ checks fail-close.
            expected_read_required=0
            expected_write_required=0
            ;;
          cuda)
            # READ stays required. WRITE is resolved from compute capability
            # and GPU/NIC/CPU topology, so do not hard-code a pre-Hopper bit.
            ;;
          *)
            echo "Unsupported visibility conformance platform: $platform_name" >&2
            visibility_status=1
            ;;
        esac

        local -a visibility_common_env=(
          -x FLAGCX_CI_GDR_VISIBILITY_RUN=1
          -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_READ="$expected_read"
          -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_WRITE="$expected_write"
          -x FLAGCX_CI_EXPECT_GDR_READ_REQUIRED="$expected_read_required"
          -x FLAGCX_CI_EXPECT_VMM_MR_ROUTE=none
        )
        if [[ -n "$expected_write_required" ]]; then
          visibility_common_env+=(
            -x FLAGCX_CI_EXPECT_GDR_WRITE_REQUIRED="$expected_write_required"
          )
        fi
        if [[ "$platform_name" != "hygon" ]]; then
          FLAGCX_CI_MPI_LABEL="rma GDR visibility ordinary" \
            make -C "$suite_dir" run-mpi-net "${args[@]}" \
            MPIRUN="$MPI_RUNNER" NET_FILTER="$visibility_filter" \
            RMA_VMM_ENABLE=0 \
            RMA_PLATFORM_ENV="$base_rma_platform_env ${visibility_common_env[*]}" || \
            visibility_status=$?
        fi

        if [[ "$platform_name" == "cuda" ]]; then
          # CUDA validates a VMM route when available. Some runners cannot
          # export DMA-BUF even though VA registration works; the fixture
          # records a capability note and skips that route on those machines.
          local route
          for route in va dmabuf; do
            local -a cuda_vmm_env=(
              -x FLAGCX_CI_GDR_VISIBILITY_RUN=1
              -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_SETUP=success_or_unsupported
              -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_READ=success
              -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_WRITE=success
              -x FLAGCX_CI_EXPECT_GDR_READ_REQUIRED=1
              # VMM route does not alter visibility policy; CUDA WRITE remains
              # topology-derived here just as it is for ordinary allocations.
              -x FLAGCX_CI_EXPECT_VMM_MR_ROUTE="$route"
              -x FLAGCX_VMM_MR_MODE="$route"
            )
            FLAGCX_CI_MPI_LABEL="rma GDR visibility VMM $route" \
              make -C "$suite_dir" run-mpi-net "${args[@]}" \
              MPIRUN="$MPI_RUNNER" NET_FILTER="$visibility_filter" \
              RMA_VMM_ENABLE=1 \
              RMA_PLATFORM_ENV="$base_rma_platform_env ${cuda_vmm_env[*]}" || \
              visibility_status=$?
          done
        elif [[ "$platform_name" == "hygon" ]]; then
          # SHCA VMM VA must fail closed. If DMA-BUF is available, GetSmall
          # checks its data path; otherwise the fixture records a capability
          # note. DU has no separate consumer kernel in this test suite.
          local route setup_expectation
          for route in va dmabuf; do
            if [[ "$route" == "va" ]]; then
              setup_expectation=unsupported
            else
              setup_expectation=success_or_unsupported
            fi
            local -a hygon_vmm_env=(
              -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_SETUP="$setup_expectation"
              -x FLAGCX_VMM_MR_MODE="$route"
              -x FLAGCX_IB_TIMEOUT=14
              -x FLAGCX_IB_RETRY_CNT=1
            )
            FLAGCX_CI_MPI_LABEL="rma Hygon VMM route $route" \
              make -C "$suite_dir" run-mpi-net "${args[@]}" \
              MPIRUN="$MPI_RUNNER" NET_FILTER="RmaTest.GetSmall" \
              RMA_VMM_ENABLE=1 \
              RMA_PLATFORM_ENV="$base_rma_platform_env ${hygon_vmm_env[*]}" || \
              visibility_status=$?
          done
        elif [[ "$platform_name" == "ppu" ]]; then
          # Forcing READ turns the temporary NONE policy into a hard contract.
          # BAREX has no real READ flush capability, so completion must fail
          # closed before a consumer kernel is launched.
          local -a ppu_forced_read_env=(
            -x FLAGCX_CI_GDR_VISIBILITY_RUN=1
            # The unsupported flush is discovered asynchronously by the RMA
            # proxy, so WaitCounter exposes its terminal flag as RemoteError.
            -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_READ=remote_error
            -x FLAGCX_CI_GDR_VISIBILITY_EXPECT_WRITE=success
            -x FLAGCX_CI_EXPECT_GDR_READ_REQUIRED=1
            -x FLAGCX_CI_EXPECT_GDR_WRITE_REQUIRED=0
            -x FLAGCX_CI_EXPECT_VMM_MR_ROUTE=none
            -x FLAGCX_GDR_READ_REQUIRES_FLUSH=1
          )
          FLAGCX_CI_MPI_LABEL="rma GDR visibility forced unsupported READ" \
            make -C "$suite_dir" run-mpi-net "${args[@]}" \
            MPIRUN="$MPI_RUNNER" \
            NET_FILTER="RmaTest.DirectConsumerReadVisibility" \
            RMA_VMM_ENABLE=0 \
            RMA_PLATFORM_ENV="$base_rma_platform_env ${ppu_forced_read_env[*]}" || \
            visibility_status=$?
        fi
      else
        network_status=$?
      fi
      if ((ipc_status != 0 || network_status != 0 || visibility_status != 0)); then
        echo "RMA MPI failures: IPC=$ipc_status network=$network_status visibility=$visibility_status" >&2
        return 1
      fi
      ;;
    runner)
      : "${FLAGCX_CI_RUNNER_NP:?The platform set_env script must define FLAGCX_CI_RUNNER_NP}"
      local platform_name
      local -a runner_net_platform_env=()
      platform_name=$(basename "$SET_ENV_SCRIPT" .sh)
      if [[ "$platform_name" == "hygon" ]]; then
        # A broken SHCA route otherwise consumes the default RC retry budget
        # for several minutes before the completion reports RETRY_EXC_ERR.
        runner_net_platform_env+=(
          -x FLAGCX_IB_TIMEOUT=14
          -x FLAGCX_IB_RETRY_CNT=1
        )
      fi
      FLAGCX_CI_TEST_LABEL="runner unit tests" \
        "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
      cd "$suite_dir"
      FLAGCX_CI_MPI_LABEL="runner default" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_CI_EXPECT_RUNNER_MODE=HOMO \
        ./build/bin/runner_mpi_tests
      FLAGCX_CI_MPI_LABEL="runner heterogeneous" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_P2P_DISABLE=0 \
        -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
        -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=P2P \
        ./build/bin/runner_mpi_tests
      FLAGCX_CI_MPI_LABEL="runner forced NET" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_P2P_DISABLE=1 \
        -x FLAGCX_PXN_DISABLE=1 \
        -x FLAGCX_VMM_ENABLE=0 \
        -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
        -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET \
        -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB \
        -x FLAGCX_IB_QPS_PER_CONNECTION=2 \
        -x FLAGCX_IB_SPLIT_DATA_ON_QPS=0 \
        -x FLAGCX_IBUC_SPLIT_DATA_ON_QPS=0 \
        -x FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1 \
        -x FLAGCX_CI_EXPECT_PXN=0 \
        "${runner_net_platform_env[@]}" \
        ./build/bin/runner_mpi_tests
      FLAGCX_CI_MPI_LABEL="runner forced NET multi-QP striping" \
        "$MPI_RUNNER" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_P2P_DISABLE=1 \
        -x FLAGCX_PXN_DISABLE=1 \
        -x FLAGCX_VMM_ENABLE=0 \
        -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
        -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB \
        -x FLAGCX_IB_QPS_PER_CONNECTION=2 \
        -x FLAGCX_IB_SPLIT_DATA_ON_QPS=1 \
        -x FLAGCX_IBUC_SPLIT_DATA_ON_QPS=1 \
        -x FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1 \
        -x FLAGCX_CI_EXPECT_COLL_QP_STRIPING=1 \
        -x FLAGCX_CI_EXPECT_PXN=0 \
        -x FLAGCX_CI_RUNNER_BYTES=67108864 \
        "${runner_net_platform_env[@]}" \
        ./build/bin/runner_mpi_tests \
        --gtest_filter=FlagCXCollTest.AlltoAll
      if [[ "${FLAGCX_CI_ENABLE_PXN:-1}" == "1" ]]; then
        local -a pxn_platform_env=()
        local -a pxn_launcher=(env)
        if [[ "$platform_name" == "hygon" ]]; then
          # The baseline uses four visible DCUs. Expose all eight here so
          # ranks 0-3 and 4-7 form the same two logical groups as other CI
          # platforms.
          pxn_launcher+=(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
          pxn_platform_env+=(
            -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            -x FLAGCX_IB_HCA="$FLAGCX_CI_HYGON_CONNECTED_HCAS"
          )
        fi
        FLAGCX_CI_MPI_LABEL="runner $platform_name PXN eight-rank" \
          "${pxn_launcher[@]}" "$MPI_RUNNER" -np 8 --allow-run-as-root \
          -x FLAGCX_MEM_ENABLE=1 \
          -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
          -x FLAGCX_P2P_DISABLE=1 \
          -x FLAGCX_PXN_DISABLE=0 \
          -x FLAGCX_VMM_ENABLE=0 \
          -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
          -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET \
          -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB \
          -x FLAGCX_CI_EXPECT_PXN=1 \
          "${pxn_platform_env[@]}" \
          "${runner_net_platform_env[@]}" \
          ./build/bin/runner_mpi_tests \
          --gtest_filter=FlagCXCollTest.SendRecv:FlagCXCollTest.AllReduce:FlagCXCollTest.AlltoAll
      fi
      ;;
    symmem)
      bash "$PROJECT_ROOT/test/script/symmem_test.sh"
      : "${FLAGCX_CI_SYMMEM_NODE_NP:=2}"
      declare -p FLAGCX_CI_NODE1_MPI_ARGS >/dev/null 2>&1 || {
        echo "symmem NET tests require FLAGCX_CI_NODE1_MPI_ARGS" >&2
        return 1
      }
      declare -p FLAGCX_CI_NODE2_MPI_ARGS >/dev/null 2>&1 || {
        echo "symmem NET tests require FLAGCX_CI_NODE2_MPI_ARGS" >&2
        return 1
      }
      local platform_name expected_adaptor
      local symmem_run_vmm_net_data=1
      local -a symmem_common_env symmem_platform_env
      platform_name=$(basename "$SET_ENV_SCRIPT" .sh)
      expected_adaptor=IB
      symmem_platform_env=()
      if [[ "$platform_name" == "ppu" ]]; then
        expected_adaptor=BAREX
        symmem_platform_env+=( -x FLAGCX_P2P_TRANSPORT=accl )
      fi
      # PPU has no VMM-capable BAREX MR route. SHCA's unsafe VMM VA route is
      # disabled and Hygon's DMA-BUF route is independently exercised by the
      # RMA immediate-consumer job. Keep the broader multi-node symmem VMM data
      # matrix disabled until that hardware job establishes a supported route.
      if [[ "$platform_name" == "ppu" || "$platform_name" == "hygon" ]]; then
        symmem_run_vmm_net_data=0
      fi
      symmem_common_env=(
        -x FLAGCX_USE_HETERO_COMM=1
        -x FLAGCX_CLUSTER_SPLIT_LIST=2
        -x FLAGCX_MEM_ENABLE=1
        -x FLAGCX_IB_DISABLE=0
        -x FLAGCX_P2P_DISABLE=1
        -x FLAGCX_CI_REQUIRE_NET_MR=1
        -x FLAGCX_CI_EXPECT_NET_ADAPTOR="$expected_adaptor"
        -x LD_LIBRARY_PATH
      )
      local symmem_bin="$PROJECT_ROOT/test/unittest/symmem/build/bin/symmem_mpi_tests"
      local symmem_filter="--gtest_filter=SymMemTest.HybridLocalAndRemoteAccess:SymMemTest.RankLocalStatusFailureConvergesDeterministically:SymMemTest.FullMeshRoundFailureConvergesAndRetries:SymMemTest.RankLocalMrFailureDoesNotPublishPartialWindow:SymMemTest.MrRollbackFailureRetainsWindowLeaseUntilRetry:SymMemTest.AsymmetricDeregisterUsesCollectivePublishSlot:SymMemTest.PublicRegistrationUsesVmmMrRouting:SymMemTest.VmmFlatFallbackPreservesMrRoute:SymMemTest.DirectGdrVmmPreservesMrRoute:SymMemTest.VmmRollbackFailureConvergesBeforeMrMetadataExchange:SymMemTest.CrossNodeCleanupFailureConvergesBeforeRelease:SymMemTest.SignalRegistrationUsesAllocationProvenance:SymMemTest.RankLocalSignalMrFailurePreservesErrorAndRetries:SymMemTest.RepeatedRegisterDeregister:SymMemTest.DuplicateWindowsShareMrUntilLastDeregister:SymMemTest.CommDestroyReleasesLiveWindow"

      FLAGCX_CI_MPI_LABEL="symmem remote without NET" \
        "$MPI_RUNNER" --allow-run-as-root \
        -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        -x FLAGCX_USE_HETERO_COMM=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_CI_REQUIRE_VMM=1 \
        -x FLAGCX_IB_DISABLE=1 \
        -x FLAGCX_P2P_DISABLE=1 \
        -x FLAGCX_CI_REQUIRE_REMOTE_NO_NET=1 \
        -x LD_LIBRARY_PATH \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
        "$symmem_bin" \
        --gtest_filter=SymMemTest.RemotePeersWithoutNetworkDoNotPublishWindow \
        : -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        -x FLAGCX_USE_HETERO_COMM=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_CI_REQUIRE_VMM=1 \
        -x FLAGCX_IB_DISABLE=1 \
        -x FLAGCX_P2P_DISABLE=1 \
        -x FLAGCX_CI_REQUIRE_REMOTE_NO_NET=1 \
        -x LD_LIBRARY_PATH \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
        "$symmem_bin" \
        --gtest_filter=SymMemTest.RemotePeersWithoutNetworkDoNotPublishWindow

      FLAGCX_CI_MPI_LABEL="symmem local IPC + NET fallback" \
        "$MPI_RUNNER" --allow-run-as-root \
        -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=0 \
        -x FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK=1 \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
        "$symmem_bin" \
        --gtest_filter=SymMemTest.RankLocalIpcFailureUsesNetworkMrFallback:SymMemTest.P2pDisabledLocalWindowAcquiresNetworkMrBeforePublish

      FLAGCX_CI_MPI_LABEL="symmem IPC + NET" \
        "$MPI_RUNNER" --allow-run-as-root \
        -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=0 \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
        "$symmem_bin" "$symmem_filter" \
        : -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=0 \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
        "$symmem_bin" "$symmem_filter"

      if ((symmem_run_vmm_net_data != 0)); then
        FLAGCX_CI_MPI_LABEL="symmem VMM + NET auto" \
          "$MPI_RUNNER" --allow-run-as-root \
          -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
          "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=1 \
          -x FLAGCX_CI_REQUIRE_VMM=1 \
          -x FLAGCX_VMM_MR_MODE=auto \
          "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
          "$symmem_bin" "$symmem_filter" \
          : -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
          "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=1 \
          -x FLAGCX_CI_REQUIRE_VMM=1 \
          -x FLAGCX_VMM_MR_MODE=auto \
          "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
          "$symmem_bin" "$symmem_filter"
      else
        echo "Skipping $platform_name VMM + NET data tests: no validated VMM MR data route"
      fi

      local -a symmem_route_union_env=()
      if ((symmem_run_vmm_net_data == 0)); then
        symmem_route_union_env+=( -x FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED=1 )
      fi
      FLAGCX_CI_MPI_LABEL="symmem VMM + NET route union" \
        "$MPI_RUNNER" --allow-run-as-root \
        -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_CI_REQUIRE_VMM=1 \
        -x FLAGCX_CI_REQUIRE_VMM_ROUTE_UNION=1 \
        -x FLAGCX_VMM_MR_MODE=auto \
        "${symmem_route_union_env[@]}" \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
        "$symmem_bin" \
        --gtest_filter=SymMemTest.VmmNetRouteCapabilityUnion \
        : -np "$FLAGCX_CI_SYMMEM_NODE_NP" \
        "${symmem_common_env[@]}" -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_CI_REQUIRE_VMM=1 \
        -x FLAGCX_CI_REQUIRE_VMM_ROUTE_UNION=1 \
        -x FLAGCX_VMM_MR_MODE=auto \
        "${symmem_route_union_env[@]}" \
        "${symmem_platform_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
        "$symmem_bin" \
        --gtest_filter=SymMemTest.VmmNetRouteCapabilityUnion
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
