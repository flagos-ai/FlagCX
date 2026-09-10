#!/usr/bin/env bash

# MetaX-specific unit-test environment setup.

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

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

export PATH="/opt/maca/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_RUNNER_NP=8
export NP=8

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    p2p)
      # The MetaX CI RoCE environment cannot establish IB_P2P QPs reliably yet.
      # Keep structure/bootstrap/slice tests enabled and skip real IB_P2P paths.
      export GTEST_FILTER="-FlagcxP2pEngineReadTest.*:P2pLoopbackTest.*:P2pBatchTest.*:P2pEngineRpcIbTest.*"
      ;;
    rma)
      FLAGCX_CI_TEST_MAKE_ARGS+=(
        "RMA_PLATFORM_ENV=-x FLAGCX_USE_TUNER=1 -x TUNNING_WITH_SINGLE_COMM=1 -x FLAGCX_USE_HOST_COMM=1 -x FLAGCX_P2P_DISABLE=1"
      )
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing MetaX environment for unit-test suite: $suite"
  command -v mpirun
  command -v mxcc

  if [[ "$suite" == "adaptor" || "$suite" == "p2p" ||
        "$suite" == "rma" ]]; then
    local -a hca_paths=()
    local -a hca_names=()
    local hca_path

    shopt -s nullglob
    # Prefer the native RoCE devices while retaining compatibility with MetaX
    # runners that expose the same links through the bonded RDMA driver.
    hca_paths=(/sys/class/infiniband/bnxt_roce*)
    if [[ ${#hca_paths[@]} -eq 0 ]]; then
      hca_paths=(/sys/class/infiniband/bnxt_re_bond*)
    fi
    shopt -u nullglob

    if [[ ${#hca_paths[@]} -eq 0 ]]; then
      if [[ "$suite" != "rma" ]]; then
        echo "MetaX $suite tests require bnxt_roce* or bnxt_re_bond* RDMA devices." >&2
        return 1
      fi
      # The RMA suite runs an explicit IPC invocation before its RDMA
      # preflight. Leave HCA selection unset here so missing RDMA does not hide
      # IPC regressions; flagcx_ci_validate_rdma will reject the NET phase.
      echo "MetaX RMA IPC phase will run without a detected Broadcom RDMA HCA."
    else
      for hca_path in "${hca_paths[@]}"; do
        hca_names+=("${hca_path##*/}")
      done
      if [[ -z "${FLAGCX_IB_HCA:-}" ]]; then
        local IFS=,
        export FLAGCX_IB_HCA="${hca_names[*]}"
      fi
    fi

    if [[ -d /sys/class/net/bond0 ]]; then
      export FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-bond0}
    fi

    export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
    export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,ENV}

    echo "MetaX network diagnostics:"
    echo "FLAGCX_IB_HCA=${FLAGCX_IB_HCA:-<unset>}"
    echo "FLAGCX_IB_GID_INDEX=${FLAGCX_IB_GID_INDEX:-<unset>}"
    echo "FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-<unset>}"
    echo "Network interfaces visible inside the CI container:"
    ls /sys/class/net 2>/dev/null || true
    echo "RDMA devices visible inside the CI container:"
    ls /sys/class/infiniband 2>/dev/null || true
    ls /dev/infiniband 2>/dev/null || true
    ip -o addr show 2>/dev/null || true
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1

  if ! compgen -G "/sys/class/infiniband/bnxt_roce*" >/dev/null &&
    ! compgen -G "/sys/class/infiniband/bnxt_re_bond*" >/dev/null; then
    echo "MetaX $suite tests require bnxt_roce* or bnxt_re_bond* RDMA devices." >&2
    return 1
  fi
}

flagcx_ci_build_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

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

  if [[ "$suite" == "runner" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    FLAGCX_CI_TEST_LABEL="runner unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}"
    echo "Skipping MetaX runner MPI tests: mcclAllGather segfaults in the current MCCL backend."
    return
  fi

  if [[ "$suite" == "symmem" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    FLAGCX_CI_TEST_LABEL="symmem unit tests" \
      "$TEST_RUNNER" "$suite_dir/build/bin/symmem_unit_tests"
    echo "Skipping MetaX symmem MPI tests: symmetric windows are not supported by the current MetaX backend."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}
