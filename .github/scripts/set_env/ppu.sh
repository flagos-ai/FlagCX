#!/usr/bin/env bash

# T-Head PPU-specific unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

export FLAGCX_ADAPTOR=ppu

export DEVICE_HOME="${DEVICE_HOME:-/usr/local/PPU_SDK}"
export CCL_HOME="${CCL_HOME:-/usr/local/PPU_SDK}"
export PATH="$DEVICE_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CCL_HOME/lib:$DEVICE_HOME/lib:$DEVICE_HOME/CUDA_SDK/lib64:${LD_LIBRARY_PATH:-}"
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
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
FLAGCX_CI_PPU_FOUR_GPU_DEVICES=0,1,2,3
export NP=4
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,3,4,7

# Two logical nodes on the shared PPU host. Symmem uses these app contexts to
# require a real BAREX MR for remote peers while retaining local flat mappings.
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
    perf-test-experimental)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      export CUDA_VISIBLE_DEVICES="$FLAGCX_CI_PPU_FOUR_GPU_DEVICES"
      ;;
    device_api|device_api_host|device_api_unified_ir)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      if [[ "$suite" == device_api_host ]]; then
        export FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=${FLAGCX_CI_TEST_DIAGNOSTICS_AFTER:-2}
      fi
      if [[ "$suite" != device_api_host ]]; then
        # BAREX has no remote signal atomic for Device API inter tests yet.
        # Keep the normal intra runs; forced-Net intra also needs that atomic.
        FLAGCX_CI_DEVICE_API_INTRA_ONLY=1
        export CUDA_VISIBLE_DEVICES="$FLAGCX_CI_PPU_FOUR_GPU_DEVICES"
        FLAGCX_CI_DEVICE_API_MPI_ARGS=(-x CUDA_VISIBLE_DEVICES)
        FLAGCX_CI_UNIFIED_IR_MPI_ARGS=(-x CUDA_VISIBLE_DEVICES)
      fi
      ;;
    adaptor)
      # Fail rather than silently falling back to IBRC/socket: this suite is
      # the integration coverage for the BAREX one-sided contract.
      export FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX
      ;;
    p2p)
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

  if [[ "$suite" == adaptor ]]; then
    local pccl_mr_xml="$suite_dir/build/ppu-pccl-host-mr.xml"
    rm -f "$pccl_mr_xml"
    FLAGCX_CI_TEST_LABEL="PPU PCCL and BAREX host MR interoperability" \
      "$TEST_RUNNER" "$suite_dir/build/bin/adaptor_unit_tests" \
        --gtest_filter=PpuPcclLoading.NativeCommBeforeBarexHostMrRegistration \
        --gtest_output="xml:$pccl_mr_xml" || return $?
    python3 "$PROJECT_ROOT/.github/scripts/ci/check_required_gtests.py" \
      --case PpuPcclLoading.NativeCommBeforeBarexHostMrRegistration \
      "$pccl_mr_xml" || return $?
  fi

  if [[ "$suite" == device_api ]]; then
    FLAGCX_CI_TEST_LABEL="PPU native/Torch stream interoperability" \
      "$TEST_RUNNER" make -C "$PROJECT_ROOT/test/kernel/ppu" \
        stream_compat "${args[@]}"
    FLAGCX_CI_TEST_LABEL="PPU SDK platform traits" \
      "$TEST_RUNNER" make -C "$PROJECT_ROOT/test/kernel/ppu" \
        device_traits "${args[@]}"
  fi

  if [[ "$suite" == "runner" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    # Exercise every collective with PCCL's built-in kernels, including when
    # the runner's inherited environment explicitly enables the plugin.
    export PCCL_ENABLE_EXT_KERNEL=0
    local inherited_filter=${GTEST_FILTER:-*}
    export GTEST_FILTER='*-CoverageSupplemental.*'
    if [[ "$inherited_filter" == *-* ]]; then
      GTEST_FILTER+=":${inherited_filter#*-}"
    fi
    local runner_status=0
    local runner_mpi_runner="${RUNNER_MPI_RUNNER:-$MPI_RUNNER}"
    FLAGCX_CI_TEST_LABEL="runner unit tests" \
      "$TEST_RUNNER" make -C "$suite_dir" run-unit "${args[@]}" || runner_status=1
    cd "$suite_dir"
    FLAGCX_CI_MPI_LABEL="runner default" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      -u FLAGCX_CLUSTER_SPLIT_LIST -u FLAGCX_MEM_ENABLE \
      -u FLAGCX_P2P_TRANSPORT -u FLAGCX_P2P_DISABLE \
      "$runner_mpi_runner" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      -x PCCL_ENABLE_EXT_KERNEL=0 \
      -x FLAGCX_CI_EXPECT_RUNNER_MODE=HOMO \
      ./build/bin/runner_mpi_tests || runner_status=1
    FLAGCX_CI_MPI_LABEL="runner BAREX heterogeneous" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      "$runner_mpi_runner" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      -x PCCL_ENABLE_EXT_KERNEL=0 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl \
      -x FLAGCX_P2P_DISABLE=0 \
      -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
      -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=P2P \
      ./build/bin/runner_mpi_tests || runner_status=1
    FLAGCX_CI_MPI_LABEL="runner BAREX forced NET" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      "$runner_mpi_runner" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
      -x PCCL_ENABLE_EXT_KERNEL=0 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl \
      -x FLAGCX_P2P_DISABLE=1 \
      -x FLAGCX_PXN_DISABLE=1 \
      -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
      -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET \
      -x FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX \
      -x FLAGCX_CI_EXPECT_COLL_MULTICHANNEL=1 \
      -x FLAGCX_CI_EXPECT_PXN=0 \
      ./build/bin/runner_mpi_tests || runner_status=1
    if [[ "${FLAGCX_CI_ENABLE_PXN:-1}" == "1" ]]; then
      FLAGCX_CI_MPI_LABEL="runner PPU PXN four-rank" \
        env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
        CUDA_VISIBLE_DEVICES="$FLAGCX_CI_PXN_VISIBLE_DEVICES" \
        "$runner_mpi_runner" -np "$FLAGCX_CI_RUNNER_NP" --allow-run-as-root \
        -x PCCL_ENABLE_EXT_KERNEL=0 \
        -x CUDA_VISIBLE_DEVICES="$FLAGCX_CI_PXN_VISIBLE_DEVICES" \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_VMM_ENABLE=0 \
        -x FLAGCX_P2P_TRANSPORT=accl \
        -x FLAGCX_P2P_DISABLE=1 \
        -x FLAGCX_PXN_DISABLE=0 \
        -x FLAGCX_CI_PXN_RANKS_PER_NODE="$FLAGCX_CI_NODE_NP" \
        -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID \
        -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET \
        -x FLAGCX_CI_EXPECT_NET_ADAPTOR=BAREX \
        -x FLAGCX_CI_EXPECT_PXN=1 \
        ./build/bin/runner_mpi_tests \
        --gtest_filter=FlagCXCollTest.SendRecv:FlagCXCollTest.AllReduce:FlagCXCollTest.AlltoAll || runner_status=1
    fi
    return "$runner_status"
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}


flagcx_ci_prepare() {
  local suite=$1
  local project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
  echo "Preparing T-Head PPU environment for CI workload: $suite"

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
  command -v hgcc
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
  flagcx_ci_validate_rdma_static PPU "$suite" \
    "/sys/class/infiniband/vsolar_*" || return
  if ! compgen -G "/dev/infiniband/fic2_soe_ucmd*" >/dev/null; then
    echo "PPU $suite tests require /dev/infiniband/fic2_soe_ucmd* control nodes." >&2
    return 1
  fi
}
