#!/usr/bin/env bash

set -euo pipefail

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

MPI_HOME=${MPI_HOME:-/usr/local/mpich-3.3.2}
CCL_HOME=${CCL_HOME:-/usr/local/xccl_Linux_x86_64}
DEVICE_HOME=${DEVICE_HOME:-/usr/local/xpu}

export MPI_HOME CCL_HOME DEVICE_HOME
export JSON_INCLUDE_DIR="${JSON_INCLUDE_DIR:-/usr/include}"
export PATH="$MPI_HOME/bin:$DEVICE_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CCL_HOME/so:$MPI_HOME/lib:$DEVICE_HOME/so:$DEVICE_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_CI_MPIRUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../ci" && pwd)/run_mpich.sh"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_KUNLUNXIN=1 MPI_HOME="$MPI_HOME" CCL_HOME="$CCL_HOME" DEVICE_HOME="$DEVICE_HOME" "HOST_CXXFLAGS=-include cstddef")
FLAGCX_CI_TEST_MAKE_ARGS=(USE_KUNLUNXIN=1 MPI_HOME="$MPI_HOME" CCL_HOME="$CCL_HOME" DEVICE_HOME="$DEVICE_HOME" "HOST_CXXFLAGS=-include cstddef")
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_REQUIRE_ADAPTOR_GPU_TESTS=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=4
export NP=4

FLAGCX_CI_NODE1_MPI_ARGS=(-x FLAGCX_HOSTID=node0)
FLAGCX_CI_NODE2_MPI_ARGS=(-x FLAGCX_HOSTID=node1)

flagcx_ci_configure_suite() {
  local suite=$1
  if [[ "$suite" == device_api || "$suite" == device_api_unified_ir ||
        "$suite" == kernel ]]; then
    FLAGCX_CI_PROJECT_MAKE_ARGS+=(USE_SHMEM=1)
    FLAGCX_CI_TEST_MAKE_ARGS+=(USE_SHMEM=1)
  fi
}

flagcx_ci_find_xpu_toolchain() {
  local root
  local -a candidates=(
    "${XTDK_HOME:-}"
    /workspace/my_flagcx/xtdk-llvm15-ubuntu2004_x86_64
    /usr/local/xtdk-llvm15-ubuntu2004_x86_64
    /opt/xtdk-llvm15-ubuntu2004_x86_64
  )
  for root in "${candidates[@]}"; do
    [[ -n "$root" ]] || continue
    if [[ -x "$root/bin/clang++" && -f "$root/include/xpu/kernel/xtdk.h" ]]; then
      printf '%s\n' "$root"
      return 0
    fi
  done
  return 1
}

flagcx_ci_build_suite_override() {
  local suite=$1
  local requires_shmem=0
  case "$suite" in
    device_api|device_api_unified_ir|kernel) requires_shmem=1 ;;
  esac

  if ((requires_shmem != 0)) &&
     [[ "${FLAGCX_CI_XPU_TOOLCHAIN_AVAILABLE:-0}" != 1 ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping $suite: XTDK clang++ and xpu/kernel/xtdk.h are not present in the CI image"
  elif ((requires_shmem != 0)) &&
       [[ ! -f "$CCL_HOME/include/xshmem/xshmem.h" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    echo "Skipping $suite: XSHMEM headers are not present under $CCL_HOME/include"
  else
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
  fi
}

flagcx_ci_run_suite_override() {
  local suite=$1
  if [[ ( "$suite" == device_api || "$suite" == device_api_unified_ir ) &&
        "${FLAGCX_CI_XPU_TOOLCHAIN_AVAILABLE:-0}" != 1 ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    echo "SKIPPED $suite: install or mount the KunlunXin XTDK toolchain to enable this suite"
  else
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static KunlunXin "$suite" \
    "/sys/class/infiniband/*"
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing KunlunXin environment for CI workload: $suite"
  command -v mpicxx
  command -v mpirun
  mpirun --version
  command -v xpu-smi
  xpu-smi -L
  test -d "$CCL_HOME/include"
  test -d "$CCL_HOME/so"
  local toolchain
  if toolchain=$(flagcx_ci_find_xpu_toolchain); then
    export FLAGCX_CI_XPU_TOOLCHAIN_AVAILABLE=1
    FLAGCX_CI_PROJECT_MAKE_ARGS+=(XTDK_HOME="$toolchain")
    FLAGCX_CI_TEST_MAKE_ARGS+=(XTDK_HOME="$toolchain")
    echo "KunlunXin XTDK toolchain: $toolchain"
  else
    export FLAGCX_CI_XPU_TOOLCHAIN_AVAILABLE=0
    echo "KunlunXin XTDK toolchain not found; device_api and device_api_unified_ir will be skipped"
  fi
}
