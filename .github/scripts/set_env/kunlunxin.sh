#!/usr/bin/env bash

set -euo pipefail

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

MPI_HOME=${MPI_HOME:-/usr/local/openmpi}
CCL_HOME=${CCL_HOME:-/usr/local/xccl_Linux_x86_64}
DEVICE_HOME=${DEVICE_HOME:-/opt/flagcx-sdk/xre}
export XCCL_HOME="$CCL_HOME"
export XRE_HOME="$DEVICE_HOME"

export MPI_HOME CCL_HOME DEVICE_HOME
export OMPI_MCA_plm=isolated
export JSON_INCLUDE_DIR="${JSON_INCLUDE_DIR:-/usr/include}"
export PATH="$MPI_HOME/bin:$DEVICE_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$CCL_HOME/so:$DEVICE_HOME/so:$DEVICE_HOME/lib:/usr/local/xcudart/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_CI_MPIRUN="$MPI_HOME/bin/mpirun"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_KUNLUNXIN=1 MPI_HOME="$MPI_HOME" CCL_HOME="$CCL_HOME" DEVICE_HOME="$DEVICE_HOME" "HOST_CXXFLAGS=-include cstddef")
FLAGCX_CI_TEST_MAKE_ARGS=(USE_KUNLUNXIN=1 MPI_FLAVOR=openmpi MPI_HOME="$MPI_HOME" CCL_HOME="$CCL_HOME" DEVICE_HOME="$DEVICE_HOME" "HOST_CXXFLAGS=-include cstddef")
FLAGCX_CI_ENABLE_IBUC=0
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_REQUIRE_ADAPTOR_GPU_TESTS=1
# The documented XRE rejects cudaMallocManaged (801), SYNC_MEMOPS (1), and
# cudaHostAlloc(Mapped, 4 MiB) (1). Keep ordinary device/host and IPC tests;
# exclude only the paths whose prerequisites fail in this SDK/backend pair.
export FLAGCX_CI_MANAGED_MEMORY_SUPPORTED=0
export FLAGCX_CI_GDR_ALLOC_SUPPORTED=0
FLAGCX_CI_GPU_MR_POLICY=gpu-mr-optional
FLAGCX_CI_ADAPTOR_IPC_EXCLUDE='IpcMemHandleMpiTest.*Gdr*'
# No validated XPU VMM MR data route is available for this CI platform yet.
export FLAGCX_CI_SYMMEM_VMM_SUPPORTED=0
# XSHMEM on P800 is limited to intra-node communication within one four-XPU group.
export FLAGCX_CI_DEVICE_API_SHMEM_INTRANODE_ONLY=1
export XSHMEM_SYMMETRIC_SIZE=17179869184
FLAGCX_CI_UNIFIED_IR_MPI_ARGS=(-x XPU_VISIBLE_DEVICES=4,5,6,7)
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=4
export NP=4

# xcclAdaptorGather returns flagcxNotSupported, so flagcxGather reports 9 on
# every rank. It is the only collective the XCCL/BKCL adaptor leaves
# unimplemented; every other operation in the shared perf list runs. Remove
# this line once the adaptor forwards a bkcl gather primitive.
export FLAGCX_CI_PERF_SKIP_OPERATIONS="gather"

FLAGCX_CI_NODE1_MPI_ARGS=(-x FLAGCX_HOSTID=node0)
FLAGCX_CI_NODE2_MPI_ARGS=(-x FLAGCX_HOSTID=node1)

flagcx_ci_phase_skip_reason() {
  case "$1" in
    p2p-write-mpi|rma-mpi|symmem-mpi)
      echo "KunlunXin GDR allocation fails at cuPointerSetAttribute(SYNC_MEMOPS); ordinary device IPC and host MR remain covered by adaptor"
      ;;
    runner-mpi)
      echo "KunlunXin XRE rejects cudaHostAlloc(Mapped, 4 MiB); Runner setup then calls memset on a null pointer"
      ;;
  esac
}

flagcx_ci_configure_suite() {
  local suite=$1
  if [[ "$suite" == device_api || "$suite" == device_api_unified_ir ||
        "$suite" == kernel ]]; then
    FLAGCX_CI_PROJECT_MAKE_ARGS+=(USE_SHMEM=1)
    FLAGCX_CI_TEST_MAKE_ARGS+=(USE_SHMEM=1)
    FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
    FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
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
    echo "KunlunXin XTDK toolchain not found; device_api, device_api_unified_ir and kernel will be skipped"
  fi
}
