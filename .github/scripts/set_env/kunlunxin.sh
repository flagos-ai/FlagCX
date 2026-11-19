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
export FLAGCX_ADAPTOR=klx

# Keep KunlunXin capability exclusions with its platform environment setup.
FLAGCX_CI_IGNORE_TEST='{
  "gtest": [
    {
      "suite": "adaptor",
      "tests": ["IpcMemHandleMpiTest.*Gdr*"],
      "reason": "KunlunXin GDR allocation fails at cuPointerSetAttribute(SYNC_MEMOPS)"
    },
    {
      "suite": "adaptor",
      "tests": ["DeviceAdaptorTest.GetManagedPointerType"],
      "reason": "XRE does not support managed allocation"
    },
    {
      "suite": "p2p",
      "tests": [
        "FlagcxP2pEngineReadTest.ReadsWholeRegisteredGpuBufferAfterMetadataHandshake",
        "FlagcxP2pEngineReadTest.TwoIndependent2KiBReadsCover4KiBBuffer",
        "FlagcxP2pEngineReadTest.RejectsOutOfBoundsReadBeforeSubmission",
        "FlagcxP2pEngineReadTest.RejectsInvalidRegistrationAndDescriptorRanges",
        "FlagcxP2pEngineReadTest.AcceptsZeroLengthDescriptorAtRegistrationEnd",
        "FlagcxP2pEngineReadTest.RejectsOutOfBoundsWriteBeforeSubmission",
        "FlagcxP2pEngineReadTest.ReadsRetargetedRemoteGpuSubrangeIntoLocalWindow"
      ],
      "reason": "KunlunXin shared P2P Engine READ tests fail during GPU memory registration or data validation; restore after the P800 registration and remote-window path is validated."
    }
  ],
  "phases": [
    {
      "suite": "p2p",
      "name": "p2p-write-mpi",
      "reason": "KunlunXin P2P WRITE MPI tests fail on both ranks while allocating the GPU test buffer; restore after the device allocation path is validated."
    },
    {
      "suite": "perf",
      "name": "heterogeneous",
      "reason": "XRE cudaLaunchHostFunc is unsupported"
    },
    {
      "suite": "device_api_unified_ir",
      "name": "inter-and-forced-net",
      "reason": "XSHMEM supports intra-node communication only"
    },
    {
      "suite": "rma",
      "name": "gdr-visibility",
      "reason": "No validated KunlunXin XPU consumer route for GDR visibility"
    },
    {
      "suite": "rma",
      "name": "rma-mpi",
      "reason": "P800 IPC and NET RMA MPI tests cannot create the data window: windowAvailable=false at rma_test.cpp:449; restore after XPU symmetric-window registration and IPC/MR paths are validated."
    },
    {
      "suite": "symmem",
      "name": "symmem-mpi",
      "reason": "P800 SymMem MPI tests fail during device allocation and symmetric window registration (flagcxMemAlloc failure and return code 4); restore after the XPU IPC/window path is validated."
    },
    {
      "suite": "symmem",
      "name": "vmm-local-and-remote-without-net",
      "reason": "No validated KunlunXin VMM MR data route"
    },
    {
      "suite": "symmem",
      "name": "vmm-net-data",
      "reason": "No validated KunlunXin VMM MR data route"
    },
    {
      "suite": "torch-api",
      "name": "heterogeneous",
      "reason": "XRE cudaLaunchHostFunc is unsupported"
    },
    {
      "suite": "runner",
      "name": "runner-mpi",
      "reason": "KunlunXin Runner MPI modes (default, heterogeneous, forced NET, multi-QP and PXN) exit unsuccessfully; restore after mode-specific transport failures are diagnosed."
    }
  ],
  "cases": [
    {
      "suite": "device_api_unified_ir",
      "tests": ["S16"],
      "reason": "P800 four-rank unified IR intra timed out twice after 20m (exit=124) before any S16 result in run 44002251935, artifact 14297650820. Barrier/PE mapping/state lifecycle diagnosis is pending; no runtime stack was captured. This is skipped coverage, not a passing test."
    }
  ],
  "perf": {
    "operations": ["gather"],
    "modes": ["homogeneous", "heterogeneous"],
    "reason": "XCCL Gather adaptor returns flagcxNotSupported"
  },
  "torch": {
    "operations": ["gather"],
    "modes": ["homogeneous", "heterogeneous"],
    "reason": "Gather is unsupported and heterogeneous collectives require unsupported cudaLaunchHostFunc"
  }
}'
export FLAGCX_CI_IGNORE_TEST

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
FLAGCX_CI_GPU_MR_POLICY=gpu-mr-optional
# XSHMEM on P800 is limited to intra-node communication within one four-XPU group.
export FLAGCX_CI_DEVICE_API_SHMEM_INTRANODE_ONLY=1
export XSHMEM_SYMMETRIC_SIZE=${XSHMEM_SYMMETRIC_SIZE:-17179869184}
# Keep one physical four-XPU group consistent across the device, kernel and
# performance workloads, and allow another runner to select its own group.
export FLAGCX_CI_KUNLUNXIN_DEVICES=${FLAGCX_CI_KUNLUNXIN_DEVICES:-4,5,6,7}
FLAGCX_CI_UNIFIED_IR_MPI_ARGS=(-x XPU_VISIBLE_DEVICES="$FLAGCX_CI_KUNLUNXIN_DEVICES")
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=4
export NP=4

FLAGCX_CI_NODE1_MPI_ARGS=(-x FLAGCX_HOSTID=node0)
FLAGCX_CI_NODE2_MPI_ARGS=(-x FLAGCX_HOSTID=node1)

flagcx_ci_configure_suite() {
  local suite=$1
  if [[ "$suite" == device_api || "$suite" == device_api_unified_ir ]]; then
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
    device_api|device_api_unified_ir) requires_shmem=1 ;;
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
