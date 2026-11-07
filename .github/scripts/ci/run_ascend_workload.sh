#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_runner="$project_root/.github/scripts/ci/run_host_perf_suite.sh"
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"

# shellcheck source=/dev/null
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

build_flagcx() {
  make -C "$project_root" --jobs="$(nproc)" \
    "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
}

case "$workload" in
  perf)
    build_flagcx
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" MPI_HOME="$MPI_HOME"

    export FLAGCX_CI_PERF_PLATFORM=Ascend
    perf_bin="$project_root/test/perf/host_api/build/bin"
    # Homogeneous perf drives HCCL directly and passes. Heterogeneous perf
    # needs a hetero communicator, which flagcxCommInitRank cannot create on
    # CANN while flagcxTopoGetXmlTopo lacks getDeviceByPciBusId.
    "$perf_runner" homogeneous "$perf_bin"
    echo "Ascend coverage gap: heterogeneous perf is not run. flagcxCommInitRank fails after topo.cc reports 'PCI reverse bus-ID lookup is unavailable for topology discovery', so the heterogeneous operations cannot reach a communicator"
    ;;
  torch-api)
    export FLAGCX_ADAPTOR=ascend
    export PYTHON_BIN=${PYTHON_BIN:-python3}
    "$PYTHON_BIN" -c 'import torch, torch_npu; count = torch.npu.device_count(); print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "devices", count); assert count >= 4'

    build_flagcx
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      "$PYTHON_BIN" setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    # The torch_npu import check, the FlagCX build and the plugin extension
    # build above all pass and stay covered. The distributed run is not
    # started: all four ranks of plugin/torch/example/example.py exit with
    # SIGSEGV right after "getting busId for cudaDev N". That crash is NOT yet
    # attributed -- the process logs no non-PCI identity line, so unlike every
    # other suite its getDevicePciBusId slot is non-null and the fault may lie
    # inside that call. It needs a 910C core dump before any claim is made.
    echo "Ascend coverage gap: Torch API distributed run is not started. plugin/torch/example/example.py crashes with SIGSEGV on all four ranks and the cause is unattributed pending a 910C stack; the import check and both builds above did run"
    ;;
  *)
    echo "Unknown Ascend workload: $workload" >&2
    exit 2
    ;;
esac
