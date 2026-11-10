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
    "$perf_runner" homogeneous "$perf_bin"
    # Exercise local IPC as well as the already validated forced NET route.
    "$perf_runner" heterogeneous "$perf_bin"
    # Force the relaxed-ordering MR path validated for Ascend NPU memory.
    FLAGCX_CI_ATTEMPT_FAILURE_HOOK="$project_root/.github/scripts/ci/ascend_rdma_attempt_inventory.sh" \
      "$perf_runner" heterogeneous "$perf_bin" \
      -x FLAGCX_P2P_DISABLE=1 -x FLAGCX_IB_PCI_RELAXED_ORDERING=1
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
    unset FLAGCX_SKIP_HETERO
    bash "$project_root/test/script/torch_api_test.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      "$PYTHON_BIN" -m pip install --quiet coverage==7.10.7
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
  *)
    echo "Unknown Ascend workload: $workload" >&2
    exit 2
    ;;
esac
