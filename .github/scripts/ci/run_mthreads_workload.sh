#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf|perf-test-experimental>" >&2
  exit 2
fi
workload=$1
case "$workload" in
  torch-api|perf|perf-test-experimental) ;;
  *) echo "Unknown MUSA workload: $workload" >&2; exit 2 ;;
esac
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
source "$project_root/.github/scripts/ci/timeouts.sh"
perf_runner="$project_root/.github/scripts/ci/run_host_perf_suite.sh"
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
# shellcheck source=/dev/null
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"
export LD_LIBRARY_PATH="$project_root/build/lib:$LD_LIBRARY_PATH"
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY,UNIRUNNER}

case "$workload" in
  perf-test-experimental)
    source "$project_root/.github/scripts/ci/run_mthreads_perf_experimental.sh"
    ;;
  perf)
    flagcx_ci_run build make -C "$project_root" --jobs="$(nproc)" \
      "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
    flagcx_ci_run build-tests make -C "$project_root/test/perf" --jobs="$(nproc)" \
      "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"

    export FLAGCX_CI_PERF_PLATFORM=Mthreads
    perf_bin="$project_root/test/perf/host_api/build/bin"
    "$perf_runner" homogeneous "$perf_bin"
    "$perf_runner" heterogeneous "$perf_bin"
    ;;
  torch-api)
    # Use the existing Torch API suite so registering MUSA cannot silently
    # omit its Torch workload from the common CI completion gate.
    python3 -c 'import torch, torch_musa; print("torch", torch.__version__, "devices", torch.musa.device_count()); assert torch.musa.device_count() >= 4'
    make -C "$project_root" --jobs="$(nproc)" "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
    python3 -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      python3 setup.py build_ext --inplace
    )
    export PYTHON_BIN=python3
    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    bash "$project_root/test/script/torch_api_test.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      "$PYTHON_BIN" -m pip install --quiet coverage==7.10.7
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
esac
