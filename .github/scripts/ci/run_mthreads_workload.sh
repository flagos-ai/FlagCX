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
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-TRACE}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-ALL}

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
    # The FlagOS image keeps Torch in /flagos, while a login shell can put
    # system Python first. Resolve the interpreter before building the plugin.
    if [[ -z "${PYTHON_BIN:-}" ]]; then
      python_candidates=()
      if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        python_candidates+=("$VIRTUAL_ENV/bin/python3")
      fi
      python_candidates+=(/flagos/bin/python3 python3 python)
      for candidate in "${python_candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import torch, torch_musa' >/dev/null 2>&1; then
          PYTHON_BIN=$(command -v "$candidate")
          break
        fi
      done
      if [[ -z "${PYTHON_BIN:-}" ]]; then
        echo "No Python interpreter with torch and torch_musa was found; set PYTHON_BIN to the MUSA environment's interpreter." >&2
        exit 1
      fi
    fi
    export PYTHON_BIN
    flagcx_ci_run preflight "$PYTHON_BIN" -c 'import sys, torch, torch_musa; print("python", sys.executable, "torch", torch.__version__, "devices", torch.musa.device_count()); assert torch.musa.device_count() >= 4'
    flagcx_ci_run build make -C "$project_root" --jobs="$(nproc)" "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
    flagcx_ci_run torch-build "$PYTHON_BIN" -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      flagcx_ci_run torch-build "$PYTHON_BIN" setup.py build_ext --inplace
    )
    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    bash "$project_root/test/script/torch_api_test.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      flagcx_ci_run torch-build "$PYTHON_BIN" -m pip install --quiet coverage==7.10.7
      flagcx_ci_run torch-coverage bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
esac
