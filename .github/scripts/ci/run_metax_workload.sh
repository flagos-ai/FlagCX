#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
source "$project_root/.github/scripts/ci/timeouts.sh"
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="/opt/maca/mxgpu_llvm/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_ADAPTOR=metax
export TORCH_DISTRIBUTED_BACKEND=mccl

case "$workload" in
  perf)
    python3 "$project_root/.github/scripts/ci/check_metax_devices.py" perf
    flagcx_ci_run build make -C "$project_root" --jobs="$(nproc)" USE_METAX=1
    flagcx_ci_run build-tests make -C "$project_root/test/perf" --jobs="$(nproc)" \
      USE_METAX=1 MPI_HOME="$MPI_HOME" CCL_HOME=/opt/maca

    export FLAGCX_CI_PERF_PLATFORM=MetaX
    # Collect all operation results while preserving a failing suite status.
    export FLAGCX_CI_PERF_CONTINUE_ON_FAILURE=1
    perf_bin="$project_root/test/perf/host_api/build/bin"
    perf_result=0
    if ! MCCL_DEBUG=TRACE MCCL_DEBUG_SUBSYS=ALL \
      "$project_root/.github/scripts/ci/run_host_perf_suite.sh" homogeneous "$perf_bin"; then
      perf_result=1
    fi
    if ! "$project_root/.github/scripts/ci/run_host_perf_suite.sh" heterogeneous "$perf_bin"; then
      perf_result=1
    fi
    exit "$perf_result"
    ;;
  torch-api)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGCX_DEBUG=INFO
    export FLAGCX_DEBUG_SUBSYS=ALL
    export USE_METAX=1
    export PATH="/flagos/bin:$PATH"
    export PYTHON_BIN=/flagos/bin/python3

    flagcx_ci_run preflight "$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"MetaX Torch environment: Python={sys.executable}", flush=True)
print(f"MetaX Torch environment: torch={torch.__version__}", flush=True)
print(f"MetaX Torch environment: torch_path={torch.__file__}", flush=True)
PY

    flagcx_ci_run build make -C "$project_root" --jobs="$(nproc)" USE_METAX=1
    flagcx_ci_run torch-build "$PYTHON_BIN" -m pip install --break-system-packages 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      flagcx_ci_run torch-build "$PYTHON_BIN" setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    bash "$project_root/test/script/torch_api_test.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      flagcx_ci_run torch-build "$PYTHON_BIN" -m pip install --break-system-packages --quiet coverage==7.10.7
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
  *)
    echo "Unknown MetaX workload: $workload" >&2
    exit 2
    ;;
esac
