#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != perf && "$1" != torch-api ) ]]; then
  echo "Usage: $0 <perf|torch-api>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export LD_LIBRARY_PATH="$project_root/build/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

if [[ "$workload" == torch-api ]]; then
  export PYTHON_BIN=${PYTHON_BIN:-python3}
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_TEST_LABEL="Iluvatar Torch preflight" \
    bash "$project_root/.github/scripts/ci/run_with_timeout.sh" "$PYTHON_BIN" -c '
import sys
import torch

print(f"Iluvatar Torch: Python={sys.executable}", flush=True)
print(f"Iluvatar Torch: version={torch.__version__} path={torch.__file__}", flush=True)
print(f"Iluvatar Torch: devices={torch.cuda.device_count()}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CoreX PyTorch cannot access the accelerator")
if torch.cuda.device_count() < 4:
    raise RuntimeError("Iluvatar Torch tests require four devices")
for device in range(4):
    tensor = torch.ones(4, device=f"cuda:{device}")
    if tensor.sum().item() != 4:
        raise RuntimeError(f"CoreX device {device} produced an invalid result")
    torch.cuda.synchronize(device)
'
fi

# Build failures are deterministic; retain logs without rebuilding on retry.
FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_TEST_LABEL="Iluvatar build FlagCX" \
  bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
  make -C "$project_root" --jobs="$(nproc)" \
    "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"

case "$workload" in
  perf)
    FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_TEST_LABEL="Iluvatar build host perf" \
      bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
      make -C "$project_root/test/perf" host_api --jobs="$(nproc)" \
        "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"
    export FLAGCX_CI_PERF_PLATFORM=Iluvatar
    perf_bin="$project_root/test/perf/host_api/build/bin"
    result=0
    bash "$project_root/.github/scripts/ci/run_host_perf_suite.sh" \
      homogeneous "$perf_bin" -x FLAGCX_VMM_ENABLE=0 || result=$?
    bash "$project_root/.github/scripts/ci/run_host_perf_suite.sh" \
      heterogeneous "$perf_bin" -x FLAGCX_VMM_ENABLE=0 || result=$?
    exit "$result"
    ;;
  torch-api)
    export PIP_BREAK_SYSTEM_PACKAGES=1
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8'
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      "$PYTHON_BIN" -m pip install coverage==7.10.7
    fi
    (
      cd "$project_root"
      export FLAGCX_TORCH_BACKEND=vendor
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_TEST_LABEL="Iluvatar build Torch extension" \
        bash .github/scripts/ci/run_with_timeout.sh \
        "$PYTHON_BIN" setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    export FLAGCX_TORCH_BACKEND=vendor
    FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_TEST_LABEL="Iluvatar import Torch extension" \
      bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
      "$PYTHON_BIN" -c 'import flagcx, flagcx._C; print("Iluvatar extension:", flagcx._C.__file__, flush=True)'
    result=0
    bash "$project_root/test/script/torch_api_test.sh" || result=$?
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      coverage_result=0
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh" || coverage_result=$?
      if ((result == 0)); then
        result=$coverage_result
      fi
    fi
    exit "$result"
    ;;
esac
