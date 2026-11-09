#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "torch-api" && "$1" != "perf" ) ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
source "$project_root/$FLAGCX_CI_SET_ENV"

flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"
# Host API performance tests and the Torch extension do not compile XPU
# kernels. XTDK and XSHMEM are only required by the device API/kernel suites.
export FLAGCX_ADAPTOR=klx
export JSON_INCLUDE_DIR=${JSON_INCLUDE_DIR:-/usr/include}
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-ALL}

build_flagcx() {
  make -C "$project_root" --jobs="$(nproc)" \
    "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
}

case "$workload" in
  perf)
    build_flagcx
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"
    export FLAGCX_CI_PERF_PLATFORM=KunlunXin
    perf_bin="$project_root/test/perf/host_api/build/bin"
    "$project_root/.github/scripts/ci/run_host_perf_suite.sh" homogeneous "$perf_bin"
    "$project_root/.github/scripts/ci/run_host_perf_suite.sh" heterogeneous "$perf_bin"
    ;;
  torch-api)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    # PyTorch's CUDA-facing headers require the complete CUDA SDK, including
    # cuda_fp16.h. The plugin's legacy /opt/kunlun paths do not exist in the
    # CI image. Link to the image's XPU compatibility runtime.
    cuda_include="${CUDA_HOME:-/usr/local/cuda}/include"
    for header in cuda.h cuda_fp16.h; do
      if [[ ! -f "$cuda_include/$header" ]]; then
        echo "KunlunXin Torch build requires $cuda_include/$header" >&2
        exit 1
      fi
    done
    export CPATH="$cuda_include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="/usr/local/xcudart/lib:$DEVICE_HOME/so:$DEVICE_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export HOST_CXXFLAGS="${HOST_CXXFLAGS:+$HOST_CXXFLAGS }-include cstddef"
    python3 - <<'PY'
import torch

print("torch", torch.__version__, "devices", torch.cuda.device_count())
assert torch.cuda.is_available(), "KunlunXin Torch runtime reports no available devices"
assert torch.cuda.device_count() >= 4, "Torch API tests require four visible KunlunXin devices"
PY

    python3 -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      python3 setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    bash "$project_root/test/script/torch_api_test.sh"
    ;;
esac
