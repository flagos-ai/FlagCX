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
    log_dir="${FLAGCX_CI_LOG_DIR:-$project_root/.ci-logs}/hccl-comm-name"
    mkdir -p "$log_dir"
    export FLAGCX_ADAPTOR=ascend
    export FLAGCX_TORCH_BACKEND=vendor
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export PYTHON_BIN=${PYTHON_BIN:-python3}
    "$PYTHON_BIN" -c 'import torch, torch_npu; count = torch.npu.device_count(); print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "devices", count); assert count >= 4'

    build_flagcx 2>&1 | tee "$log_dir/build.log"
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      "$PYTHON_BIN" setup.py build_ext --inplace
    ) 2>&1 | tee "$log_dir/build-extension.log"

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    {
      git -C "$project_root" rev-parse HEAD
      git -C "$project_root" diff --stat
      "$PYTHON_BIN" - "$project_root" <<'PY'
import pathlib
import sys

import torch
import torch_npu
import flagcx
import flagcx._C

root = pathlib.Path(sys.argv[1]).resolve()
package = root / "src" / "flagcx"
for module in (flagcx, flagcx._C):
    path = pathlib.Path(module.__file__).resolve()
    print(module.__name__, path)
    assert path.is_relative_to(package), path
libraries = {
    pathlib.Path(line.split()[-1]).resolve()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if line.split()[-1].endswith("/libflagcx.so")
}
print("loaded FlagCX libraries", sorted(map(str, libraries)))
assert libraries == {(root / "build/lib/libflagcx.so").resolve()}, libraries
print("Python", sys.version)
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
PY
      sha256sum "$project_root/build/lib/libflagcx.so" \
        "$project_root"/src/flagcx/_C*.so \
        "$project_root/plugin/torch/tests/distributed_smoke.py"
    } 2>&1 | tee "$log_dir/versions.log"

    # These cases exercise the homogeneous HCCL path in both runner modes.
    unset FLAGCX_USE_HOST_COMM FLAGCX_USE_HETERO_COMM FLAGCX_USE_TUNER
    unset FLAGCX_CLUSTER_SPLIT_LIST
    for ranks in 2 4; do
      for route in default host; do
        if [[ "$route" == host ]]; then
          export FLAGCX_USE_HOST_COMM=1
        else
          unset FLAGCX_USE_HOST_COMM
        fi
        timeout --signal=TERM --kill-after="${FLAGCX_CI_TEST_KILL_AFTER:-30s}" \
          "${FLAGCX_CI_HCCL_COMM_NAME_TIMEOUT:-10m}" \
          "$PYTHON_BIN" -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
          --nproc-per-node="$ranks" \
          "$project_root/plugin/torch/tests/distributed_smoke.py" \
          --adaptor ascend --hccl-comm-name \
          2>&1 | tee "$log_dir/${ranks}-ranks-${route}.log"
      done
    done
    ;;
  *)
    echo "Unknown Ascend workload: $workload" >&2
    exit 2
    ;;
esac
