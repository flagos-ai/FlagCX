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
    export FLAGCX_TORCH_BACKEND=vendor
    export PYTHON_BIN=${PYTHON_BIN:-python3}
    # Use the plugin built from this checkout, not the CI image's wheel.
    "$PYTHON_BIN" -m pip uninstall --yes flagcx
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    "$PYTHON_BIN" -c 'import torch, torch_npu; count = torch.npu.device_count(); print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "devices", count); assert count >= 4'

    build_flagcx
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8' wheel
    (
      cd "$project_root"
      "$PYTHON_BIN" -m pip install --force-reinstall --no-build-isolation --no-deps .
    )

    # Run HCCL-name tests first, with route settings isolated from the generic suite.
    (
      log_dir="${FLAGCX_CI_LOG_DIR:-$project_root/.ci-logs}/hccl-comm-name"
      mkdir -p "$log_dir"
      {
        git -C "$project_root" rev-parse HEAD
        git -C "$project_root" diff --stat
        "$PYTHON_BIN" - "$project_root" <<'PYTHON'
import hashlib
import pathlib
import sys
from importlib.metadata import distribution

import torch
import torch_npu
import flagcx
import flagcx._C

root = pathlib.Path(sys.argv[1]).resolve()
package = distribution("flagcx").locate_file("flagcx").resolve()
for module in (flagcx, flagcx._C):
    path = pathlib.Path(module.__file__).resolve()
    print(module.__name__, path)
    assert path.is_relative_to(package), path
extension = pathlib.Path(flagcx._C.__file__).resolve()
extension_hash = hashlib.sha256(extension.read_bytes()).hexdigest()
assert any(
    hashlib.sha256(path.read_bytes()).hexdigest() == extension_hash
    for path in root.glob(f"build/lib*/flagcx/{extension.name}")
), extension
print("extension sha256", extension_hash)
libraries = {
    pathlib.Path(line.split()[-1]).resolve()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if line.split()[-1].endswith("/libflagcx.so")
}
print("loaded FlagCX libraries", sorted(map(str, libraries)))
assert libraries == {(root / "build/lib/libflagcx.so").resolve()}, libraries
print("Python", sys.version)
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
PYTHON
        sha256sum "$project_root/build/lib/libflagcx.so" \
          "$project_root/plugin/torch/tests/distributed_smoke.py"
      } 2>&1 | tee "$log_dir/versions.log"

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
    )

    export FLAGCX_TORCH_USE_INSTALLED_PACKAGE=1
    unset FLAGCX_SKIP_HETERO
    # Match the Ascend runner quarantine: C2C refresh deviceMemset fails for
    # these three heterogeneous collectives before their transfers begin.
    export FLAGCX_TORCH_HETERO_SKIP_OPS=reduce,allreduce,reducescatter
    bash "$project_root/test/script/torch_api_test.sh"
    ;;
  *)
    echo "Unknown Ascend workload: $workload" >&2
    exit 2
    ;;
esac
