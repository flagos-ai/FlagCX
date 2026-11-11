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
    # Remove the CI image's older FlagCX wheel before the first Torch import.
    # This runs only inside the disposable hardware-test container.
    "$PYTHON_BIN" -m pip uninstall --yes flagcx
    # Keep PyTorch from loading a FlagCX backend entry point before this
    # checkout's extension is imported.
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    "$PYTHON_BIN" -c 'import torch, torch_npu; count = torch.npu.device_count(); print("torch", torch.__version__, "torch_npu", torch_npu.__version__, "devices", count); assert count >= 4'

    build_flagcx
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8' wheel
    (
      cd "$project_root"
      "$PYTHON_BIN" -m pip install --force-reinstall --no-build-isolation --no-deps .
    )

    export FLAGCX_TORCH_USE_INSTALLED_PACKAGE=1
    echo "[INFO] Torch runtime build and import inventory"
    plugin_dir=$("$PYTHON_BIN" -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("flagcx"); print(pathlib.Path(spec.origin).parent if spec and spec.origin else "")')
    if [[ -z "$plugin_dir" || "$plugin_dir" == "$project_root"/* ]]; then
      echo "[ERROR] FlagCX must resolve to the newly installed package, got: $plugin_dir" >&2
      exit 1
    fi
    for library in "$project_root/build/lib/libflagcx.so" "$plugin_dir"/lib/libflagcx.so "$plugin_dir"/_C*.so; do
      [[ -f "$library" ]] || continue
      echo "[INFO] ELF: $library"
      sha256sum "$library"
      if command -v readelf >/dev/null 2>&1; then
        readelf -d "$library" | grep -E 'NEEDED|RPATH|RUNPATH' || true
      fi
      ldd "$library" || true
    done
    "$PYTHON_BIN" - <<'PY'
import hashlib
import importlib
import os
from pathlib import Path

import torch
import flagcx
import torch.distributed as dist
from torch_npu.contrib import transfer_to_npu

root = Path(os.environ["GITHUB_WORKSPACE"])
module_file = getattr(flagcx, "__file__", None)
if not module_file:
    raise RuntimeError("Torch CI imported a FlagCX namespace or placeholder instead of the installed package")
package = Path(module_file).resolve()
extension = importlib.import_module("flagcx._C")
installed = package.parent / "lib/libflagcx.so"
built = root / "build/lib/libflagcx.so"

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

print("FlagCX package:", package, "extension:", extension.__file__)
print("FlagCX library:", installed, "built:", built)
print("FlagCX Torch backend built:", extension.torch_backend)
print("Registered backends:", sorted(dist.Backend._plugins))
if extension.torch_backend != os.environ["FLAGCX_TORCH_BACKEND"]:
    raise RuntimeError("Installed FlagCX Torch extension has the wrong build backend")
if root in package.parents:
    raise RuntimeError("Torch CI did not import the installed FlagCX package")
if Path(extension.__file__).resolve().parent != package.parent:
    raise RuntimeError("FlagCX Torch extension was imported from another package")
if not installed.is_file() or digest(installed) != digest(built):
    raise RuntimeError("Installed FlagCX library differs from this checkout's build")
if not any(name.lower() == "flagcx" for name in dist.Backend._plugins):
    raise RuntimeError("Installed FlagCX Torch extension did not register its backend")
PY
    unset FLAGCX_SKIP_HETERO
    # Match the Ascend runner quarantine: C2C refresh deviceMemset fails for
    # these three heterogeneous collectives before their transfers begin.
    export FLAGCX_TORCH_HETERO_SKIP_OPS=reduce,allreduce,reducescatter
    diagnostic_root=$(mktemp -d "${FLAGCX_CI_LOG_DIR:-/tmp}/torch-native.XXXXXX")
    torch_status=0
    FLAGCX_CI_TORCH_DIAGNOSTIC_ROOT="$diagnostic_root" \
      FLAGCX_TORCH_RANK_BACKTRACE=1 \
      FLAGCX_TORCH_GDB_PROPAGATE_FAILURE=1 \
      FLAGCX_TORCH_CONTINUE_ON_FAILURE=1 \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=300 \
      bash "$project_root/test/script/torch_api_test.sh" || torch_status=$?
    if ((torch_status != 0)); then
      echo "[ERROR] Torch rank traces: $diagnostic_root" >&2
      for trace in "$diagnostic_root"/*/rank-*.log; do
        [[ -f "$trace" ]] || continue
        echo "----- $trace (last 80 lines) -----" >&2
        tail -n 80 "$trace" >&2
      done
      exit "$torch_status"
    fi
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
