#!/usr/bin/env bash

set -euo pipefail

suite_dir=${1:?Usage: run_ppu_pccl_diagnostics.sh <runner suite directory>}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpi_runner=${MPI_RUNNER:-$script_dir/run_mpi_with_timeout.sh}
log_root=${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-/tmp}/flagcx-ci-attempts}
mkdir -p "$log_root"
log_dir=$(mktemp -d "$log_root/ppu-pccl-bindings.XXXXXX")
status=0

printf 'ext_kernel\tsequence\texit_status\n' > "$log_dir/results.tsv"
cd "$suite_dir"

# Each invocation starts fresh workers. Reopening a bound library does not
# repair its relocations. Keep the ordinary full runner matrix enabled too.
for enabled in 1 0; do
  for sequence in allreduce recreate; do
    filter=FlagCXCollTest.AllReduce
    if [[ "$sequence" == recreate ]]; then
      filter=FlagCXCollTest.AllGather:FlagCXCollTest.AllReduce
    fi
    case_dir="$log_dir/ext-$enabled-$sequence"
    mkdir -p "$case_dir"
    case_status=0
    FLAGCX_CI_MPI_LABEL="PPU PCCL ext=$enabled $sequence" \
      FLAGCX_CI_LOG_DIR="$case_dir" \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_REQUIRE_GTEST_PASS=1 \
      FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_PPU_PCCL_TIMEOUT:-5m}" \
      env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM \
      -u FLAGCX_CLUSTER_SPLIT_LIST -u FLAGCX_MEM_ENABLE \
      -u FLAGCX_P2P_TRANSPORT -u FLAGCX_P2P_DISABLE \
      -u PCCL_EXT_KERNEL_PLUGIN -u NCCL_EXT_KERNEL_PLUGIN \
      -u PCCL_ALGO -u NCCL_ALGO -u PCCL_PROTO -u NCCL_PROTO \
      "$mpi_runner" -np "${FLAGCX_CI_RUNNER_NP:-4}" --allow-run-as-root \
      -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
      -x FLAGCX_CI_EXPECT_RUNNER_MODE=HOMO \
      -x FLAGCX_CI_RUNNER_BYTES=4194304 \
      -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=INIT \
      -x PCCL_DEBUG=INFO -x PCCL_DEBUG_SUBSYS=INIT,EXT,COLL \
      -x PCCL_ENABLE_EXT_KERNEL="$enabled" \
      -x LD_DEBUG=libs,bindings,versions \
      -x LD_DEBUG_OUTPUT="$case_dir/bindings" \
      ./build/bin/runner_mpi_tests --gtest_filter="$filter" || case_status=$?
    printf '%s\t%s\t%s\n' "$enabled" "$sequence" "$case_status" >> "$log_dir/results.tsv"
    ((case_status == 0)) || status=1

    # Retain the complete worker logs in the existing CI artifacts and print
    # the actual relocation records, rather than inferring bindings via dlsym.
    python3 - "$case_dir" "$enabled" <<'PY' || status=1
from pathlib import Path
import sys

root = Path(sys.argv[1])
logs = list(root.glob("bindings.*"))
symbols = ("__hggcRegisterFatBinary", "__hggcRegisterFunction",
           "__hggcUnregisterFatBinary", "hggcFuncSetAttribute", "hggcLaunchKernel")
summary = []
for path in sorted(logs):
    with path.open(errors="replace") as source:
        for line in source:
            if any(symbol in line for symbol in symbols):
                summary.append(f"{path.name}: {line.rstrip()}")
(root / "binding-summary.log").write_text("\n".join(summary) + "\n")
print(f"PPU PCCL binding diagnostics: {root}")
for line in summary:
    print(line)
if not logs or not summary:
    raise SystemExit("Missing HGGC worker binding diagnostics")
if sys.argv[2] == "1":
    loaded = any("PPU/PCCL: retained kernel plugin" in path.read_text(errors="replace")
                 for path in root.rglob("attempt-*.log"))
    if not loaded:
        raise SystemExit("Enabled PCCL kernel plugin was not preloaded")
PY
  done
done

cat "$log_dir/results.tsv"
exit "$status"
