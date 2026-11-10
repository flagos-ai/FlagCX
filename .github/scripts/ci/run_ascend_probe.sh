#!/usr/bin/env bash
set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
probe_dir="$project_root/test/probe/ascend"
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"
log_dir="${FLAGCX_CI_LOG_DIR:-/tmp/flagcx-ci-attempts}/ascend-probe"
: "${FLAGCX_CI_SET_ENV:?Ascend platform environment is required}"

# shellcheck source=/dev/null
source "$project_root/$FLAGCX_CI_SET_ENV"
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$DEVICE_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$log_dir"

device_pair=${FLAGCX_CI_PROBE_DEVICE_PAIR:-0,1}
soc_version=${FLAGCX_CI_PROBE_SOC_VERSION:-${SOC_VERSION:-Ascend910_9362}}
ascendc_dir=${FLAGCX_CI_PROBE_ASCENDC_CMAKE_DIR:-${ASCENDC_CMAKE_DIR:-$DEVICE_HOME/compiler/tikcpp/ascendc_kernel_cmake}}
pass_count=0
fail_count=0
unsupported_count=0
not_run_count=0
printf 'case\tstatus\texit_code\tdetail\n' > "$log_dir/results.tsv"
printf 'case\tmode\tstatus\texit_code\tlog_label\n' > "$log_dir/results-detail.tsv"
printf 'build\tstatus\n' > "$log_dir/build-results.tsv"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo '### Ascend independent probe results'
    echo '| Probe | Status | Exit | Detail |'
    echo '| --- | --- | ---: | --- |'
  } >> "$GITHUB_STEP_SUMMARY"
fi

record_build() {
  local case_name=$1 status=$2
  printf 'Ascend probe %s: %s\n' "$case_name" "$status"
  printf '%s\t%s\n' "$case_name" "$status" >> "$log_dir/build-results.tsv"
}

record_case() {
  local case_name=$1 status=$2 exit_code=$3 detail=$4
  printf 'Ascend probe %s: %s (exit=%s; %s)\n' \
    "$case_name" "$status" "$exit_code" "$detail"
  printf '%s\t%s\t%s\t%s\n' \
    "$case_name" "$status" "$exit_code" "$detail" >> "$log_dir/results.tsv"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '| %s | %s | %s | %s |\n' \
      "$case_name" "$status" "$exit_code" "$detail" >> "$GITHUB_STEP_SUMMARY"
  fi
  case "$status" in
    PASS) pass_count=$((pass_count + 1)) ;;
    FAIL) fail_count=$((fail_count + 1)) ;;
    UNSUPPORTED) unsupported_count=$((unsupported_count + 1)) ;;
    NOT_RUN) not_run_count=$((not_run_count + 1)) ;;
    *) echo "Unknown probe status: $status" >&2; exit 2 ;;
  esac
}

build_probe() {
  local label=$1
  shift
  if timeout --signal=TERM --kill-after=30s 25m "$@" \
      2>&1 | tee "$log_dir/build-$label.log"; then
    record_build "$label" PASS
    return 0
  fi
  record_build "$label" FAIL
  return 1
}

run_probe() {
  local label=$1 binary=$2 pair=$3 mode=$4 status=0
  local -a mpi_args=(--allow-run-as-root --tag-output -np 2)
  if [[ "$mode" == ib ]]; then
    mpi_args+=(-x IB_DEVICE -x IB_PORT -x IB_GID_INDEX)
  fi
  FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
    FLAGCX_CI_MPI_TIMEOUT=3m \
    FLAGCX_CI_MPI_LABEL="Ascend probe $label" \
    "$mpi_runner" "${mpi_args[@]}" "$binary" --devices "$pair" || status=$?
  case "$status" in
    0) record_case "$label" PASS 0 'data and cleanup verified' ;;
    3) record_case "$label" UNSUPPORTED 3 'CANN reported unsupported capability' ;;
    *) record_case "$label" FAIL "$status" 'see tagged rank log' ;;
  esac
}

# Every mode has a fresh MPI process and fresh device resources. A timed-out
# ValueWait cannot prevent later diagnostic modes from running.
run_probe_group() {
  local label=$1 binary=$2 pair=$3 transport=$4
  shift 4
  local mode exit_code mode_status group_status=PASS group_exit=0 mode_summary=
  local -a mpi_args=(--allow-run-as-root --tag-output -np 2)
  if [[ "$transport" == ib ]]; then
    mpi_args+=(-x IB_DEVICE -x IB_PORT -x IB_GID_INDEX)
  fi
  for mode in "$@"; do
    exit_code=0
    FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_MPI_TIMEOUT=3m \
      FLAGCX_CI_MPI_LABEL="Ascend probe $label $mode" \
      "$mpi_runner" "${mpi_args[@]}" "$binary" --devices "$pair" \
        --mode "$mode" || exit_code=$?
    case "$exit_code" in
      0) mode_status=PASS ;;
      3) mode_status=UNSUPPORTED ;;
      *) mode_status=FAIL ;;
    esac
    printf 'Ascend probe %s mode=%s: %s (exit=%s)\n' \
      "$label" "$mode" "$mode_status" "$exit_code"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$mode" "$mode_status" "$exit_code" \
      "Ascend_probe_${label// /_}_${mode}" >> "$log_dir/results-detail.tsv"
    mode_summary="${mode_summary}${mode_summary:+, }${mode}=${mode_status}"
    if [[ "$mode_status" == FAIL ]]; then
      group_status=FAIL
      group_exit=2
    elif [[ "$mode_status" == UNSUPPORTED && "$group_status" == PASS ]]; then
      group_status=UNSUPPORTED
      group_exit=3
    fi
  done
  record_case "$label" "$group_status" "$group_exit" "$mode_summary"
}

select_ib_device() {
  export IB_DEVICE=${FLAGCX_CI_PROBE_IB_DEVICE:-${IB_DEVICE:-}}
  export IB_PORT=${FLAGCX_CI_PROBE_IB_PORT:-${IB_PORT:-1}}
  export IB_GID_INDEX=${FLAGCX_CI_PROBE_IB_GID_INDEX:-${IB_GID_INDEX:-0}}
  if [[ -n "$IB_DEVICE" ]]; then
    return 0
  fi
  local state_file state rest
  for state_file in /sys/class/infiniband/*/ports/*/state; do
    [[ -f "$state_file" ]] || continue
    IFS= read -r state < "$state_file"
    [[ "$state" == 4:* ]] || continue
    rest=${state_file#/sys/class/infiniband/}
    IB_DEVICE=${rest%%/*}
    rest=${state_file#*/ports/}
    IB_PORT=${rest%%/*}
    export IB_DEVICE IB_PORT
    return 0
  done
  return 1
}

printf 'Ascend probe configuration: devices=%s soc=%s toolkit=%s ascendc=%s\n' \
  "$device_pair" "$soc_version" "$DEVICE_HOME" "$ascendc_dir"
npu-smi info | tee "$log_dir/npu-smi-info.log"

if build_probe vmm make -C "$probe_dir" all \
    DEVICE_HOME="$DEVICE_HOME" MPI_HOME="$MPI_HOME"; then
  run_probe 'VMM same device' "$probe_dir/build/vmm_probe" 0,0 ipc
  run_probe 'VMM cross device' "$probe_dir/build/vmm_probe" "$device_pair" ipc
else
  record_case 'VMM same device' NOT_RUN - 'VMM build failed'
  record_case 'VMM cross device' NOT_RUN - 'VMM build failed'
fi

if build_probe stream make -C "$probe_dir" stream-build \
    DEVICE_HOME="$DEVICE_HOME" MPI_HOME="$MPI_HOME" \
    SOC_VERSION="$soc_version" ASCENDC_CMAKE_DIR="$ascendc_dir"; then
  run_probe_group 'stream IPC same device' \
    "$probe_dir/build/stream/stream_visibility_probe" 0,0 ipc \
    valuewrite kernel-readback kernel-wait
  run_probe_group 'stream IPC cross device' \
    "$probe_dir/build/stream/stream_visibility_probe" "$device_pair" ipc \
    valuewrite kernel-readback kernel-wait

  if build_probe ib cmake --build "$probe_dir/build/stream" \
      --target ib_visibility_probe --parallel "$(nproc)"; then
    if select_ib_device; then
      printf 'Ascend probe NIC: device=%s port=%s gid_index=%s\n' \
        "$IB_DEVICE" "$IB_PORT" "$IB_GID_INDEX"
      run_probe_group 'NIC RDMA visibility' \
        "$probe_dir/build/stream/ib_visibility_probe" "$device_pair" ib \
        readback wait
    else
      record_case 'NIC RDMA visibility' NOT_RUN - 'no active IB device'
    fi
  else
    record_case 'NIC RDMA visibility' NOT_RUN - 'NIC probe build failed'
  fi
else
  record_case 'stream IPC same device' NOT_RUN - 'stream kernel build failed'
  record_case 'stream IPC cross device' NOT_RUN - 'stream kernel build failed'
  record_case 'NIC RDMA visibility' NOT_RUN - 'stream kernel build failed'
fi

total_count=$((pass_count + fail_count + unsupported_count + not_run_count))
printf 'Ascend probe summary: PASS=%d FAIL=%d UNSUPPORTED=%d NOT_RUN=%d TOTAL=%d; logs: %s\n' \
  "$pass_count" "$fail_count" "$unsupported_count" "$not_run_count" \
  "$total_count" "$log_dir"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '\nPASS=%d, FAIL=%d, UNSUPPORTED=%d, NOT_RUN=%d, TOTAL=%d.\n' \
    "$pass_count" "$fail_count" "$unsupported_count" "$not_run_count" \
    "$total_count" >> "$GITHUB_STEP_SUMMARY"
fi
if ((total_count != 5 || not_run_count != 0)); then
  echo 'Ascend probe report is incomplete; inspect build and device inventory logs.' >&2
  exit 1
fi
# Probe failures are findings. A completed five-case report is a successful CI
# diagnostic run, independent of the PASS/FAIL/UNSUPPORTED distribution.
exit 0
