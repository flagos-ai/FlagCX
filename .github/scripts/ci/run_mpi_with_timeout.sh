#!/usr/bin/env bash

set -euo pipefail
umask 022

script_dir=$(cd "$(dirname "$0")" && pwd)
source "$script_dir/timeouts.sh"
mpi_timeout="$FLAGCX_CI_MPI_TIMEOUT"
mpi_kill_after="${FLAGCX_CI_MPI_KILL_AFTER:-30s}"
mpi_launcher="${FLAGCX_CI_MPIRUN:-mpirun}"
mpi_label="${FLAGCX_CI_MPI_LABEL:-MPI invocation}"
test_runner="$script_dir/run_with_timeout.sh"
export FLAGCX_CI_LIFECYCLE_CASE="${FLAGCX_CI_LIFECYCLE_CASE:-$mpi_label}"

if ! command -v "$mpi_launcher" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_launcher" >&2
  exit 1
fi

# Keep this build prefix out of UCX's runtime option parser in MPI and workers.
unset UCX_HOME

# Lifecycle conformance must be reported by every rank. Restrict this wrapper
# to runner_mpi_tests so other MPI applications retain their existing contract.
args=("$@")
runner_index=-1
filter=${GTEST_FILTER:-*}
np=0
for ((i=0; i<${#args[@]}; ++i)); do
  if [[ "${args[i]##*/}" == runner_mpi_tests ]]; then
    runner_index=$i
  elif [[ "${args[i]}" == --gtest_filter=* ]]; then
    filter=${args[i]#--gtest_filter=}
    if [[ "${GTEST_FILTER:-*}" == *-* ]]; then
      if [[ "$filter" == *-* ]]; then
        filter+=":${GTEST_FILTER#*-}"
      else
        filter+="-${GTEST_FILTER#*-}"
      fi
    fi
    args[i]="--gtest_filter=$filter"
  elif [[ "${args[i]}" == -np || "${args[i]}" == -n ]]; then
    np=${args[i+1]}
  fi
done
require_lifecycle=0
if ((runner_index >= 0)); then
  require_lifecycle=$(python3 - "$filter" <<'PYFILTER'
import fnmatch, sys
case = "FlagCXCollTest.FinalizeThenDestroy"
positive, _, negative = sys.argv[1].partition("-")
print(int(any(fnmatch.fnmatchcase(case, p) for p in (positive or "*").split(":"))
          and not any(fnmatch.fnmatchcase(case, p) for p in negative.split(":"))))
PYFILTER
  )
fi
if ((require_lifecycle)); then
  lifecycle_log_root="${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}"
  result_root="$lifecycle_log_root/runner-lifecycle"
  mkdir -p "$result_root"
  result_dir=$(mktemp -d "$result_root/results.XXXXXX")
  # The container runs as root, while upload-artifact runs as the host user.
  # mktemp -d always starts at 0700, independently of the caller's umask.
  chmod 0755 "$lifecycle_log_root" "$result_root" "$result_dir"
  publish_lifecycle_xml() {
    local original_status=$? permission_status=0 report
    trap - EXIT
    for report in "$result_dir"/rank-*.xml; do
      [[ -f "$report" ]] || continue
      chmod 0644 "$report" || permission_status=$?
    done
    if ((original_status != 0)); then
      exit "$original_status"
    fi
    exit "$permission_status"
  }
  trap publish_lifecycle_xml EXIT
  args=("${args[@]:0:runner_index}" bash "$script_dir/run_gtest_rank_xml.sh"
        "$result_dir" "${args[@]:runner_index}")
  status=0
  FLAGCX_CI_TEST_TIMEOUT="$mpi_timeout" \
    FLAGCX_CI_TEST_KILL_AFTER="$mpi_kill_after" \
    FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="${FLAGCX_CI_MPI_DIAGNOSTICS_AFTER:-${FLAGCX_CI_TEST_DIAGNOSTICS_AFTER:-}}" \
    FLAGCX_CI_TEST_LABEL="$mpi_label" \
    "$test_runner" "$mpi_launcher" "${args[@]}" || status=$?
  xml=("$result_dir"/rank-*.xml)
  if [[ ! -f "${xml[0]}" || ${#xml[@]} -ne $np ]]; then
    echo "$mpi_label: expected $np lifecycle XML reports in $result_dir" >&2
    if ((status == 0)); then status=1; fi
  else
    python3 "$script_dir/check_required_gtests.py" \
      --case FlagCXCollTest.FinalizeThenDestroy "${xml[@]}" || {
        if ((status == 0)); then status=1; fi
      }
  fi
  exit "$status"
fi

FLAGCX_CI_TEST_TIMEOUT="$mpi_timeout" \
  FLAGCX_CI_TEST_KILL_AFTER="$mpi_kill_after" \
  FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="${FLAGCX_CI_MPI_DIAGNOSTICS_AFTER:-${FLAGCX_CI_TEST_DIAGNOSTICS_AFTER:-}}" \
  FLAGCX_CI_TEST_LABEL="$mpi_label" \
  exec "$test_runner" "$mpi_launcher" "${args[@]}"
