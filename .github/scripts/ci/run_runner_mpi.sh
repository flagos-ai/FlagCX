#!/usr/bin/env bash
# Keep a stuck case from consuming the budget of the remaining runner tests.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/timeouts.sh"
binary=
filter=${GTEST_FILTER:-*}
args=()
mpi_env=()
if [[ -n "${FLAGCX_CI_SENDRECV_SYNC_BEFORE_D2H:-}" ]]; then
  mpi_env+=(
    -x "FLAGCX_CI_SENDRECV_SYNC_BEFORE_D2H=$FLAGCX_CI_SENDRECV_SYNC_BEFORE_D2H"
  )
fi
for arg in "$@"; do
  if [[ "$arg" == --gtest_filter=* ]]; then
    filter=${arg#--gtest_filter=}
    continue
  fi
  args+=("$arg")
  if [[ "${arg##*/}" == runner_mpi_tests ]]; then
    binary=$arg
  fi
done
if [[ -z "$binary" || ! -x "$binary" ]]; then
  echo 'Runner binary is missing' >&2
  exit 1
fi

per_test_timeout=$FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT
stage_budget=$FLAGCX_CI_RUNNER_STAGE_BUDGET
sendrecv_timeout=${FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT:-$per_test_timeout}
flagcx_ci_start_budget FLAGCX_CI_MODE_DEADLINE "$stage_budget"
# Keep the existing SendRecv-only diagnostic enablement.
diagnostics_after=${FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER:-$(
  awk -v t="$(flagcx_ci_seconds "$sendrecv_timeout")" 'BEGIN { printf "%.3f", t * 0.75 }'
)}

# GTest applies the original positive and negative filters before listing.
# Listing does not initialize MPI or run any hardware fixture.
listing=$(timeout --kill-after=5s "$(flagcx_ci_limit_timeout 30s)" "$binary" \
  --gtest_list_tests --gtest_filter="$filter")
# Accept every suite the binary reports. Hard-coding one suite name would
# silently drop any case added to this binary later.
# Build the array with a read loop: mapfile needs Bash 4 and this repository
# still supports the Bash 3.2 shipped on macOS.
tests=()
while IFS= read -r discovered; do
  [[ -n "$discovered" ]] && tests+=("$discovered")
done < <(awk '
  /^[^[:space:]]/ {
    suite = ""
    if ($1 ~ /\.$/)
      suite = $1
    next
  }
  suite != "" && /^[[:space:]]+[A-Za-z0-9_]/ { print suite $1 }
' <<< "$listing")
if ((${#tests[@]} == 0)); then
  echo "Runner filter selected no tests: $filter" >&2
  exit 1
fi

# The published CI image may predate the Dockerfile's gdb dependency. Install
# it for this runner job so a SendRecv timeout yields user-space thread stacks.
# Diagnostics remain best effort if the package mirror is unavailable.
needs_sendrecv_diagnostics=0
for test in "${tests[@]}"; do
  if [[ "$test" == *.SendRecv ]]; then
    needs_sendrecv_diagnostics=1
    break
  fi
done
if ((needs_sendrecv_diagnostics)) &&
   [[ "${FLAGCX_CI_INSTALL_GDB:-0}" == 1 ]] &&
   ! command -v gdb >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if ! timeout --signal=TERM --kill-after=5s 90s bash -c \
      'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gdb'; then
      echo 'Iluvatar runner: gdb installation failed; timeout diagnostics will use /proc only' >&2
    fi
  else
    echo 'Iluvatar runner: apt-get unavailable; timeout diagnostics will use /proc only' >&2
  fi
fi

status=0
label=${FLAGCX_CI_MPI_LABEL:-runner}
executed=0
for index in "${!tests[@]}"; do
  test=${tests[index]}
  if ! flagcx_ci_limit_timeout "$per_test_timeout" >/dev/null; then
    echo "$label exhausted its ${stage_budget}s stage budget; these cases did" \
      "not run: ${tests[*]:index}" >&2
    if ((status == 0)); then
      status=124
    fi
    break
  fi
  skip_reason=$(python3 "$script_dir/ignore_test.py" runner 0 "$label" "$test")
  if [[ -n "$skip_reason" ]]; then
    echo "Skipping $label $test: $skip_reason"
    continue
  fi
  executed=$((executed + 1))
  test_status=0
  require_pass=0
  if [[ "${FLAGCX_CI_PLATFORM:-}" == iluvatar &&
        ( "$test" == FlagCXCollTest.SendRecv ||
          "$test" == FlagCXCollTest.AlltoAll ) ]]; then
    # GTest returns zero for an all-skipped case. These restored cases must
    # report an actual pass, not merely a successful MPI exit.
    require_pass=1
  fi
  if [[ "$test" == *.SendRecv && "${FLAGCX_CI_PLATFORM:-}" == iluvatar ]]; then
    # Keep Iluvatar's existing pre-timeout diagnostics.
    FLAGCX_CI_MPI_TIMEOUT="$sendrecv_timeout" \
      FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="$diagnostics_after" \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_REQUIRE_GTEST_PASS="$require_pass" \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${mpi_env[@]}" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  else
    FLAGCX_CI_MPI_TIMEOUT="$per_test_timeout" \
      FLAGCX_CI_REQUIRE_GTEST_PASS="$require_pass" \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${mpi_env[@]}" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  fi
  echo "$label $test result=$test_status"
  if [[ "$test_status" == 130 || "$test_status" == 143 ]]; then
    exit "$test_status"
  fi
  if ((status == 0 && test_status != 0)); then
    status=$test_status
  fi
done
echo "$label summary: executed=$executed result=$status"
exit "$status"
