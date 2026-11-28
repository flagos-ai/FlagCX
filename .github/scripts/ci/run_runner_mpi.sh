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
# Each platform and case uses the same pre-timeout diagnostics. The timeout
# wrapper computes "auto" from the effective (including cumulative) limit.
diagnostics_after=${FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER:-auto}

# GTest applies the original positive and negative filters before listing.
# Listing does not initialize MPI or run any hardware fixture.
listing_info=$(flagcx_ci_limit_timeout 30s details) || {
  listing_status=$?
  if [[ "$listing_status" == 124 ]]; then
    echo "Runner discovery: $listing_info cumulative budget exhausted; invocation not run" >&2
  else
    echo 'Runner discovery: invalid timeout or deadline; invocation not run' >&2
  fi
  exit "$listing_status"
}
read -r listing_timeout listing_source <<< "$listing_info"
listing=$(timeout --kill-after=5s "$listing_timeout" "$binary" \
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

# Diagnostics remain best effort when an opt-in gdb installation is unavailable.
if [[ "${FLAGCX_CI_INSTALL_GDB:-0}" == 1 ]] &&
   ! command -v gdb >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if ! timeout --signal=TERM --kill-after=5s 90s bash -c \
      'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gdb'; then
      echo 'Runner diagnostics: gdb installation failed; timeout diagnostics will use /proc only' >&2
    fi
  else
    echo 'Runner diagnostics: apt-get unavailable; timeout diagnostics will use /proc only' >&2
  fi
fi

status=0
label=${FLAGCX_CI_MPI_LABEL:-runner}
executed=0
for index in "${!tests[@]}"; do
  test=${tests[index]}
  if budget_info=$(flagcx_ci_limit_timeout "$per_test_timeout" details); then
    :
  else
    budget_status=$?
    if [[ "$budget_status" == 124 ]]; then
      echo "$label: $budget_info cumulative budget exhausted; these cases did not run: ${tests[*]:index}" >&2
    else
      echo "$label: invalid timeout or deadline; these cases did not run: ${tests[*]:index}" >&2
    fi
    if ((status == 0)); then
      status=$budget_status
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
  case_timeout=$per_test_timeout
  if [[ "$test" == *.SendRecv && "${FLAGCX_CI_PLATFORM:-}" == iluvatar ]]; then
    case_timeout=$sendrecv_timeout
  fi
  FLAGCX_CI_MPI_TIMEOUT="$case_timeout" \
    FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="$diagnostics_after" \
    FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
    FLAGCX_CI_REQUIRE_GTEST_PASS="$require_pass" \
    FLAGCX_CI_LIFECYCLE_CASE="$test" \
    FLAGCX_CI_MPI_LABEL="$label $test" \
    "$script_dir/run_mpi_with_timeout.sh" "${mpi_env[@]}" "${args[@]}" \
    --gtest_filter="$test" || test_status=$?

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
