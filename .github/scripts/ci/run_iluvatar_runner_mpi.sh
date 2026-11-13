#!/usr/bin/env bash
# Keep a stuck case from consuming the budget of the remaining runner tests.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
binary=
filter=${GTEST_FILTER:-*}
args=()
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
  echo 'Iluvatar runner binary is missing' >&2
  exit 1
fi

seconds_of() {
  local value=$1
  case "$value" in
    *h) awk -v v="${value%h}" 'BEGIN { printf "%.3f", v * 3600 }' ;;
    *m) awk -v v="${value%m}" 'BEGIN { printf "%.3f", v * 60 }' ;;
    *s) printf '%s' "${value%s}" ;;
    *) printf '%s' "$value" ;;
  esac
}

# Running one case per MPI launch loses the in-process short circuit that
# main_mpi.cpp uses after a permanent transport error, so every case now owns a
# full timeout budget. Bound each case well below the suite default and stop the
# stage once the cumulative budget is spent: the unit_tests_common.yml job is
# capped at 120 minutes and must not be consumed by a second stuck case.
per_test_timeout=${FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT:-5m}
stage_budget=${FLAGCX_CI_RUNNER_STAGE_BUDGET:-900}
sendrecv_timeout=${FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT:-2m}
# Collect diagnostics before, not after, timeout terminates the process group.
# Derive the delay from the timeout actually in effect so overriding one value
# cannot push collection past the kill.
diagnostics_after=${FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER:-$(
  awk -v t="$(seconds_of "$sendrecv_timeout")" 'BEGIN { printf "%.3f", t * 0.75 }'
)}

# GTest applies the original positive and negative filters before listing.
# Listing does not initialize MPI or run any hardware fixture.
listing=$(timeout --kill-after=5s 30s "$binary" \
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
  echo "Iluvatar runner filter selected no tests: $filter" >&2
  exit 1
fi

status=0
label=${FLAGCX_CI_MPI_LABEL:-Iluvatar runner}
executed=0
start=$SECONDS
for index in "${!tests[@]}"; do
  test=${tests[index]}
  if ((SECONDS - start >= stage_budget)); then
    echo "$label exhausted its ${stage_budget}s stage budget; these cases did" \
      "not run: ${tests[*]:index}" >&2
    if ((status == 0)); then
      status=1
    fi
    break
  fi
  executed=$((executed + 1))
  test_status=0
  require_pass=0
  if [[ "$test" == FlagCXCollTest.SendRecv ||
        "$test" == FlagCXCollTest.AlltoAll ]]; then
    # GTest returns zero for an all-skipped case. These restored cases must
    # report an actual pass, not merely a successful MPI exit.
    require_pass=1
  fi
  if [[ "$test" == *.SendRecv ]]; then
    # Give a stuck P2P case a short timeout and collect pre-timeout stacks.
    FLAGCX_CI_MPI_TIMEOUT="$sendrecv_timeout" \
      FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="$diagnostics_after" \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_REQUIRE_GTEST_PASS="$require_pass" \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  else
    FLAGCX_CI_MPI_TIMEOUT="$per_test_timeout" \
      FLAGCX_CI_REQUIRE_GTEST_PASS="$require_pass" \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${args[@]}" \
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
