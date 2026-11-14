#!/usr/bin/env bash
# Keep a stuck case from consuming the budget of the remaining runner tests.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
binary=
filter=${GTEST_FILTER:-*}
runner_mode=HOMO
pxn_enabled=0
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
  case "$arg" in
    FLAGCX_CI_EXPECT_RUNNER_MODE=*) runner_mode=${arg#*=} ;;
    FLAGCX_CI_EXPECT_PXN=*) pxn_enabled=${arg#*=} ;;
  esac
done
if [[ "$pxn_enabled" == 1 ]]; then
  runner_mode=PXN
elif [[ " ${args[*]} " == *"FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET"* ]]; then
  runner_mode=forced-NET
fi
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
skipped=0
start=$SECONDS
for index in "${!tests[@]}"; do
  test=${tests[index]}
  reason=$(python3 "$script_dir/ignore_test.py" runner \
    "$pxn_enabled" "$runner_mode" "$test")
  if [[ -n "$reason" ]]; then
    echo "::warning title=Iluvatar known-issue quarantine::$label $test SKIPPED: $reason (PR #670)"
    quarantine_log_root=${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}
    mkdir -p "$quarantine_log_root"
    printf '%s\t%s\t%s\n' "$label" "$test" "$reason" \
      >> "$quarantine_log_root/iluvatar-runner-quarantine.tsv"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
      printf -- '- **Temporarily skipped:** %s / `%s` — %s (PR #670)\n' \
        "$label" "$test" "$reason" >> "$GITHUB_STEP_SUMMARY"
    fi
    skipped=$((skipped + 1))
    continue
  fi
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
  if [[ "$test" == *.SendRecv ]]; then
    # This case currently hangs in CoreX stream synchronization. Still run
    # it and fail on timeout, but do not repeat a known deterministic hang.
    FLAGCX_CI_MPI_TIMEOUT="$sendrecv_timeout" \
      FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="$diagnostics_after" \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  else
    FLAGCX_CI_MPI_TIMEOUT="$per_test_timeout" \
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
if ((executed == 0 && skipped > 0)); then
  echo "$label ran no tests: all selected cases are quarantined" >&2
  status=1
fi
echo "$label summary: executed=$executed quarantined=$skipped result=$status"
exit "$status"
