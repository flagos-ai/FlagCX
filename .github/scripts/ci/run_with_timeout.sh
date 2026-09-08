#!/usr/bin/env bash

set -euo pipefail

test_timeout="${FLAGCX_CI_TEST_TIMEOUT:-20m}"
test_kill_after="${FLAGCX_CI_TEST_KILL_AFTER:-30s}"
test_label="${FLAGCX_CI_TEST_LABEL:-test invocation}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [arguments...]" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required to run CI tests" >&2
  exit 1
fi

# Each call owns a fresh timeout process. A stalled test therefore cannot
# consume the entire job timeout or reduce the budget of later invocations.
echo "Starting $test_label (timeout=$test_timeout, kill-after=$test_kill_after)"
if timeout --signal=TERM --kill-after="$test_kill_after" \
  "$test_timeout" "$@"; then
  exit 0
else
  status=$?
  if [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "$test_label exceeded its $test_timeout timeout" >&2
  fi
  exit "$status"
fi
