#!/usr/bin/env bash
# Harness-only mock of `mthreads-gmi -L`. NEVER used for real acceptance.
# MOCK_DEVICES controls the reported GPU count (default 8).
n="${MOCK_DEVICES:-8}"
for ((i = 0; i < n; i++)); do
  printf 'GPU %d : MOCK MUSA Device (UUID : 00000000-0000-0000-0000-%012d)\n' "$i" "$i"
done
