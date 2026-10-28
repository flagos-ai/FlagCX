#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 || "$2" != exclude ]]; then
  echo "Usage: $0 <gtest-filter> exclude" >&2
  exit 2
fi

filter=$1
if [[ "$filter" == *-* ]]; then
  positive=${filter%%-*}
  negative=${filter#*-}
else
  positive=$filter
  negative=
fi

[[ -n "$positive" ]] || positive='*'
if [[ -n "$negative" ]]; then
  negative+=:
fi
printf '%s-%sCoverageSupplemental.*\n' "$positive" "$negative"
