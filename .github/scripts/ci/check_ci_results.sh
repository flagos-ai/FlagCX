#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 <active> <has-platforms> <prepare> <unit> <perf> <torch-api> <format>" >&2
  exit 2
fi

active=$1
has_platforms=$2
prepare_result=$3
unit_result=$4
perf_result=$5
torch_result=$6
format_result=$7

if [[ "$active" == false && "$prepare_result" == success ]]; then
  echo "No authorized CI request; no platform jobs were scheduled."
  exit 0
fi
if [[ "$prepare_result" != success ]]; then
  echo "Platform matrix preparation ended with result: $prepare_result" >&2
  exit 1
fi

if [[ "$has_platforms" == true && "$unit_result" != success ]]; then
  echo "Selected platform unit tests ended with result: $unit_result" >&2
  exit 1
fi
if [[ "$has_platforms" != true && "$unit_result" != success && "$unit_result" != skipped ]]; then
  echo "Unit-test job ended with result: $unit_result" >&2
  exit 1
fi
if [[ "$has_platforms" == true ]]; then
  for result in "$perf_result" "$torch_result"; do
    [[ "$result" == success ]] || {
      echo "Selected platform workflow ended with result: $result" >&2
      exit 1
    }
  done
else
  for result in "$perf_result" "$torch_result"; do
    case "$result" in
      success|skipped) ;;
      *) echo "Platform workflow ended with result: $result" >&2; exit 1 ;;
    esac
  done
fi
if [[ "$format_result" != success ]]; then
  echo "Common checks ended with result: $format_result" >&2
  exit 1
fi

echo "All selected FlagCX CI jobs completed successfully."
