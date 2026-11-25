#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 7 && $# -ne 8 ]]; then
  echo "Usage: $0 <active> <has-platforms> <prepare> <unit> <perf> <torch-api> <format> [level]" >&2
  exit 2
fi

active=$1
has_platforms=$2
prepare_result=$3
unit_result=$4
perf_result=$5
torch_result=$6
format_result=$7
level=${8:-L2}

case "$level" in
  L0|L1|L2|L3) ;;
  *) echo "Unknown CI level: $level" >&2; exit 2 ;;
esac

if [[ "$active" == false && "$prepare_result" == success ]]; then
  echo "No authorized CI request; no platform jobs were scheduled."
  exit 0
fi
if [[ "$prepare_result" != success ]]; then
  echo "Platform matrix preparation ended with result: $prepare_result" >&2
  exit 1
fi

if [[ "$has_platforms" == true && "$level" != L0 && "$unit_result" != success ]]; then
  echo "Selected platform unit tests ended with result: $unit_result" >&2
  exit 1
fi
if [[ ( "$has_platforms" != true || "$level" == L0 ) && "$unit_result" != success && "$unit_result" != skipped ]]; then
  echo "Unit-test job ended with result: $unit_result" >&2
  exit 1
fi
if [[ "$has_platforms" == true ]]; then
  case "$level" in
    L2|L3)
      for result in "$perf_result" "$torch_result"; do
        [[ "$result" == success ]] || {
          echo "Selected platform workflow ended with result: $result" >&2
          exit 1
        }
      done
      ;;
    *)
      # At L0/L1 perf/torch are intentionally skipped. Unexpected cancellation
      # or failure still needs attention.
      for result in "$perf_result" "$torch_result"; do
        case "$result" in
          success|skipped) ;;
          *) echo "Platform workflow failed at level $level: $result" >&2; exit 1 ;;
        esac
      done
      ;;
  esac
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
