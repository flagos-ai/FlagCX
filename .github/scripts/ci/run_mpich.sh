#!/usr/bin/env bash

set -euo pipefail

launcher="${MPI_HOME:-/usr/local/mpich-3.3.2}/bin/mpirun"
args=()
while (($#)); do
  case "$1" in
    --allow-run-as-root)
      shift
      ;;
    -x)
      shift
      [[ $# -gt 0 ]] || { echo "-x requires KEY=VALUE" >&2; exit 2; }
      key=${1%%=*}
      if [[ "$1" == *=* ]]; then
        value=${1#*=}
      else
        value=${!key:-}
      fi
      args+=(-env "$key" "$value")
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

exec "$launcher" "${args[@]}"
