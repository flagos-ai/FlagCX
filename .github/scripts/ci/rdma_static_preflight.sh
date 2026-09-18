#!/usr/bin/env bash

# Source-only helper shared by accelerator-specific set_env scripts.

flagcx_ci_validate_rdma_static() {
  if [[ $# -lt 3 ]]; then
    echo "Usage: flagcx_ci_validate_rdma_static <platform> <suite> <HCA glob>..." >&2
    return 2
  fi

  local platform=$1
  local suite=$2
  shift 2
  local pattern hca port state physical_state link_layer pci_path
  local -a hcas=()
  local found_usable_port=0

  for pattern in "$@"; do
    while IFS= read -r hca; do
      [[ -n "$hca" ]] && hcas+=("$hca")
    done < <(compgen -G "$pattern" || true)
  done

  if [[ ${#hcas[@]} -eq 0 ]]; then
    echo "$platform $suite tests require an RDMA HCA matching: $*" >&2
    return 1
  fi
  if ! compgen -G "/sys/class/infiniband_verbs/uverbs*" >/dev/null; then
    echo "$platform $suite tests require uverbs entries in /sys/class/infiniband_verbs." >&2
    return 1
  fi
  if ! compgen -G "/dev/infiniband/uverbs*" >/dev/null; then
    echo "$platform $suite tests require /dev/infiniband/uverbs* device nodes." >&2
    return 1
  fi
  if [[ ! -e /dev/infiniband/rdma_cm ]]; then
    echo "$platform $suite tests require /dev/infiniband/rdma_cm." >&2
    return 1
  fi

  for hca in "${hcas[@]}"; do
    pci_path=$(readlink -f "$hca/device" 2>/dev/null || true)
    if [[ -z "$pci_path" ]]; then
      pci_path="<unavailable>"
    fi
    while IFS= read -r port; do
      [[ -n "$port" ]] || continue
      if [[ -r "$port/state" ]]; then
        state=$(<"$port/state")
      else
        state="<unavailable>"
      fi
      if [[ -r "$port/link_layer" ]]; then
        link_layer=$(<"$port/link_layer")
      else
        link_layer="<unavailable>"
      fi
      if [[ -r "$port/phys_state" ]]; then
        physical_state=$(<"$port/phys_state")
      else
        physical_state="<unavailable>"
      fi
      echo "$platform RDMA port: hca=${hca##*/} pci=$pci_path port=${port##*/} state=$state physical_state=$physical_state link_layer=$link_layer"
      if [[ "$state" == 4:* ]]; then
        case "${link_layer,,}" in
          ethernet|infiniband) found_usable_port=1 ;;
        esac
      fi
    done < <(compgen -G "$hca/ports/*" || true)
  done

  if [[ "$found_usable_port" != 1 ]]; then
    echo "$platform $suite tests require at least one ACTIVE Ethernet (RoCE) or InfiniBand port." >&2
    return 1
  fi
}
