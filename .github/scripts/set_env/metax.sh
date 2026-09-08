#!/usr/bin/env bash

# MetaX-specific unit-test environment setup.

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

# Use the real OpenMPI launcher if the image provides a wrapper.
if [[ -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" ]]; then
  FLAGCX_CI_MPI_HOME=$(mktemp -d)
  mkdir -p "$FLAGCX_CI_MPI_HOME/bin"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" \
    "$FLAGCX_CI_MPI_HOME/bin/mpirun"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/include" "$FLAGCX_CI_MPI_HOME/include"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/lib" "$FLAGCX_CI_MPI_HOME/lib"
  export MPI_HOME=$FLAGCX_CI_MPI_HOME
else
  export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
fi

export PATH="/opt/maca/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_RUNNER_NP=8
# MetaX heterogeneous runner currently faults when a progress thread writes to
# an IPC mapping imported by another thread. Keep the runner build and its
# non-MPI unit tests enabled, but skip MPI coverage until that runtime path is
# fixed and validated by the focused cross-thread IPC adaptor test.
FLAGCX_CI_RUNNER_MPI_SUPPORTED=0
FLAGCX_CI_RUNNER_MPI_SKIP_REASON="cross-thread IPC mappings are not yet writable on MetaX"
# Symmetric-memory MPI coverage currently hangs in
# SymMemTest.DevMemCreateWithWindow. Keep its build and non-MPI unit tests
# enabled while avoiding the full invocation timeout on every CI run.
export FLAGCX_CI_SYMMEM_MPI_SUPPORTED=0
export FLAGCX_CI_SYMMEM_MPI_SKIP_REASON="DevMemCreateWithWindow currently hangs on MetaX"
export NP=8

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    adaptor|p2p|rma)
      export FLAGCX_DEBUG=TRACE
      export FLAGCX_DEBUG_SUBSYS=ALL
      ;;
  esac

  case "$suite" in
    adaptor|p2p|rma)
      # bnxt_roce2 is attached to the PCIe/NUMA side containing physical GPUs
      # 4 and 5 on the MetaX CI runner. Use both CUDA- and MACA-compatible
      # visibility controls so the runtime consistently exposes that pair as
      # logical devices 0 and 1 to all focused two-GPU network suites.
      export CUDA_VISIBLE_DEVICES=4,5
      export MACA_VISIBLE_DEVICES=4,5
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  local require_roce=0
  echo "Preparing MetaX environment for test suite: $suite"
  command -v mpirun
  command -v mxcc

  case "$suite" in
    adaptor|p2p|rma|symmem) require_roce=1 ;;
  esac

  if compgen -G "/sys/class/infiniband/bnxt_roce*" >/dev/null; then
    local detected_hcas
    detected_hcas=$(printf '%s\n' /sys/class/infiniband/bnxt_roce* | xargs -n1 basename | paste -sd, -)
    if [[ "$suite" == "rma" ]]; then
      if [[ ! -d /sys/class/infiniband/bnxt_roce2 ]]; then
        echo "MetaX RMA tests require bnxt_roce2, but it was not found." >&2
        echo "Detected RoCE HCAs: $detected_hcas" >&2
        return 1
      fi
      # bnxt_roce3 timed out while creating a CQ on the RMA runner. Restrict
      # this focused suite to bnxt_roce2 instead of allowing topology
      # selection to route one rank through the unhealthy HCA.
      export FLAGCX_IB_HCA=bnxt_roce2
    elif [[ "$require_roce" == "1" ]]; then
      # Network-sensitive suites must use the validated MetaX RoCE HCAs even
      # when the runner or container supplies a different FLAGCX_IB_HCA.
      export FLAGCX_IB_HCA=$detected_hcas
    else
      export FLAGCX_IB_HCA=${FLAGCX_IB_HCA:-$detected_hcas}
    fi
  elif [[ "$require_roce" == "1" ]]; then
    echo "MetaX $suite tests require bnxt_roce*, but none was found." >&2
    echo "RDMA devices visible in /sys/class/infiniband:" >&2
    ls -la /sys/class/infiniband >&2 || true
    return 1
  fi

  if [[ -d /sys/class/net/bond0 ]]; then
    export FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-bond0}
  elif [[ -d /sys/class/net/eth0 ]]; then
    export FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-eth0}
  else
    local fallback_ifname
    fallback_ifname=$(ip -o link show 2>/dev/null | awk -F': ' '$2 != "lo" {print $2; exit}')
    if [[ -n "$fallback_ifname" ]]; then
      export FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-$fallback_ifname}
    fi
  fi

  echo "MetaX network diagnostics:"
  echo "FLAGCX_IB_HCA=${FLAGCX_IB_HCA:-<unset>}"
  echo "FLAGCX_IB_GID_INDEX=${FLAGCX_IB_GID_INDEX:-<unset>}"
  echo "FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-<unset>}"
  echo "net interfaces:"
  ls /sys/class/net 2>/dev/null || true
  echo "infiniband devices:"
  ls /dev/infiniband 2>/dev/null || true
  ibv_devices 2>/dev/null || true
  ibv_devinfo 2>/dev/null || true
  ip -o addr show 2>/dev/null || true
}
