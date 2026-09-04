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
export NP=8

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    adaptor|p2p)
      export FLAGCX_DEBUG=TRACE
      export FLAGCX_DEBUG_SUBSYS=ALL
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing MetaX environment for test suite: $suite"
  command -v mpirun
  command -v mxcc

  if compgen -G "/sys/class/infiniband/bnxt_roce*" >/dev/null; then
    local detected_hcas
    detected_hcas=$(printf '%s\n' /sys/class/infiniband/bnxt_roce* | xargs -n1 basename | paste -sd, -)
    if [[ "$suite" == "adaptor" || "$suite" == "p2p" ]]; then
      export FLAGCX_IB_HCA=$detected_hcas
    else
      export FLAGCX_IB_HCA=${FLAGCX_IB_HCA:-$detected_hcas}
    fi
  elif [[ "$suite" == "adaptor" || "$suite" == "p2p" ]]; then
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
  echo "net interfaces:"
  ls /sys/class/net 2>/dev/null || true
  echo "infiniband devices:"
  ls /dev/infiniband 2>/dev/null || true
  ibv_devices 2>/dev/null || true
  ibv_devinfo 2>/dev/null || true
  ip -o addr show 2>/dev/null || true
}
