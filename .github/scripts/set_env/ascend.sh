#!/usr/bin/env bash
FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export DEVICE_HOME=${DEVICE_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
export CCL_HOME=${CCL_HOME:-$DEVICE_HOME}
export FLAGCX_ADAPTOR=ascend
export FLAGCX_CI_RMA_GDR_VISIBILITY_CONSUMER_SUPPORTED=0
export MPI_HOME=${MPI_HOME:-/usr/local/mpi}
export PATH="$DEVICE_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$DEVICE_HOME/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"


FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ASCEND=1)
FLAGCX_CI_ENABLE_IBUC=1
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_ENABLE_PXN=${FLAGCX_CI_ENABLE_PXN:-0}
# NPU memory registration is part of the Ascend IBRC contract. Require the
# four one-sided GPU cases to transfer and verify data.
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
export NP=4

# One Ascend host provides enough devices to run two logical host contexts
# through MPI MPMD for inter-mode transport coverage.
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)

# Ascend builds only test_device_api_intra for now (test/unittest/device_api
# compiles the IR and inter cases out until the device peer-memory and atomic
# layers land), so the device_api suite runs the intra step only.
FLAGCX_CI_DEVICE_API_IR=0
FLAGCX_CI_DEVICE_API_INTER=0

# The unified IR suite runs INTRA only on Ascend.  The Device IR bindings
# declare the NET transport arm absent (FLAGCX_DEVICE_HAS_NET_TRANSPORT=0 in
# the shared impl), so the inter product is not built and neither the inter
# step nor the FORCE_NET fallback runs have a transport to exercise here;
# cross-node unified IR is deferred to the T2 inter stage.  Both gates default
# to 1 elsewhere, so this only narrows Ascend.
FLAGCX_CI_UNIFIED_IR_INTER=0
FLAGCX_CI_UNIFIED_IR_FALLBACK_INTRA=0

# 910C hardware contract (measured 2026-11-17): AI-core scalar stores into
# IPC-imported peer GM fault-free the kernel even with
# ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS set (bounded-wait probe:
# the wait never runs; AICore idle while the stream stays active; the
# import itself returns no error), so device-side cross-rank rendezvous
# is NOT a supported capability on this backend.  Support policy
# (MC952 review P1): the cross-rank cases S16-S18 are capability-gated to
# SKIP at compile time (FLAGCX_TEST_NO_CROSS_RANK_COMM in the ascend block
# of test/unittest/device_api/Makefile) - a world-size-1 barrier iterates
# over zero peers and a world-size-1 put resolves to self, so PASS there
# would fake a certification this hardware cannot give.  UNIFIED_IR_NP=1
# (now honored by BOTH the runner and the Makefile recipe) runs the suite
# end-to-end as launch/init + skip-contract smoke: every cross-rank,
# remote-read and atomic case reports SKIP, and the final verdict says
# "NO CASES GRADED" instead of an empty ALL PASS.  The four-rank
# communication certification target stays open and needs a working peer
# transport (MTE/SSLL track), tracked as follow-up outside this PR.
UNIFIED_IR_NP=1
export UNIFIED_IR_NP

flagcx_ci_configure_suite() {
  case "$1" in
    device_api|device_api_unified_ir)
      # The intra device kernels live in test/kernel/ascend and the library's
      # device-API helpers (flagcxDevNetLaunchConstruct et al.) are only
      # emitted with COMPILE_KERNEL=1; same contract as cuda/hygon.
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;&
    core|adaptor|device_api_host|p2p|rma|runner|service|symmem|perf|torch-api) ;;
    *) echo "Ascend CI does not enable suite: $1" >&2; return 1 ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  if [[ "$suite" == rma || "$suite" == symmem ]]; then
    export FLAGCX_CI_REQUIRE_GTEST_PASS=1
  fi
  echo "Preparing Ascend environment for CI workload: $suite"
  test -d "$DEVICE_HOME/include"
  test -e /dev/davinci_manager
  test -e /dev/devmm_svm
  test -e /dev/hisi_hdc
  test -x /usr/local/sbin/npu-smi
  test -e /dev/infiniband/rdma_cm
  echo "Ascend host and container device inventory:"
  npu-smi info
  ls -l /dev/davinci* /dev/infiniband
  command -v g++
  command -v python3
  # Device-toolchain inventory (reviewer ask of 2026-11-18): the qualifier
  # selection in device_utils.h keys on the makefile macro now, but the
  # exact bisheng a job ran belongs in the log so a future build failure
  # can be attributed without a re-run.
  command -v bisheng || true
  bisheng --version 2>&1 | sed -n '1,3p' || true
  python3 "$FLAGCX_CI_ENV_DIR/../ci/check_ascend_pci_runtime.py"
  if [[ "$suite" == core ]]; then
    flagcx_ci_validate_rdma core
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static Ascend "$suite" "/sys/class/infiniband/*"
}
