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
export MPI_HOME=${MPI_HOME:-/usr/local/mpi}
export PATH="$DEVICE_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$DEVICE_HOME/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_IGNORE_TEST='{
  "phases": [
    {
      "suite": "rma",
      "name": "gdr-visibility",
      "reason": "CUDA immediate-consumer conformance cases are outside Ascend RMA/SymMem coverage"
    }
  ],
  "torch": {
    "modes": ["heterogeneous"],
    "operations": ["reduce", "allreduce", "reducescatter"],
    "reason": "C2C refresh deviceMemset fails before heterogeneous transfer"
  },
  "runner": [
    {
      "test": "FlagCXCollTest.AlltoAllvAsymmetric",
      "modes": ["runner default"],
      "reason": "HCCL AlltoAllV fails at stream synchronization"
    },
    {
      "test": "FlagCXCollTest.AllReduce",
      "modes": ["runner heterogeneous", "runner forced NET"],
      "reason": "C2C refresh deviceMemset fails before transfer"
    },
    {
      "test": "FlagCXCollTest.Reduce",
      "modes": ["runner heterogeneous", "runner forced NET"],
      "reason": "C2C refresh deviceMemset fails before transfer"
    },
    {
      "test": "FlagCXCollTest.ReduceScatter",
      "modes": ["runner heterogeneous", "runner forced NET"],
      "reason": "C2C refresh deviceMemset fails before transfer"
    }
  ]
}'
export FLAGCX_CI_IGNORE_TEST

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

# cann-shmem (aclshmem) backend: when the runner carries the runtime the
# Ascend suites build with USE_SHMEM=1 and the unified IR suite grades its
# cross-rank cases at the same four-chip world the intra suite already runs
# (UNIFIED_IR_NP follows FLAGCX_CI_INTRA_NP).  When it is absent the
# IPC-gated build above stays and the lost shmem coverage is surfaced as a
# warning - a missing runtime must never render as silently green coverage.
if [[ "${FLAGCX_CI_ASCEND_SHMEM:-1}" == 1 ]]; then
  if ASCEND_SHMEM_HOME=$("$FLAGCX_CI_ENV_DIR/../ci/check_shmem.sh"); then
    export SHMEM_HOME=$ASCEND_SHMEM_HOME
    export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-$DEVICE_HOME}
    export LD_LIBRARY_PATH="$SHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
    FLAGCX_CI_PROJECT_MAKE_ARGS+=(USE_SHMEM=1)
    FLAGCX_CI_TEST_MAKE_ARGS+=(USE_SHMEM=1)
    # Do not flip FLAGCX_CI_DEVICE_API_IR here: that gate launches the legacy
    # build/bin/test_device_ir_intra, which the Ascend Makefile never builds
    # (SKIPPED_TARGETS).  The unified IR suite has its own matrix entry
    # (device_api_unified_ir) and its own runner path (MC952 review P1).
    UNIFIED_IR_NP=$FLAGCX_CI_INTRA_NP
    export UNIFIED_IR_NP
    echo "cann-shmem backend enabled: SHMEM_HOME=$SHMEM_HOME"
  else
    echo "::warning title=Ascend shmem backend not exercised::cann-shmem runtime not found in the CI image; the IPC-gated build runs and the shmem cross-rank cases stay skipped"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
      printf -- '- **Ascend shmem backend not exercised:** cann-shmem runtime not found in the CI image\n' \
        >> "$GITHUB_STEP_SUMMARY"
    fi
  fi
fi

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
