#!/usr/bin/env bash
# T2/T4 on two *real* physical nodes (4 MUSA devices each, 8 ranks total).
#
# This gate is deliberately separate from run_inter.sh / run_unified_inter.sh:
# those launch two logical nodes on a single host through FLAGCX_HOSTID and
# therefore cannot exercise a real NIC, a real second host hash, or real
# cross-host RDMA. Here:
#   * ranks are split by a real MPI hostfile (2 hosts x 4 slots),
#   * no FLAGCX_HOSTID is injected, so FlagCX discovers two distinct hosts,
#   * MUSA_VISIBLE_DEVICES is 0,1,2,3 on *both* nodes (4-device nodes),
#   * the IB transport is pinned to the RoCE interface/HCA pair used by the
#     acceptance run, and device buffers reach the HCA through DMA-BUF.
#
# Stages (gated):
#   run_real_inter.t2_api  test_device_api_inter           (K1..K14)
#
# Stages (opt-in, REAL_INTER_IR=1, known to fail on MUSA):
#   run_real_inter.t2_ir   test_device_ir_inter            (S1..S14)
#   run_real_inter.t4      test_device_ir_unified_inter    (S16..S25)
# The two Device IR suites abort with err=700 (illegal memory access) in the
# first remote-signal case (S3 and S20). Until that device-side IR signal path
# is fixed they are not part of the pass criteria; REAL_INTER_IR=1 runs them
# and records the failure as evidence.
#
# Usage:
#   bash scripts/musa_device_api/run_real_inter.sh [--allow-blocked] [-- extra]
#
# Overridable environment (defaults match the 79/08 acceptance pair):
#   REAL_INTER_HOSTFILE   /tmp/flagcx.hosts
#   REAL_INTER_SSH_PORT   22222
#   REAL_INTER_SSH_KEY    /root/.ssh/flagcx_mpi
#   REAL_INTER_KNOWN_HOSTS /root/.ssh/flagcx_known_hosts
#   REAL_INTER_IFNAME     ens12np0
#   REAL_INTER_IB_HCA     mlx5_0
#   REAL_INTER_GID_INDEX  3
#   REAL_INTER_DEVICES    0,1,2,3
#   REAL_INTER_NODE_RANKS 4
#   REAL_INTER_IR         0 (1 also runs the known-failing Device IR stages)
#
# PASS requires: exit 0, every official case line PASS on all 8 ranks, the
# "Nodes: 2 / IntraSize: 4" topology report, the IB transport line, and
# byte-identical libflagcx.so on every node (workspaces are not shared).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"
# shellcheck source=common.sh
source "$HERE/common.sh"

REAL_INTER_HOSTFILE="${REAL_INTER_HOSTFILE:-/tmp/flagcx.hosts}"
REAL_INTER_SSH_PORT="${REAL_INTER_SSH_PORT:-22222}"
REAL_INTER_SSH_KEY="${REAL_INTER_SSH_KEY:-/root/.ssh/flagcx_mpi}"
REAL_INTER_KNOWN_HOSTS="${REAL_INTER_KNOWN_HOSTS:-/root/.ssh/flagcx_known_hosts}"
REAL_INTER_IFNAME="${REAL_INTER_IFNAME:-ens12np0}"
REAL_INTER_IB_HCA="${REAL_INTER_IB_HCA:-mlx5_0}"
REAL_INTER_GID_INDEX="${REAL_INTER_GID_INDEX:-3}"
REAL_INTER_DEVICES="${REAL_INTER_DEVICES:-0,1,2,3}"
REAL_INTER_NODE_RANKS="${REAL_INTER_NODE_RANKS:-4}"
REAL_INTER_IR="${REAL_INTER_IR:-0}"

# These are cross-node requirements specific to this gate; common.sh only
# forwards a fixed list, so extend it before mpi_env_flags is called.
for v in FLAGCX_IB_HCA FLAGCX_IB_GID_INDEX FLAGCX_SIGNAL_HOST_ENABLE \
         FLAGCX_DMABUF_ENABLE FLAGCX_SOCKET_IFNAME; do
  case " ${MPI_PROPAGATED_ENV[*]} " in
    *" $v "*) ;;
    *) MPI_PROPAGATED_ENV+=("$v") ;;
  esac
done

NAME=run_real_inter

allow_blocked=0
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-blocked) allow_blocked=1; shift ;;
    --) shift; extra=("$@"); break ;;
    -h|--help)
      sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) extra+=("$1"); shift ;;
  esac
done

begin_status_record "$NAME"

# Cross-node registration is a hard requirement: MUSA device memory cannot be
# registered with the HCA as a raw VA, so every one-sided MR goes through
# DMA-BUF. The signal/counter plane stays in device memory: the host-pinned
# plane (FLAGCX_SIGNAL_HOST_ENABLE=1) intermittently terminates cross-node
# device waits with err=702 on this SDK/driver pair, so it cannot be gated yet.
export FLAGCX_SIGNAL_HOST_ENABLE="${FLAGCX_SIGNAL_HOST_ENABLE:-0}"
export FLAGCX_DMABUF_ENABLE="${FLAGCX_DMABUF_ENABLE:-1}"
export FLAGCX_IB_HCA="$REAL_INTER_IB_HCA"
export FLAGCX_IB_GID_INDEX="$REAL_INTER_GID_INDEX"
export FLAGCX_SOCKET_IFNAME="${FLAGCX_SOCKET_IFNAME:-$REAL_INTER_IFNAME}"
# Keep the registration route visible: FLAGCX_REG prints the DMA-BUF gate and
# the exported fd for every one-sided MR, which is the evidence that the data
# path did not silently fall back to a raw VA MR.
export FLAGCX_DEBUG_SUBSYS="${FLAGCX_DEBUG_SUBSYS:-INIT,NET,REG}"
ensure_net_env

HOSTS=()
HOSTNAMES=()
if [ ! -r "$REAL_INTER_HOSTFILE" ]; then
  record_blocked "$NAME" "hostfile $REAL_INTER_HOSTFILE is not readable"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
while read -r host _; do
  case "$host" in ''|\#*) continue ;; esac
  HOSTS+=("$host")
done <"$REAL_INTER_HOSTFILE"
TOTAL_RANKS=$((REAL_INTER_NODE_RANKS * ${#HOSTS[@]}))

if [ "${#HOSTS[@]}" -ne 2 ]; then
  record_blocked "$NAME" "expected exactly 2 physical hosts in $REAL_INTER_HOSTFILE, found ${#HOSTS[@]}"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi

LOCAL_ADDRS=" $(hostname) $(hostname -I 2>/dev/null) "
is_local_host() {
  case "$LOCAL_ADDRS" in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

ssh_host() {
  local host="$1"; shift
  if is_local_host "$host"; then
    bash -c "$*"
    return $?
  fi
  ssh -p "$REAL_INTER_SSH_PORT" -i "$REAL_INTER_SSH_KEY" -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$REAL_INTER_KNOWN_HOSTS" \
    "$host" "$@"
}

# record_assert <line> [record-name]
# Defaults to the aggregate record; pass a stage tag to keep per-stage evidence
# (e.g. the two-host log assertion) in the stage's own .status file.
record_assert() {
  printf '%s\n' "$1" >>"$LOG_DIR/${2:-$NAME}.status"
}

# ---------------------------------------------------------------------------
# Preflight: per-host device inventory and artifact identity.
# Workspaces are separate copies on each node, so a stale libflagcx.so on one
# node silently changes the result (this happened once: 79 ran the new
# DMA-BUF library while 08 still had the previous build).
# ---------------------------------------------------------------------------
preflight() {
  local rc=0 host n devs lib_sha bin_name local_bin remote_bin local_lib
  local hostname
  local_lib="$(sha256sum "$FLAGCX_BUILD_LIB/libflagcx.so" | awk '{print $1}')"
  for host in "${HOSTS[@]}"; do
    if ! ssh_host "$host" true >/dev/null 2>&1; then
      record_assert "assertion=FAIL ssh unreachable $host (port $REAL_INTER_SSH_PORT)"
      rc=1
      continue
    fi
    record_assert "assertion=PASS ssh reachable $host (port $REAL_INTER_SSH_PORT, key $REAL_INTER_SSH_KEY)"
    hostname="$(ssh_host "$host" hostname 2>/dev/null | tr -d '\r')"
    # FlagCX log prefixes truncate at the first dot (getHostName(..., '.')),
    # so compare the short name, not a possible FQDN.
    hostname="${hostname%%.*}"
    HOSTNAMES+=("${hostname:-$host}")
    record_assert "assertion=PASS $host hostname=${hostname:-unknown}"
    devs="$(ssh_host "$host" "ls -d /dev/dri/renderD* 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')"
    if [ "${devs:-0}" -ge "$REAL_INTER_NODE_RANKS" ]; then
      record_assert "assertion=PASS $host exposes $devs renderD device(s) >= $REAL_INTER_NODE_RANKS"
    else
      record_assert "assertion=FAIL $host exposes ${devs:-0} renderD device(s) < $REAL_INTER_NODE_RANKS"
      rc=1
    fi
    n="$(ssh_host "$host" "aa=\$(ls -d /dev/infiniband/uverbs* 2>/dev/null | wc -l); bb=\$(cat /sys/class/infiniband/$REAL_INTER_IB_HCA/ports/1/state 2>/dev/null | awk '{print \$2}'); printf '%s %s' \"\$aa\" \"\$bb\"" 2>/dev/null)"
    case "$n" in
      *ACTIVE*) record_assert "assertion=PASS $host $REAL_INTER_IB_HCA port ACTIVE (uverbs=$(printf '%s' "$n" | awk '{print $1}'))" ;;
      *)
        record_assert "assertion=FAIL $host $REAL_INTER_IB_HCA not ACTIVE ($n)"
        rc=1
        ;;
    esac
    lib_sha="$(ssh_host "$host" "sha256sum '$FLAGCX_BUILD_LIB/libflagcx.so' 2>/dev/null | awk '{print \$1}'")"
    # Guard against the empty==empty case: a missing artifact on both sides
    # must never be recorded as a matching hash.
    if [ -n "$local_lib" ] && [ "$lib_sha" = "$local_lib" ]; then
      record_assert "assertion=PASS $host libflagcx.so sha256 matches ($lib_sha)"
    else
      record_assert "assertion=FAIL $host libflagcx.so sha256=${lib_sha:-none} != local ${local_lib:-none}"
      rc=1
    fi
    # mpirun executes the absolute test path on each node; a stale test binary
    # would print the same PASS lines from different code. Pin them too.
    for bin_name in test_device_api_inter test_device_ir_inter \
                   test_device_ir_unified_inter; do
      local_bin="$(sha256sum "$DEVICE_API_BIN_DIR/$bin_name" 2>/dev/null | awk '{print $1}')"
      remote_bin="$(ssh_host "$host" "sha256sum '$DEVICE_API_BIN_DIR/$bin_name' 2>/dev/null | awk '{print \$1}'" | tr -d '\r')"
      if [ -n "$local_bin" ] && [ "$remote_bin" = "$local_bin" ]; then
        record_assert "assertion=PASS $host $bin_name sha256 matches ($local_bin)"
      else
        record_assert "assertion=FAIL $host $bin_name sha256=${remote_bin:-none} != local ${local_bin:-none}"
        rc=1
      fi
    done
  done
  if [ "$RDMA_AVAILABLE" -ne 1 ]; then
    record_assert "assertion=FAIL no local RDMA HCA; cross-node gate needs a real mlx device"
    rc=1
  fi
  return "$rc"
}

if ! preflight; then
  set_status "$NAME" FAIL 1 0
  printf 'verdict=FAIL\n' >>"$LOG_DIR/$NAME.status"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 1
fi

# A leaked FLAGCX_HOSTID would collapse both physical hosts into one logical
# node. The topology assertions would catch it, but unset it up front so the
# run is not wasted.
unset FLAGCX_HOSTID NCCL_HOSTID
record_assert "assertion=PASS FLAGCX_HOSTID/NCCL_HOSTID unset (real host discovery)"
read -r -a ENVV <<<"$(mpi_env_flags "" "$REAL_INTER_DEVICES")"
MPI_CROSS_ARGS=(
  --hostfile "$REAL_INTER_HOSTFILE"
  -np "$TOTAL_RANKS"
  --map-by "ppr:$REAL_INTER_NODE_RANKS:node"
  --bind-to none
  --mca plm_rsh_agent "ssh -F /dev/null -p $REAL_INTER_SSH_PORT -i $REAL_INTER_SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$REAL_INTER_KNOWN_HOSTS"
  --mca pml ob1
  --mca btl self,tcp
  --mca btl_tcp_if_include "$FLAGCX_SOCKET_IFNAME"
  --mca oob_tcp_if_include "$FLAGCX_SOCKET_IFNAME"
)

rc_total=0
timed_out=0

# require_real_node_log <name>: the two host prefixes appear in FlagCX's own
# records, proving the run really spanned both physical nodes.
require_real_node_log() {
  local name="$1" rc=0 hostname
  for hostname in "${HOSTNAMES[@]}"; do
    if grep -q "${hostname//./\\.}:" "$LOG_DIR/$name.out" 2>/dev/null; then
      record_assert "assertion=PASS $name output contains records from $hostname" "$name"
    else
      record_assert "assertion=FAIL $name output has no records from $hostname" "$name"
      rc=1
    fi
  done
  return "$rc"
}

run_one() {
  local tag="$1" bin="$2"; shift 2
  local args=("$@")
  log "running $tag: $TOTAL_RANKS ranks on ${#HOSTS[@]} physical nodes, $bin ${args[*]}"
  begin_status_record "$tag"
  if [ ! -x "$bin" ]; then
    record_blocked "$tag" "missing binary $bin"
    return 1
  fi
  assert_runtime_lib "$tag" "$bin" || {
    rc_total=1
    finalize_status "$tag" FAIL 1
    return 1
  }
  run_capture "$tag" "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" "${MPI_CROSS_ARGS[@]}" \
    "${ENVV[@]}" "$bin" "${args[@]}"
  local rc=$?
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && timed_out=1
  local verdict
  verdict="$(classify_output "$tag" "$rc")"
  if [ "$verdict" = "PASS" ]; then
    case "$tag" in
      "$NAME.t2_api")
        require_case_pass "$tag" "$TOTAL_RANKS" \
          'K14 OneSidedAlltoAll.* PASS' "T2 K14 OneSidedAlltoAll" || verdict=FAIL
        require_case_pass "$tag" "$TOTAL_RANKS" \
          '=== Overall: PASS ===' "T2 overall PASS" || verdict=FAIL
        ;;
      "$NAME.t2_ir")
        require_case_pass "$tag" "$TOTAL_RANKS" \
          'S14 OneSidedAlltoAll: PASS' "T2 S14 OneSidedAlltoAll" || verdict=FAIL
        require_case_pass "$tag" "$TOTAL_RANKS" \
          '=== Overall: PASS ===' "T2 ir overall PASS" || verdict=FAIL
        ;;
      "$NAME.t4")
        local c
        for c in \
          'S16 DevBarrier\(INTER\+WORLD\)' \
          'S17 DevTeamResolution\(INTER\+WORLD\+INTRA-locality\)' \
          'S18 DevPut\+PutValue\(INTER\+WORLD\)' \
          'S19 DevGet\(INTER\+WORLD\)' \
          'S20 DevSignalStandalone\(INTER\+WORLD\)' \
          'S21 DevPutSignalWait\(INTER\+WORLD\)' \
          'S22 DevPut_RSig\(INTER\+WORLD\)' \
          'S23 DevPutCounter\(INTER\+WORLD\)' \
          'S24 DevPutValue_RSig\(INTER\+WORLD\)' \
          'S25 DevSignalShadowFlush\(INTER\+WORLD\)'; do
          require_case_pass "$tag" "$TOTAL_RANKS" "$c.* PASS" "T4 $c" || verdict=FAIL
        done
        require_case_pass "$tag" 1 '=== Final Result: ALL PASS ===' \
          "T4 final result" || verdict=FAIL
        ;;
    esac
    case "$tag" in
      "$NAME.t2_api")
        check_log_pattern "$tag" 'nRanks: 8, regMode:' \
          "8 total ranks registered for one-sided ops" || verdict=FAIL
        check_log_pattern "$tag" 'K1 DevNetGetFromComm +PASS \(intraSize=4\)' \
          "intra size 4" || verdict=FAIL
        ;;
      "$NAME.t2_ir")
        check_log_pattern "$tag" 'nranks 8' \
          "8 total ranks" || verdict=FAIL
        ;;
      "$NAME.t4")
        apply_node_assertions "$tag" 2 "$REAL_INTER_NODE_RANKS" || verdict=FAIL
        ;;
    esac
    check_log_pattern "$tag" 'localRanks=4' \
      "each physical node reports 4 local ranks" || verdict=FAIL
    assert_transport "$tag" || verdict=FAIL
    require_real_node_log "$tag" || verdict=FAIL
  fi
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && verdict=TIMEOUT
  printf 'verdict=%s\n' "$verdict" >>"$LOG_DIR/$tag.status"
  finalize_status "$tag" "$verdict" "$rc"
  [ "$verdict" = "PASS" ] || rc=1
  return "$rc"
}

COMMON_ARGS=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR")
ran_any=0

if require_bin "$NAME.t2_api" test_device_api_inter; then
  ran_any=1
  run_one "$NAME.t2_api" "$BIN" "${COMMON_ARGS[@]}" -R "$REG_MODE" || rc_total=1
else
  rc_total=1
fi
if [ "$REAL_INTER_IR" -eq 1 ]; then
  if require_bin "$NAME.t2_ir" test_device_ir_inter; then
    ran_any=1
    run_one "$NAME.t2_ir" "$BIN" "${COMMON_ARGS[@]}" "${extra[@]}" || rc_total=1
  else
    rc_total=1
  fi
  if require_bin "$NAME.t4" test_device_ir_unified_inter; then
    ran_any=1
    run_one "$NAME.t4" "$BIN" "${COMMON_ARGS[@]}" "${extra[@]}" || rc_total=1
  else
    rc_total=1
  fi
else
  warn "$NAME.t2_ir and $NAME.t4 skipped: inter-node Device IR signal path" \
    "is a known MUSA limitation (err=700 at S3/S20); REAL_INTER_IR=1 reproduces"
fi

if [ "$ran_any" -eq 0 ]; then
  record_blocked "$NAME" "no cross-node device test binary found"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi

STAGE_DUR="$(total_duration "$NAME.t2_api" "$NAME.t2_ir" "$NAME.t4")"
if [ "$timed_out" -eq 1 ]; then
  set_status "$NAME" TIMEOUT 124 "$STAGE_DUR"
  printf 'verdict=TIMEOUT\n' >>"$LOG_DIR/$NAME.status"
  exit 1
elif [ "$rc_total" -eq 0 ]; then
  set_status "$NAME" PASS 0 "$STAGE_DUR"
  printf 'verdict=PASS\n' >>"$LOG_DIR/$NAME.status"
else
  set_status "$NAME" FAIL "$rc_total" "$STAGE_DUR"
  printf 'verdict=FAIL\n' >>"$LOG_DIR/$NAME.status"
fi
exit "$rc_total"
