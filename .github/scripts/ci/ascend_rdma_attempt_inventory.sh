#!/usr/bin/env bash
# Capture the container's RDMA view immediately after a failed MPI attempt.
set -uo pipefail

printf 'Ascend RDMA attempt: time=%s label=%s attempt=%s exit=%s\n' \
  "$(date -Is)" "${1:-unknown}" "${2:-unknown}" "${3:-unknown}"
for path in /dev/infiniband /sys/class/infiniband /sys/class/infiniband_verbs; do
  echo "Ascend RDMA attempt: $path"
  ls -l "$path" 2>&1 || true
done
if command -v rdma >/dev/null 2>&1; then
  echo 'Ascend RDMA attempt: rdma link show'
  timeout 5s rdma link show 2>&1 || true
fi
if command -v ibv_devices >/dev/null 2>&1; then
  echo 'Ascend RDMA attempt: ibv_devices'
  devices=$(timeout 10s env IBV_SHOW_WARNINGS=1 ibv_devices 2>&1)
  status=$?
  printf '%s\n' "$devices"
  if ((status == 0)) &&
      printf '%s\n' "$devices" | grep -Eq '^[[:space:]]*device[[:space:]]+node GUID'; then
    count=$(printf '%s\n' "$devices" | awk '
      /^[[:space:]]*device[[:space:]]+node GUID/ { header = 1; next }
      header && NF >= 2 && $1 !~ /^-+$/ { n++ }
      END { print n + 0 }
    ')
    if ((count == 0)); then
      echo 'Ascend RDMA attempt: no verbs devices; aborting retry before Socket fallback'
      exit 86
    fi
  fi
fi
if command -v python3 >/dev/null 2>&1; then
  echo 'Ascend RDMA attempt: direct ibv_get_device_list count'
  count_result=$(timeout 10s env IBV_SHOW_WARNINGS=1 python3 -c '
import ctypes
import sys
try:
    verbs = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    get_list = verbs.ibv_get_device_list
    get_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_list.restype = ctypes.c_void_p
    free_list = verbs.ibv_free_device_list
    free_list.argtypes = [ctypes.c_void_p]
    device_count = ctypes.c_int()
    device_list = get_list(ctypes.byref(device_count))
    if not device_list:
        raise OSError(ctypes.get_errno(), "ibv_get_device_list failed")
    print("verbs_device_count=%d" % device_count.value)
    free_list(device_list)
except (AttributeError, OSError) as error:
    print("verbs_device_count_error=%s" % error)
    sys.exit(2)
' 2>&1)
  printf '%s\n' "$count_result"
  if printf '%s\n' "$count_result" | grep -Eq '^verbs_device_count=0$'; then
    echo 'Ascend RDMA attempt: no verbs devices; aborting retry before Socket fallback'
    exit 86
  fi
fi
if command -v journalctl >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
  echo 'Ascend RDMA attempt: recent mlx5 and peer-memory kernel messages'
  timeout 5s journalctl -k --no-pager -n 300 2>/dev/null | \
    grep -Ei 'mlx5|rdma|uverbs|npu.peermem|pci.*reset' | tail -80 || true
fi
exit 0
